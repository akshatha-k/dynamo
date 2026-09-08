# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failure injection for N-2 deployment lifecycle without a cluster."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.compatibility.runner import ContractError
from tests.deploy import test_n2_compatibility as suite


@pytest.mark.pre_merge
@pytest.mark.sglang
@pytest.mark.core
@pytest.mark.framework_agnostic
@pytest.mark.unit
@pytest.mark.gpu_0
@pytest.mark.parametrize(
    "failure,pair_index",
    [("startup", 1), ("response", 1), ("cleanup", 1), ("response", 0)],
)
async def test_n2_failure_is_reported_and_teardown_attempted(
    monkeypatch, tmp_path, failure, pair_index
):
    output = tmp_path / "evidence"
    image = {"pinned": "registry/image@sha256:" + "a" * 64, "version": "1.5.0"}
    plan = {
        "pairs": [
            ("candidate", "frontend", "worker"),
            ("old-frontend-1", "frontend", "worker"),
        ],
        "namespace": "isolated",
        "images": {"frontend": image, "worker": image},
        "client_image": image["pinned"],
        "models": {
            "embedding": {"id": "test/model", "revision": "a" * 40, "dimensions": 2}
        },
    }
    deployment = SimpleNamespace(
        get_pods=Mock(
            return_value={
                "Frontend": [SimpleNamespace(name="fe")],
                "decode": [SimpleNamespace(name="wk")],
            }
        ),
        port_forward=Mock(return_value=SimpleNamespace(local_port=12345)),
    )
    context = SimpleNamespace()

    class Deployment:
        async def __aenter__(self):
            if failure == "startup":
                raise RuntimeError("startup failed")
            return deployment

        async def __aexit__(self, *args):
            context.exited = True

    def kubectl(*args, **kwargs):
        if failure == "cleanup" and "delete" in args:
            raise RuntimeError("cleanup failed")
        return '{"items": []}'

    commands = Mock(side_effect=kubectl)
    monkeypatch.setattr(suite, "ManagedDeployment", Mock(return_value=Deployment()))
    monkeypatch.setattr(suite, "command", commands)
    monkeypatch.setattr(suite, "resolve_test_output_path", lambda _: str(output))
    monkeypatch.setattr(suite, "wait_ready", Mock())
    monkeypatch.setattr(
        suite,
        "probe",
        Mock(
            return_value=[{"status": "failed" if failure == "response" else "passed"}]
        ),
    )
    expected = (
        pytest.exit.Exception if pair_index == 0 else (RuntimeError, ContractError)
    )
    with pytest.raises(expected):
        await suite.test_n2_compatibility(plan, tmp_path, pair_index, "embedding")
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed"
    assert any("delete" in call.args for call in commands.call_args_list)
    if failure == "cleanup":
        assert report["cleanup_error"] == "cleanup failed"
    if failure != "startup":
        assert context.exited


@pytest.mark.pre_merge
@pytest.mark.sglang
@pytest.mark.core
@pytest.mark.framework_agnostic
@pytest.mark.unit
@pytest.mark.gpu_0
@pytest.mark.parametrize(
    "reason,restarts,fatal",
    [
        ("CrashLoopBackOff", 2, True),
        ("CrashLoopBackOff", 1, False),
        ("ContainerCreating", 0, False),
        ("InvalidImageName", 0, True),
    ],
)
async def test_startup_failure_detection(
    monkeypatch, tmp_path, reason, restarts, fatal
):
    from unittest.mock import AsyncMock

    from tests.deploy.dgd_utils import (
        DeploymentStartupError,
        ManagedDeployment,
        PodStatusDetail,
    )

    deployment = ManagedDeployment(
        str(tmp_path),
        SimpleNamespace(name="test", api_version="v1beta1"),
        "test",
        fail_fast_startup=True,
    )
    # A successful status lets non-fatal cases finish without sleeping.
    deployment._custom_api = SimpleNamespace(
        get_namespaced_custom_object=AsyncMock(
            return_value={
                "status": {
                    "state": "successful",
                    "conditions": [{"type": "Ready", "status": "True"}],
                }
            }
        )
    )
    monkeypatch.setattr(
        deployment,
        "_get_pod_status_details",
        AsyncMock(
            return_value=[
                PodStatusDetail(
                    "pod", "main", "Waiting", reason, restart_count=restarts
                )
            ]
        ),
    )
    if fatal:
        with pytest.raises(DeploymentStartupError):
            await deployment._wait_for_ready(timeout=1)
    else:
        assert await deployment._wait_for_ready(timeout=1)


@pytest.mark.pre_merge
@pytest.mark.sglang
@pytest.mark.core
@pytest.mark.framework_agnostic
@pytest.mark.unit
@pytest.mark.gpu_0
@pytest.mark.parametrize(
    "shared,failed", [(True, False), (False, False), (False, True)]
)
def test_cache_ownership_and_gpu_release(monkeypatch, tmp_path, shared, failed):
    from scripts.compatibility.runner import ContractError
    from tests.deploy import n2_utils

    calls = []

    def kubectl(*args, **kwargs):
        calls.append(args)
        if "get" in args and "pod" in args:
            return json.dumps(
                {
                    "metadata": {"name": "cache"},
                    "spec": {"nodeName": "node"},
                    "status": {"phase": "Failed" if failed else "Succeeded"},
                }
            )
        if "get" in args and "node" in args:
            return json.dumps(
                {"metadata": {"labels": {"kubernetes.io/hostname": "gpu-node"}}}
            )
        return "{}"

    monkeypatch.setattr(n2_utils, "command", kubectl)
    report = {}
    context = n2_utils.prepared_cache(
        "test", "cache", "worker", {}, "shared" if shared else "", tmp_path, report
    )
    if failed:
        with pytest.raises(ContractError):
            with context:
                pytest.fail("Failed preparation must not allow inference")
    else:
        with context as cache:
            assert cache["pvc"] == ("shared" if shared else "cache")
            assert any("delete" in c and "pod" in c for c in calls)
            if not shared:
                assert cache["hostname"] == "gpu-node"
    assert any("delete" in c and "pvc" in c for c in calls) == (not shared)
    manifest = json.loads((tmp_path / "cache.json").read_text())
    pod = next(item for item in manifest["items"] if item["kind"] == "Pod")
    assert ("resources" in pod["spec"]["containers"][0]) == (not shared)
    assert "prepare_cache" in report["timings_seconds"]
