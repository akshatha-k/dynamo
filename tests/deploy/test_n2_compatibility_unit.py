# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failure injection for N-2 deployment lifecycle without a cluster."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from scripts.compatibility.runner import ContractError
from tests.deploy import n2_utils
from tests.deploy import test_n2_compatibility as suite
from tests.deploy.dgd_utils import (
    DeploymentStartupError,
    ManagedDeployment,
    PodStatusDetail,
)


@pytest.mark.pre_merge
@pytest.mark.sglang
@pytest.mark.core
@pytest.mark.framework_agnostic
@pytest.mark.unit
@pytest.mark.gpu_0
@pytest.mark.parametrize(
    "failure,pair_index",
    [
        ("startup", 1),
        ("response", 1),
        ("cleanup", 1),
        ("response", 0),
        ("response+cleanup", 0),
        ("managed-cleanup", 1),
    ],
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
        cleanup_errors = []

        async def __aenter__(self):
            if failure == "startup":
                raise RuntimeError("startup failed")
            return deployment

        async def __aexit__(self, *args):
            context.exited = True
            if failure == "managed-cleanup":
                error = RuntimeError("managed cleanup failed")
                self.cleanup_errors.append(error)
                raise error

    def kubectl(*args, **kwargs):
        if "cleanup" in failure and "delete" in args:
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
            return_value=[{"status": "failed" if "response" in failure else "passed"}]
        ),
    )
    expected = (
        ExceptionGroup
        if failure in ("cleanup", "managed-cleanup")
        else (ContractError if "response" in failure else RuntimeError)
    )
    with pytest.raises(expected):
        await suite.test_n2_compatibility(plan, tmp_path, pair_index, "embedding")
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed"
    assert any("delete" in call.args for call in commands.call_args_list)
    assert any("wait" in call.args for call in commands.call_args_list)
    if "cleanup" in failure:
        assert any(e["message"] == "cleanup failed" for e in report["cleanup_errors"])
    if "response" in failure:
        assert report["primary_error"]["type"] == "ContractError"
        assert "ContractError" in report["primary_error"]["traceback"]
    if pair_index == 0:
        before = commands.call_count
        with pytest.raises(pytest.skip.Exception, match="not validated"):
            await suite.test_n2_compatibility(plan, tmp_path, 1, "embedding")
        assert commands.call_count == before
    if failure != "startup":
        assert context.exited


@pytest.mark.pre_merge
@pytest.mark.sglang
@pytest.mark.core
@pytest.mark.framework_agnostic
@pytest.mark.unit
@pytest.mark.gpu_0
@pytest.mark.parametrize(
    "reason,restarts,exit_code,fatal",
    [
        ("CrashLoopBackOff", 2, None, True),
        ("CrashLoopBackOff", 1, None, False),
        ("ContainerCreating", 0, None, False),
        ("InvalidImageName", 0, None, True),
        ("Completed", 2, 0, False),
        ("Error", 2, 1, True),
    ],
)
async def test_startup_failure_detection(
    monkeypatch, tmp_path, reason, restarts, exit_code, fatal
):
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
                    "pod",
                    "init" if exit_code is not None else "main",
                    "Terminated" if exit_code is not None else "Waiting",
                    reason,
                    exit_code=exit_code,
                    restart_count=restarts,
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
    "shared,failed,cleanup_failed",
    [
        (True, False, False),
        (False, False, False),
        (False, True, False),
        (False, True, True),
        (False, False, True),
        (True, True, True),
    ],
)
def test_cache_ownership_and_gpu_release(
    monkeypatch, tmp_path, shared, failed, cleanup_failed
):
    calls = []

    def kubectl(*args, **kwargs):
        calls.append(args)
        if (
            cleanup_failed
            and "delete" in args
            and "pod" in args
            and "--ignore-not-found" in args
        ):
            raise RuntimeError("cache Pod deletion failed")
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
    elif cleanup_failed:
        with pytest.raises(ExceptionGroup):
            with context:
                pass
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
    if cleanup_failed:
        assert any(e["stage"] == "delete cache Pod" for e in report["cleanup_errors"])
    if failed:
        assert report["primary_error"]["type"] == "ContractError"
    assert (tmp_path / "cache-report.json").exists()


@pytest.mark.pre_merge
@pytest.mark.sglang
@pytest.mark.core
@pytest.mark.framework_agnostic
@pytest.mark.unit
@pytest.mark.gpu_0
@pytest.mark.parametrize("startup", [True, False])
async def test_managed_cleanup_preserves_primary(monkeypatch, tmp_path, startup):
    deployment = ManagedDeployment(str(tmp_path), SimpleNamespace(name="test"), "test")
    primary = RuntimeError("original failure")
    cleanup = RuntimeError("cleanup failure")
    monkeypatch.setattr(deployment, "_cleanup", AsyncMock(side_effect=cleanup))
    if startup:
        monkeypatch.setattr(
            deployment, "_init_kubernetes", AsyncMock(side_effect=primary)
        )
        with pytest.raises(RuntimeError) as raised:
            await deployment.__aenter__()
        assert raised.value is primary
    else:
        await deployment.__aexit__(type(primary), primary, primary.__traceback__)
    assert deployment.cleanup_errors == [cleanup]
