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
@pytest.mark.unit
@pytest.mark.gpu_0
@pytest.mark.parametrize("failure", ["startup", "response", "cleanup"])
async def test_n2_failure_is_reported_and_teardown_attempted(
    monkeypatch, tmp_path, failure
):
    output = tmp_path / "evidence"
    image = {"pinned": "registry/image@sha256:" + "a" * 64, "version": "1.5.0"}
    plan = {
        "pairs": [("candidate", "frontend", "worker")],
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
    with pytest.raises((RuntimeError, ContractError)):
        await suite.test_n2_compatibility(plan, tmp_path, 0, "embedding")
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed"
    assert any("delete" in call.args for call in commands.call_args_list)
    if failure == "cleanup":
        assert report["cleanup_error"] == "cleanup failed"
    if failure != "startup":
        assert context.exited
