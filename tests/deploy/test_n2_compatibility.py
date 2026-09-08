# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Five static version pairs on Kubernetes, sequentially using one GPU."""

import asyncio
import json
import os
import re
import uuid
from pathlib import Path

import pytest

from scripts.compatibility.kubernetes import manifest, resolve_image
from scripts.compatibility.runner import check, command, matrix, probe, wait_ready
from tests.deploy.dgd_utils import DeploymentSpec, ManagedDeployment
from tests.utils.test_output import resolve_test_output_path

ROOT = Path(__file__).resolve().parents[2]
VERSIONS = (
    "import importlib.metadata as m,json; print(json.dumps({"
    'd.metadata["Name"]:d.version for d in m.distributions() '
    'if d.metadata["Name"] in ("ai-dynamo", "ai-dynamo-runtime", "sglang")}))'
)


@pytest.fixture(scope="module")
def compatibility_plan(request, tmp_path_factory):
    check(not hasattr(request.config, "workerinput"), "Run the N-2 suite with -n 0")
    frontend = request.config.getoption("--frontend-image")
    worker = request.config.getoption("--image")
    check(bool(frontend and worker), "Pass --frontend-image and --image")
    config = json.loads((ROOT / "scripts/compatibility/releases.json").read_text())
    line = (
        os.environ.get("N2_RELEASE_LINE")
        or re.search(
            r'^version = "(\d+\.\d+)\.', (ROOT / "Cargo.toml").read_text(), re.M
        )[1]
    )
    pairs = matrix(config["releases"], line, frontend, worker)
    images = {
        ref: resolve_image(ref)
        for ref in sorted({ref for _, fe, wk in pairs for ref in (fe, wk)})
    }
    namespace = request.config.getoption("--namespace") or "default"
    # The workflow owns this vCluster. The quota prevents a leaked terminating
    # worker from allowing another GPU allocation in the next parameter.
    quotas = json.loads(
        command("kubectl", "-n", namespace, "get", "resourcequota", "-o", "json")
    )
    check(not quotas["items"], "Use a dedicated namespace without existing quotas")
    path = tmp_path_factory.mktemp("n2-quota") / "quota.json"
    name = "n2-" + uuid.uuid4().hex[:10]
    path.write_text(
        json.dumps(
            {
                "apiVersion": "v1",
                "kind": "ResourceQuota",
                "metadata": {"name": name},
                "spec": {"hard": {"requests.nvidia.com/gpu": "1"}},
            }
        )
    )
    output = Path(resolve_test_output_path("n2-compatibility"))
    output.mkdir(parents=True, exist_ok=True)
    plan = {
        "line": line,
        "pairs": pairs,
        "images": images,
        "models": config["models"],
        "namespace": namespace,
        "client_image": images[worker]["pinned"],
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2))
    try:
        command("kubectl", "-n", namespace, "create", "-f", str(path))
        yield plan
    finally:
        command(
            "kubectl",
            "-n",
            namespace,
            "delete",
            "resourcequota",
            name,
            "--ignore-not-found",
        )


@pytest.mark.k8s
@pytest.mark.deploy
@pytest.mark.sglang
@pytest.mark.framework_agnostic
@pytest.mark.core
# The dedicated PR/nightly workflow selects this file with -m k8s. Keep it
# outside the ordinary pre_merge GPU lane, which has no cluster or image inputs.
@pytest.mark.post_merge
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "pair_index",
    range(5),
    ids=[
        "candidate",
        "old-frontend-1",
        "old-worker-1",
        "old-frontend-2",
        "old-worker-2",
    ],
)
@pytest.mark.parametrize("scenario", ["embedding", "chat"])
async def test_n2_compatibility(compatibility_plan, tmp_path, pair_index, scenario):
    plan = compatibility_plan
    pair, frontend, worker = plan["pairs"][pair_index]
    namespace = plan["namespace"]
    name = "n2-" + uuid.uuid4().hex[:10]
    label = "nvidia.com/dynamo-graph-deployment-name=" + name
    output = Path(resolve_test_output_path(f"n2-compatibility/{pair}/{scenario}"))
    output.mkdir(parents=True, exist_ok=True)
    images = {"frontend": plan["images"][frontend], "worker": plan["images"][worker]}
    model = plan["models"][scenario]
    source = tmp_path / "dgd.json"
    source.write_text(
        json.dumps(manifest(name, images, scenario, model, plan["client_image"]))
    )
    (output / "dgd.json").write_text(source.read_text())
    report = {
        "pair": pair,
        "scenario": scenario,
        "images": images,
        "model": model,
        "status": "failed",
    }

    async def kubectl(*args, timeout=120):
        return await asyncio.to_thread(
            command, "kubectl", "-n", namespace, *args, timeout=timeout
        )

    async def collect():
        # Always collect before ManagedDeployment removes Pods, including when
        # inference assertions fail. Never let a diagnostics error hide them.
        for resource in ("pods", "events"):
            if (output / f"{resource}.json").exists():
                continue
            try:
                data = await kubectl(
                    "get",
                    resource,
                    "-o",
                    "json",
                    *(["-l", label] if resource == "pods" else []),
                )
                (output / f"{resource}.json").write_text(data)
            except Exception as error:
                (output / f"{resource}-error.txt").write_text(str(error))

    try:
        async with ManagedDeployment(
            str(output),
            DeploymentSpec(str(source)),
            namespace,
            skip_service_restart=True,
            readiness_timeout=1200,
        ) as deployment:
            try:
                pods = await asyncio.to_thread(deployment.get_pods)
                for role in ("Frontend", "decode"):
                    check(len(pods[role]) == 1, f"Expected one {role} Pod")
                    version = await kubectl(
                        "exec",
                        pods[role][0].name,
                        "-c",
                        "main",
                        "--",
                        "python3",
                        "-c",
                        VERSIONS,
                    )
                    (output / f"{role}-versions.json").write_text(version)
                pf = await asyncio.to_thread(
                    deployment.port_forward, pods["Frontend"][0], 8000
                )
                check(pf is not None, "Frontend port-forward failed")
                base = f"http://127.0.0.1:{pf.local_port}"
                await asyncio.to_thread(
                    wait_ready, base, model["id"], lambda: None, timeout=60
                )
                report["cases"] = await asyncio.to_thread(
                    probe, base, scenario, model, output
                )
                check(
                    all(case["status"] == "passed" for case in report["cases"]),
                    report["cases"],
                )
            finally:
                await collect()
        report["status"] = "passed"
    except Exception as error:
        report["error"] = str(error)
        await collect()
        raise
    finally:
        try:
            # Cover failed __aenter__ and partial setup. DGD deletion is
            # asynchronous: wait for its Pods before releasing the next pair.
            await kubectl(
                "delete",
                "dynamographdeployment",
                name,
                "--ignore-not-found",
                "--wait=true",
                "--timeout=300s",
                timeout=330,
            )
            await kubectl(
                "wait",
                "--for=delete",
                "pod",
                "-l",
                label,
                "--timeout=300s",
                timeout=330,
            )
        except Exception as error:
            report["status"] = "failed"
            report["cleanup_error"] = str(error)
            raise
        finally:
            (output / "report.json").write_text(json.dumps(report, indent=2))
