# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Five static version pairs on Kubernetes, sequentially using one GPU."""

import asyncio
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

import pytest

from scripts.compatibility.kube_manifest import manifest, resolve_image
from scripts.compatibility.runner import check, command, matrix, probe, wait_ready
from tests.deploy.dgd_utils import DeploymentSpec, ManagedDeployment
from tests.deploy.n2_utils import cleanup_step, error_details, phase, prepared_cache
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
    output = Path(resolve_test_output_path("n2-compatibility"))
    output.mkdir(parents=True, exist_ok=True)
    setup_report = {"status": "failed"}
    try:
        with phase(setup_report, "resolve_images"):
            images = {
                ref: resolve_image(ref)
                for ref in sorted({ref for _, fe, wk in pairs for ref in (fe, wk)})
            }
    except Exception as error:
        setup_report["primary_error"] = error_details(error, "resolve_images")
        (output / "setup-report.json").write_text(json.dumps(setup_report, indent=2))
        raise
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
        with prepared_cache(
            namespace,
            name + "-cache",
            plan["client_image"],
            plan["models"],
            request.config.getoption("--model-cache-pvc"),
            output,
            setup_report,
        ) as cache:
            plan["cache"] = cache
            (output / "plan.json").write_text(json.dumps(plan, indent=2))
            setup_report["status"] = "passed"
            yield plan
    except Exception as error:
        setup_report["status"] = "failed"
        setup_report["error"] = str(error)
        raise
    finally:
        primary = sys.exception()
        errors = []
        if primary is not None:
            setup_report["primary_error"] = error_details(primary, "setup or teardown")
        with cleanup_step(errors, setup_report, "delete quota"):
            command(
                "kubectl",
                "-n",
                namespace,
                "delete",
                "resourcequota",
                name,
                "--ignore-not-found",
            )
        if errors:
            setup_report["status"] = "failed"
        with cleanup_step(errors, setup_report, "write setup report"):
            (output / "setup-report.json").write_text(
                json.dumps(setup_report, indent=2)
            )
        if errors and primary is None:
            raise ExceptionGroup("Setup cleanup failed", errors)


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
@pytest.mark.timeout(2700)  # First baseline includes one-time model preparation.
@pytest.mark.parametrize("scenario", ["embedding", "chat"])
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
async def test_n2_compatibility(compatibility_plan, tmp_path, pair_index, scenario):
    plan = compatibility_plan
    if plan.get("baseline_failed"):
        pytest.skip(
            "Candidate baseline failed; remaining compatibility matrix not validated. "
            + plan["baseline_failed"]
        )
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
        json.dumps(
            manifest(
                name, images, scenario, model, plan["client_image"], plan.get("cache")
            )
        )
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
                if resource == "pods":
                    report["pod_timeline"] = [
                        {
                            "name": pod["metadata"]["name"],
                            "created": pod["metadata"].get("creationTimestamp"),
                            "conditions": pod.get("status", {}).get("conditions", []),
                            "containers": pod.get("status", {}).get(
                                "containerStatuses", []
                            ),
                            "init_containers": pod.get("status", {}).get(
                                "initContainerStatuses", []
                            ),
                        }
                        for pod in json.loads(data)["items"]
                    ]
            except Exception as error:
                (output / f"{resource}-error.txt").write_text(str(error))

    startup_clock = time.monotonic()
    print(f"N-2 starting {pair}/{scenario}", flush=True)
    managed = ManagedDeployment(
        str(output),
        DeploymentSpec(str(source)),
        namespace,
        skip_service_restart=True,
        readiness_timeout=1200,
        fail_fast_startup=True,
    )
    try:
        async with managed as deployment:
            report.setdefault("timings_seconds", {})["deployment_ready"] = round(
                time.monotonic() - startup_clock, 3
            )
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
                with phase(report, "requests"):
                    report["cases"] = await asyncio.to_thread(
                        probe, base, scenario, model, output
                    )
                check(
                    all(case["status"] == "passed" for case in report["cases"]),
                    report["cases"],
                )
            finally:
                serving_finished = time.monotonic()
                await collect()
        report["status"] = "passed"
    except Exception as error:
        report["error"] = str(error)
        if error not in managed.cleanup_errors:
            report["primary_error"] = error_details(error, "deployment or requests")
        report.setdefault("timings_seconds", {}).setdefault(
            "deployment_ready", round(time.monotonic() - startup_clock, 3)
        )
        await collect()
        if error not in managed.cleanup_errors:
            raise
    finally:
        primary = sys.exception()
        errors = list(managed.cleanup_errors)
        for error in errors:
            report.setdefault("cleanup_errors", []).append(
                error_details(error, "managed deployment cleanup")
            )
        cleanup_clock = locals().get("serving_finished", time.monotonic())
        with cleanup_step(errors, report, "delete DGD"):
            await kubectl(
                "delete",
                "dynamographdeployment",
                name,
                "--ignore-not-found",
                "--wait=true",
                "--timeout=300s",
                timeout=330,
            )
        with cleanup_step(errors, report, "wait for Pod deletion"):
            await kubectl(
                "wait",
                "--for=delete",
                "pod",
                "-l",
                label,
                "--timeout=300s",
                timeout=330,
            )
        if primary is not None or errors:
            report["status"] = "failed"
        report.setdefault("timings_seconds", {})["cleanup"] = round(
            time.monotonic() - cleanup_clock, 3
        )
        with cleanup_step(errors, report, "write report"):
            (output / "report.json").write_text(json.dumps(report, indent=2))
        if errors:
            report["status"] = "failed"
        if pair_index == 0 and report["status"] != "passed":
            plan["baseline_failed"] = str(output / "report.json")
        print(
            f"N-2 {pair}/{scenario}: {report['status']} {report['timings_seconds']}",
            flush=True,
        )
        if errors and primary is None:
            raise ExceptionGroup("Deployment cleanup failed", errors)
