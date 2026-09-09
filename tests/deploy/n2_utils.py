# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run-scoped model preparation and phase timing for the N-2 suite."""

import json
import sys
import time
import traceback
from contextlib import contextmanager

from scripts.compatibility.runner import check, command


def error_details(error, stage):
    return {
        "stage": stage,
        "type": type(error).__name__,
        "message": str(error),
        "traceback": "".join(traceback.format_exception(error)),
    }


@contextmanager
def cleanup_step(errors, report, stage):
    """Attempt one cleanup without preventing the remaining owned-resource cleanup."""
    try:
        yield
    except Exception as error:
        errors.append(error)
        report.setdefault("cleanup_errors", []).append(error_details(error, stage))
        print(f"N-2 cleanup failed ({stage}): {error}", flush=True)


@contextmanager
def phase(report, name):
    start = time.monotonic()
    print(f"N-2 phase started: {name}", flush=True)
    try:
        yield
    finally:
        elapsed = round(time.monotonic() - start, 3)
        report.setdefault("timings_seconds", {})[name] = elapsed
        print(f"N-2 phase finished: {name} ({elapsed}s)", flush=True)


@contextmanager
def prepared_cache(namespace, name, image, models, shared_pvc, directory, report):
    """Warm once; use shared NFS or a run-owned RWO volume on one GPU node."""

    def kubectl(*args, timeout=120):
        return command("kubectl", "-n", namespace, *args, timeout=timeout)

    claim = shared_pvc or name
    cache = {"pvc": claim}
    code = (
        "import time\nfrom huggingface_hub import snapshot_download\n"
        f"models = {models!r}\n"
        "for model in models.values():\n"
        "    started = time.monotonic()\n"
        "    print('Downloading ' + model['id'] + '@' + model['revision'], flush=True)\n"
        "    snapshot_download(model['id'], revision=model['revision'], cache_dir='/model/hub')\n"
        "    print('Model ready in %.3fs' % (time.monotonic() - started), flush=True)\n"
    )
    container = {
        "name": "prepare",
        "image": image,
        "command": ["python3", "-c", code],
        "envFrom": [{"secretRef": {"name": "hf-token-secret", "optional": True}}],
        "volumeMounts": [{"name": "model", "mountPath": "/model"}],
    }
    if not shared_pvc:
        # Schedule the RWO volume on a GPU-capable node. No inference runs in
        # this Pod; its single GPU reservation is released before any DGD starts.
        container["resources"] = {
            "limits": {"nvidia.com/gpu": "1"},
            "requests": {"nvidia.com/gpu": "1"},
        }
    pod = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": name},
        "spec": {
            "restartPolicy": "Never",
            "containers": [container],
            "volumes": [
                {"name": "model", "persistentVolumeClaim": {"claimName": claim}}
            ],
        },
    }
    resources = [pod]
    if not shared_pvc:
        resources.insert(
            0,
            {
                "apiVersion": "v1",
                "kind": "PersistentVolumeClaim",
                "metadata": {"name": claim},
                "spec": {
                    "accessModes": ["ReadWriteOnce"],
                    "resources": {"requests": {"storage": "10Gi"}},
                },
            },
        )
    source = directory / "cache.json"
    source.write_text(
        json.dumps({"apiVersion": "v1", "kind": "List", "items": resources})
    )
    try:
        with phase(report, "prepare_cache"):
            kubectl("create", "-f", str(source))
            deadline = time.monotonic() + 900
            previous = None
            while True:
                state = json.loads(kubectl("get", "pod", name, "-o", "json"))
                status = state.get("status", {})
                check(status.get("phase") != "Failed", "Model preparation failed")
                statuses = status.get("containerStatuses", [])
                progress = (
                    status.get("phase"),
                    [item.get("state") for item in statuses],
                )
                if progress != previous:
                    print(f"N-2 cache Pod: {progress}", flush=True)
                    previous = progress
                for item in statuses:
                    reason = item.get("state", {}).get("waiting", {}).get("reason")
                    check(
                        reason
                        not in (
                            "InvalidImageName",
                            "CreateContainerConfigError",
                            "ImagePullBackOff",
                        ),
                        f"Cache Pod cannot start: {reason}",
                    )
                if status.get("phase") == "Succeeded":
                    break
                check(
                    time.monotonic() < deadline,
                    "Cache preparation exceeded 900s; inspect cache Pod/PVC events",
                )
                time.sleep(2)
            if not shared_pvc:
                node = json.loads(
                    kubectl("get", "node", state["spec"]["nodeName"], "-o", "json")
                )
                cache["hostname"] = node["metadata"]["labels"]["kubernetes.io/hostname"]
            (directory / "cache-prepare.log").write_text(
                kubectl("logs", name, "-c", "prepare")
            )
            kubectl("delete", "pod", name, "--wait=true", "--timeout=120s", timeout=150)
        yield cache
    finally:
        primary = sys.exception()
        errors = []
        if primary is not None:
            report["primary_error"] = error_details(
                primary, "cache preparation or test body"
            )
        # Cache logs are also available from the failed Pod, before deletion.
        # A completed Pod's timestamps are retained below via its last snapshot.
        with cleanup_step(errors, report, "cache logs"):
            if not (directory / "cache-prepare.log").exists():
                (directory / "cache-prepare.log").write_text(
                    kubectl("logs", name, "-c", "prepare")
                )
        with cleanup_step(errors, report, "cache Pod snapshot"):
            if "state" in locals():
                (directory / "cache-pod.json").write_text(json.dumps(state, indent=2))
        with cleanup_step(errors, report, "cache events"):
            (directory / "cache-events.json").write_text(
                kubectl("get", "events", "-o", "json")
            )
        with cleanup_step(errors, report, "delete cache Pod"):
            kubectl(
                "delete",
                "pod",
                name,
                "--ignore-not-found",
                "--wait=true",
                "--timeout=120s",
                timeout=150,
            )
        if not shared_pvc:
            with cleanup_step(errors, report, "delete owned PVC"):
                kubectl(
                    "delete",
                    "pvc",
                    claim,
                    "--ignore-not-found",
                    "--wait=true",
                    "--timeout=120s",
                    timeout=150,
                )
        if primary is not None or errors:
            report["status"] = "failed"
        with cleanup_step(errors, report, "write cache report"):
            (directory / "cache-report.json").write_text(json.dumps(report, indent=2))
        if errors and primary is None:
            raise ExceptionGroup("Cache cleanup failed", errors)
