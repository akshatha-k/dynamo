# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes manifests for the static N-2 matrix; no inference code is injected."""

import re
import subprocess
import time

from scripts.compatibility.runner import ContractError, check, command


def resolve_image(reference):
    """Pin registry content without a Docker daemon and retain runtime semantics."""
    version = re.search(r":(\d+\.\d+\.\d+)(?:[-@]|$)", reference)
    check(
        version is not None, "Image tag must start with a runtime version: " + reference
    )
    for attempt in range(1, 4):
        print(f"Resolving image digest: {reference} (attempt {attempt}/3)", flush=True)
        try:
            digest = command(
                "skopeo",
                "inspect",
                "--override-os",
                "linux",
                "--override-arch",
                "amd64",
                "--format",
                "{{.Digest}}",
                "docker://" + reference,
                timeout=60,
            ).strip()
            break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            detail = error.output or str(error)
            if isinstance(detail, bytes):
                detail = detail.decode(errors="replace")
            authentication_error = re.search(
                r"unauthorized|forbidden|denied|authentication required|\b(?:401|403)\b",
                detail,
                re.IGNORECASE,
            )
            transient = isinstance(error, subprocess.TimeoutExpired) or re.search(
                r"timeout|timed out|connection reset|connection refused|"
                r"temporary failure|TLS handshake timeout|\b(?:429|502|503|504)\b",
                detail,
                re.IGNORECASE,
            )
            if authentication_error or not transient or attempt == 3:
                raise ContractError(
                    f"Image digest resolution failed for {reference} "
                    f"on attempt {attempt}/3: {detail}"
                ) from error
            print(f"Image registry temporarily unavailable: {detail}", flush=True)
            time.sleep(2 * attempt)
    check(
        re.fullmatch(r"sha256:[0-9a-f]{64}", digest),
        f"Invalid image digest for {reference}: {digest}",
    )
    repository = reference.split("@", 1)[0].rsplit(":", 1)[0]
    return {
        "reference": reference,
        "pinned": repository + "@" + digest,
        "version": version[1],
    }


def manifest(name, images, scenario, model, client_image, cache=None):
    # One immutable model snapshot per Pod; no code from the checkout is
    # installed in either inference component. Init containers use no GPU.
    download = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download({model['id']!r}, revision={model['revision']!r}, local_dir='/model')"
    )
    components = []
    for role, kind, image in [
        ("Frontend", "frontend", images["frontend"]["pinned"]),
        ("decode", "worker", images["worker"]["pinned"]),
    ]:
        args = (
            ["python3", "-m", "dynamo.frontend", "--http-port", "8000"]
            if kind == "frontend"
            else [
                "python3",
                "-m",
                "dynamo.sglang",
                "--model-path",
                "/model",
                "--served-model-name",
                model["id"],
                "--tp",
                "1",
                "--mem-fraction-static",
                "0.65",
                "--enable-metrics",
            ]
        )
        if kind == "worker" and scenario == "embedding":
            args += [
                "--embedding-worker",
                "--use-sglang-tokenizer",
                "--page-size",
                "16",
            ]
        container = {
            "name": "main",
            "image": image,
            "command": args[:1],
            "args": args[1:],
            "workingDir": "/tmp",
            "env": [
                {"name": "HF_HUB_OFFLINE", "value": "1"},
                {"name": "TRANSFORMERS_OFFLINE", "value": "1"},
                {"name": "DYN_REQUEST_PLANE", "value": "tcp"},
            ],
            "volumeMounts": [
                {"name": "model", "mountPath": "/model"},
                {"name": "shared-memory", "mountPath": "/dev/shm"},
            ],
        }
        if kind == "worker":
            container["resources"] = {
                "requests": {"nvidia.com/gpu": "1"},
                "limits": {"nvidia.com/gpu": "1"},
            }
        components.append(
            {
                "name": role,
                "type": kind,
                "replicas": 1,
                "runtimeVersionOverride": images[kind]["version"],
                "podTemplate": {
                    "spec": {
                        "containers": [container],
                        "initContainers": [
                            {
                                "name": "model",
                                "image": client_image,
                                "command": ["python3", "-c", download],
                                "envFrom": [
                                    {
                                        "secretRef": {
                                            "name": "hf-token-secret",
                                            "optional": True,
                                        }
                                    }
                                ],
                                "volumeMounts": [
                                    {"name": "model", "mountPath": "/model"}
                                ],
                            }
                        ],
                        "volumes": [
                            {"name": "model", "emptyDir": {}},
                            {
                                "name": "shared-memory",
                                "emptyDir": {"medium": "Memory", "sizeLimit": "2Gi"},
                            },
                        ],
                    },
                },
            }
        )
    if cache:
        for component in components:
            pod = component["podTemplate"]["spec"]
            pod.pop("initContainers")
            pod["volumes"][0] = {
                "name": "model",
                "persistentVolumeClaim": {"claimName": cache["pvc"]},
            }
            container = pod["containers"][0]
            container["volumeMounts"][0]["readOnly"] = True
            path = (
                "/model/hub/models--"
                + model["id"].replace("/", "--")
                + "/snapshots/"
                + model["revision"]
            )
            container["args"] = [
                path if arg == "/model" else arg for arg in container["args"]
            ]
            if cache.get("hostname"):
                pod["nodeSelector"] = {"kubernetes.io/hostname": cache["hostname"]}
    return {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {
            "name": name,
            "annotations": {
                "nvidia.com/enable-grove": "false",
                "nvidia.com/dynamo-discovery-backend": "kubernetes",
            },
        },
        "spec": {"components": components},
    }
