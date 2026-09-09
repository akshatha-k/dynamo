# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.compatibility.kube_manifest import manifest, resolve_image
from scripts.compatibility.runner import ContractError


class KubernetesCompatibilityTests(unittest.TestCase):
    def test_resolve_preserves_registry_port_and_runtime_version(self):
        reference = "registry:5000/dynamo:1.5.0-ci-abcdef-sglang-runtime"
        digest = "sha256:" + "a" * 64
        with patch(
            "scripts.compatibility.kube_manifest.command", return_value=digest
        ) as inspect:
            result = resolve_image(reference)
        self.assertEqual(result["pinned"], "registry:5000/dynamo@" + digest)
        self.assertEqual(result["version"], "1.5.0")
        self.assertIn("--no-tags", inspect.call_args.args)

    def test_unknown_runtime_or_bad_digest_is_not_accepted(self):
        with self.assertRaises(ContractError):
            resolve_image("registry/dynamo:latest")
        with patch(
            "scripts.compatibility.kube_manifest.command", return_value="invalid"
        ):
            with self.assertRaises(ContractError):
                resolve_image("registry/dynamo:1.4.2")

    def test_transient_resolution_failure_retries_then_succeeds(self):
        digest = "sha256:" + "a" * 64
        with patch(
            "scripts.compatibility.kube_manifest.command",
            side_effect=[subprocess.TimeoutExpired("skopeo", 60), digest],
        ) as inspect, patch("scripts.compatibility.kube_manifest.time.sleep"):
            self.assertEqual(
                resolve_image("registry/image:1.4.2")["pinned"],
                "registry/image@" + digest,
            )
        self.assertEqual(inspect.call_count, 2)

    def test_resolution_failures_are_bounded_and_identify_image(self):
        reference = "registry/image:1.4.2"
        for message, attempts in [
            ("401 Unauthorized: connection reset", 1),
            ("403 Forbidden", 1),
            ("manifest unknown", 1),
            ("503 Service Unavailable", 3),
        ]:
            with self.subTest(message=message), patch(
                "scripts.compatibility.kube_manifest.command",
                side_effect=subprocess.CalledProcessError(1, "skopeo", output=message),
            ) as inspect, patch("scripts.compatibility.kube_manifest.time.sleep"):
                with self.assertRaises(ContractError) as raised:
                    resolve_image(reference)
                self.assertIn(reference, str(raised.exception))
                self.assertIn(message, str(raised.exception))
                self.assertEqual(inspect.call_count, attempts)

    def test_scenarios_keep_distinct_versions_and_use_one_gpu(self):
        config = json.loads(Path(__file__).with_name("releases.json").read_text())
        images = {
            "frontend": {
                "pinned": "acr/frontend@sha256:" + "a" * 64,
                "version": "1.5.0",
            },
            "worker": {"pinned": "ngc/worker@sha256:" + "b" * 64, "version": "1.3.1"},
        }
        for scenario, model in config["models"].items():
            with self.subTest(scenario=scenario):
                dgd = manifest(
                    "isolated", images, scenario, model, images["worker"]["pinned"]
                )
                fe, wk = dgd["spec"]["components"]
                for component, kind in [(fe, "frontend"), (wk, "worker")]:
                    self.assertEqual(component["replicas"], 1)
                    self.assertEqual(
                        component["runtimeVersionOverride"], images[kind]["version"]
                    )
                    pod = component["podTemplate"]["spec"]
                    self.assertEqual(pod["nodeSelector"]["kubernetes.io/arch"], "amd64")
                    container = pod["containers"][0]
                    self.assertEqual(container["image"], images[kind]["pinned"])
                    self.assertEqual(container["workingDir"], "/tmp")
                    self.assertEqual(container["command"], ["python3"])
                    self.assertEqual(
                        container["args"][:2],
                        [
                            "-m",
                            (
                                "dynamo.frontend"
                                if kind == "frontend"
                                else "dynamo.sglang"
                            ),
                        ],
                    )
                    self.assertNotIn(
                        "DYN_SYSTEM_PORT", {env["name"] for env in container["env"]}
                    )
                    self.assertIn(
                        model["revision"], pod["initContainers"][0]["command"][-1]
                    )
                    self.assertNotIn("resources", pod["initContainers"][0])
                    self.assertTrue(
                        all(
                            "configMap" not in volume and "hostPath" not in volume
                            for volume in pod["volumes"]
                        )
                    )
                self.assertNotIn(
                    "resources", fe["podTemplate"]["spec"]["containers"][0]
                )
                worker_container = wk["podTemplate"]["spec"]["containers"][0]
                self.assertEqual(
                    worker_container["resources"]["limits"]["nvidia.com/gpu"], "1"
                )
                self.assertEqual(
                    "--embedding-worker" in worker_container["args"],
                    scenario == "embedding",
                )

    def test_cached_manifest_reuses_snapshot_without_download_init(self):
        model = {"id": "org/model", "revision": "a" * 40}
        images = {
            role: {"pinned": role, "version": "1.5.0"}
            for role in ("frontend", "worker")
        }
        dgd = manifest(
            "test",
            images,
            "embedding",
            model,
            "client",
            {"pvc": "cache", "hostname": "gpu-node"},
        )
        self.assertEqual(
            dgd["metadata"]["annotations"]["nvidia.com/dynamo-discovery-backend"],
            "kubernetes",
        )
        for component in dgd["spec"]["components"]:
            pod = component["podTemplate"]["spec"]
            self.assertNotIn("initContainers", pod)
            self.assertEqual(
                pod["nodeSelector"],
                {"kubernetes.io/hostname": "gpu-node", "kubernetes.io/arch": "amd64"},
            )
            self.assertEqual(
                pod["volumes"][0]["persistentVolumeClaim"]["claimName"], "cache"
            )
            self.assertTrue(pod["containers"][0]["volumeMounts"][0]["readOnly"])
        worker_args = dgd["spec"]["components"][1]["podTemplate"]["spec"]["containers"][
            0
        ]["args"]
        self.assertEqual(
            worker_args[worker_args.index("--model-path") + 1],
            "/model/hub/models--org--model/snapshots/" + model["revision"],
        )
