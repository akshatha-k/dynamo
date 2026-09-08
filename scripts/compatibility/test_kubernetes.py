# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.compatibility.kubernetes import manifest, resolve_image
from scripts.compatibility.runner import ContractError, matrix


class KubernetesCompatibilityTests(unittest.TestCase):
    def test_resolve_preserves_registry_port_and_runtime_version(self):
        reference = "registry:5000/dynamo:1.5.0-ci-abcdef-sglang-runtime"
        digest = "sha256:" + "a" * 64
        with patch("scripts.compatibility.kubernetes.command", return_value=digest):
            result = resolve_image(reference)
        self.assertEqual(result["pinned"], "registry:5000/dynamo@" + digest)
        self.assertEqual(result["version"], "1.5.0")

    def test_unknown_runtime_or_bad_digest_is_not_accepted(self):
        with self.assertRaises(ContractError):
            resolve_image("registry/dynamo:latest")
        with patch("scripts.compatibility.kubernetes.command", return_value="invalid"):
            with self.assertRaises(ContractError):
                resolve_image("registry/dynamo:1.4.2")

    def test_all_ten_deployments_keep_versions_isolated_and_use_one_gpu(self):
        config = json.loads(Path(__file__).with_name("releases.json").read_text())
        pairs = matrix(
            config["releases"], "1.5", "acr/frontend:1.5.0", "acr/worker:1.5.0"
        )
        for pair, frontend, worker in pairs:
            for scenario, model in config["models"].items():
                with self.subTest(pair=pair, scenario=scenario):
                    with patch(
                        "scripts.compatibility.kubernetes.command",
                        return_value="sha256:" + "b" * 64,
                    ):
                        images = {
                            "frontend": resolve_image(frontend),
                            "worker": resolve_image(worker),
                        }
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
                        container = pod["containers"][0]
                        self.assertEqual(container["image"], images[kind]["pinned"])
                        self.assertEqual(container["workingDir"], "/tmp")
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
                        "--embedding-worker" in worker_container["command"],
                        scenario == "embedding",
                    )
