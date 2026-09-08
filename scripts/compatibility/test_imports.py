# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise import isolation in a fresh interpreter, as spawned workers do."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ImportIsolationTests(unittest.TestCase):
    def test_harness_does_not_shadow_kubernetes_or_lose_to_other_scripts_package(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            packages = Path(directory)
            # A dependency can ship a regular 'scripts' package. It must not
            # win over our checkout as it does against a namespace package.
            for name in ("scripts", "kubernetes"):
                (packages / name).mkdir()
                (packages / name / "__init__.py").touch()
            (packages / "kubernetes/client.py").write_text("sentinel = True\n")
            code = (
                "import sys; "
                f"sys.path[:0] = {[str(root / 'scripts/compatibility'), str(root), directory]!r}; "
                "import kubernetes.client; "
                "from scripts.compatibility.kube_manifest import manifest; "
                "assert kubernetes.client.sentinel; "
                "assert callable(manifest)"
            )
            subprocess.run([sys.executable, "-c", code], cwd=directory, check=True)
