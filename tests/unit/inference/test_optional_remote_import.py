# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_inference_package_imports_without_zenoh() -> None:
    script = textwrap.dedent(
        """
        import builtins

        original_import = builtins.__import__

        def block_zenoh(name, *args, **kwargs):
            if name == "zenoh" or name.startswith("zenoh."):
                raise ModuleNotFoundError("simulated missing optional zenoh dependency")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = block_zenoh
        import physicalai.inference as inference

        assert inference.InferenceModel is not None
        assert "RemoteInferenceModel" not in inference.__all__
        assert "InferenceServer" not in inference.__all__
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
