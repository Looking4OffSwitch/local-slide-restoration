from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

import slide_pipeline


class ProfileTests(unittest.TestCase):
    def test_cuda_profiles_have_explicit_memory_settings(self) -> None:
        fp16 = slide_pipeline.seedvr_arguments(
            slide_pipeline.PROFILES["archival-fp16"], "cuda", None
        )
        fp8 = slide_pipeline.seedvr_arguments(
            slide_pipeline.PROFILES["balanced-fp8"], "cuda", None
        )
        self.assertEqual(fp16[fp16.index("--blocks_to_swap") + 1], "32")
        self.assertEqual(fp8[fp8.index("--blocks_to_swap") + 1], "16")
        for arguments in (fp16, fp8):
            self.assertIn("--dit_offload_device", arguments)
            self.assertIn("--vae_offload_device", arguments)
            self.assertIn("--tensor_offload_device", arguments)
            self.assertIn("--swap_io_components", arguments)

    def test_mps_profile_does_not_enable_cuda_only_options(self) -> None:
        arguments = slide_pipeline.seedvr_arguments(
            slide_pipeline.PROFILES["archival-fp16"], "mps", None
        )
        self.assertNotIn("--cuda_device", arguments)
        self.assertNotIn("--blocks_to_swap", arguments)
        self.assertNotIn("--swap_io_components", arguments)

    def test_zero_blockswap_override_disables_io_swapping(self) -> None:
        arguments = slide_pipeline.seedvr_arguments(
            slide_pipeline.PROFILES["balanced-fp8"], "cuda", 0
        )
        self.assertEqual(arguments[arguments.index("--blocks_to_swap") + 1], "0")
        self.assertNotIn("--swap_io_components", arguments)


class BenchmarkComparisonTests(unittest.TestCase):
    def test_identical_outputs_compare_as_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pixels = np.full((16, 16, 3), 127, dtype=np.uint8)
            fp16 = root / "fp16.png"
            fp8 = root / "fp8.png"
            Image.fromarray(pixels, "RGB").save(fp16)
            Image.fromarray(pixels, "RGB").save(fp8)
            comparison = slide_pipeline.compare_profile_outputs(fp16, fp8)
        self.assertEqual(comparison["mean_absolute_channel_difference"], 0.0)
        self.assertEqual(comparison["maximum_channel_difference"], 0.0)
        self.assertIsNone(comparison["psnr_db"])
        self.assertEqual(comparison["ssim"], 1.0)


if __name__ == "__main__":
    unittest.main()
