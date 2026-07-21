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


class OriginalSafetyTests(unittest.TestCase):
    def test_repository_originals_directory_is_protected(self) -> None:
        originals = (slide_pipeline.ROOT / "originals").resolve()
        self.assertIn(originals, slide_pipeline.ORIGINAL_DIRS)

        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory)
            with self.assertRaisesRegex(SystemExit, "Refusing to process originals"):
                slide_pipeline.guard_paths(
                    originals,
                    scratch / "output",
                    scratch / "work",
                )

    def test_symlink_to_repository_originals_is_protected(self) -> None:
        originals = (slide_pipeline.ROOT / "originals").resolve()
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory)
            linked_input = scratch / "input"
            linked_input.symlink_to(originals, target_is_directory=True)
            with self.assertRaisesRegex(SystemExit, "Refusing to process originals"):
                slide_pipeline.guard_paths(
                    linked_input,
                    scratch / "output",
                    scratch / "work",
                )


class CommandLineDefaultsTests(unittest.TestCase):
    def test_run_only_requires_input_and_output(self) -> None:
        arguments = slide_pipeline.parser().parse_args(
            ["run", "--input-dir", "input", "--output-dir", "output"]
        )
        self.assertEqual(arguments.profile, "archival-fp16")
        self.assertTrue(arguments.recursive)
        self.assertIsNone(arguments.work_dir)

    def test_recursion_can_be_disabled(self) -> None:
        arguments = slide_pipeline.parser().parse_args(
            [
                "run",
                "--input-dir",
                "input",
                "--output-dir",
                "output",
                "--no-recursive",
            ]
        )
        self.assertFalse(arguments.recursive)

    def test_default_work_directory_is_hidden_output_sibling(self) -> None:
        output = Path("/jobs/restored")
        self.assertEqual(
            slide_pipeline.default_work_dir(output), Path("/jobs/.restored-work")
        )


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
