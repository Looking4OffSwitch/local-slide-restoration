import json
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import comfy_restore

ROOT = Path(__file__).resolve().parents[1]


class ComfyWorkflowTests(unittest.TestCase):
    def test_simple_parser_needs_only_an_input_image(self) -> None:
        args = comfy_restore.parser().parse_args(
            ["simple", "--input-image", "originals/manual/IMG_6219.jpeg"]
        )
        self.assertEqual(args.profile, "q4ks")
        self.assertEqual(args.engine, "fidelity")
        self.assertEqual(args.max_dimension, 1200)
        self.assertEqual(args.demoire, "auto")
        self.assertEqual(args.steps, 4)
        self.assertFalse(args.overwrite)

    def test_simple_generative_is_explicit_opt_in(self) -> None:
        args = comfy_restore.parser().parse_args(
            ["simple", "--input-image", "input.jpeg", "--generative"]
        )
        self.assertEqual(args.engine, "qwen")

    def test_simple_destination_uses_caller_directory_and_source_extension(self) -> None:
        destination = comfy_restore.simple_destination(
            Path("/archive/IMG_6219.jpeg"), Path("/caller")
        )
        self.assertEqual(destination, Path("/caller/IMG_6219_restored.jpeg"))

    def test_atomic_convert_writes_real_jpeg_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "generated.png"
            destination = root / "restored.jpeg"
            Image.new("RGB", (8, 6), "navy").save(source)

            comfy_restore.atomic_convert(source, destination)

            with Image.open(destination) as restored:
                self.assertEqual(restored.format, "JPEG")
                self.assertEqual(restored.size, (8, 6))

    def test_fidelity_restore_is_non_generative_and_limits_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            destination = root / "restored.png"
            Image.new("RGB", (120, 80), (120, 80, 60)).save(source)

            report = comfy_restore.fidelity_restore_image(
                source, destination, max_dimension=60, demoire="filter"
            )

            self.assertEqual(report["engine"], "fidelity")
            self.assertEqual(report["source_size"], [120, 80])
            self.assertEqual(report["output_size"], [60, 40])
            with Image.open(destination) as restored:
                self.assertEqual(restored.size, (60, 40))

    def test_resolve_python_preserves_virtualenv_launcher_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            comfy_root = root / "ComfyUI"
            launcher = comfy_root / ".venv" / "bin" / "python"
            launcher.parent.mkdir(parents=True)
            launcher.symlink_to(Path(sys.executable))

            resolved = comfy_restore.resolve_python(comfy_root, None)

            self.assertEqual(resolved, launcher.absolute())
            self.assertNotEqual(resolved, launcher.resolve())
            self.assertTrue(os.access(resolved, os.X_OK))

    def test_api_workflow_has_no_post_restoration_ai_upscaler(self) -> None:
        workflow = json.loads(comfy_restore.WORKFLOW.read_text(encoding="utf-8"))
        node_types = {node["class_type"] for node in workflow.values()}
        self.assertTrue(
            {
                "FluxKontextImageScale",
                "UnetLoaderGGUF",
                "LoraLoaderModelOnly",
                "TextEncodeQwenImageEditPlus",
                "KSampler",
                "SaveImage",
            }.issubset(node_types)
        )
        self.assertNotIn("ImageUpscaleWithModel", node_types)
        self.assertNotIn("UpscaleModelLoader", node_types)
        self.assertEqual(workflow["13"]["inputs"]["steps"], 4)
        self.assertEqual(workflow["13"]["inputs"]["cfg"], 1.0)
        self.assertEqual(workflow["13"]["inputs"]["denoise"], 1.0)

    def test_loadable_workflow_matches_prompt_and_has_valid_links(self) -> None:
        workflow_path = ROOT / "workflows" / "photo_restoration_qwen_2511.json"
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        prompt = comfy_restore.DEFAULT_PROMPT.read_text(encoding="utf-8").strip()
        nodes = {node["id"]: node for node in workflow["nodes"]}
        self.assertEqual(nodes[8]["widgets_values"], [prompt])
        self.assertEqual(nodes[5]["type"], "UnetLoaderGGUF")
        self.assertEqual(nodes[13]["widgets_values"][2:4], [4, 1.0])

        for link_id, origin, origin_slot, target, target_slot, link_type in workflow["links"]:
            self.assertIn(link_id, nodes[origin]["outputs"][origin_slot]["links"])
            self.assertEqual(nodes[target]["inputs"][target_slot]["link"], link_id)
            self.assertEqual(nodes[origin]["outputs"][origin_slot]["type"], link_type)
            self.assertEqual(nodes[target]["inputs"][target_slot]["type"], link_type)

    def test_model_names_match_api_workflow(self) -> None:
        workflow = json.loads(comfy_restore.WORKFLOW.read_text(encoding="utf-8"))
        names = {path.name for path in comfy_restore.MODEL_FILES.values()}
        self.assertIn(workflow["5"]["inputs"]["unet_name"], names)
        self.assertIn(workflow["4"]["inputs"]["clip_name"], names)
        self.assertIn(workflow["3"]["inputs"]["vae_name"], names)
        self.assertIn(workflow["18"]["inputs"]["lora_name"], names)

    def test_profiles_are_limited_to_the_pc_candidate_and_quality_reference(self) -> None:
        self.assertEqual(comfy_restore.PROFILE_NAMES, ("q4ks", "q4km"))
        q4ks = comfy_restore.profile_configuration("q4ks")[1]["diffusion model"]
        q4km = comfy_restore.profile_configuration("q4km")[1]["diffusion model"]
        self.assertEqual(q4ks.name, "qwen-image-edit-2511-Q4_K_S.gguf")
        self.assertEqual(q4km.name, "qwen-image-edit-2511-Q4_K_M.gguf")

    def test_batch_comfyui_uses_an_ephemeral_database(self) -> None:
        command = comfy_restore.build_batch_server_command(
            python=Path("/venv/python"),
            comfy_root=Path("/ComfyUI"),
            port=8189,
            input_dir=Path("/scratch/input"),
            output_dir=Path("/scratch/output"),
            temp_dir=Path("/scratch/temp"),
            user_dir=Path("/scratch/user"),
            extra_models=Path("/scratch/models.yaml"),
        )
        database_option = command.index("--database-url")
        self.assertEqual(command[database_option + 1], "sqlite:///:memory:")


class BatchSafetyTests(unittest.TestCase):
    def test_protected_originals_can_never_be_an_output(self) -> None:
        with self.assertRaises(SystemExit):
            comfy_restore.validate_destination(comfy_restore.PROTECTED_ORIGINALS / "bad.png")

    def test_batch_preserves_relative_paths_and_uses_one_server_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            output_root = (root / "output").resolve()
            (input_root / "nested").mkdir(parents=True)
            (input_root / "one.jpeg").write_bytes(b"one")
            (input_root / "nested" / "two.png").write_bytes(b"two")
            args = Namespace(
                input_dir=input_root,
                output_dir=output_root,
                prompt_file=comfy_restore.DEFAULT_PROMPT,
                seed=42,
                steps=4,
                port=8189,
                recursive=True,
                overwrite=False,
                fail_fast=False,
                profile="q4ks",
                engine="fidelity",
                max_dimension=1600,
                demoire="filter",
            )

            with patch.object(
                comfy_restore, "fidelity_restore_image", return_value={"engine": "fidelity"}
            ) as restore:
                comfy_restore.restore_batch(args, Path("/unused"), Path("/unused"))

            self.assertEqual(restore.call_count, 2)
            destinations = {
                Path(call.args[1]).relative_to(output_root) for call in restore.call_args_list
            }
            self.assertEqual(
                destinations,
                {Path("one_restored.png"), Path("nested/two_restored.png")},
            )
            manifest = json.loads(
                (output_root / "restoration_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(manifest["results"]), 2)

    def test_benchmark_runs_both_profiles_and_writes_a_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jpeg"
            source.write_bytes(b"source")
            output = root / "comparison"
            args = Namespace(
                input_image=source,
                output_dir=output,
                prompt_file=comfy_restore.DEFAULT_PROMPT,
                seed=42,
                steps=4,
                port=8189,
                overwrite=False,
            )

            def completed(profile_args, _comfy_root, _python, jobs, **_kwargs):
                job = jobs[0]
                job.destination.write_bytes(profile_args.profile.encode("ascii"))
                return [
                    {
                        "source": str(job.source),
                        "destination": str(job.destination),
                        "status": "completed",
                        "elapsed_seconds": 1.0,
                        "error": None,
                    }
                ]

            with (
                patch.object(
                    comfy_restore,
                    "validate_installation",
                    return_value={"status": "verified"},
                ),
                patch.object(comfy_restore, "execute_jobs", side_effect=completed) as execute,
            ):
                comfy_restore.benchmark(args, Path("/unused"), Path("/unused"))

            self.assertEqual(execute.call_count, 2)
            report = json.loads((output / "benchmark.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [result["profile"] for result in report["results"]],
                ["q4ks", "q4km"],
            )
            self.assertTrue(all(result["status"] == "completed" for result in report["results"]))


if __name__ == "__main__":
    unittest.main()
