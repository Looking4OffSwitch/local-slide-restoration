#!/usr/bin/env python3
"""Run the repository's local ComfyUI photo-restoration workflows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
WORKFLOW = ROOT / "workflows" / "photo_restoration_qwen_2511_api.json"
DEFAULT_PROMPT = ROOT / "workflows" / "photo_restoration_prompt.txt"
MODEL_ROOT = ROOT / ".slide_pipeline" / "comfyui_models"
COMFYUI_COMMIT = "8b099de36acd81acd1afa3b5442951dc847e0a52"
GGUF_COMMIT = "6ea2651e7df66d7585f6ffee804b20e92fb38b8a"
SHARED_MODEL_FILES = {
    "text encoder": MODEL_ROOT / "text_encoders" / "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "VAE": MODEL_ROOT / "vae" / "qwen_image_vae.safetensors",
    "Lightning LoRA": MODEL_ROOT
    / "loras"
    / "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
}
SHARED_MODEL_SHA256 = {
    "text encoder": "cb5636d852a0ea6a9075ab1bef496c0db7aef13c02350571e388aea959c5c0b4",
    "VAE": "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f",
    "Lightning LoRA": "22226e8d05d354bb356627d428809f5afd7819399b077238a2b70a82883a904f",
}
QWEN_Q4KS_MODEL_FILES = {
    **SHARED_MODEL_FILES,
    "diffusion model": MODEL_ROOT / "diffusion_models" / "qwen-image-edit-2511-Q4_K_S.gguf",
}
QWEN_Q4KS_MODEL_SHA256 = {
    **SHARED_MODEL_SHA256,
    "diffusion model": "df952ef0d2b46463bd95d9afbb78e045ec5412316f453a7ad5a3d7bcbb111b72",
}
QWEN_Q4KM_MODEL_FILES = {
    **SHARED_MODEL_FILES,
    "diffusion model": MODEL_ROOT / "diffusion_models" / "qwen-image-edit-2511-Q4_K_M.gguf",
}
QWEN_Q4KM_MODEL_SHA256 = {
    **SHARED_MODEL_SHA256,
    "diffusion model": "8677bac90627adbbc11efab87b1870e701c4eb3689ee865a3de8ab81b705a723",
}
MODEL_FILES = QWEN_Q4KS_MODEL_FILES
MODEL_SHA256 = QWEN_Q4KS_MODEL_SHA256
PROFILE_NAMES = ("q4ks", "q4km")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
PROTECTED_ORIGINALS = (ROOT / "originals").resolve()


@dataclass(frozen=True)
class RestorationJob:
    source: Path
    destination: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def candidate_comfy_roots() -> list[Path]:
    candidates: list[Path] = []
    configured = os.environ.get("COMFYUI_DIR")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            ROOT / ".slide_pipeline" / "ComfyUI",
            Path.home() / "ComfyUI",
            Path.home() / "dev" / "ComfyUI",
            Path.home()
            / "dev"
            / "3rd_party"
            / "ComfyUI-Easy-Install"
            / "ComfyUI-Easy-Install"
            / "ComfyUI",
        ]
    )
    return candidates


def resolve_comfy_root(configured: Path | None) -> Path:
    candidates = [configured] if configured is not None else candidate_comfy_roots()
    for candidate in candidates:
        if candidate is not None:
            resolved = candidate.expanduser().resolve()
            if (resolved / "main.py").is_file():
                return resolved
    rendered = "\n".join(f"  - {path}" for path in candidates if path is not None)
    raise SystemExit(
        "ComfyUI was not found. Set COMFYUI_DIR or pass --comfyui-dir. Checked:\n" + rendered
    )


def resolve_python(comfy_root: Path, configured: Path | None) -> Path:
    if configured is not None:
        candidates = [configured]
    else:
        sibling = comfy_root.parent
        candidates = [
            sibling / "python_embeded" / "python",
            sibling / "python_embeded" / "python.exe",
            comfy_root / ".venv" / "bin" / "python",
            comfy_root / "venv" / "bin" / "python",
            comfy_root / "venv" / "Scripts" / "python.exe",
        ]
    for candidate in candidates:
        # Do not resolve symlinks here. Virtual-environment launchers are commonly
        # symlinks to a base interpreter, and dereferencing one makes Python lose
        # the venv's pyvenv.cfg and site-packages when ComfyUI is started.
        executable = candidate.expanduser().absolute()
        if executable.is_file() and os.access(executable, os.X_OK):
            return executable
    if configured is None and Path(sys.executable).is_file():
        return Path(sys.executable)
    raise SystemExit("A working ComfyUI Python executable was not found; pass --comfyui-python.")


def profile_configuration(profile: str) -> tuple[Path, dict[str, Path], dict[str, str], str, str]:
    if profile == "q4ks":
        return WORKFLOW, QWEN_Q4KS_MODEL_FILES, QWEN_Q4KS_MODEL_SHA256, "1", "17"
    if profile == "q4km":
        return WORKFLOW, QWEN_Q4KM_MODEL_FILES, QWEN_Q4KM_MODEL_SHA256, "1", "17"
    raise ValueError(f"Unknown restoration profile: {profile}")


def validate_installation(
    comfy_root: Path, python: Path, profile: str = "q4ks", *, emit: bool = True
) -> dict[str, Any]:
    problems: list[str] = []
    workflow_path, model_files, model_sha256, _, _ = profile_configuration(profile)
    if not workflow_path.is_file() or not DEFAULT_PROMPT.is_file():
        problems.append("repository workflow files are missing")
    gguf = comfy_root / "custom_nodes" / "ComfyUI-GGUF"
    if not (gguf / "nodes.py").is_file():
        problems.append(f"ComfyUI-GGUF is missing: {gguf}")
    model_report: dict[str, dict[str, Any]] = {}
    for label, path in model_files.items():
        if not path.is_file() or path.stat().st_size == 0:
            problems.append(f"{label} is missing: {path}")
            continue
        actual_sha256 = sha256_file(path)
        expected_sha256 = model_sha256[label]
        if actual_sha256 != expected_sha256:
            problems.append(
                f"{label} checksum mismatch: {path}\n"
                f"  expected {expected_sha256}\n"
                f"  actual   {actual_sha256}"
            )
        model_report[label] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual_sha256,
        }
    if problems:
        raise SystemExit("ComfyUI restoration setup is incomplete:\n- " + "\n- ".join(problems))
    report = {
        "comfyui": str(comfy_root),
        "comfyui_commit": git_head(comfy_root),
        "gguf_commit": git_head(gguf),
        "python": str(python),
        "platform": platform.platform(),
        "profile": profile,
        "models": model_report,
    }
    if emit:
        print(json.dumps(report, indent=2))
    return report


def bazzite_cuda_report(comfy_root: Path, python: Path) -> dict[str, Any]:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SystemExit("The Bazzite CUDA check requires x86-64 Linux.")
    os_release = Path("/etc/os-release")
    fields: dict[str, str] = {}
    if os_release.is_file():
        for line in os_release.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                fields[key] = value.strip().strip('"')
    if fields.get("ID") != "bazzite":
        raise SystemExit(f"Expected Bazzite; /etc/os-release reports ID={fields.get('ID')!r}.")
    if shutil.which("nvidia-smi") is None:
        raise SystemExit("nvidia-smi is unavailable; use the Bazzite NVIDIA image.")
    nvidia = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    first_gpu = [item.strip() for item in nvidia.splitlines()[0].split(",")]
    if len(first_gpu) != 3:
        raise SystemExit(f"Could not parse nvidia-smi output: {nvidia}")
    gpu_name, memory_mib_text, driver = first_gpu
    memory_mib = int(memory_mib_text)
    if "RTX 4080" not in gpu_name or not 11264 <= memory_mib <= 13312:
        raise SystemExit(
            f"Expected the 12 GB RTX 4080 target; detected {gpu_name} with {memory_mib} MiB."
        )
    probe = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import json, torch; "
                "print(json.dumps({'torch': torch.__version__, "
                "'cuda_available': torch.cuda.is_available(), "
                "'cuda_version': torch.version.cuda, "
                "'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    torch_report = json.loads(probe.stdout.strip())
    if not torch_report["cuda_available"]:
        raise SystemExit("ComfyUI's Python environment cannot access CUDA.")
    comfyui_commit = git_head(comfy_root)
    gguf_commit = git_head(comfy_root / "custom_nodes" / "ComfyUI-GGUF")
    if comfyui_commit != COMFYUI_COMMIT:
        raise SystemExit(
            f"ComfyUI commit mismatch: expected {COMFYUI_COMMIT}, found {comfyui_commit}."
        )
    if gguf_commit != GGUF_COMMIT:
        raise SystemExit(
            f"ComfyUI-GGUF commit mismatch: expected {GGUF_COMMIT}, found {gguf_commit}."
        )
    return {
        "os": fields.get("PRETTY_NAME", fields.get("ID")),
        "gpu": gpu_name,
        "vram_mib": memory_mib,
        "nvidia_driver": driver,
        **torch_report,
    }


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI rejected the request ({error.code}): {body}") from error


def get_json(url: str, timeout: float = 30) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def stream_process_output(process: subprocess.Popen[str]) -> None:
    if process.stdout is None:
        return
    for line in process.stdout:
        print(f"[ComfyUI] {line}", end="", flush=True)


def wait_for_server(base_url: str, process: subprocess.Popen[str], timeout: float = 180) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"ComfyUI exited during startup with code {process.returncode}")
        try:
            get_json(f"{base_url}/system_stats", timeout=3)
            return
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            time.sleep(1)
    raise TimeoutError(f"ComfyUI did not start within {timeout:.0f} seconds")


def wait_for_result(
    base_url: str, prompt_id: str, process: subprocess.Popen[str]
) -> dict[str, Any]:
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"ComfyUI exited during restoration with code {process.returncode}")
        history = get_json(f"{base_url}/history/{prompt_id}")
        if prompt_id in history:
            record = history[prompt_id]
            status = record.get("status", {})
            if status.get("status_str") == "error":
                messages = status.get("messages", [])
                raise RuntimeError("ComfyUI workflow failed: " + json.dumps(messages, indent=2))
            return record
        time.sleep(2)


def write_extra_model_config(path: Path) -> None:
    root = str(MODEL_ROOT.resolve())
    path.write_text(
        "slide_restoration:\n"
        f"  base_path: {json.dumps(root)}\n"
        "  diffusion_models: diffusion_models\n"
        "  text_encoders: text_encoders\n"
        "  vae: vae\n"
        "  loras: loras\n",
        encoding="utf-8",
    )


def serve(args: argparse.Namespace, comfy_root: Path, python: Path) -> None:
    validate_installation(comfy_root, python, args.profile)
    extra_models = ROOT / ".slide_pipeline" / "comfy_extra_model_paths.yaml"
    extra_models.parent.mkdir(parents=True, exist_ok=True)
    write_extra_model_config(extra_models)
    command = [
        str(python),
        str(comfy_root / "main.py"),
        "--listen",
        args.listen,
        "--port",
        str(args.port),
        "--extra-model-paths-config",
        str(extra_models),
    ]
    environment = os.environ.copy()
    if sys.platform == "darwin":
        environment.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    print(f"Starting ComfyUI at http://{args.listen}:{args.port}")
    ui_workflow = ROOT / "workflows" / "photo_restoration_qwen_2511.json"
    print(f"Load workflow: {ui_workflow}")
    os.execve(str(python), command, environment)


def validate_destination(destination: Path, source: Path | None = None) -> None:
    if source is not None and destination == source:
        raise SystemExit("Output image must not replace the input image.")
    if destination == PROTECTED_ORIGINALS or destination.is_relative_to(PROTECTED_ORIGINALS):
        raise SystemExit(f"Refusing to write inside the protected originals archive: {destination}")


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as temporary_output:
        temporary_output_path = Path(temporary_output.name)
    try:
        shutil.copyfile(source, temporary_output_path)
        temporary_output_path.replace(destination)
    except BaseException:
        temporary_output_path.unlink(missing_ok=True)
        raise


def execute_job(
    *,
    base_url: str,
    process: subprocess.Popen[str],
    workflow_template: dict[str, Any],
    input_dir: Path,
    output_dir: Path,
    job: RestorationJob,
    index: int,
    input_node: str,
    output_node: str,
) -> None:
    input_name = f"source-{index:06d}-{uuid.uuid4().hex}{job.source.suffix.lower()}"
    staged_input = input_dir / input_name
    shutil.copy2(job.source, staged_input)
    workflow = json.loads(json.dumps(workflow_template))
    workflow[input_node]["inputs"]["image"] = input_name
    try:
        requested_prompt_id = str(uuid.uuid4())
        response = post_json(
            f"{base_url}/prompt",
            {
                "prompt": workflow,
                "client_id": str(uuid.uuid4()),
                "prompt_id": requested_prompt_id,
            },
        )
        if response.get("error"):
            raise RuntimeError("ComfyUI rejected the workflow: " + json.dumps(response))
        prompt_id = response.get("prompt_id", requested_prompt_id)
        result = wait_for_result(base_url, prompt_id, process)
        images = result.get("outputs", {}).get(output_node, {}).get("images", [])
        if len(images) != 1:
            raise RuntimeError(f"Expected one restored image, received {len(images)}")
        image_record = images[0]
        generated = output_dir / image_record.get("subfolder", "") / image_record["filename"]
        if not generated.is_file():
            raise RuntimeError(f"ComfyUI output was not found: {generated}")
        atomic_copy(generated, job.destination)
        generated.unlink()
    finally:
        staged_input.unlink(missing_ok=True)


def build_batch_server_command(
    *,
    python: Path,
    comfy_root: Path,
    port: int,
    input_dir: Path,
    output_dir: Path,
    temp_dir: Path,
    user_dir: Path,
    extra_models: Path,
) -> list[str]:
    return [
        str(python),
        str(comfy_root / "main.py"),
        "--listen",
        "127.0.0.1",
        "--port",
        str(port),
        "--input-directory",
        str(input_dir),
        "--output-directory",
        str(output_dir),
        "--temp-directory",
        str(temp_dir),
        "--user-directory",
        str(user_dir),
        "--database-url",
        "sqlite:///:memory:",
        "--extra-model-paths-config",
        str(extra_models),
        "--disable-all-custom-nodes",
        "--whitelist-custom-nodes",
        "ComfyUI-GGUF",
    ]


def execute_jobs(
    args: argparse.Namespace,
    comfy_root: Path,
    python: Path,
    jobs: list[RestorationJob],
    *,
    validate: bool = True,
) -> list[dict[str, Any]]:
    if validate:
        validate_installation(comfy_root, python, args.profile)
    prompt_path = args.prompt_file.expanduser().resolve()
    if not prompt_path.is_file():
        raise SystemExit(f"Prompt file does not exist: {prompt_path}")
    workflow_path, _, _, input_node, output_node = profile_configuration(args.profile)
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    workflow["5"]["inputs"]["unet_name"] = profile_configuration(args.profile)[1][
        "diffusion model"
    ].name
    workflow["8"]["inputs"]["prompt"] = prompt
    workflow["13"]["inputs"]["seed"] = args.seed
    workflow["13"]["inputs"]["steps"] = args.steps
    results: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="comfy-slide-restoration-") as temporary:
        scratch = Path(temporary)
        input_dir = scratch / "input"
        output_dir = scratch / "output"
        temp_dir = scratch / "temp"
        user_dir = scratch / "user"
        for directory in (input_dir, output_dir, temp_dir, user_dir):
            directory.mkdir()
        extra_models = scratch / "extra_model_paths.yaml"
        write_extra_model_config(extra_models)
        base_url = f"http://127.0.0.1:{args.port}"
        command = build_batch_server_command(
            python=python,
            comfy_root=comfy_root,
            port=args.port,
            input_dir=input_dir,
            output_dir=output_dir,
            temp_dir=temp_dir,
            user_dir=user_dir,
            extra_models=extra_models,
        )
        environment = os.environ.copy()
        if sys.platform == "darwin":
            environment.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        process = subprocess.Popen(
            command,
            cwd=comfy_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output_thread = threading.Thread(target=stream_process_output, args=(process,), daemon=True)
        output_thread.start()
        try:
            wait_for_server(base_url, process)
            for index, job in enumerate(jobs, start=1):
                started = time.monotonic()
                print(f"[{index}/{len(jobs)}] Restoring {job.source}", flush=True)
                try:
                    execute_job(
                        base_url=base_url,
                        process=process,
                        workflow_template=workflow,
                        input_dir=input_dir,
                        output_dir=output_dir,
                        job=job,
                        index=index,
                        input_node=input_node,
                        output_node=output_node,
                    )
                    status = "completed"
                    error = None
                    print(f"[{index}/{len(jobs)}] Wrote {job.destination}", flush=True)
                except Exception as problem:
                    status = "failed"
                    error = str(problem)
                    print(f"[{index}/{len(jobs)}] FAILED: {error}", file=sys.stderr, flush=True)
                    if getattr(args, "fail_fast", False):
                        raise
                results.append(
                    {
                        "source": str(job.source),
                        "destination": str(job.destination),
                        "status": status,
                        "elapsed_seconds": round(time.monotonic() - started, 2),
                        "error": error,
                    }
                )
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            output_thread.join(timeout=2)
    return results


def restore(args: argparse.Namespace, comfy_root: Path, python: Path) -> Path:
    started = time.monotonic()
    source = args.input_image.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Input image does not exist: {source}")
    destination = args.output_image.expanduser().resolve()
    validate_destination(destination, source)
    if destination.exists() and not args.overwrite:
        raise SystemExit(f"Output exists; pass --overwrite after reviewing it: {destination}")
    results = execute_jobs(args, comfy_root, python, [RestorationJob(source, destination)])
    if results[0]["status"] != "completed":
        raise SystemExit(results[0]["error"])
    print(f"Restored image: {destination}")
    print(f"Elapsed seconds: {time.monotonic() - started:.1f}")
    return destination


def restore_batch(args: argparse.Namespace, comfy_root: Path, python: Path) -> None:
    input_root = args.input_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    if not input_root.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_root}")
    validate_destination(output_root)
    if output_root == input_root or output_root.is_relative_to(input_root):
        raise SystemExit("Batch output directory must be outside the input directory.")
    iterator = input_root.rglob("*") if args.recursive else input_root.glob("*")
    sources = sorted(
        path.resolve()
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not sources:
        raise SystemExit(f"No supported images found in: {input_root}")

    jobs: list[RestorationJob] = []
    results: list[dict[str, Any]] = []
    for source in sources:
        relative = source.relative_to(input_root)
        destination = output_root / relative.parent / f"{relative.stem}_restored.png"
        validate_destination(destination, source)
        if destination.exists() and not args.overwrite:
            results.append(
                {
                    "source": str(source),
                    "destination": str(destination),
                    "status": "skipped_existing",
                    "elapsed_seconds": 0.0,
                    "error": None,
                }
            )
        else:
            jobs.append(RestorationJob(source, destination))
    output_root.mkdir(parents=True, exist_ok=True)
    if jobs:
        results.extend(execute_jobs(args, comfy_root, python, jobs))
    manifest = {
        "workflow": str(profile_configuration(args.profile)[0]),
        "profile": args.profile,
        "prompt": str(args.prompt_file.expanduser().resolve()),
        "seed": args.seed,
        "steps": args.steps,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "results": results,
    }
    manifest_path = output_root / "restoration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    failures = [result for result in results if result["status"] == "failed"]
    completed = [result for result in results if result["status"] == "completed"]
    skipped = [result for result in results if result["status"] == "skipped_existing"]
    print(
        f"Batch complete: {len(completed)} restored, {len(skipped)} skipped, "
        f"{len(failures)} failed. Manifest: {manifest_path}"
    )
    if failures:
        raise SystemExit(1)


def benchmark(args: argparse.Namespace, comfy_root: Path, python: Path) -> None:
    source = args.input_image.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Input image does not exist: {source}")
    output_root = args.output_dir.expanduser().resolve()
    validate_destination(output_root, source)
    output_root.mkdir(parents=True, exist_ok=True)

    installation = {
        profile: validate_installation(comfy_root, python, profile, emit=False)
        for profile in PROFILE_NAMES
    }
    for profile in PROFILE_NAMES:
        destination = output_root / f"{source.stem}_{profile}_restored.png"
        validate_destination(destination, source)
        if destination.exists() and not args.overwrite:
            raise SystemExit(
                f"Benchmark output exists; pass --overwrite after reviewing it: {destination}"
            )

    results: list[dict[str, Any]] = []
    for profile in PROFILE_NAMES:
        destination = output_root / f"{source.stem}_{profile}_restored.png"
        profile_args = argparse.Namespace(**vars(args), profile=profile, fail_fast=False)
        started = time.monotonic()
        print(f"Benchmarking {profile} -> {destination}", flush=True)
        try:
            run_results = execute_jobs(
                profile_args,
                comfy_root,
                python,
                [RestorationJob(source, destination)],
                validate=False,
            )
            result = run_results[0]
        except Exception as problem:
            result = {
                "source": str(source),
                "destination": str(destination),
                "status": "failed",
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "error": str(problem),
            }
        if result["status"] == "completed":
            result["sha256"] = sha256_file(destination)
            result["bytes"] = destination.stat().st_size
        result["profile"] = profile
        result["model"] = profile_configuration(profile)[1]["diffusion model"].name
        result["total_elapsed_seconds"] = round(time.monotonic() - started, 2)
        results.append(result)

    report = {
        "source": str(source),
        "prompt": str(args.prompt_file.expanduser().resolve()),
        "seed": args.seed,
        "steps": args.steps,
        "installation": installation,
        "results": results,
    }
    report_path = output_root / "benchmark.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    failures = [result for result in results if result["status"] != "completed"]
    print(f"Benchmark report: {report_path}")
    if failures:
        raise SystemExit(1)


def add_runtime_paths(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--comfyui-dir", type=Path)
    subparser.add_argument("--comfyui-python", type=Path)


def add_generation_options(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    subparser.add_argument("--seed", type=int, default=42)
    subparser.add_argument("--steps", type=int, default=4)
    subparser.add_argument("--port", type=int, default=8189)
    subparser.add_argument("--overwrite", action="store_true")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor", help="verify ComfyUI, custom node, and model files")
    add_runtime_paths(doctor)
    doctor.add_argument("--profile", choices=PROFILE_NAMES, default="q4ks")
    doctor.add_argument("--all-profiles", action="store_true")
    doctor.add_argument("--require-bazzite-cuda", action="store_true")
    serve_parser = subparsers.add_parser(
        "serve", help="start ComfyUI with the restoration models available"
    )
    serve_parser.add_argument("--listen", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8188)
    serve_parser.add_argument("--profile", choices=PROFILE_NAMES, default="q4ks")
    add_runtime_paths(serve_parser)
    run = subparsers.add_parser("run", help="restore one slide photograph")
    add_runtime_paths(run)
    run.add_argument("--input-image", type=Path, required=True)
    run.add_argument("--output-image", type=Path, required=True)
    add_generation_options(run)
    run.add_argument("--profile", choices=PROFILE_NAMES, required=True)
    batch = subparsers.add_parser(
        "batch", help="restore a directory while keeping ComfyUI and models resident"
    )
    add_runtime_paths(batch)
    batch.add_argument("--input-dir", type=Path, required=True)
    batch.add_argument("--output-dir", type=Path, required=True)
    add_generation_options(batch)
    batch.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    batch.add_argument("--fail-fast", action="store_true")
    batch.add_argument("--profile", choices=PROFILE_NAMES, required=True)
    compare = subparsers.add_parser(
        "benchmark", help="restore one slide with both candidate Qwen quantizations"
    )
    add_runtime_paths(compare)
    compare.add_argument("--input-image", type=Path, required=True)
    compare.add_argument("--output-dir", type=Path, required=True)
    add_generation_options(compare)
    return root


def main() -> None:
    args = parser().parse_args()
    comfy_root = resolve_comfy_root(args.comfyui_dir)
    python = resolve_python(comfy_root, args.comfyui_python)
    if args.command == "doctor":
        profiles = PROFILE_NAMES if args.all_profiles else (args.profile,)
        report = {
            "profiles": [
                validate_installation(comfy_root, python, profile, emit=False)
                for profile in profiles
            ]
        }
        if args.require_bazzite_cuda:
            report["bazzite_cuda"] = bazzite_cuda_report(comfy_root, python)
        print(json.dumps(report, indent=2))
    elif args.command == "serve":
        serve(args, comfy_root, python)
    elif args.command == "batch":
        restore_batch(args, comfy_root, python)
    elif args.command == "benchmark":
        benchmark(args, comfy_root, python)
    else:
        restore(args, comfy_root, python)


if __name__ == "__main__":
    main()
