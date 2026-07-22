#!/usr/bin/env python3
"""Build the loadable ComfyUI graph that mirrors the checked-in API workflow."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPT = (ROOT / "workflows" / "photo_restoration_prompt.txt").read_text(encoding="utf-8").strip()
OUTPUT = ROOT / "workflows" / "photo_restoration_qwen_2511.json"


links: list[list[object]] = []
next_link = 1


def node(
    node_id: int,
    node_type: str,
    pos: tuple[int, int],
    size: tuple[int, int],
    inputs: list[tuple[str, str, int | None]],
    outputs: list[tuple[str, str]],
    widgets: list[object],
    *,
    title: str | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "id": node_id,
        "type": node_type,
        "pos": list(pos),
        "size": list(size),
        "flags": {},
        "order": node_id - 1,
        "mode": 0,
        "inputs": [{"name": name, "type": kind, "link": link} for name, kind, link in inputs],
        "outputs": [{"name": name, "type": kind, "links": []} for name, kind in outputs],
        "properties": {"Node name for S&R": node_type},
        "widgets_values": widgets,
    }
    if title:
        value["title"] = title
    return value


nodes = [
    node(
        1,
        "LoadImage",
        (-940, 460),
        (310, 314),
        [("image", "COMBO", None), ("upload", "IMAGEUPLOAD", None)],
        [("IMAGE", "IMAGE"), ("MASK", "MASK")],
        ["original.jpeg", "image"],
    ),
    node(
        2,
        "FluxKontextImageScale",
        (-560, 460),
        (230, 46),
        [("image", "IMAGE", None)],
        [("IMAGE", "IMAGE")],
        [],
    ),
    node(
        3,
        "VAELoader",
        (-940, 280),
        (330, 58),
        [("vae_name", "COMBO", None)],
        [("VAE", "VAE")],
        ["qwen_image_vae.safetensors"],
    ),
    node(
        4,
        "CLIPLoader",
        (-940, 100),
        (330, 106),
        [("clip_name", "COMBO", None), ("type", "COMBO", None), ("device", "COMBO", None)],
        [("CLIP", "CLIP")],
        ["qwen_2.5_vl_7b_fp8_scaled.safetensors", "qwen_image", "default"],
    ),
    node(
        5,
        "UnetLoaderGGUF",
        (-940, -80),
        (330, 58),
        [("unet_name", "COMBO", None)],
        [("MODEL", "MODEL")],
        ["qwen-image-edit-2511-Q4_K_S.gguf"],
    ),
    node(
        18,
        "LoraLoaderModelOnly",
        (-560, -80),
        (340, 82),
        [("model", "MODEL", None), ("lora_name", "COMBO", None), ("strength_model", "FLOAT", None)],
        [("MODEL", "MODEL")],
        ["Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", 1.0],
    ),
    node(
        6,
        "ModelSamplingAuraFlow",
        (-150, -80),
        (250, 58),
        [("model", "MODEL", None), ("shift", "FLOAT", None)],
        [("MODEL", "MODEL")],
        [3.1],
    ),
    node(
        7,
        "CFGNorm",
        (160, -80),
        (250, 82),
        [("model", "MODEL", None), ("strength", "FLOAT", None), ("pre_cfg", "BOOLEAN", None)],
        [("patched_model", "MODEL")],
        [1.0, False],
    ),
    node(
        8,
        "TextEncodeQwenImageEditPlus",
        (-250, 100),
        (520, 350),
        [
            ("clip", "CLIP", None),
            ("vae", "VAE", None),
            ("image1", "IMAGE", None),
            ("image2", "IMAGE", None),
            ("image3", "IMAGE", None),
            ("prompt", "STRING", None),
        ],
        [("CONDITIONING", "CONDITIONING")],
        [PROMPT],
        title="Restoration instructions (positive)",
    ),
    node(
        9,
        "FluxKontextMultiReferenceLatentMethod",
        (340, 170),
        (330, 58),
        [("conditioning", "CONDITIONING", None), ("reference_latents_method", "COMBO", None)],
        [("CONDITIONING", "CONDITIONING")],
        ["index_timestep_zero"],
    ),
    node(
        10,
        "TextEncodeQwenImageEditPlus",
        (-250, 510),
        (520, 210),
        [
            ("clip", "CLIP", None),
            ("vae", "VAE", None),
            ("image1", "IMAGE", None),
            ("image2", "IMAGE", None),
            ("image3", "IMAGE", None),
            ("prompt", "STRING", None),
        ],
        [("CONDITIONING", "CONDITIONING")],
        [""],
        title="Reference conditioning (negative)",
    ),
    node(
        11,
        "FluxKontextMultiReferenceLatentMethod",
        (340, 560),
        (330, 58),
        [("conditioning", "CONDITIONING", None), ("reference_latents_method", "COMBO", None)],
        [("CONDITIONING", "CONDITIONING")],
        ["index_timestep_zero"],
    ),
    node(
        12,
        "VAEEncode",
        (340, 690),
        (210, 58),
        [("pixels", "IMAGE", None), ("vae", "VAE", None)],
        [("LATENT", "LATENT")],
        [],
    ),
    node(
        13,
        "KSampler",
        (730, 90),
        (280, 510),
        [
            ("model", "MODEL", None),
            ("positive", "CONDITIONING", None),
            ("negative", "CONDITIONING", None),
            ("latent_image", "LATENT", None),
            ("seed", "INT", None),
            ("steps", "INT", None),
            ("cfg", "FLOAT", None),
            ("sampler_name", "COMBO", None),
            ("scheduler", "COMBO", None),
            ("denoise", "FLOAT", None),
        ],
        [("LATENT", "LATENT")],
        [42, "fixed", 4, 1.0, "euler", "simple", 1.0],
    ),
    node(
        14,
        "VAEDecode",
        (1070, 90),
        (210, 58),
        [("samples", "LATENT", None), ("vae", "VAE", None)],
        [("IMAGE", "IMAGE")],
        [],
    ),
    node(
        17,
        "SaveImage",
        (1380, 90),
        (420, 470),
        [("images", "IMAGE", None), ("filename_prefix", "STRING", None)],
        [],
        ["photo_restoration/restored"],
    ),
]
by_id = {item["id"]: item for item in nodes}


def connect(origin: int, origin_slot: int, target: int, target_slot: int, kind: str) -> None:
    global next_link
    links.append([next_link, origin, origin_slot, target, target_slot, kind])
    by_id[origin]["outputs"][origin_slot]["links"].append(next_link)
    by_id[target]["inputs"][target_slot]["link"] = next_link
    next_link += 1


connect(1, 0, 2, 0, "IMAGE")
connect(5, 0, 18, 0, "MODEL")
connect(18, 0, 6, 0, "MODEL")
connect(6, 0, 7, 0, "MODEL")
connect(4, 0, 8, 0, "CLIP")
connect(3, 0, 8, 1, "VAE")
connect(2, 0, 8, 2, "IMAGE")
connect(8, 0, 9, 0, "CONDITIONING")
connect(4, 0, 10, 0, "CLIP")
connect(3, 0, 10, 1, "VAE")
connect(2, 0, 10, 2, "IMAGE")
connect(10, 0, 11, 0, "CONDITIONING")
connect(2, 0, 12, 0, "IMAGE")
connect(3, 0, 12, 1, "VAE")
connect(7, 0, 13, 0, "MODEL")
connect(9, 0, 13, 1, "CONDITIONING")
connect(11, 0, 13, 2, "CONDITIONING")
connect(12, 0, 13, 3, "LATENT")
connect(13, 0, 14, 0, "LATENT")
connect(3, 0, 14, 1, "VAE")
connect(14, 0, 17, 0, "IMAGE")

workflow = {
    "id": "ad7ea8f9-1d48-4ffc-bf31-2511aa097057",
    "revision": 0,
    "last_node_id": 18,
    "last_link_id": next_link - 1,
    "nodes": nodes,
    "links": links,
    "groups": [
        {
            "id": 1,
            "title": "Qwen Image Edit 2511 Q4_K_S + 4-step Lightning",
            "bounding": [-980, -120, 1420, 480],
            "color": "#3f789e",
            "font_size": 24,
            "flags": {},
        },
        {
            "id": 2,
            "title": "Faithful restoration prompt and source conditioning",
            "bounding": [-600, 60, 1310, 730],
            "color": "#3f9e68",
            "font_size": 24,
            "flags": {},
        },
        {
            "id": 3,
            "title": "Restoration and direct output",
            "bounding": [690, 50, 1150, 590],
            "color": "#9e783f",
            "font_size": 24,
            "flags": {},
        },
    ],
    "config": {},
    "extra": {"ds": {"scale": 0.72, "offset": [760, 360]}, "frontendVersion": "1.37.11"},
    "version": 0.4,
}
OUTPUT.write_text(json.dumps(workflow, indent=2) + "\n", encoding="utf-8")
print(OUTPUT)
