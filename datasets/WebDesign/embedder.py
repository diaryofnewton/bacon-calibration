from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


MODEL_ID_DEFAULT = "google/siglip-base-patch16-224"

DOMAIN_CONFIG = {
    "unis": {
        "data_dir": Path("datasets/WebDesign/Universitites/data"),
        "results_dir": Path("datasets/WebDesign/Universitites/results"),
        "glob": "universities.part*/*.jpg",
    },
    "banks": {
        "data_dir": Path("datasets/WebDesign/Commercial Banks/data"),
        "results_dir": Path("datasets/WebDesign/Commercial Banks/results"),
        "glob": "banks.part*/*.jpg",
    },
    "fashion": {
        "data_dir": Path("datasets/WebDesign/eCommerce/data"),
        "results_dir": Path("datasets/WebDesign/eCommerce/results"),
        "glob": "fashion.part*/*.jpg",
    },
    "homeware": {
        "data_dir": Path("datasets/WebDesign/eCommerce/data"),
        "results_dir": Path("datasets/WebDesign/eCommerce/results"),
        "glob": "homeware.part*/*.jpg",
    },
}


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace("-", "_")


def discover_images(data_dir: Path, glob_expr: str) -> list[Path]:
    files = sorted(data_dir.glob(glob_expr))
    return [p for p in files if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]


def to_embedding_tensor(obj: object) -> torch.Tensor:
    if torch.is_tensor(obj):
        return obj
    if hasattr(obj, "image_embeds") and getattr(obj, "image_embeds") is not None:
        return getattr(obj, "image_embeds")
    if hasattr(obj, "pooler_output") and getattr(obj, "pooler_output") is not None:
        return getattr(obj, "pooler_output")
    if isinstance(obj, (tuple, list)) and len(obj) > 0:
        return to_embedding_tensor(obj[0])
    raise RuntimeError("Unable to extract tensor embeddings from model output.")


def embed_category(
    model: AutoModel,
    processor: AutoImageProcessor,
    image_paths: list[Path],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Embedding", unit="batch"):
            batch_paths = image_paths[i : i + batch_size]
            images = []
            for p in batch_paths:
                with Image.open(p) as img:
                    images.append(img.convert("RGB"))

            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            if hasattr(model, "get_image_features"):
                out = model.get_image_features(**inputs)
            else:
                out = model(**inputs)
            emb = to_embedding_tensor(out)
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            chunks.append(emb.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(chunks, axis=0)


def run_one_domain(
    domain: str,
    model_id: str,
    batch_size: int,
    device: torch.device,
) -> None:
    cfg = DOMAIN_CONFIG[domain]
    data_dir = cfg["data_dir"].resolve()
    results_dir = cfg["results_dir"].resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    image_paths = discover_images(data_dir, cfg["glob"])
    if not image_paths:
        raise RuntimeError(f"No images found for domain={domain} in {data_dir} via {cfg['glob']}")

    slug = model_slug(model_id)
    out_npz = results_dir / f"siglip_{slug}_embeddings.npz"
    out_manifest = results_dir / f"siglip_{slug}_manifest.json"

    print(f"[{domain}] n_images={len(image_paths)} model={model_id} device={device}", flush=True)
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    embeddings = embed_category(model, processor, image_paths, batch_size=batch_size, device=device)

    np.savez_compressed(
        out_npz,
        embeddings=embeddings,
        paths=np.asarray([str(p.resolve()) for p in image_paths], dtype=object),
    )
    manifest = {
        "domain": domain,
        "model": model_id,
        "n_images": int(len(image_paths)),
        "embedding_dim": int(embeddings.shape[1]),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_file": out_npz.name,
    }
    out_manifest.write_text(json.dumps(manifest, indent=2))
    print(f"[{domain}] Saved: {out_npz}")
    print(f"[{domain}] Saved: {out_manifest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SigLIP image embeddings for WebDesign.")
    parser.add_argument(
        "--domains",
        nargs="*",
        choices=list(DOMAIN_CONFIG.keys()) + ["all"],
        default=["all"],
    )
    parser.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    domains = list(DOMAIN_CONFIG.keys()) if "all" in args.domains else args.domains
    for d in domains:
        run_one_domain(d, args.model_id, args.batch_size, device)


if __name__ == "__main__":
    main()
