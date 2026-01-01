from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_MODEL = "text-embedding-3-large"
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_RETRIES = 5
DEFAULT_REQUEST_TIMEOUT = 60.0

EMBED_MODES = ("source_only", "concat", "separate")


def model_slug(model_id: str) -> str:
    return model_id.replace("-", "_").replace("/", "_")


def request_embeddings(
    client: OpenAI,
    model: str,
    texts: list[str],
    max_retries: int,
) -> np.ndarray:
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(
                model=model,
                input=texts,
                timeout=DEFAULT_REQUEST_TIMEOUT,
            )
            rows = [d.embedding for d in resp.data]
            return np.asarray(rows, dtype=np.float32)
        except Exception as e:
            print(
                f"Embedding request failed (attempt {attempt + 1}/{max_retries}): {e}",
                flush=True,
            )
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    raise RuntimeError("Unreachable retry loop.")


def embed_texts(
    client: OpenAI,
    model: str,
    texts: list[str],
    batch_size: int,
    max_retries: int,
    label: str = "",
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        emb = request_embeddings(client, model, batch, max_retries)
        chunks.append(emb)
        done = min(i + batch_size, len(texts))
        print(f"  [{label}] {done}/{len(texts)}", flush=True)
    return np.concatenate(chunks, axis=0)


def load_segments(tsv_path: Path) -> pd.DataFrame:
    """Load unique (system, seg_id) translation pairs from the human MQM TSV."""
    df = pd.read_csv(tsv_path, sep="\t")
    segs = (
        df.drop_duplicates(subset=["system", "seg_id"])
        [["system", "seg_id", "doc", "doc_id", "source", "target"]]
        .sort_values(["system", "seg_id"])
        .reset_index(drop=True)
    )
    segs["task_id"] = segs["system"] + "::" + segs["seg_id"].astype(str)
    return segs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build OpenAI text embeddings for MQM newstest2020 en-de translation tasks."
    )
    parser.add_argument(
        "--tsv-path",
        type=Path,
        default=Path("datasets/mqm/data/mqm_newstest2020_ende.tsv"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("datasets/mqm/results"),
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--mode",
        type=str,
        choices=EMBED_MODES,
        default="separate",
        help=(
            "source_only: embed source sentence only. "
            "concat: embed 'source [SEP] target' as one string. "
            "separate: embed source and target independently, concatenate vectors."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("PLUTO_ENDPOINT")
    if not api_key or not base_url:
        raise RuntimeError("Missing API_KEY or PLUTO_ENDPOINT in .env/environment.")

    tsv_path = args.tsv_path.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    segs = load_segments(tsv_path)
    if args.max_tasks is not None:
        segs = segs.head(args.max_tasks)
    segs = segs.reset_index(drop=True)

    task_ids = segs["task_id"].tolist()
    sources = segs["source"].astype(str).tolist()
    targets = segs["target"].astype(str).tolist()

    if not task_ids:
        raise RuntimeError("No translation tasks found.")

    slug = model_slug(args.model)
    mode = args.mode
    out_npz = results_dir / f"{slug}_mqm_embeddings_{mode}.npz"
    out_manifest = results_dir / f"{slug}_mqm_embeddings_{mode}_manifest.json"

    print(
        f"Embedding MQM tasks: n={len(task_ids)}, model={args.model}, "
        f"mode={mode}, batch={args.batch_size}",
        flush=True,
    )
    client = OpenAI(api_key=api_key, base_url=base_url)

    if mode == "source_only":
        mat = embed_texts(client, args.model, sources, args.batch_size, args.max_retries, "source")

    elif mode == "concat":
        combined = [f"{s} [SEP] {t}" for s, t in zip(sources, targets)]
        mat = embed_texts(client, args.model, combined, args.batch_size, args.max_retries, "concat")

    elif mode == "separate":
        # Deduplicate sources: all systems share the same source per seg_id
        unique_sources = segs.drop_duplicates("seg_id")[["seg_id", "source"]].reset_index(drop=True)
        src_texts = unique_sources["source"].astype(str).tolist()
        src_seg_ids = unique_sources["seg_id"].tolist()

        print(f"  Source dedup: {len(sources)} -> {len(src_texts)} unique source sentences", flush=True)
        src_emb_unique = embed_texts(
            client, args.model, src_texts, args.batch_size, args.max_retries, "source"
        )

        src_lookup = dict(zip(src_seg_ids, range(len(src_seg_ids))))
        src_indices = [src_lookup[sid] for sid in segs["seg_id"]]
        src_emb = src_emb_unique[src_indices]

        tgt_emb = embed_texts(
            client, args.model, targets, args.batch_size, args.max_retries, "target"
        )

        mat = np.concatenate([src_emb, tgt_emb], axis=1)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    np.savez_compressed(
        out_npz,
        task_ids=np.asarray(task_ids, dtype=object),
        embeddings=mat,
    )

    manifest = {
        "model": args.model,
        "mode": mode,
        "tsv_path": str(tsv_path),
        "n_tasks": len(task_ids),
        "embedding_dim": int(mat.shape[1]),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_file": out_npz.name,
    }
    if mode == "separate":
        manifest["n_unique_sources"] = len(src_texts)
        manifest["source_dim"] = int(src_emb.shape[1])
        manifest["target_dim"] = int(tgt_emb.shape[1])

    out_manifest.write_text(json.dumps(manifest, indent=2))
    print(f"Saved: {out_npz}  (shape={mat.shape})")
    print(f"Saved: {out_manifest}")


if __name__ == "__main__":
    main()
