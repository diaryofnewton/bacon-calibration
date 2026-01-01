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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OpenAI text embeddings for PERSUADE essays.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("datasets/persuade/data/persuade_corpus_2.0_train.csv"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("datasets/persuade/results"),
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--grade", type=int, default=10, help="Use -1 for all grades.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--max-essays", type=int, default=None)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("PLUTO_ENDPOINT")
    if not api_key or not base_url:
        raise RuntimeError("Missing API_KEY or PLUTO_ENDPOINT in .env/environment.")

    csv_path = args.csv_path.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    slug = model_slug(args.model)
    out_npz = results_dir / f"{slug}_essay_embeddings.npz"
    out_manifest = results_dir / f"{slug}_essay_embeddings_manifest.json"

    grade = None if args.grade == -1 else args.grade
    df = pd.read_csv(csv_path, low_memory=False)
    req = {"essay_id_comp", "full_text", "grade_level"}
    missing = req - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    essays = df.drop_duplicates("essay_id_comp")[["essay_id_comp", "full_text", "grade_level"]].copy()
    if grade is not None:
        essays = essays[essays["grade_level"] == grade]
    if args.max_essays is not None:
        essays = essays.head(args.max_essays)
    essays = essays.reset_index(drop=True)

    ids = essays["essay_id_comp"].astype(str).tolist()
    texts = essays["full_text"].astype(str).tolist()
    if not ids:
        raise RuntimeError("No essays selected for embedding.")

    print(f"Embedding essays: n={len(ids)}, model={args.model}, batch={args.batch_size}", flush=True)
    client = OpenAI(api_key=api_key, base_url=base_url)

    chunks: list[np.ndarray] = []
    for i in range(0, len(texts), args.batch_size):
        batch_texts = texts[i : i + args.batch_size]
        emb = request_embeddings(client, args.model, batch_texts, args.max_retries)
        chunks.append(emb)
        print(f"Done {min(i + args.batch_size, len(texts))}/{len(texts)}", flush=True)

    mat = np.concatenate(chunks, axis=0)
    np.savez_compressed(
        out_npz,
        essay_ids=np.asarray(ids, dtype=object),
        embeddings=mat,
    )

    manifest = {
        "model": args.model,
        "csv_path": str(csv_path),
        "grade_filter": grade,
        "n_essays": int(len(ids)),
        "embedding_dim": int(mat.shape[1]),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_file": out_npz.name,
    }
    out_manifest.write_text(json.dumps(manifest, indent=2))
    print(f"Saved: {out_npz}")
    print(f"Saved: {out_manifest}")


if __name__ == "__main__":
    main()
