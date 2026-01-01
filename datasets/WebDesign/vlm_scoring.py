from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI


MODELS = [
    "gemini-2.5-flash-lite",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
]

OUTCOMES = ["AE", "TRU", "TYP", "EXMPL", "AVG", "US"]
MAX_CONCURRENT = 10
BATCH_DELAY = 1.0

DOMAIN_CONFIG = {
    "unis": {
        "data_dir": Path("datasets/WebDesign/Universitites/data"),
        "results_dir": Path("datasets/WebDesign/Universitites/results"),
        "glob": "universities.part*/*.jpg",
        "output": "vlm_scores_unis.jsonl",
        "label": "university/college website",
        "prompt_typ": "This webpage looks like a typical homepage of a university/college website.",
        "prompt_exmpl": "This webpage is a representative example of homepages of university/college websites.",
        "prompt_avg": "This webpage has many visual aspects in common with homepages of other university/college websites.",
    },
    "banks": {
        "data_dir": Path("datasets/WebDesign/Commercial Banks/data"),
        "results_dir": Path("datasets/WebDesign/Commercial Banks/results"),
        "glob": "banks.part*/*.jpg",
        "output": "vlm_scores_banks.jsonl",
        "label": "commercial-bank website",
        "prompt_typ": "This webpage looks like a typical commercial-bank homepage.",
        "prompt_exmpl": "This webpage is a representative example of commercial-bank homepages.",
        "prompt_avg": "This webpage has many visual aspects in common with homepages of other commercial-bank websites.",
    },
    "fashion": {
        "data_dir": Path("datasets/WebDesign/eCommerce/data"),
        "results_dir": Path("datasets/WebDesign/eCommerce/results"),
        "glob": "fashion.part*/*.jpg",
        "output": "vlm_scores_fashion.jsonl",
        "label": "online fashion-shopping website",
        "prompt_typ": "This webpage looks like a typical homepage of an online fashion-shopping website.",
        "prompt_exmpl": "This webpage is a representative example of a homepage of online fashion-shopping websites.",
        "prompt_avg": "This webpage has many visual aspects in common with homepages of other online fashion-shopping websites.",
    },
    "homeware": {
        "data_dir": Path("datasets/WebDesign/eCommerce/data"),
        "results_dir": Path("datasets/WebDesign/eCommerce/results"),
        "glob": "homeware.part*/*.jpg",
        "output": "vlm_scores_homeware.jsonl",
        "label": "online homeware-shopping website",
        "prompt_typ": "This webpage looks like a typical homepage of an online homeware-shopping website.",
        "prompt_exmpl": "This webpage is a representative example of a homepage of online homeware-shopping websites.",
        "prompt_avg": "This webpage has many visual aspects in common with homepages of other online homeware-shopping websites.",
    },
}


def build_prompt(cfg: dict[str, str]) -> str:
    return f"""Rate this screenshot from a {cfg['label']} on six dimensions.
Return ONLY six integer scores in [-3, -2, -1, 0, 1, 2, 3], one per line, exactly:
AE: <score>
TRU: <score>
TYP: <score>
EXMPL: <score>
AVG: <score>
US: <score>

Definitions:
- AE (Aesthetics): visually appealing overall design.
- TRU (Trustworthiness): appears credible and trustworthy.
- TYP (Typicality): {cfg['prompt_typ']}
- EXMPL (Exemplar Goodness): {cfg['prompt_exmpl']}
- AVG (Family Resemblance): {cfg['prompt_avg']}
- US (Usability): easy to navigate and understand.
"""


def parse_scores(text: str) -> dict[str, int | None]:
    out: dict[str, int | None] = {k: None for k in OUTCOMES}
    for k in OUTCOMES:
        m = re.search(rf"{k}\s*:\s*([+-]?[0-3])\b", text or "", flags=re.IGNORECASE)
        if m:
            out[k] = int(m.group(1))
    return out


def entropy_from_top_logprobs(top_logprobs: list[Any]) -> float | None:
    if not top_logprobs:
        return None
    probs = []
    for t in top_logprobs:
        lp = getattr(t, "logprob", None)
        if lp is None:
            continue
        probs.append(math.exp(lp))
    s = sum(probs)
    if s <= 0:
        return None
    probs = [p / s for p in probs]
    return float(-sum(p * math.log(max(p, 1e-12)) for p in probs))


def load_done(path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            stim = str(row.get("stimulus_id", ""))
            model = str(row.get("model", ""))
            if not stim or not model:
                continue
            ok = all(row.get(f"score_{k}") not in (None, "NA") for k in OUTCOMES)
            if ok:
                done.add((stim, model))
    return done


def encode_image(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


async def score_image(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    model: str,
    image_path: Path,
    prompt: str,
) -> dict[str, Any]:
    stim = image_path.name
    try:
        img_b64 = encode_image(image_path)
    except Exception as e:
        return {
            "stimulus_id": stim,
            "model": model,
            "raw_output": "NA",
            **{f"score_{k}": "NA" for k in OUTCOMES},
            "token_logprobs": "NA",
            "mean_logprob": "NA",
            "perplexity": "NA",
            "mean_entropy": "NA",
            "prompt_tokens": "NA",
            "completion_tokens": "NA",
            "error": f"image_read_error: {e}",
        }

    async with sem:
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a strict visual evaluator. Return only the requested six lines."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                        ],
                    },
                ],
                "temperature": 0.0,
                "max_tokens": 128,
                "logprobs": True,
                "top_logprobs": 5,
            }
            if model.startswith("gemini"):
                kwargs["extra_body"] = {"thinking": {"type": "enabled", "budget_tokens": 0}}

            resp = await client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            text = (choice.message.content or "").strip()
            scores = parse_scores(text)

            token_logprobs = []
            entropy_list = []
            if getattr(choice, "logprobs", None) and getattr(choice.logprobs, "content", None):
                for tok in choice.logprobs.content:
                    lp = getattr(tok, "logprob", None)
                    if lp is not None:
                        token_logprobs.append(float(lp))
                    ent = entropy_from_top_logprobs(getattr(tok, "top_logprobs", None) or [])
                    if ent is not None:
                        entropy_list.append(ent)
            mean_logprob = float(sum(token_logprobs) / len(token_logprobs)) if token_logprobs else None
            perplexity = float(math.exp(-mean_logprob)) if mean_logprob is not None else None
            mean_entropy = float(sum(entropy_list) / len(entropy_list)) if entropy_list else None

            row = {
                "stimulus_id": stim,
                "model": model,
                "raw_output": text if text else "NA",
                **{f"score_{k}": (scores[k] if scores[k] is not None else "NA") for k in OUTCOMES},
                "token_logprobs": token_logprobs if token_logprobs else "NA",
                "mean_logprob": mean_logprob if mean_logprob is not None else "NA",
                "perplexity": perplexity if perplexity is not None else "NA",
                "mean_entropy": mean_entropy if mean_entropy is not None else "NA",
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
            }
            return row
        except Exception as e:
            return {
                "stimulus_id": stim,
                "model": model,
                "raw_output": "NA",
                **{f"score_{k}": "NA" for k in OUTCOMES},
                "token_logprobs": "NA",
                "mean_logprob": "NA",
                "perplexity": "NA",
                "mean_entropy": "NA",
                "prompt_tokens": "NA",
                "completion_tokens": "NA",
                "error": str(e),
            }


async def run(args: argparse.Namespace) -> None:
    cfg = DOMAIN_CONFIG[args.domain]
    data_dir = args.data_dir.resolve() if args.data_dir else cfg["data_dir"].resolve()
    results_dir = args.results_dir.resolve() if args.results_dir else cfg["results_dir"].resolve()
    out_path = results_dir / cfg["output"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(cfg)

    files = sorted(data_dir.glob(cfg["glob"]))
    if args.test is not None and args.test > 0:
        files = files[: args.test]
    if not files:
        raise RuntimeError(f"No images found in {data_dir} with glob {cfg['glob']}")

    load_dotenv()
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("PLUTO_ENDPOINT")
    if not api_key or not base_url:
        raise RuntimeError("Missing API_KEY or PLUTO_ENDPOINT in environment/.env")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    models = args.models if args.models else MODELS
    done = load_done(out_path)
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    with out_path.open("a", encoding="utf-8") as f:
        for model in models:
            pending = [p for p in files if (p.name, model) not in done]
            if not pending:
                print(f"[{model}] nothing to do")
                continue
            print(f"[{model}] pending={len(pending)}")

            for i in range(0, len(pending), MAX_CONCURRENT):
                batch = pending[i : i + MAX_CONCURRENT]
                tasks = [score_image(client, sem, model, p, prompt) for p in batch]
                rows = await asyncio.gather(*tasks)
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if all(row.get(f"score_{k}") not in ("NA", None) for k in OUTCOMES):
                        done.add((str(row["stimulus_id"]), str(row["model"])))
                f.flush()
                await asyncio.sleep(BATCH_DELAY)

    print(f"Saved: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Score WebDesign screenshots with VLM judges.")
    p.add_argument("--domain", choices=list(DOMAIN_CONFIG.keys()), default="unis")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--test", type=int, default=None)
    p.add_argument("--models", nargs="*", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
