from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI


MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "llama-3-1-8b",
]

SYSTEM_MSG = """You are an essay grader. You will be given a student essay and its writing prompt.
Rate the essay's holistic quality on a scale of 1 to 6:
1 = Very poor, 2 = Poor, 3 = Below average, 4 = Above average, 5 = Good, 6 = Excellent.

You MUST respond with ONLY a single digit (1, 2, 3, 4, 5, or 6).
No explanation, no reasoning, no other text - just the number."""

USER_TEMPLATE = """Essay prompt: {prompt_name}

Essay:
{full_text}"""

MAX_CONCURRENT = 10
BATCH_DELAY = 1.0


def parse_score(text: str) -> int | None:
    m = re.search(r"\b([1-6])\b", (text or "").strip())
    return int(m.group(1)) if m else None


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
    ent = -sum(p * math.log(max(p, 1e-12)) for p in probs)
    return float(ent)


def safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


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
            essay_id = str(row.get("essay_id_comp", ""))
            model = str(row.get("model", ""))
            pred = row.get("predicted_score")
            if essay_id and model and pred is not None:
                done.add((essay_id, model))
    return done


async def score_one(
    client: AsyncOpenAI,
    model: str,
    essay_id: str,
    prompt_name: str,
    full_text: str,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    user_msg = USER_TEMPLATE.format(prompt_name=prompt_name, full_text=full_text)
    async with sem:
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": user_msg},
                ],
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": 5,
                "stop": ["\n"],
            }
            if model.startswith("gemini"):
                kwargs["extra_body"] = {"thinking": {"type": "enabled", "budget_tokens": 0}}

            resp = await client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            text = (choice.message.content or "").strip()
            pred = parse_score(text)

            token_logprobs = []
            entropy_list = []
            if getattr(choice, "logprobs", None) and getattr(choice.logprobs, "content", None):
                for tok in choice.logprobs.content:
                    lp = safe_float(getattr(tok, "logprob", None))
                    if lp is not None:
                        token_logprobs.append(lp)
                    ent = entropy_from_top_logprobs(getattr(tok, "top_logprobs", None) or [])
                    if ent is not None:
                        entropy_list.append(ent)

            mean_logprob = float(sum(token_logprobs) / len(token_logprobs)) if token_logprobs else None
            perplexity = float(math.exp(-mean_logprob)) if mean_logprob is not None else None
            mean_entropy = float(sum(entropy_list) / len(entropy_list)) if entropy_list else None

            return {
                "essay_id_comp": essay_id,
                "model": model,
                "raw_output": text if text else "NA",
                "predicted_score": pred if pred is not None else "NA",
                "token_logprobs": token_logprobs if token_logprobs else "NA",
                "mean_logprob": mean_logprob if mean_logprob is not None else "NA",
                "perplexity": perplexity if perplexity is not None else "NA",
                "mean_entropy": mean_entropy if mean_entropy is not None else "NA",
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
            }
        except Exception as e:
            return {
                "essay_id_comp": essay_id,
                "model": model,
                "raw_output": "NA",
                "predicted_score": "NA",
                "token_logprobs": "NA",
                "mean_logprob": "NA",
                "perplexity": "NA",
                "mean_entropy": "NA",
                "prompt_tokens": "NA",
                "completion_tokens": "NA",
                "error": str(e),
            }


async def run(args: argparse.Namespace) -> None:
    load_dotenv()
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("PLUTO_ENDPOINT")
    if not api_key or not base_url:
        raise RuntimeError("Missing API_KEY or PLUTO_ENDPOINT in environment/.env")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    csv_path = args.csv_path.resolve()
    out_path = args.output_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, low_memory=False)
    keep_cols = ["essay_id_comp", "prompt_name", "full_text", "grade_level"]
    for col in keep_cols:
        if col not in df.columns:
            raise RuntimeError(f"Missing required column: {col}")

    df = df.drop_duplicates("essay_id_comp")[keep_cols].copy()
    if args.grade is not None:
        df = df[df["grade_level"] == args.grade]
    if args.test is not None and args.test > 0:
        df = df.head(args.test)

    models = args.models if args.models else MODELS
    done = load_done(out_path)
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    with out_path.open("a", encoding="utf-8") as f:
        for model in models:
            rows = []
            for _, r in df.iterrows():
                essay_id = str(r["essay_id_comp"])
                if (essay_id, model) in done:
                    continue
                rows.append((essay_id, str(r["prompt_name"]), str(r["full_text"])))

            if not rows:
                print(f"[{model}] nothing to do")
                continue

            print(f"[{model}] pending={len(rows)}")
            for i in range(0, len(rows), MAX_CONCURRENT):
                batch = rows[i : i + MAX_CONCURRENT]
                tasks = [
                    score_one(client, model, essay_id=eid, prompt_name=pname, full_text=txt, sem=sem)
                    for (eid, pname, txt) in batch
                ]
                results = await asyncio.gather(*tasks)
                for row in results:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    done.add((str(row["essay_id_comp"]), str(row["model"])))
                f.flush()
                await asyncio.sleep(BATCH_DELAY)

    print(f"Saved: {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Score PERSUADE essays with LLM judges.")
    p.add_argument(
        "--csv-path",
        type=Path,
        default=Path("datasets/persuade/data/persuade_corpus_2.0_train.csv"),
    )
    p.add_argument(
        "--output-path",
        type=Path,
        default=Path("datasets/persuade/results/llm_scores.jsonl"),
    )
    p.add_argument("--grade", type=int, default=10, help="Use -1 for all grades.")
    p.add_argument("--test", type=int, default=None, help="First N essays only.")
    p.add_argument("--models", nargs="*", default=None)
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.grade == -1:
        args.grade = None
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
