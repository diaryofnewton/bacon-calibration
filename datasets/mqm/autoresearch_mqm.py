"""
autoresearch_mqm.py — Orchestrator for LLM-driven research loop.

Structure (Karpathy autoresearch style):
  program.md  — living algorithm description; LLM reads this to understand the problem
  train.py    — standalone experiment script; LLM edits this to implement hypotheses
  autoresearch_mqm.py  — this file; runs the loop

Each iteration:
  1. Read program.md and current train.py
  2. Ask LLM to produce a modified train.py implementing its next hypothesis
  3. Write the new train.py, run it, capture JSON result
  4. Update program.md experiment log
  5. Repeat until dr_rmse < baseline2_rmse or budget exhausted

Usage:
  python datasets/mqm/autoresearch_mqm.py [--iterations 15] [--trials 25] [--n-jobs 4]
  python datasets/mqm/autoresearch_mqm.py --dry-run   # skip LLM, run current train.py only
"""

from __future__ import annotations

import os
import sys

# Re-exec with venv python if openai isn't available in the current interpreter
_VENV_PYTHON = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), ".venv", "bin", "python")
if os.path.exists(_VENV_PYTHON) and _VENV_PYTHON != sys.executable:
    try:
        import openai  # noqa: F401
    except ImportError:
        os.execv(_VENV_PYTHON, [_VENV_PYTHON] + sys.argv)

import argparse
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# LLM prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert ML researcher improving a two-stage adaptive sampling estimator.
You will be given:
  1. program.md — describes the problem, algorithm, current performance, and hypotheses
  2. The current train.py — the experiment script

Your job: propose ONE specific improvement, then output a complete modified train.py.

Rules for editing train.py:
- You may change any module-level constants (PI, ETA, SIGMA_*, etc.)
- You may rewrite `compute_dr()`, `transform_residuals()`, or `run_trial()`
- Do NOT change the CLI argument interface or the output JSON schema keys
- Keep LABEL and NOTES updated to describe your change
- The file must be complete and runnable as-is

OUTPUT FORMAT — THIS IS MANDATORY:
Your response MUST contain exactly two parts, in this order:
  PART 1: One paragraph starting with "## Hypothesis" explaining why you expect improvement.
  PART 2: The COMPLETE modified train.py inside a fenced code block, like this:

```python
[full contents of train.py here — every line, from the docstring to if __name__ == "__main__"]
```

DO NOT truncate the code. DO NOT say "rest of code unchanged". Write EVERY line.
DO NOT end your response before providing the ```python block.
"""


def build_prompt(program_md: str, train_py: str) -> str:
    return f"""## program.md
{program_md}

## Current train.py
```python
{train_py}
```

Now propose the next improvement. Write your ## Hypothesis paragraph, then the COMPLETE train.py in a ```python block. Every single line must be present — do not truncate."""


# ─────────────────────────────────────────────────────────────────────────────
# LLM call
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(client, program_md: str, train_py: str, model: str) -> tuple[str, str]:
    """Returns (hypothesis_text, new_train_py_source)."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_prompt(program_md, train_py)},
        ],
        temperature=0.7,
        max_tokens=8192,
    )
    raw = response.choices[0].message.content

    # Extract hypothesis (text before the code block)
    hypothesis = raw.split("```")[0].strip()

    # Extract Python code block
    match = re.search(r"```python\s*(.*?)```", raw, re.DOTALL)
    if not match:
        raise ValueError(f"LLM response did not contain a ```python block.\nRaw:\n{raw[:500]}")
    new_src = match.group(1).strip()
    return hypothesis, new_src


# ─────────────────────────────────────────────────────────────────────────────
# Run train.py and capture JSON output
# ─────────────────────────────────────────────────────────────────────────────

def run_train(train_path: Path, trials: int, n_jobs: int, seed: int) -> dict:
    """Runs train.py as a subprocess, returns parsed JSON result."""
    venv_python = ROOT / ".venv" / "bin" / "python"
    python = str(venv_python) if venv_python.exists() else sys.executable

    cmd = [
        python, str(train_path),
        "--trials", str(trials),
        "--n-jobs", str(n_jobs),
        "--seed", str(seed),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))

    if proc.returncode != 0:
        raise RuntimeError(
            f"train.py exited {proc.returncode}\n"
            f"stderr:\n{proc.stderr[-2000:]}\n"
            f"stdout:\n{proc.stdout[-500:]}"
        )

    # Find the JSON object in stdout (may have log lines before it)
    stdout = proc.stdout.strip()
    json_start = stdout.rfind("{")
    if json_start == -1:
        raise ValueError(f"No JSON found in train.py output:\n{stdout[-500:]}")
    return json.loads(stdout[json_start:])


# ─────────────────────────────────────────────────────────────────────────────
# Update program.md experiment log
# ─────────────────────────────────────────────────────────────────────────────

def update_program_md(program_path: Path, iteration: int, result: dict,
                      hypothesis: str, beats_b2: bool) -> None:
    content = program_path.read_text()

    # Append to experiment log table
    label   = result.get("label", "?")
    dr_rmse = result.get("dr_rmse", 0)
    dr_bias = result.get("dr_bias", 0)
    b2_rmse = result.get("baseline2_rmse", 0)
    ratio   = result.get("dr_vs_b2_ratio", 0)
    beat    = "✓ BEATS B2" if beats_b2 else "✗"
    short_hyp = hypothesis[:60].replace("|", "/") + ("…" if len(hypothesis) > 60 else "")

    new_row = f"| {iteration} | {label} | {dr_rmse:.4f} | {dr_bias:.4f} | {ratio:.2f}× | {short_hyp} {beat} |"

    # Insert before the closing of the table (after last | row)
    table_marker = "| 0 | baseline_default"
    if table_marker in content:
        # Find end of the table block — insert after the last table row
        lines = content.splitlines()
        last_table_line = max(
            (i for i, l in enumerate(lines) if l.strip().startswith("|")),
            default=None
        )
        if last_table_line is not None:
            lines.insert(last_table_line + 1, new_row)
            content = "\n".join(lines)
    else:
        content += f"\n{new_row}\n"

    program_path.write_text(content)


# ─────────────────────────────────────────────────────────────────────────────
# Save versioned train.py backup
# ─────────────────────────────────────────────────────────────────────────────

def archive_train(train_path: Path, iteration: int, result: dict) -> None:
    archive_dir = train_path.parent / "train_versions"
    archive_dir.mkdir(exist_ok=True)
    label = re.sub(r"[^\w\-]", "_", result.get("label", "exp"))
    dst = archive_dir / f"train_v{iteration:02d}_{label}.py"
    shutil.copy2(train_path, dst)


# ─────────────────────────────────────────────────────────────────────────────
# Load API credentials from .env
# ─────────────────────────────────────────────────────────────────────────────

def load_env(base_dir: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    env_path = base_dir / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    # Also check process environment
    for k in ("API_KEY", "PLUTO_ENDPOINT"):
        if k in __import__("os").environ and k not in env:
            env[k] = __import__("os").environ[k]
    return env


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Autoresearch loop for two-stage DR.")
    parser.add_argument("--iterations",  type=int, default=15)
    parser.add_argument("--trials",      type=int, default=25,
                        help="Quick trials per experiment")
    parser.add_argument("--final-trials",type=int, default=100,
                        help="Trials for final validation")
    parser.add_argument("--n-jobs",      type=int, default=1)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--model",       type=str, default="claude-haiku-4.5")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Skip LLM; run current train.py only and exit")
    args = parser.parse_args()

    train_path   = HERE / "train.py"
    program_path = HERE / "program.md"
    results_dir  = HERE / "results"
    results_dir.mkdir(exist_ok=True)

    # ── Baseline run ─────────────────────────────────────────────────────────
    print(f"[{datetime.now():%H:%M:%S}] Running baseline (current train.py)...")
    baseline = run_train(train_path, trials=args.trials, n_jobs=args.n_jobs, seed=args.seed)
    print(
        f"  dr_rmse={baseline['dr_rmse']:.4f}  dr_bias={baseline['dr_bias']:.4f}  "
        f"baseline2_rmse={baseline['baseline2_rmse']:.4f}  ratio={baseline['dr_vs_b2_ratio']:.2f}"
    )

    if args.dry_run:
        print("Dry run done.")
        return

    # ── LLM client ────────────────────────────────────────────────────────────
    if OpenAI is None:
        raise RuntimeError("pip install openai")
    env = load_env(ROOT)
    api_key  = env.get("API_KEY", "")
    endpoint = env.get("PLUTO_ENDPOINT", "").rstrip("/") + "/v1"
    client   = OpenAI(api_key=api_key, base_url=endpoint)
    print(f"LLM: {args.model} via {endpoint}")

    # ── History ───────────────────────────────────────────────────────────────
    best_result   = baseline
    best_train_src = train_path.read_text()
    history: list[dict] = [{"iteration": 0, "result": baseline, "hypothesis": "baseline"}]
    overall_log_path = results_dir / "autoresearch_log.json"

    # ── Research loop ─────────────────────────────────────────────────────────
    for iteration in range(1, args.iterations + 1):
        print(f"\n{'='*60}")
        print(f"[{datetime.now():%H:%M:%S}] Iteration {iteration}/{args.iterations}")

        program_md = program_path.read_text()
        train_py   = train_path.read_text()

        # Ask LLM to propose and implement next change
        print(f"  Querying {args.model}...")
        try:
            hypothesis, new_train_src = call_llm(client, program_md, train_py, args.model)
        except Exception as e:
            print(f"  [LLM error] {e}")
            continue

        print(f"  Hypothesis: {hypothesis[:120]}...")

        # Write new train.py
        train_path.write_text(new_train_src)

        # Run experiment
        print(f"  Running {args.trials} trials (n_jobs={args.n_jobs})...")
        try:
            result = run_train(train_path, trials=args.trials, n_jobs=args.n_jobs,
                               seed=args.seed + iteration)
        except Exception as e:
            print(f"  [Run error] {e}")
            # Restore previous train.py
            train_path.write_text(best_train_src)
            history.append({"iteration": iteration, "result": {"error": str(e)},
                             "hypothesis": hypothesis})
            continue

        beats_b2  = result["dr_rmse"] < result["baseline2_rmse"]
        improved  = result["dr_rmse"] < best_result["dr_rmse"]
        delta     = best_result["dr_rmse"] - result["dr_rmse"]

        print(
            f"  label={result['label']}  "
            f"dr_rmse={result['dr_rmse']:.4f}  dr_bias={result['dr_bias']:.4f}  "
            f"ratio={result['dr_vs_b2_ratio']:.2f}  "
            f"{'*** BEATS BASELINE2 ***' if beats_b2 else ''}"
        )
        if improved:
            print(f"  (+{delta:.4f} improvement — new best)")

        # Archive this train.py version
        archive_train(train_path, iteration, result)

        # Update program.md log
        update_program_md(program_path, iteration, result, hypothesis, beats_b2)

        # Track history
        history.append({"iteration": iteration, "result": result, "hypothesis": hypothesis})
        overall_log_path.write_text(json.dumps(history, indent=2))

        if improved:
            best_result    = result
            best_train_src = new_train_src
        else:
            # Revert to best train.py so next iteration builds on best known version
            train_path.write_text(best_train_src)

        if beats_b2:
            print(f"\n  *** Two-stage DR beats baseline2! Stopping early. ***")
            break

    # ── Final validation ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[{datetime.now():%H:%M:%S}] Final validation with {args.final_trials} trials...")
    train_path.write_text(best_train_src)
    final = run_train(train_path, trials=args.final_trials, n_jobs=args.n_jobs,
                      seed=args.seed + 99999)
    beats_b2_final = final["dr_rmse"] < final["baseline2_rmse"]

    print(json.dumps(final, indent=2))
    print(
        f"\n*** Final: {'BEATS' if beats_b2_final else 'DOES NOT BEAT'} baseline2 ***\n"
        f"  dr_rmse={final['dr_rmse']:.4f} vs baseline2_rmse={final['baseline2_rmse']:.4f}"
    )

    # Save final summary
    summary = {
        "best_label":       final.get("label"),
        "final_result":     final,
        "beat_baseline2":   beats_b2_final,
        "history":          history,
    }
    (results_dir / "autoresearch_final.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {results_dir}/autoresearch_final.json")
    print(f"Saved: {results_dir}/autoresearch_log.json")
    print(f"Train versions: {HERE}/train_versions/")


if __name__ == "__main__":
    main()
