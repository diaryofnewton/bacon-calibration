#!/usr/bin/env bash
set -euo pipefail

doc_arg="${1:?missing document argument}"
out_dir="${2:?missing output directory}"

resolve_doc() {
  local candidate="$1"

  if [[ -f "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi

  if [[ -f "${candidate}.tex" ]]; then
    printf '%s\n' "${candidate}.tex"
    return 0
  fi

  if [[ -f "$out_dir/$candidate" ]]; then
    printf '%s\n' "$out_dir/$candidate"
    return 0
  fi

  if [[ -f "$out_dir/${candidate}.tex" ]]; then
    printf '%s\n' "$out_dir/${candidate}.tex"
    return 0
  fi

  return 1
}

doc_path="$(resolve_doc "$doc_arg" || true)"

if [[ -z "${doc_path:-}" ]]; then
  printf 'Could not resolve TeX input from "%s" with output dir "%s"\n' "$doc_arg" "$out_dir" >&2
  exit 1
fi

exec /home/colligo/bin/tectonic --synctex -o "$out_dir" "$doc_path"
