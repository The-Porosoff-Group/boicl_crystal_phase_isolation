#!/usr/bin/env python3
"""
Design-space inverse prompt generator (Step #A only) — with role preamble
=========================================================================

Drop-in module that creates **three inverse-design prompt options** from
(1) user parameters (dict/JSON/plain text or a path) and (2) an optional SOP
(.txt/.docx/.pdf or None). Includes:

- Optional **style-only few-shot examples** to copy *format* without leaking domain
- Optional **forbid_terms** to stop specific vocabulary (e.g., "OCM", "CH4")
- Optional **role_preamble** injected verbatim at the top of each option
- Raw-text-safe parameter handling (won't mistake long text for a filesystem path)
- SOP is optional; a stub is used if not supplied
- Lightweight domain-leak detector with a single guarded retry
- Notebook-friendly wrapper + CLI

Install
-------
    pip install openai python-docx PyPDF2 pandas

Auth
----
    export OPENAI_API_KEY=...

CLI
---
    python design_space_A_only.py \
      --params "/path/to/params.txt" \
      --outdir out/run1 \
      --sop "/path/to/sop.pdf" \
      --example-pair "/path/style_params.txt:/path/style_inverse.txt" \
      --style-only-examples \
      --forbid OCM --forbid CH4 --forbid Na2WO4 \
      --preamble-file "/path/preamble.txt"

Notebook
--------
    from design_space_A_only import InversePromptBuilder
    b = InversePromptBuilder(
        params_input={"raw_parameters": "..."}, sop_path=None,
        example_pairs=[], style_only_examples=True,
        forbid_terms=["OCM","CH4"],
        role_preamble="You are an expert in heterogeneous catalysis..."
    )
    df = b.generate(outdir="out/test")

"""
from __future__ import annotations

import os
import re
import ast
import json
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# -----------------------------
# File readers (txt/docx/pdf) & param parsing
# -----------------------------

def _read_text_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore").strip()


def _read_docx(pathlike: Path) -> str:
    try:
        from docx import Document  # type: ignore
    except Exception as e:
        raise ImportError("python-docx is required to read .docx files. `pip install python-docx`.") from e
    doc = Document(pathlike)
    return "\n".join(par.text for par in doc.paragraphs).strip()


def _read_pdf(pathlike: Path) -> str:
    try:
        from PyPDF2 import PdfReader  # type: ignore
    except Exception as e:
        raise ImportError("PyPDF2 is required to read .pdf files. `pip install PyPDF2`.") from e
    reader = PdfReader(str(pathlike))
    buf: List[str] = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        buf.append(txt)
    return "\n".join(buf).strip()


def read_any_sop(path: Optional[str]) -> str:
    """Return SOP text; accept None or missing path by returning a stub."""
    if not path:
        return "SOP: [not provided]; rely on USER PARAMETERS; use '[value needed]' for unknowns."
    p = Path(path)
    if not p.exists():
        logging.warning("SOP file not found: %s — using stub.", p)
        return "SOP: [missing file]; rely on USER PARAMETERS; use '[value needed]'."
    suf = p.suffix.lower()
    if suf == ".pdf":
        return _read_pdf(p)
    if suf == ".docx":
        return _read_docx(p)
    return _read_text_file(p)


def read_any_params(maybe_path: str | Dict[str, Any]) -> str:
    """Return pretty-printed JSON-ish string to embed in prompts.
    Accepts dict, JSON string, Python-literal string, or a filesystem path.
    Robust to huge strings that are *not* paths.
    """
    if isinstance(maybe_path, dict):
        return json.dumps(maybe_path, indent=2, ensure_ascii=False)

    s = str(maybe_path).strip()

    # Guard: a very long "path" is almost surely raw content
    if len(s) > 260:  # typical PATH_MAX-ish heuristic
        raw = s
    else:
        try:
            p = Path(s)
            is_file = p.exists() and p.is_file()
        except OSError:
            is_file = False
        raw = _read_text_file(p) if is_file else s

    # Try JSON -> Python literal -> fallback raw
    for loader in (json.loads, ast.literal_eval):
        try:
            obj = loader(raw)
            return json.dumps(obj, indent=2, ensure_ascii=False)
        except Exception:
            pass
    return raw


# -----------------------------
# Few-shot utilities & prompts
# -----------------------------

FEWSHOT_SEP = "\n\n" + ("-" * 78) + "\n\n"


def _load_example_pairs(pairs: Iterable[str]) -> List[Tuple[str, str]]:
    """Each item: '/path/params.txt:/path/inv.txt' → (params_text, inv_text)"""
    out: List[Tuple[str, str]] = []
    for item in pairs:
        l, r = item.split(":", 1)
        params_text = read_any_params(l)
        inv_text = _read_text_file(Path(r))
        out.append((params_text, inv_text))
    return out


def build_fewshot_block(pairs: List[Tuple[str, str]], style_only: bool) -> str:
    if not pairs:
        return ""
    preface = (
        "The following examples are provided to illustrate **formatting and structure**. "
        "Do not copy domain specifics unless they also appear in USER PARAMETERS or SOP."
        if style_only else
        "Use the following examples for content and style when consistent with USER PARAMETERS and SOP."
    )
    blocks: List[str] = [preface]
    for i, (params_text, inv_text) in enumerate(pairs, start=1):
        blocks.append(
            f"""### EXAMPLE {i}
PARAMETERS:\n{params_text}\n\nINVERSE_DESIGN_PROMPT (gold):\n{inv_text}\n"""
        )
    return FEWSHOT_SEP.join(blocks)


def build_main_prompt(
    user_params_text: str,
    sop_text: str,
    fewshot_block: str,
    forbid_terms: Optional[List[str]] = None,
    role_preamble: Optional[str] = None,        # NEW
) -> str:
    header = (
        "You are an expert prompt engineer collaborating with a scientist. "
        "Produce **three concise, fully-specified inverse-design prompts** that constrain a "
        "downstream model to a permissible design space derived strictly from USER PARAMETERS and SOP."
    )

    forbid = forbid_terms or []
    forbid_clause = ("\n• You must NOT mention or import any of these terms: " + ", ".join(sorted(set(forbid)))
                     if forbid else "")

    preamble_rule = (
        "\n• Each option must begin with the following PREAMBLE verbatim, then continue with the sections:"
        if role_preamble else ""
    )

    rules = (
        "\nRULES:\n"
        "• Derive constraints **only** from USER PARAMETERS and SOP; do not import topics from examples unless they also appear in inputs.\n"
        "• Provide **exactly three** options; each must be self-contained and ≤ 400 tokens (excluding the PREAMBLE).\n"
        "• Each option must include: (a) goal/target, (b) allowed parameters/value sets, (c) boundary/guard-rails,"
        " (d) required output fields/schema, (e) explicit handling of unknowns via '[value needed]'.\n"
        "• Separate options with these exact tags on their own lines:\n  === INVERSE_PROMPT 1 ===\n  === INVERSE_PROMPT 2 ===\n  === INVERSE_PROMPT 3 ===\n"
        f"{forbid_clause}{preamble_rule}"
    )

    preamble_block = f"\nPREAMBLE:\n{role_preamble.strip()}\n" if role_preamble else ""

    io_block = f"""
USER PARAMETERS (verbatim or normalized):
{user_params_text}

USER SOP (text or stub):
{sop_text}
{preamble_block}
"""

    return header + "\n\n" + (fewshot_block + "\n\n" if fewshot_block else "") + rules + "\n\n" + io_block


# -----------------------------
# LLM call and output splitting
# -----------------------------

class _LLM:
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.3, max_tokens: int = 6000):
        from openai import OpenAI  # local import to avoid hard dep if unused
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set in environment.")
        self.client = OpenAI(api_key=key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        txt = resp.choices[0].message.content or ""
        # strip accidental code fences
        lines = [l for l in txt.splitlines() if not l.strip().startswith("```")]
        return "\n".join(lines).strip()


TAG_1 = re.compile(r"^===\s*INVERSE_PROMPT\s*1\s*===\s*$", re.IGNORECASE)
TAG_2 = re.compile(r"^===\s*INVERSE_PROMPT\s*2\s*===\s*$", re.IGNORECASE)
TAG_3 = re.compile(r"^===\s*INVERSE_PROMPT\s*3\s*===\s*$", re.IGNORECASE)


def _split_three_options(full: str) -> List[str]:
    lines = full.splitlines()
    idx: Dict[int, int] = {}
    for i, ln in enumerate(lines):
        if TAG_1.match(ln): idx[1] = i
        elif TAG_2.match(ln): idx[2] = i
        elif TAG_3.match(ln): idx[3] = i
    if len(idx) != 3:
        raise ValueError("Model output missing one or more INVERSE_PROMPT tags. See raw_output.txt.")
    i1, i2, i3 = idx[1], idx[2], idx[3]
    return [
        "\n".join(lines[i1+1:i2]).strip(),
        "\n".join(lines[i2+1:i3]).strip(),
        "\n".join(lines[i3+1:]).strip(),
    ]


# -----------------------------
# Domain-leak detection (lightweight)
# -----------------------------

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,}")


def _token_set(s: str) -> Set[str]:
    return set(w.lower() for w in _WORD_RE.findall(s))


def _detect_leak(examples: List[Tuple[str, str]], params_text: str, sop_text: str, output_text: str) -> List[str]:
    if not examples:
        return []
    ex_vocab: Set[str] = set()
    for p, inv in examples:
        ex_vocab.update(_token_set(p))
        ex_vocab.update(_token_set(inv))
    allowed = _token_set(params_text) | _token_set(sop_text)
    suspicious = sorted(list(ex_vocab - allowed))
    leaks = [t for t in suspicious if re.search(rf"\b{re.escape(t)}\b", output_text, flags=re.I)]
    return leaks[:20]  # cap for readability


# -----------------------------
# Public API
# -----------------------------

SYSTEM_MSG = "You write compact, precise inverse-design prompts with strict formatting."


def generate_inverse_prompts(
    params: str | Dict[str, Any],
    sop_path: Optional[str],
    outdir: str,
    model: str = "gpt-4o",
    temperature: float = 0.3,
    max_tokens: int = 6000,
    example_pairs: Optional[List[str]] = None,
    style_only_examples: bool = True,
    forbid_terms: Optional[List[str]] = None,
    role_preamble: Optional[str] = None,
) -> List[Path]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    user_params_text = read_any_params(params)
    sop_text = read_any_sop(sop_path)

    pairs = _load_example_pairs(example_pairs or [])
    fewshot_block = build_fewshot_block(pairs, style_only_examples)

    llm = _LLM(model=model, temperature=temperature, max_tokens=max_tokens)

    # First attempt
    prompt = build_main_prompt(user_params_text, sop_text, fewshot_block, forbid_terms, role_preamble)
    full = llm.complete(SYSTEM_MSG, prompt)

    # Leak check; one retry with stronger language if needed
    leaks = _detect_leak(pairs, user_params_text, sop_text, full)
    violated_forbid = []
    if forbid_terms:
        violated_forbid = [t for t in forbid_terms if re.search(rf"\b{re.escape(t)}\b", full, re.I)]
    if leaks or violated_forbid:
        logging.warning("Detected potential domain leakage: %s", ", ".join(leaks or violated_forbid))
        stronger = prompt + "\n\nHARD CONSTRAINT: Do not include any terms or concepts not present in USER PARAMETERS or SOP. " \
                              "If constraint conflicts, replace with '[value needed]'."
        full = llm.complete(SYSTEM_MSG, stronger)

    # Save raw output
    raw_path = out / "raw_output.txt"
    raw_path.write_text(full, encoding="utf-8")

    # Split & write three options
    options = _split_three_options(full)

    # Ensure preamble is present verbatim at the top of each option
    if role_preamble:
        pre = role_preamble.strip()
        fixed: List[str] = []
        for content in options:
            head = content[:200].lower()
            if pre[:40].lower() not in head:
                content = pre + "\n\n" + content
            fixed.append(content)
        options = fixed

    written: List[Path] = [raw_path]
    for i, content in enumerate(options, start=1):
        p = out / f"inverse_prompt_option_{i}.txt"
        p.write_text(content, encoding="utf-8")
        written.append(p)

    logging.info("Wrote %d files to %s", len(written), out)
    return written


# Notebook-friendly wrapper
try:
    import pandas as pd  # noqa: F401
except Exception:
    pd = None  # type: ignore


class InversePromptBuilder:
    def __init__(
        self,
        params_input: str | Dict[str, Any],
        sop_path: Optional[str],
        model: str = "gpt-4o",
        temperature: float = 0.3,
        max_tokens: int = 6000,
        example_pairs: Optional[List[str]] = None,
        style_only_examples: bool = True,
        forbid_terms: Optional[List[str]] = None,
        role_preamble: Optional[str] = None,
    ) -> None:
        self.params_input = params_input
        self.sop_path = sop_path
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.example_pairs = example_pairs or []
        self.style_only_examples = style_only_examples
        self.forbid_terms = forbid_terms or []
        self.role_preamble = role_preamble

    def generate(self, outdir: str = "out/inv_prompts"):
        paths = generate_inverse_prompts(
            params=self.params_input,
            sop_path=self.sop_path,
            outdir=outdir,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            example_pairs=self.example_pairs,
            style_only_examples=self.style_only_examples,
            forbid_terms=self.forbid_terms,
            role_preamble=self.role_preamble,
        )
        rows = []
        for p in paths:
            txt = Path(p).read_text(encoding="utf-8")
            preview = txt[:200].replace("\n", "\\n") + ("..." if len(txt) > 200 else "")
            rows.append({"file": Path(p).name, "path": str(p), "chars": len(txt), "preview": preview})
        if pd is not None:
            return pd.DataFrame(rows)
        return rows


# -----------------------------
# CLI
# -----------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate 3 inverse-design prompts (Step #A only)")
    ap.add_argument("--params", required=True, help="Path to params file OR literal JSON/dict/text")
    ap.add_argument("--sop", default=None, help="Path to SOP (.txt/.docx/.pdf) or omit for stub")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--example-pair", action="append", default=[], help="Few-shot pair: params_path:inverse_prompt_path (repeatable)")
    ap.add_argument("--style-only-examples", action="store_true", help="Treat examples as style/format ONLY; do not import domain content")
    ap.add_argument("--forbid", action="append", default=[], help="Add a forbidden term (repeatable)")
    ap.add_argument("--preamble", default=None, help="Literal preamble text to prepend to each option")
    ap.add_argument("--preamble-file", default=None, help="Path to a file containing the preamble text")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=6000)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    preamble_text = args.preamble
    if args.preamble_file and not preamble_text:
        preamble_text = Path(args.preamble_file).read_text(encoding="utf-8", errors="ignore").strip()
    generate_inverse_prompts(
        params=args.params,
        sop_path=args.sop,
        outdir=args.outdir,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        example_pairs=args.example_pair,
        style_only_examples=args.style_only_examples,
        forbid_terms=args.forbid,
        role_preamble=preamble_text,
    )


if __name__ == "__main__":
    main()
