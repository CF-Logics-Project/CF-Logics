from __future__ import annotations

import csv
import importlib
import json
import os
import re
from pathlib import Path
from typing import Any

# All paths are relative to the repository root, where this script lives.
REPO_ROOT = Path(__file__).resolve().parent
OCR_OUTPUT_DIR = REPO_ROOT / "out" / "ocr"
TAGGING_OUTPUT_DIR = REPO_ROOT / "out" / "tagging"

PLACEHOLDER_MAP: dict[str, str] = {
    "person": "[person]",
    "community_foundation": "[community_foundation]",
    "community": "[community]",
}

REPLACEMENTS_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "replacements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "entity": {"type": "string"},
                },
                "required": ["text", "entity"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["replacements"],
    "additionalProperties": False,
}


def _normalize_entity_label(raw_label: str) -> str:
    label = raw_label.strip().lower()
    if label.startswith("[") and label.endswith("]"):
        label = label[1:-1]
    return label


def _extract_first_json_object(text: str) -> dict[str, Any]:
    fenced_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if fenced_match:
        return json.loads(fenced_match.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        preview = re.sub(r"\s+", " ", text).strip()
        raise ValueError(
            "Model output does not contain a JSON object. "
            f"Raw output preview: {preview[:300]!r}"
        )

    return json.loads(text[start : end + 1])


def _extract_replacements_fallback(text: str) -> list[dict[str, str]]:
    """Best-effort extraction of replacement pairs from malformed JSON-like output."""
    replacements: list[dict[str, str]] = []

    # Typical order: {"text": "...", "entity": "..."}
    pattern_text_entity = re.compile(
        r'"text"\s*:\s*"(?P<text>.*?)"\s*,\s*"entity"\s*:\s*"(?P<entity>[^"]+)"',
        re.DOTALL,
    )
    # Alternate order: {"entity": "...", "text": "..."}
    pattern_entity_text = re.compile(
        r'"entity"\s*:\s*"(?P<entity>[^"]+)"\s*,\s*"text"\s*:\s*"(?P<text>.*?)"',
        re.DOTALL,
    )

    for pattern in (pattern_text_entity, pattern_entity_text):
        for match in pattern.finditer(text):
            raw_text = match.group("text").strip()
            entity = match.group("entity").strip()
            if not raw_text or not entity:
                continue

            # Minimal unescaping for common model output artifacts.
            cleaned_text = (
                raw_text.replace('\\"', '"')
                .replace("\\n", "\n")
                .replace("\\t", "\t")
                .strip()
            )

            replacements.append({"text": cleaned_text, "entity": entity})

    # De-duplicate while preserving order.
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in replacements:
        key = (item["text"], item["entity"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped


def _build_tagging_prompt(prompt_instructions: str, markdown_text: str) -> str:
    return (
        f"{prompt_instructions}\n\n"
        "Return ONLY valid JSON in this exact format:\n"
        "{\n"
        '  "replacements": [\n'
        "    {\"text\": \"exact span from markdown\", \"entity\": \"person|community_foundation|community\"}\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "1) Use exact substrings from the markdown for each text value.\n"
        "2) Do not include items not present in the markdown.\n"
        "3) Do not add explanations or any text outside JSON.\n\n"
        "Markdown input:\n"
        f"{markdown_text}"
    )


def _generate_with_local_model(
    prompt: str,
    model_id: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError(
            "transformers is required for local inference. Add it to requirements and install."
        ) from exc

    text_generator = pipeline(
        task="text-generation",
        model=model_id,
        tokenizer=model_id,
        device_map="auto",
    )

    outputs = text_generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        return_full_text=False,
    )
    return outputs[0]["generated_text"]


def _generate_with_gemini_api(
    prompt: str,
    model_id: str,
    api_key: str | None,
    max_new_tokens: int,
    temperature: float,
    thinking_level: str | None = "high",
) -> str:
    resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_api_key:
        raise ValueError("Missing Gemini API key. Set GEMINI_API_KEY or pass api_key.")

    try:
        genai = importlib.import_module("google.genai")
        types = importlib.import_module("google.genai.types")
    except ImportError as exc:
        raise ImportError(
            "google-genai is required for Gemini API mode. Add it to requirements and install."
        ) from exc

    client = genai.Client(api_key=resolved_api_key)
    config_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_new_tokens,
        "response_mime_type": "application/json",
        "response_json_schema": REPLACEMENTS_RESPONSE_SCHEMA,
    }

    # Gemma API quality is often better when explicit thinking is enabled.
    if thinking_level and model_id.strip().lower().startswith("gemma"):
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)

    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )

    # Prefer SDK-level parsed output when a schema is provided.
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, dict):
        return json.dumps(parsed)

    return response.text or ""


def _substitute_entities(
    markdown_text: str,
    replacements: list[dict[str, str]],
) -> tuple[str, int, list[dict[str, str]]]:
    updated_text = markdown_text
    substitution_count = 0
    substitutions_log: list[dict[str, str]] = []

    # Replace longer matches first to reduce nested overlap issues.
    ordered = sorted(replacements, key=lambda item: len(item["text"]), reverse=True)
    seen_pairs: set[tuple[str, str]] = set()

    for item in ordered:
        source_text = item["text"].strip()
        entity_label = _normalize_entity_label(item["entity"])

        if not source_text or entity_label not in PLACEHOLDER_MAP:
            continue

        pair = (source_text, entity_label)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        placeholder = PLACEHOLDER_MAP[entity_label]

        # Use direct escaped replacement so punctuation-heavy names still match reliably.
        pattern = re.escape(source_text)
        updated_text, hits = re.subn(pattern, placeholder, updated_text)
        substitution_count += hits

        if hits > 0:
            substitutions_log.extend(
                {"recognized_text": source_text, "entity": placeholder} for _ in range(hits)
            )

    return updated_text, substitution_count, substitutions_log


def _write_substitutions_csv(csv_path: Path, rows: list[dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["recognized_text", "entity"])
        writer.writeheader()
        writer.writerows(rows)


def process_markdown_entities(
    input_md_path: str | Path,
    output_md_path: str | Path,
    prompt_instructions: str,
    output_csv_path: str | Path | None = None,
    mode: str = "local",
    local_model_id: str = "google/gemma-4-26B-A4B-it",
    api_model_id: str = "gemini-2.5-pro",
    api_key: str | None = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    api_thinking_level: str | None = "high",
) -> dict[str, Any]:
    """Find target entities in markdown and replace them with placeholders.

    The entity definition policy comes from prompt_instructions provided by caller.
    """
    input_path = Path(input_md_path)
    output_path = Path(output_md_path)
    csv_path = Path(output_csv_path) if output_csv_path is not None else output_path.with_suffix(".csv")

    markdown_text = input_path.read_text(encoding="utf-8")
    llm_prompt = _build_tagging_prompt(prompt_instructions=prompt_instructions, markdown_text=markdown_text)

    mode_normalized = mode.strip().lower()
    if mode_normalized == "local":
        raw_response = _generate_with_local_model(
            prompt=llm_prompt,
            model_id=local_model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        backend_model = local_model_id
    elif mode_normalized == "api":
        raw_response = _generate_with_gemini_api(
            prompt=llm_prompt,
            model_id=api_model_id,
            api_key=api_key,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            thinking_level=api_thinking_level,
        )
        backend_model = api_model_id
    else:
        raise ValueError("mode must be either 'local' or 'api'.")

    try:
        parsed = _extract_first_json_object(raw_response)
        replacements = parsed.get("replacements", [])
        if not isinstance(replacements, list):
            raise ValueError("Model JSON must include a list field named 'replacements'.")
    except (ValueError, json.JSONDecodeError):
        replacements = _extract_replacements_fallback(raw_response)
        if not replacements:
            raise

    normalized: list[dict[str, str]] = []
    for item in replacements:
        if not isinstance(item, dict):
            continue
        text_value = str(item.get("text", "")).strip()
        entity_value = str(item.get("entity", "")).strip()
        if text_value and entity_value:
            normalized.append({"text": text_value, "entity": entity_value})

    tagged_text, substitutions, substitutions_log = _substitute_entities(
        markdown_text=markdown_text,
        replacements=normalized,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(tagged_text, encoding="utf-8")
    _write_substitutions_csv(csv_path=csv_path, rows=substitutions_log)

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "output_csv_path": str(csv_path),
        "backend": mode_normalized,
        "model": backend_model,
        "replacement_candidates": len(normalized),
        "substitutions_applied": substitutions,
        "csv_rows_written": len(substitutions_log),
    }


DEFAULT_TAGGING_PROMPT = """
You are an information extraction assistant. There are three types of entities you must identify in a markdown text, everything that is not one of those entities, should be left unchanged.
Once you identify one of these three types of entities, you need to substitute the text with its placeholder:
1) placeholder [person]: any specific human person name and potentially surname, the entire name and surname is one single instance of the entity.
2) placeholder [community_foundation]: a named community foundation (X) or any language used by the community foundation to refer to itself (e.g. "community foundation", "the foundation", "we", community foundation X, acronym of community foundation X etc.)
3) placeholder [community]: a named community, neighborhood, local population group, or locality-based collective. Any language used by the community foundation X to refer to the community (e.g. "community", "the community", "our community", "the residents of X", "the people of X", etc.) is also considered a mention of the entity [community].

Substitution policy:
- Replace each detected specific entity mention with its placeholder:
  - person -> [person]
  - community_foundation -> [community_foundation]
  - community -> [community]

- Return exact text spans as they appear in the markdown.
- Do not invent entities.
- Do not include placeholders in the extracted text field.
""".strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Replace entity names with tags ("out_<report_id>_AR.md" -> "tagging_out_<report_id>_AR.md").'
    )
    parser.add_argument(
        "--report",
        help="Report identifier, e.g. 363466802_2021. Default: every Markdown file in the input folder.",
    )
    parser.add_argument("--input-dir", default=str(OCR_OUTPUT_DIR), help=f"OCR output folder (default: {OCR_OUTPUT_DIR}).")
    parser.add_argument("--output-dir", default=str(TAGGING_OUTPUT_DIR), help=f"Output folder (default: {TAGGING_OUTPUT_DIR}).")
    parser.add_argument("--mode", default="api", choices=["api", "local"], help="'api' calls Gemini (needs GEMINI_API_KEY); 'local' runs Gemma.")
    parser.add_argument("--prompt-file", help="Text file with the tagging instructions. Default: DEFAULT_TAGGING_PROMPT.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else DEFAULT_TAGGING_PROMPT
    md_paths = [input_dir / f"out_{args.report}_AR.md"] if args.report else sorted(input_dir.glob("out_*_AR.md"))

    for md_path in md_paths:
        if not md_path.is_file():
            raise FileNotFoundError(f"Markdown file not found: {md_path}")
        result = process_markdown_entities(
            input_md_path=md_path,
            output_md_path=output_dir / f"tagging_{md_path.name}",
            output_csv_path=output_dir / f"tagging_{md_path.stem}.csv",
            prompt_instructions=prompt,
            mode=args.mode,
        )
        print(result)
