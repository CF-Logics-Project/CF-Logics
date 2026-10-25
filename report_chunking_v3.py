"""Split an annual-report Markdown file into semantically grouped CSV chunks.

Run this script with its Python dependencies installed. It prompts for the report folder and the
identifier in `tagging_out_<identifier>_AR.md`, then writes both CSV outputs beside that report.
EmbeddingGemma runs locally, so no embedding-service API key is used. Hugging Face access is required
for this gated model; accept Google's model license and provide a Hugging Face token if not already
logged in.
"""

from getpass import getpass
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from huggingface_hub import get_token
from transformers import AutoModel, AutoTokenizer
from wtpsplit import SaT

MODEL_NAME = "google/embeddinggemma-300m"
THRESHOLD = 0.70
MIN_CHUNK_LENGTH = 0
MAX_CHUNK_LENGTH = 3000


def load_text(path):
    source_path = Path(path)
    if source_path.exists():
        return source_path.read_text(encoding="utf-8", errors="ignore")
    return path


def preprocess_text(text):
    """Normalize markdown/ocr noise while preserving paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?m)^[ \t]*<!--[ \t]*image[ \t]*-->[ \t]*$", "", text)
    text = re.sub(r"(?m)^[ \t]*#{1,6}[ \t]*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_sentence_splitter():
    """Create the wtpsplit sentence splitter and move it to GPU if available."""
    sat = SaT("sat-12l-sm")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        sat.to(device)
        if device == "cuda":
            sat.half()
    except Exception:
        pass
    return sat, device


def looks_like_header_fragment(sentence):
    """Return True when a split fragment looks like a heading or label rather than a complete sentence.

    Examples of the intended pattern:
    - NECAT's
    - Hack.Diversity
    - Program X
    - The Neighborhood Developers
    - The [individual_tag] Honorary Fund

    These are usually short, lack terminal sentence punctuation, and are immediately followed by a
    sentence that continues the same title or narrative. Placeholder tags such as [individual_tag]
    can be part of a legitimate report title or donor name, but they are not complete sentences by
    themselves.
    """
    text = sentence.strip()
    if not text or len(text) > 60:
        return False
    if text.endswith((".", "!", "?")):
        return False

    # A bare placeholder like [individual_tag] is a valid heading fragment when it opens a donor or
    # award name, but it is not a complete sentence.
    if re.fullmatch(r"\[[A-Za-z0-9_]+\]", text):
        return True

    # Legitimate annual-report titles often contain spaces and title-case words, such as:
    # "The Neighborhood Developers"
    # "The [individual_tag] Honorary Fund"
    if re.search(r"\s", text):
        tokens = re.split(r"\s+", text)
        stopwords = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with"}

        for token in tokens:
            token = token.strip(" ,;:()")
            if not token:
                continue
            if re.fullmatch(r"\[[A-Za-z0-9_]+\]", token):
                continue
            # Preserve placeholder tags while still validating the rest of the token.
            raw_token = token
            if raw_token.startswith("[") and raw_token.endswith("]"):
                if re.fullmatch(r"\[[A-Za-z0-9_]+\]", raw_token):
                    continue
            if not re.fullmatch(r"[A-Za-z0-9&/.'’()\-]+", token):
                return False
            if token.lower() in stopwords:
                continue
            if token[0].isupper():
                continue
            return False

        return True

    if not re.search(r"[A-Za-z]", text):
        return False

    # This deliberately only accepts compact header-like tokens: titles, names, initials, acronyms,
    # apostrophes, slashes, ampersands, punctuation-heavy organization names, and similar labels.
    if re.fullmatch(r"[A-Za-z0-9&/.'’()\-]+", text):
        return True
    return False


def starts_with_lowercase_continuation(text):
    """Return True when the sentence begins with a continuation, ignoring leading quotes."""
    if not text:
        return False
    trimmed = text.lstrip("'\"“”‘’")
    if not trimmed:
        return False

    # A title may be followed by a sentence beginning with a placeholder tag, such as
    # "[individual_tag] was raised...". The continuation check should look at the first lowercase
    # character after the tag, not reject the whole case outright.
    if trimmed.startswith("[") and "]" in trimmed:
        remainder = trimmed[trimmed.index("]") + 1 :].lstrip()
        if not remainder:
            return False
        return remainder[0].islower()

    return trimmed[0].islower()


def merge_header_continuations(sentences, paragraph_breaks):
    """Merge a heading-like fragment with the following sentence when it clearly continues the thought.

    This is a conservative repair step for cases like:
        "NECAT's"
        "goal is long-term success..."

    We keep the original wording exactly, we only join the adjacent fragments into a single string.
    """
    merged_sentences = []
    merged_breaks = []
    i = 0

    while i < len(sentences):
        current = sentences[i]

        if i + 1 < len(sentences):
            next_sentence = sentences[i + 1]
            # Rule: a heading-like fragment should merge only when the next sentence begins with a
            # lowercase continuation and the current fragment is not itself a complete sentence.
            if (
                looks_like_header_fragment(current)
                and next_sentence
                and starts_with_lowercase_continuation(next_sentence)
            ):
                merged_sentences.append(f"{current} {next_sentence}")
                merged_breaks.append(False)
                i += 2
                continue

        merged_sentences.append(current)
        merged_breaks.append(bool(i < len(paragraph_breaks) and paragraph_breaks[i]))
        i += 1

    return merged_sentences, merged_breaks


def split_sentences_wtpsplit(text, sat):
    """Split report text with wtpsplit and then apply a narrow structural repair pass.

    Important design choice: sentence splitting happens first, then a conservative repair step fixes
    header-like boundary errors without rewriting text. This keeps the pipeline transparent and easy
    to debug before embedding and cosine clustering happen.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    sentences = []
    paragraph_breaks = []

    for para_index, paragraph in enumerate(paragraphs):
        try:
            chunks = sat.split(paragraph)
        except TypeError:
            chunks = sat.split([paragraph])[0]

        if not isinstance(chunks, list):
            chunks = list(chunks)

        for idx, sentence in enumerate(chunks):
            cleaned = str(sentence).strip()
            if not cleaned:
                continue
            # We keep track of true paragraph transitions so the later cosine clustering does not
            # merge unrelated sections that happen to be adjacent in the document.
            paragraph_breaks.append(bool(para_index > 0 and idx == 0))
            sentences.append(cleaned)

    # This post-pass is intentionally narrow and deterministic: it only repairs cases where a short
    # header fragment was split away from the sentence that actually follows it.
    sentences, paragraph_breaks = merge_header_continuations(sentences, paragraph_breaks)
    return sentences, paragraph_breaks


def export_split_only_debug_csv(
    sentences,
    paragraph_breaks,
    path=None,
):
    """Persist only the sentence-split output; no embeddings or cosine clustering included.

    This is the most important debug artifact for this pipeline: it lets us inspect exactly what the
    splitter produced before semantic clustering happens. If a header is split from its sentence, we
    can see it here and fix it without changing the underlying wording.
    """
    if path is None:
        path = Path.cwd() / "sentence_check_v5.csv"

    debug_df = pd.DataFrame({
        "sentence_index": range(len(sentences)),
        "paragraph_break": paragraph_breaks,
        "sentence": sentences,
    })
    debug_df.to_csv(path, index=False)
    print(f"Saved split-only debug export to {path} with {len(sentences)} sentences")


def export_sentence_debug_csv(sentences, paragraph_breaks, path=None):
    """Backward-compatible alias for the split-only export stage."""
    export_split_only_debug_csv(sentences, paragraph_breaks, path=path)


# ============================================================
# 1. Load embedding model
# ============================================================

def load_embedding_model(token=None):
    """Download EmbeddingGemma and prepare it for local inference."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)
    model = AutoModel.from_pretrained(MODEL_NAME, token=token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


# ============================================================
# 2. Sentence embedding
# ============================================================

def embed_sentences(sentences, tokenizer, model, device, batch_size=16):
    """Generate normalized sentence embeddings with EmbeddingGemma.

    After splitting, we convert each sentence into an embedding. The embedding stage is intentionally
    separate from the split stage so that we can inspect the split output before any semantic merging
    occurs. This separation is critical for debugging whether an issue is in splitting or in later
    clustering.
    """
    if not sentences:
        return np.empty((0, 0))

    all_embeddings = []
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start:start + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        summed = (token_embeddings * attention_mask).sum(dim=1)
        divisor = attention_mask.sum(dim=1).clamp_min(1)
        pooled = summed / divisor
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        all_embeddings.append(normalized.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


# ============================================================
# 3. Cluster adjacent sentences by cosine similarity
# ============================================================

def cluster_text(sents, vecs, paragraph_breaks, threshold=THRESHOLD):
    """Create semantic chunks from adjacent sentence embeddings.

    This stage is intentionally kept separate from the sentence splitter. The split stage should be
    solved first, and the cluster stage should only merge adjacent sentences that are actually similar.
    Paragraph breaks are respected so section headers and new blocks do not get merged by accident.
    """
    if not sents:
        return []

    clusters = [[0]]
    for i in range(1, len(sents)):
        similarity = float(np.dot(vecs[i], vecs[i - 1]))
        if paragraph_breaks[i] or similarity < threshold:
            clusters.append([])
        clusters[-1].append(i)
    return clusters


# ============================================================
# 4. Materialize final chunks
# ============================================================

def clean_text(text):
    return text.strip()


def context_windows(sentences, start, end, window_size=15):
    """Return up to 15 words immediately before and after a chunk."""
    words_before = " ".join(sentences[:start]).split()
    words_after = " ".join(sentences[end + 1:]).split()
    return (
        " ".join(words_before[-window_size:]),
        " ".join(words_after[:window_size]),
    )


def materialize_chunks(sents, clusters):
    """Combine cluster members into chunk records."""
    chunks = []

    for cluster in clusters:
        cluster_text = clean_text(" ".join(sents[i] for i in cluster))
        if not cluster_text:
            continue

        if len(cluster_text) <= MAX_CHUNK_LENGTH:
            chunks.append({
                "start": cluster[0],
                "end": cluster[-1],
                "text": cluster_text,
                "length": len(cluster_text),
            })
            continue

        current = []
        current_indexes = []
        current_len = 0

        for idx in cluster:
            sentence = sents[idx]
            sentence_len = len(sentence) + 1
            if current and current_len + sentence_len > MAX_CHUNK_LENGTH:
                chunks.append({
                    "start": current_indexes[0],
                    "end": current_indexes[-1],
                    "text": clean_text(" ".join(current)),
                    "length": current_len,
                })
                current = []
                current_indexes = []
                current_len = 0

            current.append(sentence)
            current_indexes.append(idx)
            current_len += sentence_len

        if current:
            chunks.append({
                "start": current_indexes[0],
                "end": current_indexes[-1],
                "text": clean_text(" ".join(current)),
                "length": current_len,
            })

    return chunks


def process_report(
    report_path,
    save_debug=True,
    debug_path=None,
    hf_token=None,
):
    """Run the full pipeline: preprocess -> split -> export debug -> embed -> cosine cluster.

    The overall sequence matters. We split first, inspect the exact split output, then embed and only
    then cluster. This makes it possible to tell whether a problem came from the split stage or from the
    later cosine-similarity merge stage.
    """
    raw_text = load_text(report_path)
    text = preprocess_text(raw_text)
    sat, _ = build_sentence_splitter()
    sentences, paragraph_breaks = split_sentences_wtpsplit(text, sat)

    if save_debug:
        if debug_path is None:
            debug_path = Path(report_path).with_name(
                f"{Path(report_path).stem}_sentence_check_v5.csv"
            )
        export_split_only_debug_csv(sentences, paragraph_breaks, path=debug_path)

    # The embeddings are computed only after the splitting/debug stage passes. This keeps the process
    # verifiable and prevents accidental recombination of text before we inspect the split output.
    tokenizer, model, device = load_embedding_model(token=hf_token)
    vecs = embed_sentences(sentences, tokenizer, model, device)
    clusters = cluster_text(sentences, vecs, paragraph_breaks, threshold=THRESHOLD)
    chunk_records = materialize_chunks(sentences, clusters)

    for chunk in chunk_records:
        chunk["context_before"], chunk["context_after"] = context_windows(
            sentences,
            chunk["start"],
            chunk["end"],
        )

    chunks_df = pd.DataFrame(chunk_records)[[
        "context_before",
        "text",
        "context_after",
        "start",
        "end",
        "length",
    ]]
    return text, sentences, paragraph_breaks, vecs, chunks_df


if __name__ == "__main__":
    default_folder = Path(__file__).resolve().parent
    folder_input = input(
        f"Folder containing the report Markdown file [{default_folder}]: "
    ).strip()
    report_folder = Path(folder_input).expanduser() if folder_input else default_folder
    report_name = input(
        "Report name (identifier only, e.g. 541950727_2019): "
    ).strip()
    if not report_name:
        raise ValueError("Report name cannot be empty.")

    report_path = report_folder / f"tagging_out_{report_name}_AR.md"
    if not report_path.is_file():
        raise FileNotFoundError(f"Report file not found: {report_path}")

    # The model runs locally, but Hugging Face requires account access to this gated model.
    # A saved Hugging Face login is reused; otherwise ask without echoing the token.
    hf_token = get_token()
    if not hf_token:
        print(
            "EmbeddingGemma is gated on Hugging Face. Accept Google's license at "
            "https://huggingface.co/google/embeddinggemma-300m, then enter a "
            "Hugging Face user access token. This is not an embedding API key."
        )
        hf_token = getpass("Hugging Face access token: ").strip()
        if not hf_token:
            raise ValueError("A Hugging Face access token is required to download the model.")

    debug_sentence_path = report_folder / f"{report_name}_sentence_check_v5.csv"
    chunks_output_path = report_folder / f"annual_report_chunks_{report_name}_v5.csv"

    _, sentences, paragraph_breaks, _, chunks_df = process_report(
        report_path,
        save_debug=True,
        debug_path=debug_sentence_path,
        hf_token=hf_token,
    )

    chunks_df.to_csv(chunks_output_path, index=False)
    print(f"Generated {len(chunks_df)} chunks at {chunks_output_path}")
    print(f"Sentence count: {len(sentences)}")
    print(f"Paragraph break count: {sum(1 for x in paragraph_breaks if x)}")
