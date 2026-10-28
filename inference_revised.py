"""
2D_inference.py — Quality-coding pipeline for two-dimensional (element / logic)
yes-no interrogation of text chunks, using either Anthropic (Claude) or
Google (Gemini/Gemma) as the backing LLM provider.

This module defines functions only. It does not store, read, or hardcode any
API key. API keys, input/output paths, and the provider choice are supplied
by the caller (intended to be a Jupyter notebook, e.g. 2D_inference.ipynb).

Required packages (install whichever provider(s) you use):
    pip install pandas
    pip install anthropic        # for provider="anthropic"
    pip install google-genai     # for provider="google"

--------------------------------------------------------------------------
Question structure ("2D" = two nested dimensions of questions)
--------------------------------------------------------------------------
QUESTIONS is a list of "element" questions. Every text chunk is asked every
element question. Each element question carries its own nested list of
"logic" questions:
  - If the element question is answered "yes", every one of its nested logic
    questions is also asked (still one independent API call per question).
  - If the element question is answered "no", none of its nested logic
    questions are asked; the output still records why they were skipped.

Replace the placeholder "text" values below with the real questions.

--------------------------------------------------------------------------
Output structure
--------------------------------------------------------------------------
One JSON object / CSV row is produced per input text (per row of the input
CSV), carrying every original input column plus, for every element and
logic question, the columns "{question_id}_asked", "{question_id}_answer",
"{question_id}_explanation", "{question_id}_raw_response", and
"{question_id}_error".

--------------------------------------------------------------------------
Saving cadence and API rate limiting
--------------------------------------------------------------------------
- checkpoint_every: results are (re-)written to the JSON/CSV output paths
  after every `checkpoint_every` completed input texts. Default is 1 (save
  after every single text), since a single text's worth of questions can
  already be sizeable.
- calls_before_pause / pause_seconds: a running counter of model calls
  (shared across the whole run, not reset per text) triggers a
  `pause_seconds` sleep every `calls_before_pause` calls, so a strict
  per-minute API quota (e.g. a free tier capped at 30 requests/minute) is
  not exceeded even when a single text's questions alone approach the
  limit. Every real call attempt counts toward this, including retries.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger("2D_inference")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------
# Provider / model configuration
# --------------------------------------------------------------------------

DEFAULT_MODELS = {
    "anthropic": "claude-opus-4-6",
    "google": "gemma-4-31b-it",
}


# --------------------------------------------------------------------------
# Questions — text is for the element level question; logic_questions is a list of nested logic questions inside a given element question.
# Each question, both element and logic questions, has an "id" (used in the output column names) and a "text" (the actual question to ask the model).
# --------------------------------------------------------------------------

QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "element_community",
        "text": "Does this chunk describe a community, a group of people, a 'we', or the ideal of how people build and maintain community?",
        "logic_questions": [
            {
                "id": "element_community_logic_PS",
                "text": "Does this chunk align with the logic of: Community, the group of people, or 'we' currently live, used to live, or have family connections to the same place and share the same values, traditions, and heritage?",
            },
            {
                "id": "element_community_logic_DS",
                "text": "Does this chunk align well with the logic of: Community, the group of people, or the 'we' express shared interests, values, and/or identity but may not live in the same place?",
            },
            {
                "id": "element_community_logic_PPG",
                "text": "Does this chunk align with the logic of: Community, the group of people, or the 'we' currently live in the same place and depend on one another to address long-term local problems?",
            },
            {
                "id": "element_community_logic_SC",
                "text": "Does this chunk align with the logic of: Community, the group of people, or the 'we' currently live in the same place and share a commitment and moral duty to come together even though they have diverse values, traditions, identities, cultures, languages, and perspectives on local problems?",
            },
            {
                "id": "element_community_contrastive_logic",
                "text": "Reply with the two most representative id numbers separated by a '/': Which two of the following statements best align with how the chunk describes a community, a group of people, a 'we', or the ideal of how people build and maintain community? 1. Community, the group of people, or the 'we' currently live, used to live, or have family connections to the same place and cultivate shared values, traditions, and heritage; 2. Community, the group of people, or the 'we' express shared interests, values, and/or identity but may not live in the same place; 3. Community, the group of people, or the 'we' currently live in the same place and depend on one another to address long-term local problems; 4. Community, the group of people, or the 'we' currently live in the same place and share a commitment and moral duty to come together even though they have diverse values, traditions, identities, cultures, languages, and perspectives on local problems.",   
                
            },
        ],
    },
    {
        "id": "element_donor",
        "text": "Does this chunk describe the ideal of philanthropy, the importance of charitable giving, making donations to community foundation funds, or supporting causes through the community foundation?",
        "logic_questions": [
            {
                "id": "element_donor_logic_PS",
                "text": "Does this chunk align with the logic of: Philanthropy, charitable giving, and donations should be 'unrestricted' and support the values, traditions, and heritage of the place where people currently live, used to live, or have family connections so that the place can be resilient when there is an unexpected need, crisis or disaster?",
            },
            {
                "id": "element_donor_logic_DS",
                "text": "Does this chunk align with the logic of: Philanthropy, charitable giving, and donations should support causes and funds that express an individual's interests, values, and identities?",
            },
            {
                "id": "element_donor_logic_PPG",
                "text": "Does this chunk align with the logic of: Philanthropy, charitable giving, and donations should support cross-sector initiatives that bring together business, government, and foundation leaders to assess needs, evaluate solutions, and strategically address long-term problems that effect everyone such as economic development, disabilty access, and public services?",
            },
            {
                "id": "element_donor_logic_SC",
                "text": "Does this chunk align with the logic of: Philanthropy, charitable giving, and donations are a moral duty and should support causes such as justice, diversity, and issues affecting marginalized groups?",
            },
            {
                "id": "element_donor_contrastive_logic",
                "text": "Reply with the two most representative id numbers separated by a '/': Which two of the following statements best align with how the chunk describes the ideal of philanthropy, the importance of charitable giving, making donations to community foundation funds, or supporting causes through the community foundation? 1. Philanthropy, charitable giving, and donations should be 'unrestricted' and support the values, traditions, and heritage of the place where people currently live, used to live, or have family connections so that the place can be resilient when there is an unexpected need, crisis or disaster; 2. Philanthropy, charitable giving, and donations should support causes and funds that express an individual's interests, values, and identities; 3. Philanthropy, charitable giving, and donations should support cross-sector initiatives that bring together business, government, and foundation leaders to assess needs, evaluate solutions, and strategically address long-term problems that effect everyone such as economic development, disabilty access, and public services; 4. Philanthropy, charitable giving, and donations are a moral duty and should support causes such as justice, diversity, and issues affecting marginalized groups.",
                
            },
        ],
    },
    {
        "id": "element_activity",
        "text": "Does this chunk describe activities of the community foundation, the value it provides, or the way it works with other organizations?",
        "logic_questions": [
            {
                "id": "element_activity_logic_PS",
                "text": "Does this chunk align with the logic of: Community foundations are charitable banks that protect and grow endowments so that people can leave a 'community legacy' by making donations that support the values, traditions, and heritage of the place where they currently live, used to live, or have family connections; they make grants to local nonprofits and provide support when there is an unexpected need, crisis or disaster?",
            },
            {
                "id": "element_activity_logic_DS",
                "text": "Does this chunk align with the logic of: Community foundations are philanthropic advisors that provide customer and fund management services and advise donors on charitable goals so that they can leave a 'personal legacy' by supporting causes that express their individual interests, values, and identities?",
            },
            {
                "id": "element_activity_logic_PPG",
                "text": "Does this chunk align with the logic of: Community foundations are community leaders that convene, coordinate, and support cross-sector initiatives; they work with businesses, governments, other foundations, and leaders to assess needs, evaluate solutions, and strategically address long-term local problems such as economic development, disabilty access, and public services?",
            },
            {
                "id": "element_activity_logic_SC",
                "text": "Does this chunk align with the logic of: Community foundations are activists and advocates that help people with diverse values, traditions, identities, cultures, languages, and perspectives on local problems come together to do what is morally right. They support causes such as justice, diversity, and issues affecting marginalized groups?",
            },
            {
                "id": "element_activity_contrastive_logic",
                "text": "Reply with the two most representative id numbers separated by a '/': Which two of the following statements best align with how the chunk describes activities of the community foundation, the value it provides, or the way it works with other organizations? 1. Community foundations are charitable banks that protect and grow endowments so that people can leave a 'community legacy' by making donations that support the values, traditions, and heritage of the place where they currently live, used to live, or have family connections; they make grants to local nonprofits and provide support when there is an unexpected need, crisis or disaster; 2. Community foundations are philanthropic advisors that provide customer and fund management services and advise donors on charitable goals so that they can leave a 'personal legacy' by supporting causes that express their individual interests, values, and identities; 3. Community foundations are community leaders that convene, coordinate, and support cross-sector initiatives; they work with businesses, governments, other foundations, and leaders to assess needs, evaluate solutions, and strategically address long-term local problems such as economic development, disabilty access, and public services; 4. Community foundations are activists and advocates that help people with diverse values, traditions, identities, cultures, languages, and perspectives on local problems come together to do what is morally right. They support causes such as justice, diversity, and issues affecting marginalized groups.",   
                
            },
        ],
    },
    {
        "id": "element_board",
        "text": "Does this chunk describe the community foundation board members or committee members?",
        "logic_questions": [
            {
                "id": "element_board_logic_PS",
                "text": "Does this chunk align with the logic of: Community foundation board members and committee members have high social, political and economic status; they ensure that the foundation gets good investment returns and effectively manages community endowments used to make grants to local nonprofits, and provide support when there is an unexpected need, crisis or disaster?",
            },
            {   
                "id": "element_board_logic_DS",
                "text": "Does this chunk align with the logic of: Community foundation board members and committee members are wealthy individuals; they ensure that the foundation effectively manages donor-advised funds used to support that express the interests, values, and identities of different donors?",            
            },
            {
                "id": "element_board_logic_PPG",
                "text": "Does this chunk align with the logic of: Community foundation board members and committee members are business, government, and philanthropic leaders that often have professional credentials (MD, MBA, MPA, JD, Esq); they ensure that the foundation works with businesses, governments, other foundations, and leaders to do needs assessments, program evaluations, and strategically addresses long-term local problems such as economic development, disabilty access, and public services?",
            },
            {
                "id": "element_board_logic_SC",
                "text": "Does this chunk align with the logic of: Community foundation board and committee members represent diverse values, traditions, identities, cultures, languages, and perspectives on local problems; they ensure the foundation is an activist and advocate that brings diverse people together to do what is morally right, and support causes such as justice, diversity, and issues affecting marginalized groups?",
            },
            {
                "id": "element_board_contrastive_logic",
                "text": "Reply with the two most representative id numbers separated by a '/': Which two of the following statements best align with how the chunk describes the community foundation board members or committee members? 1. Community foundation board members and committee members have high social, political and economic status; they ensure that the foundation gets good investment returns and effectively manages community endowments used to make grants to local nonprofits, and provide support when there is an unexpected need, crisis or disaster; 2. Community foundation board members and committee members are wealthy individuals; they ensure that the foundation effectively manages donor-advised funds used to support that express the interests, values, and identities of different donors; 3. Community foundation board members and committee members are business, government, and philanthropic leaders that often have professional credentials (MD, MBA, MPA, JD, Esq); they ensure that the foundation works with businesses, governments, other foundations, and leaders to do needs assessments, program evaluations, and strategically addresses long-term local problems such as economic development, disabilty access, and public services; 4. Community foundation board and committee members represent diverse values, traditions, identities, cultures, languages, and perspectives on local problems; they ensure the foundation is an activist and advocate that brings diverse people together to do what is morally right, and support causes such as justice, diversity, and issues affecting marginalized groups.",   
                
            },
        ],
    },
    {
        "id": "element_grantee",
        "text": "Does this chunk describe causes, organizations, and grantees that receive funding or grants from the community foundation?",
        "logic_questions": [
            {
                "id": "element_grantee_logic_PS",
                "text": "Does this chunk align with the logic of: The community foundation provides funding and grants to the same organizations every year for causes such as 'community betterment',  'community needs', and emergency or disaster support?",    
            },
            {
                "id": "element_grantee_logic_DS",
                "text": "Does this chunk align with the logic of: The community foundation provides funding and grants to causes and organizations chosen by donors and uses donor-advised funds to support eligible charities and individuals who apply for scholarships?",    
            },
            {
                "id": "element_grantee_logic_PPG",
                "text": "Does this chunk align with the logic of: The community foundation provides funding and grants for 'long-term solutions' and 'impact investment' based on needs assessments, program evaluation, and strategic planning?",
            },
            {
                "id": "element_grantee_logic_SC",
                "text": "Does this chunk align with the logic of: The community foundation simplifies grant application and reporting so that it can provide funding and grants to under-resourced organizations and individuals, and support causes such as justice, diversity, and issues affecting marginalized groups?",
            },
            {
                "id": "element_grantee_contrastive_logic",
                "text": "Reply with the two most representative id numbers separated by a '/': Which two of the following statements best align with how the chunk describes causes, organizations, and grantees that receive funding or grants from the community foundation? 1. The community foundation provides funding and grants to the same organizations every year for causes such as 'community betterment', 'community needs', and emergency or disaster support; 2. The community foundation provides funding and grants to causes and organizations chosen by donors and uses donor-advised funds to support eligible charities and individuals who apply for scholarships; 3. The community foundation provides funding and grants for 'long-term solutions' and 'impact investment' based on needs assessments, program evaluation, and strategic planning; 4. The community foundation simplifies grant application and reporting so that it can provide funding and grants to under-resourced organizations and individuals, and support causes such as justice, diversity, and issues affecting marginalized groups.",   
                
            },
        ],
    },
]


# --------------------------------------------------------------------------
# Provider routing
# --------------------------------------------------------------------------

def _normalize_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized in ("anthropic", "claude"):
        return "anthropic"
    if normalized in ("google", "gemini", "gemma"):
        return "google"
    raise ValueError(f"Unknown provider '{provider}'. Use 'anthropic' or 'google'.")


def create_client(provider: str, api_key: str):
    """Create the provider-specific SDK client. The api_key lives only in
    this in-memory client object; it is never written to disk by this module.
    """
    provider = _normalize_provider(provider)

    if provider == "anthropic":
        import anthropic

        return anthropic.Anthropic(api_key=api_key)

    if provider == "google":
        from google import genai

        return genai.Client(api_key=api_key)

    raise ValueError(f"Unknown provider '{provider}'.")


def _call_anthropic(client, model: str, prompt: str, max_tokens: int) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    text_blocks = [block.text for block in response.content if block.type == "text"]
    return "".join(text_blocks).strip()


def _call_google(client, model: str, prompt: str, max_tokens: int) -> str:
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={"max_output_tokens": max_tokens},
    )
    return (response.text or "").strip()


def _call_model(provider: str, client, model: str, prompt: str, max_tokens: int) -> str:
    provider = _normalize_provider(provider)
    if provider == "anthropic":
        return _call_anthropic(client, model, prompt, max_tokens)
    if provider == "google":
        return _call_google(client, model, prompt, max_tokens)
    raise ValueError(f"Unknown provider '{provider}'.")


# --------------------------------------------------------------------------
# API rate limiting
# --------------------------------------------------------------------------

class RateLimiter:
    """Pauses for `pause_seconds` every `calls_before_pause` model calls.

    One instance is shared across an entire run_pipeline() call, so the
    limit applies across the whole run (across texts, not just within a
    single text's questions). Set calls_before_pause to None (or 0) to
    disable pausing entirely.
    """

    def __init__(self, calls_before_pause: Optional[int], pause_seconds: float):
        self.calls_before_pause = calls_before_pause
        self.pause_seconds = pause_seconds
        self.call_count = 0

    def record_call(self) -> None:
        if not self.calls_before_pause:
            return
        self.call_count += 1
        if self.call_count % self.calls_before_pause == 0:
            logger.info(
                "Reached %d model calls; pausing %.0fs to respect API rate limits.",
                self.call_count, self.pause_seconds,
            )
            time.sleep(self.pause_seconds)


def _call_with_retry(
    call_fn,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
    rate_limiter: Optional[RateLimiter] = None,
):
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            result = call_fn()
            if rate_limiter is not None:
                rate_limiter.record_call()
            return result
        except Exception as exc:
            if rate_limiter is not None:
                rate_limiter.record_call()
            last_exc = exc
            logger.warning("Attempt %d/%d failed: %s", attempt, max_retries, exc)
            if attempt < max_retries:
                time.sleep(backoff_seconds * attempt)
    raise last_exc


# --------------------------------------------------------------------------
# Prompt building & response parsing
# --------------------------------------------------------------------------

def _build_prompt(text_chunk: str, question_text: str) -> str:
    return (
        "You are analyzing a chunk of text to answer a specific yes/no question about it.\n\n"
        "Text:\n"
        '"""\n'
        f"{text_chunk}\n"
        '"""\n\n'
        f"Question: {question_text}\n\n"
        "Respond in exactly this format:\n"
        "Answer: Yes or No\n"
        "Explanation: <a short explanation of your reasoning>"
    )


_ANSWER_PATTERN = re.compile(r"answer\s*:\s*(yes|no)\b", re.IGNORECASE)
_EXPLANATION_PATTERN = re.compile(r"explanation\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)
_LEADING_YES_NO_PATTERN = re.compile(r"\s*(yes|no)\b", re.IGNORECASE)


def parse_answer(raw_text: str) -> "tuple[Optional[str], str]":
    """Parse a model response into (answer, explanation).

    answer is "yes", "no", or None if no yes/no could be found.
    """
    raw_text = (raw_text or "").strip()

    answer_match = _ANSWER_PATTERN.search(raw_text)
    explanation_match = _EXPLANATION_PATTERN.search(raw_text)

    answer = answer_match.group(1).lower() if answer_match else None
    explanation = explanation_match.group(1).strip() if explanation_match else raw_text

    if answer is None:
        leading = _LEADING_YES_NO_PATTERN.match(raw_text)
        if leading:
            answer = leading.group(1).lower()
            explanation = raw_text[leading.end():].strip(" .:-\n") or raw_text

    return answer, explanation


# --------------------------------------------------------------------------
# Per-question / per-text-chunk execution
# --------------------------------------------------------------------------

def ask_question(
    provider: str,
    client,
    model: str,
    text_chunk: str,
    question_text: str,
    max_tokens: int = 300,
    max_retries: int = 3,
    rate_limiter: Optional[RateLimiter] = None,
) -> Dict[str, Any]:
    """Ask a single independent yes/no question about a single text chunk."""
    prompt = _build_prompt(text_chunk, question_text)

    def _do_call():
        return _call_model(provider, client, model, prompt, max_tokens)

    try:
        raw_response = _call_with_retry(
            _do_call, max_retries=max_retries, rate_limiter=rate_limiter,
        )
        answer, explanation = parse_answer(raw_response)
        error = None if answer in ("yes", "no") else "Could not parse a yes/no answer from the model response."
    except Exception as exc:
        raw_response = None
        answer = None
        explanation = None
        error = str(exc)

    return {
        "raw_response": raw_response,
        "answer": answer,
        "explanation": explanation,
        "error": error,
    }


def _add_question_fields(
    record: Dict[str, Any],
    question_id: str,
    asked: bool,
    raw_response: Optional[str],
    answer: Optional[str],
    explanation: Optional[str],
    error: Optional[str],
) -> None:
    record[f"{question_id}_asked"] = asked
    record[f"{question_id}_answer"] = answer
    record[f"{question_id}_explanation"] = explanation
    record[f"{question_id}_raw_response"] = raw_response
    record[f"{question_id}_error"] = error


def process_text_chunk(
    provider: str,
    client,
    model: str,
    text_index: int,
    row: Dict[str, Any],
    text_column: str = "text",
    questions: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 300,
    max_retries: int = 3,
    sleep_between_calls: float = 0.0,
    rate_limiter: Optional[RateLimiter] = None,
) -> Dict[str, Any]:
    """Run the full element + logic question tree against one text chunk.

    Returns a single flat record: every original input column plus, per
    question, "{question_id}_asked/_answer/_explanation/_raw_response/_error".
    A logic question that was skipped (its element question was "no") is
    recorded with asked=False and an explanation describing why.
    """
    questions = questions if questions is not None else QUESTIONS
    text_chunk = str(row[text_column])

    record: Dict[str, Any] = dict(row)
    record["text_index"] = text_index
    record["provider"] = provider
    record["model"] = model

    for element in questions:
        element_result = ask_question(
            provider, client, model, text_chunk, element["text"],
            max_tokens=max_tokens, max_retries=max_retries, rate_limiter=rate_limiter,
        )
        _add_question_fields(record, element["id"], asked=True, **element_result)
        if sleep_between_calls:
            time.sleep(sleep_between_calls)

        element_answer = element_result["answer"]
        logic_questions = element.get("logic_questions", [])

        if element_answer == "yes":
            for logic in logic_questions:
                logic_result = ask_question(
                    provider, client, model, text_chunk, logic["text"],
                    max_tokens=max_tokens, max_retries=max_retries, rate_limiter=rate_limiter,
                )
                _add_question_fields(record, logic["id"], asked=True, **logic_result)
                if sleep_between_calls:
                    time.sleep(sleep_between_calls)
        else:
            skip_reason = (
                f"Not asked: element question '{element['id']}' was answered "
                f"'{element_answer}'."
            )
            for logic in logic_questions:
                _add_question_fields(
                    record, logic["id"], asked=False,
                    raw_response=None, answer=None, explanation=skip_reason, error=None,
                )

    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    return record


# --------------------------------------------------------------------------
# Saving results
# --------------------------------------------------------------------------

def save_results(records: List[Dict[str, Any]], output_json_path: str, output_csv_path: str) -> None:
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    pd.DataFrame(records).to_csv(output_csv_path, index=False)


# --------------------------------------------------------------------------
# Top-level pipeline entry point (called from the notebook)
# --------------------------------------------------------------------------

def run_pipeline(
    input_csv_path: str,
    output_json_path: str,
    output_csv_path: str,
    provider: str,
    api_key: str,
    model: Optional[str] = None,
    text_column: str = "text",
    questions: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 300,
    max_retries: int = 3,
    sleep_between_calls: float = 0.0,
    checkpoint_every: int = 1,
    calls_before_pause: Optional[int] = 30,
    pause_seconds: float = 60.0,
) -> pd.DataFrame:
    """Run the full 2D (element + logic) question pipeline over every row of
    the input CSV's text_column, using the given provider and API key.

    One record (JSON object / CSV row) is produced per input text. Results
    are (re-)written to output_json_path and output_csv_path every
    `checkpoint_every` completed texts, so progress survives an
    interruption partway through a run. A shared RateLimiter pauses for
    `pause_seconds` every `calls_before_pause` model calls across the whole
    run (set calls_before_pause=None to disable). Returns the results as a
    DataFrame.
    """
    provider = _normalize_provider(provider)
    model = model or DEFAULT_MODELS[provider]
    questions = questions if questions is not None else QUESTIONS

    df = pd.read_csv(input_csv_path)
    if text_column not in df.columns:
        raise ValueError(f"Input CSV is missing the required '{text_column}' column.")

    client = create_client(provider, api_key)
    rate_limiter = RateLimiter(calls_before_pause, pause_seconds)

    input_rows = df.to_dict("records")
    total = len(input_rows)
    all_records: List[Dict[str, Any]] = []

    for text_index, row in enumerate(input_rows):
        logger.info("Processing text %d/%d", text_index + 1, total)
        record = process_text_chunk(
            provider, client, model, text_index, row, text_column,
            questions=questions, max_tokens=max_tokens, max_retries=max_retries,
            sleep_between_calls=sleep_between_calls, rate_limiter=rate_limiter,
        )
        all_records.append(record)

        if checkpoint_every and (text_index + 1) % checkpoint_every == 0:
            save_results(all_records, output_json_path, output_csv_path)
            logger.info("Checkpoint saved after %d/%d texts.", text_index + 1, total)

    save_results(all_records, output_json_path, output_csv_path)
    logger.info("Done. Wrote %d records to %s and %s.", len(all_records), output_json_path, output_csv_path)

    return pd.DataFrame(all_records)
