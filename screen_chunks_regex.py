"""
Rule-based (deterministic) screening of the text chunks produced by the
sequence-chunking pipeline.

WHY THIS SCRIPT EXISTS
----------------------
An earlier version of this step (`screen_chunks.py`) asked an LLM a set of
yes/no questions about every chunk. That approach was unreliable, so it has been
replaced by this script: every screening question is answered here by an
explicit, deterministic rule (regular expressions, tokenization + word-list
matching, character counts). No model, no API key, no randomness -- the same
input always produces exactly the same output.

WHAT IT DOES
------------
For every .csv file in the input folder (default: "out/chunked reports"):
  1. read the file,
  2. look at the chunk text stored in the "text" column of each row,
  3. apply each screening rule to that text, writing "yes" or "no" into a new
     column named after the rule,
  4. save a copy of the file -- all original columns preserved, in their
     original order, plus the new screening columns appended at the end -- into
     the output folder (default: "out/screening") under the same file name with
     a "screening_" prefix.

Example: "out/chunked reports/annual_report_chunks_42104021_2023_v5.csv"
      -> "out/screening/screening_annual_report_chunks_42104021_2023_v5.csv"

SCREENING RULES
---------------
All four rules share the same polarity: "yes" flags a chunk to be screened OUT
of the later analysis, "no" means the chunk is kept.

  name          "yes" if the chunk is a single anonymisation name tag --
                "[individual_tag]" or "[community_foundation_tag]" -- on its
                own, apart from whitespace and/or punctuation around it.
  list_names    "yes" if the chunk is nothing but two or more name tags,
                separated and surrounded only by whitespace and punctuation.
                Any other word in the chunk makes it "no".
  non_english   "yes" if the chunk is largely written in a language other than
                English (English -> "no").
  min_length    "yes" if the chunk is 4 characters or shorter, OR if it
                contains no whitespace at all (i.e. it is a single unbroken
                token).

Rules are independent: a chunk can be "yes" on several of them at once.

USAGE
-----
    # screen every .csv in the default input folder
    python scr/screening/screen_chunks_regex.py

    # screen a single file
    python scr/screening/screen_chunks_regex.py \
        --input-file "out/chunked reports/annual_report_chunks_25-1650208_2021_v5.csv"

    # custom folders / custom text column
    python scr/screening/screen_chunks_regex.py \
        --input-dir "out/chunked reports" \
        --output-dir "out/screening" \
        --text-column text

Only dependency: pandas.

HOW TO ADD A NEW SCREENING RULE
-------------------------------
  1. Write a function `screen_<something>(text: str) -> str` that returns the
     string "yes" or "no" and nothing else. It must be pure and deterministic:
     same text in, same answer out.
  2. Register it in the SCREENINGS dictionary below. The dictionary key becomes
     the name of the new column in the output .csv, and the order of the keys is
     the order of the columns.
That is the only change needed -- the rest of the script is generic.
"""

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Name of the column in the input .csv that holds the chunk text to screen.
# Can be overridden on the command line with --text-column.
DEFAULT_TEXT_COLUMN = "text"

# Prefix added to the input file name to build the output file name.
OUTPUT_PREFIX = "screening_"

# The two possible answers. Kept as constants so every rule returns exactly the
# same spelling and a typo cannot slip into one rule only.
YES = "yes"
NO = "no"


# ---------------------------------------------------------------------------
# Building blocks shared by several rules
# ---------------------------------------------------------------------------

# The anonymisation placeholders used upstream. They are written in lower case
# and always wrapped in square brackets.
INDIVIDUAL_TAG = "[individual_tag]"
FOUNDATION_TAG = "[community_foundation_tag]"

# Regex fragment matching EITHER of the two "name" tags. `re.escape` turns the
# square brackets into literal characters (unescaped, "[...]" would mean "any
# one of these characters" in regex).
NAME_TAG_FRAGMENT = (
    "(?:" + re.escape(INDIVIDUAL_TAG) + "|" + re.escape(FOUNDATION_TAG) + ")"
)

# Punctuation characters that may sit around or between tags: the full ASCII
# punctuation set plus the typographic marks the PDF conversion leaves behind
# (dashes, bullets, ellipsis, curly quotes, Spanish opening marks), MINUS the
# square brackets, which belong to the tags themselves and must therefore never
# be treated as decoration around them.
PUNCTUATION_CHARS = r"""!"#$%&'()*+,\-./:;<=>?@\\^_`{|}~·•‧–—…«»“”‘’¡¿"""

# One or more whitespace and/or punctuation characters in a row ("\s" covers
# spaces, tabs, newlines and non-breaking spaces). The two HTML entities are
# listed explicitly because they appear literally in the chunked text ("&amp;"
# for "&", "&nbsp;" for a non-breaking space) and are decoration, not content.
# NOTE: words are NOT separators. The English "and" in
# "[individual_tag] and [individual_tag]" counts as other information, so that
# chunk is "no" for list_names.
SEPARATOR_FRAGMENT = r"(?:&amp;|&nbsp;|[\s" + PUNCTUATION_CHARS + r"])+"


# ---------------------------------------------------------------------------
# Rule 1: "name"
# ---------------------------------------------------------------------------
# Personal and organisation names were anonymised upstream and replaced by the
# placeholder tags "[individual_tag]" and "[community_foundation_tag]". A chunk
# that is nothing but one such tag carries no information at all and must be
# flagged.
#
# Detection method: an anchored full-match regular expression. The chunk counts
# as a name-only chunk when, apart from whitespace and/or punctuation marks
# before and after it, it consists of exactly ONE of the two tags:
#
#       [SEPARATOR?]  TAG  [SEPARATOR?]
#
#   "[individual_tag]"                       -> yes
#   "  [community_foundation_tag]\n"         -> yes   (whitespace ignored)
#   "[individual_tag]."                      -> yes   (trailing punctuation)
#   "- [individual_tag],"                    -> yes   (punctuation both sides)
#   "([community_foundation_tag])"           -> yes
#   "The [individual_tag] Honorary Fund"     -> no    (other words present)
#   "[individual_tag], [individual_tag]"     -> no    (two tags: that is rule 2)
#
# The match is case-sensitive: the tags are always written in lower case.
NAME_ONLY_PATTERN = re.compile(
    r"(?:" + SEPARATOR_FRAGMENT + r")?"
    + NAME_TAG_FRAGMENT
    + r"(?:" + SEPARATOR_FRAGMENT + r")?"
)


def screen_name(text: str) -> str:
    """Return "yes" if the whole chunk is one name tag plus decoration only.

    `fullmatch` (as opposed to `search`) requires the pattern to cover the
    ENTIRE string, which is what enforces the "and nothing else" condition.
    """
    return YES if NAME_ONLY_PATTERN.fullmatch(text) else NO


# ---------------------------------------------------------------------------
# Rule 2: "list_names"
# ---------------------------------------------------------------------------
# Donor, trustee and board pages become chunks that are just an enumeration of
# names, e.g. "[individual_tag], [individual_tag], [individual_tag]". Such
# chunks carry no analysable content and must be flagged.
#
# Detection method: an anchored full-match. The WHOLE chunk must consist of two
# or more name tags separated by whitespace and/or punctuation, with only
# whitespace and punctuation allowed before the first tag and after the last:
#
#       [SEPARATOR?]  TAG  ( SEPARATOR  TAG )+  [SEPARATOR?]
#
# The "+" means the group repeats at least once, hence at least two tags in
# total. Both tag types may be mixed inside the same chunk, because a donor list
# can name individuals and foundations side by side. Anything else in the chunk
# -- a single ordinary word -- makes the answer "no".
#
#   "[individual_tag], [individual_tag]"                  -> yes
#   "[individual_tag] [community_foundation_tag]"         -> yes
#   "[individual_tag] &amp; [individual_tag]"             -> yes
#   "- [individual_tag]; [individual_tag]."               -> yes
#   "[individual_tag]"                                    -> no  (one tag only:
#                                                             that is rule 1)
#   "Board: [individual_tag], [individual_tag]"           -> no  ("Board" is
#                                                             other information)
#   "[individual_tag] and [individual_tag]"               -> no  ("and" is a
#                                                             word, not
#                                                             punctuation)
#   "[individual_tag] Fund and [individual_tag] Fund"     -> no
LIST_NAMES_PATTERN = re.compile(
    r"(?:" + SEPARATOR_FRAGMENT + r")?"
    + NAME_TAG_FRAGMENT
    + r"(?:" + SEPARATOR_FRAGMENT + NAME_TAG_FRAGMENT + r")+"
    + r"(?:" + SEPARATOR_FRAGMENT + r")?"
)


def screen_list_names(text: str) -> str:
    """Return "yes" if the chunk is only a repetition of two or more name tags."""
    return YES if LIST_NAMES_PATTERN.fullmatch(text) else NO


# ---------------------------------------------------------------------------
# Rule 3: "non_english"
# ---------------------------------------------------------------------------
# Goal: flag chunks that are largely written in a language other than English so
# they can be screened out:
#       English text      -> "no"  (keep)
#       non-English text  -> "yes" (screen out)
#
# Detection method: word tokenization + fixed word-list (dictionary) matching.
# It does NOT identify WHICH language the text is in; it only weighs English
# evidence against non-English evidence. It is fully deterministic: no model, no
# probabilities, no external language-detection package.
#
# Procedure, step by step:
#   1. Remove text that would distort the count: the anonymisation placeholders
#      (which contain the English-looking words "individual", "community",
#      "foundation", "location", "tag"), HTML comments and HTML entities.
#   2. Split what remains into lower-case word tokens (letters only, accented
#      letters included).
#   3. Count how many tokens are English function words (ENGLISH_STOPWORDS) and
#      how many are function words specific to another language
#      (FOREIGN_STOPWORDS: Spanish, French, German, Italian, Portuguese).
#      Function words are used because they are the most frequent and most
#      language-specific words in any text, so a short list identifies the
#      language of running prose reliably.
#   4. Also count tokens carrying letters that English does not use (á, é, ñ,
#      ü, ç, ß, ...), which is independent evidence of a foreign language.
#   5. Decide (see thresholds below).
#
# Conservative by design: the answer is "yes" only when there is POSITIVE
# evidence of another language. Chunks with too few words to judge, and chunks
# with no function words at all (table headers such as "Total assets Revenue
# Grants"), stay "no" -- they are kept for analysis rather than silently
# discarded.

# Markup to strip before tokenizing: any "[..._tag]" placeholder, HTML comments
# left behind by the PDF conversion, and HTML entities ("&amp;", "&#39;").
PLACEHOLDER_TAG_PATTERN = re.compile(r"\[[a-z_]+_tag\]")
HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)
HTML_ENTITY_PATTERN = re.compile(r"&[a-z]+;|&#\d+;")

# A word token: one or more letters, optionally with apostrophes inside
# ("don't"). À-ɏ covers the accented Latin letters of Western European
# languages, so foreign words are tokenized as single words rather than split.
TOKEN_PATTERN = re.compile(r"[A-Za-zÀ-ɏ]+(?:['’][A-Za-z]+)*")

# Letters that (essentially) never occur in English words but are common in
# Spanish, French, German, Italian and Portuguese.
NON_ENGLISH_LETTER_PATTERN = re.compile(r"[à-ÿā-ɏ]")

# --- word lists -----------------------------------------------------------
# English function words. Only closed-class, very frequent words: articles,
# pronouns, prepositions, conjunctions, auxiliaries.
ENGLISH_STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those of in on at to for
    from with within without by as into onto over under about after before
    between during through across against above below off out up down again
    further once while until because since so such only own same other others
    another each every both all any some few more most many much no not nor
    is are was were be been being am has have had having do does did doing
    will would shall should can could may might must
    it its it's he him his she her hers they them their theirs we us our ours
    you your yours i me my mine who whom whose which what when where why how
    there here also very too just now new when's
    """.split()
)

# Function words that are characteristic of other languages. Any word that also
# belongs to the English list is removed further below, so shared spellings
# ("no", "a", "in", "as", "on") never count as foreign evidence.
# Very short ambiguous tokens ("a", "e", "o", "y", "i", "u") are intentionally
# left out: they collide with initials in English name lists ("J. E. Smith").
SPANISH_STOPWORDS = """
    el la los las un una unos unas del al lo
    de que en con por para como pero mas muy sin sobre entre desde hasta
    es son fue fueron era eran ha han hay ser estar esta este estos estas esa
    ese esos esas su sus nuestro nuestra nuestros nuestras se les
    tambien cuando donde porque anos ano mismo mismos todo toda todos
    todas otro otra otros otras cada quien cual cuales
    mas alli aqui asi aunque mientras segun
    """.split()

FRENCH_STOPWORDS = """
    le la les un une des du de au aux et ou que qui dans pour avec sur par
    plus pas ne est sont etait etaient ont avoir etre ce cet cette ces se son
    sa ses leur leurs nous vous ils elles il elle on mais comme tout tous toute
    toutes tres aussi entre sans chez depuis apres avant ainsi meme lors dont
    """.split()

GERMAN_STOPWORDS = """
    der die das den dem des ein eine einen einem einer eines und oder aber
    nicht ist sind war waren haben hat hatte wird werden wurde wurden mit von
    zu zum zur fur auf aus bei nach uber unter durch als auch sich sie wir ihr
    ich es im am dass wenn noch nur sehr schon kein keine mehr sowie zwischen
    """.split()

ITALIAN_STOPWORDS = """
    il lo gli le un uno una di del della dello dei degli delle ed che chi non
    per con su da come piu anche sono era erano hanno questo questa questi
    queste nel nella nelle alla alle agli ai tra fra suo sua suoi sue loro
    perche quando dove sempre ancora molto
    """.split()

PORTUGUESE_STOPWORDS = """
    os um uma uns umas do da dos das no na nos nas ao aos pelo pela pelos pelas
    que nao para com por mais como mas seu sua seus suas este esta isso isto
    aquele aquela sao foi foram tem temos muito tambem sobre entre quando onde
    porque desde ate sem ja apenas ainda
    """.split()

# Union of all foreign lists MINUS anything that is also an English function
# word. The subtraction is what keeps the test conservative: a word must be
# unambiguously non-English to count as foreign evidence.
FOREIGN_STOPWORDS = frozenset(
    SPANISH_STOPWORDS
    + FRENCH_STOPWORDS
    + GERMAN_STOPWORDS
    + ITALIAN_STOPWORDS
    + PORTUGUESE_STOPWORDS
) - ENGLISH_STOPWORDS

# --- thresholds -----------------------------------------------------------
# Below this many word tokens there is not enough material to judge a language,
# so the chunk is left as "no" (English / undecidable).
MIN_TOKENS_FOR_LANGUAGE_TEST = 5
# Share of foreign function words needed to call a chunk non-English. Running
# prose is roughly 30-50% function words, so 15% is a clear signal while still
# tolerating proper names and figures mixed into the sentence.
FOREIGN_STOPWORD_MIN_RATIO = 0.15
# Share of tokens carrying non-English letters that on its own signals a foreign
# language (used only when English evidence is essentially absent).
ACCENTED_TOKEN_MIN_RATIO = 0.30
# Below this share of English function words, a chunk is considered to carry no
# real English evidence.
ENGLISH_STOPWORD_MIN_RATIO = 0.10


def fold_accents(token: str) -> str:
    """Return the token with its diacritics removed ("más" -> "mas", "ü" -> "u").

    The word lists above are written without diacritics, while the text keeps
    them, so both sides are folded before being compared. Unicode NFKD
    decomposition splits an accented letter into "base letter + combining
    accent"; the combining accents (category "Mn") are then dropped. German "ß"
    has no decomposition, so it is mapped to "ss" explicitly.

    This is a pure string transformation -- deterministic, no dictionary
    involved -- and it is used ONLY for the word-list lookup. The accent
    evidence in `screen_non_english` is measured on the unfolded token, so
    folding here never hides the fact that the word carried an accent.
    """
    decomposed = unicodedata.normalize("NFKD", token.replace("ß", "ss"))
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def tokenize_words(text: str) -> list:
    """Strip placeholders/markup, then return the chunk's lower-case word tokens.

    Placeholders and markup are removed first so that words baked into them
    ("individual", "community", "foundation") cannot tip the language balance
    towards English.
    """
    cleaned = PLACEHOLDER_TAG_PATTERN.sub(" ", text)
    cleaned = HTML_COMMENT_PATTERN.sub(" ", cleaned)
    cleaned = HTML_ENTITY_PATTERN.sub(" ", cleaned)
    # Curly apostrophes are normalised to straight ones so that contractions
    # ("it’s") match the same way as their typed form ("it's").
    cleaned = cleaned.replace("’", "'")
    return [token.lower() for token in TOKEN_PATTERN.findall(cleaned)]


def screen_non_english(text: str) -> str:
    """Return "yes" if the chunk is largely NOT in English, "no" if it is English."""
    tokens = tokenize_words(text)
    total = len(tokens)

    # Step 1: too few words to judge -> keep the chunk ("no").
    if total < MIN_TOKENS_FOR_LANGUAGE_TEST:
        return NO

    # Step 2: count the three kinds of evidence. The word-list lookups use the
    # accent-folded token (the lists are written without diacritics); the accent
    # count uses the original token, where the diacritics are still visible.
    folded = [fold_accents(token) for token in tokens]
    english_hits = sum(1 for token in folded if token in ENGLISH_STOPWORDS)
    foreign_hits = sum(1 for token in folded if token in FOREIGN_STOPWORDS)
    accented_hits = sum(1 for token in tokens if NON_ENGLISH_LETTER_PATTERN.search(token))

    english_ratio = english_hits / total
    foreign_ratio = foreign_hits / total
    accented_ratio = accented_hits / total

    # Step 3a: clear lexical evidence -- more foreign than English function
    # words, and enough of them to matter.
    if foreign_hits > english_hits and foreign_ratio >= FOREIGN_STOPWORD_MIN_RATIO:
        return YES

    # Step 3b: orthographic evidence -- many accented words and no meaningful
    # English function-word evidence (catches foreign text whose function words
    # happen to be missing from the lists above).
    if accented_ratio >= ACCENTED_TOKEN_MIN_RATIO and english_ratio < ENGLISH_STOPWORD_MIN_RATIO:
        return YES

    # Step 4: default -- treat as English and keep the chunk.
    return NO


# ---------------------------------------------------------------------------
# Rule 4: "min_length"
# ---------------------------------------------------------------------------
# Two kinds of chunk are too thin to analyse: the very short ones (a stray
# letter, a page number, a bullet), and the ones made of a single unbroken token
# with no whitespace in them at all (a lone word, a URL, a placeholder tag, a
# table remnant) -- a chunk without a single space cannot be a sentence.
#
# Detection method: count the characters and look for whitespace. Leading and
# trailing whitespace is ignored for both tests, so a chunk made of blanks
# counts as 0 characters and as having no whitespace. Every remaining character
# counts, including inner spaces, digits and punctuation.
#
#   ""                                     -> 0 characters, no space -> yes
#   "5"                                    -> 1 character,  no space -> yes
#   "abcd"                                 -> 4 characters           -> yes
#   "[individual_tag]"                     -> 16 characters, no space-> yes
#   "www.example.org"                      -> no space               -> yes
#   "2021 was a record year for new funds" -> long and spaced        -> no
#
# Polarity, as with every other rule in this script: "yes" flags the chunk to be
# screened out.
MAX_CHARACTERS_TO_FLAG = 4
WHITESPACE_PATTERN = re.compile(r"\s")


def screen_min_length(text: str) -> str:
    """Return "yes" if the chunk has 4 characters or fewer, or no whitespace."""
    stripped = text.strip()
    if len(stripped) <= MAX_CHARACTERS_TO_FLAG:
        return YES
    if not WHITESPACE_PATTERN.search(stripped):
        return YES
    return NO


# ---------------------------------------------------------------------------
# Registry of screenings
# ---------------------------------------------------------------------------
# Maps output column name -> rule function. Insertion order is preserved by
# Python dicts, so it is also the left-to-right order of the new columns in the
# output .csv. Add future rules here.
SCREENINGS = {
    "name": screen_name,
    "list_names": screen_list_names,
    "non_english": screen_non_english,
    "min_length": screen_min_length,
}


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def normalize_text(value) -> str:
    """Turn a raw cell value into a plain string that the rules can inspect.

    Empty cells are read by pandas as NaN (a float), which would crash the
    regular expressions; they are converted to an empty string.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def screen_dataframe(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Return a copy of `df` with one new "yes"/"no" column per screening rule.

    The original columns are left untouched; the screening columns are appended
    at the end in the order given by SCREENINGS.
    """
    df = df.copy()

    if text_column not in df.columns:
        raise KeyError(
            f"column {text_column!r} not found; available columns: {list(df.columns)}"
        )

    # Pre-normalise the text column once, then apply each rule to it. This keeps
    # every rule a simple str -> str function.
    texts = df[text_column].map(normalize_text)

    for column_name, rule in SCREENINGS.items():
        if column_name in df.columns:
            # The input already had a column with this name (e.g. left over from
            # an earlier screening run). Warn loudly: it is being replaced by
            # the deterministic answer.
            print(
                f"  note: overwriting pre-existing column {column_name!r}",
                file=sys.stderr,
            )
        df[column_name] = texts.map(rule)

    return df


def process_file(csv_path: Path, output_dir: Path, text_column: str) -> Path:
    """Screen one .csv file and write the "screening_" copy. Returns its path."""
    df = pd.read_csv(csv_path)
    screened = screen_dataframe(df, text_column)

    output_path = output_dir / f"{OUTPUT_PREFIX}{csv_path.name}"
    screened.to_csv(output_path, index=False)

    # Short per-file report: how many rows got a "yes" for each rule.
    counts = ", ".join(
        f"{name}={int((screened[name] == YES).sum())}" for name in SCREENINGS
    )
    print(f"  {len(screened)} rows -> {output_path}  (yes counts: {counts})")

    return output_path


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic, rule-based yes/no screening of chunked report .csv "
            "files (no LLM involved)."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="out/chunked reports",
        help='Folder containing the chunked .csv files to screen (default: "out/chunked reports").',
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Path to a single .csv file to screen. Overrides --input-dir when given.",
    )
    parser.add_argument(
        "--output-dir",
        default="out/screening",
        help='Folder where the "screening_" copies are written (default: "out/screening").',
    )
    parser.add_argument(
        "--text-column",
        default=DEFAULT_TEXT_COLUMN,
        help=f'Name of the column holding the chunk text (default: "{DEFAULT_TEXT_COLUMN}").',
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the list of files to process: either the single file given with
    # --input-file, or every .csv directly inside --input-dir (sub-folders such
    # as "archive" are deliberately not searched).
    if args.input_file:
        csv_path = Path(args.input_file)
        if not csv_path.is_file():
            sys.exit(f"File not found: {csv_path}")
        csv_files = [csv_path]
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            sys.exit(f"Input folder not found: {input_dir}")
        csv_files = sorted(input_dir.glob("*.csv"))
        if not csv_files:
            sys.exit(f"No .csv files found in {input_dir}")

    print(f"Screenings applied: {', '.join(SCREENINGS)}")

    failures = 0
    for csv_path in csv_files:
        print(f"{csv_path.name}")
        try:
            process_file(csv_path, output_dir, args.text_column)
        except Exception as exc:  # keep going: one bad file must not stop the batch
            failures += 1
            print(f"  ERROR: skipped ({exc})", file=sys.stderr)

    print(f"\nDone: {len(csv_files) - failures}/{len(csv_files)} file(s) screened.")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
