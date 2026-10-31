#!/usr/bin/env python3
"""
01_match_and_score.py

Step 1 of the Human-vs-AI institutional-logics agreement analysis.

What it does
------------
1. Loads the human coder's CSV export (from the coding tool) and the AI
   pipeline's own raw output CSV.
2. Matches chunks between the two files by exact text (handles repeated
   text via an occurrence index, matched in row order).
3. For each matched chunk and each coder, reduces the 5 elements
   (community, donor, activity, board, grantee) to a weighted 4-logic
   points vector: primary (top-1) logic = 2 points, secondary (top-2)
   logic = 1 point, summed across applicable elements.
4. Flags a chunk as "coded_by_both" if BOTH coders marked >=1 element
   applicable for it.
5. Writes:
     - chunk_level.csv      (one row per matched chunk; points & proportions)
     - document_aggregate.json (doc-level point totals/proportions, coded_by_both only)
     - summary.json          (coverage counts + doc-level proportions)

This script does NOT compute inter-coder agreement (kappa) — that is
02_chunk_kappa.py, which consumes chunk_level.csv.

Usage
-----
    python3 01_match_and_score.py \\
        --human-csv path/to/human_coded_export.csv \\
        --ai-csv path/to/ai_pipeline_output.csv \\
        --outdir ./out

Expected input columns
-----------------------
Human coder CSV (export format of the coding-tool artifact):
    human_skip                                   "yes"/"no" (or blank)
    human_{element}_answer                        "yes"/"no" per element
    human_{element}_top1_logic                    one of PS/DS/PPG/SC
    human_{element}_top2_logic                    one of PS/DS/PPG/SC
    text                                           chunk text (for matching)

AI pipeline CSV (raw inference output):
    {element}_answer                              "yes"/"no" per element
    {element}_contrastive_logic_answer             "X/Y" digit pair, e.g. "2/1"
                                                    (1=PS, 2=DS, 3=PPG, 4=SC)
    text                                           chunk text (for matching)

ELEMENTS and the digit->logic mapping below must match the coding
instrument actually used; edit them if your instrument differs.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ELEMENTS = ['element_community', 'element_donor', 'element_activity', 'element_board', 'element_grantee']
LOGICS = ['PS', 'DS', 'PPG', 'SC']
LOGIC_NAMES = {'PS': 'Philanthropic Stewardship', 'DS': 'Donor Services',
               'PPG': 'Public-Private Governance', 'SC': 'Social Change'}
# Digit numbering used by the AI pipeline's own contrastive-logic question
# (1=PS, 2=DS, 3=PPG, 4=SC). This is also the ordinal order used later for
# weighted kappa (02_chunk_kappa.py): PS < DS < PPG < SC.
DIGIT_TO_LOGIC = {'1': 'PS', '2': 'DS', '3': 'PPG', '4': 'SC'}


def build_occurrence_index(df: pd.DataFrame, text_col: str = 'text') -> dict:
    """Map (text, occurrence_number) -> row index, in row order.

    Lets identical chunk text appear multiple times in a file and still be
    matched 1:1 against the correct occurrence in the other file, rather
    than collapsing duplicates.
    """
    counts = defaultdict(int)
    idx_map = {}
    for i, t in enumerate(df[text_col].astype(str)):
        counts[t] += 1
        idx_map[(t, counts[t])] = i
    return idx_map


def human_applicable_and_ranks(row: pd.Series) -> dict:
    """Human coder: element -> (top1_logic_or_None, top2_logic_or_None, applicable_bool)."""
    skipped = row.get('human_skip') == 'yes'
    out = {}
    for el in ELEMENTS:
        if skipped:
            out[el] = (None, None, False)
            continue
        ans = row.get(f'human_{el}_answer')
        applicable = (ans == 'yes')
        t1 = row.get(f'human_{el}_top1_logic') if applicable else None
        t2 = row.get(f'human_{el}_top2_logic') if applicable else None
        t1 = t1 if t1 in LOGICS else None
        t2 = t2 if t2 in LOGICS else None
        out[el] = (t1, t2, applicable)
    return out


def parse_contrastive(val) -> tuple:
    """Parse the AI pipeline's 'X/Y' contrastive-logic answer, e.g. '2/1' -> ('DS','PS')."""
    if not val or '/' not in str(val):
        return (None, None)
    parts = str(val).strip().split('/')
    if len(parts) < 2:
        return (None, None)
    d1, d2 = parts[0].strip(), parts[1].strip()
    return (DIGIT_TO_LOGIC.get(d1), DIGIT_TO_LOGIC.get(d2))


def ai_applicable_and_ranks(row: pd.Series) -> dict:
    """AI pipeline: element -> (top1_logic_or_None, top2_logic_or_None, applicable_bool).

    The AI pipeline has no explicit top1/top2 field, so top1/top2 are
    derived from its own contrastive-logic answer field.
    """
    out = {}
    for el in ELEMENTS:
        applicable = (row.get(f'{el}_answer') == 'yes')
        if applicable:
            t1, t2 = parse_contrastive(row.get(f'{el}_contrastive_logic_answer'))
        else:
            t1, t2 = None, None
        out[el] = (t1, t2, applicable)
    return out


def points_from_ranks(ranks_dict: dict) -> tuple:
    """Collapse one coder's per-element (top1, top2, applicable) triples into a
    4-logic weighted points vector: top1 = 2 pts, top2 = 1 pt, summed across
    applicable elements. Returns (points_dict, n_applicable, n_incomplete)."""
    pts = {l: 0 for l in LOGICS}
    n_applicable = 0
    n_incomplete = 0  # applicable but top1 missing/unparseable
    for el, (t1, t2, applicable) in ranks_dict.items():
        if not applicable:
            continue
        n_applicable += 1
        if t1:
            pts[t1] += 2
        else:
            n_incomplete += 1
        if t2:
            pts[t2] += 1
    return pts, n_applicable, n_incomplete


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--human-csv', required=True, help="Human coder's export CSV")
    ap.add_argument('--ai-csv', required=True, help="AI pipeline's raw output CSV")
    ap.add_argument('--outdir', default='.', help='Directory to write outputs into')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cd = pd.read_csv(args.human_csv, dtype=str, keep_default_na=False, na_values=['', 'nan', 'NaN'])
    ai = pd.read_csv(args.ai_csv, dtype=str, keep_default_na=False, na_values=['', 'nan', 'NaN'], low_memory=False)
    print(f"Human coder rows: {len(cd)}   AI pipeline rows: {len(ai)}")

    # --- 1. Match chunks by exact text (stable, occurrence-aware) ---
    ai_idx = build_occurrence_index(ai)
    matches, used_ai = [], set()
    cd_occurrence = defaultdict(int)
    for i, t in enumerate(cd['text'].astype(str)):
        cd_occurrence[t] += 1
        key = (t, cd_occurrence[t])
        if key in ai_idx:
            ai_row = ai_idx[key]
            matches.append((i, ai_row, t))
            used_ai.add(ai_row)

    n_unmatched_cd = len(cd) - len(matches)
    n_unmatched_ai = len(ai) - len(used_ai)
    print(f"Matched {len(matches)} chunks   (unmatched human rows: {n_unmatched_cd}, unmatched AI rows: {n_unmatched_ai})")

    # --- 2. Per-chunk weighted points, per coder ---
    records = []
    for cd_i, ai_i, text in matches:
        cd_row, ai_row = cd.iloc[cd_i], ai.iloc[ai_i]
        cd_ranks = human_applicable_and_ranks(cd_row)
        ai_ranks = ai_applicable_and_ranks(ai_row)
        cd_pts, cd_napp, cd_incomplete = points_from_ranks(cd_ranks)
        ai_pts, ai_napp, ai_incomplete = points_from_ranks(ai_ranks)

        rec = {
            'chunk_id_cd': cd_row.get('chunk_id'),
            'text': text,
            'human_n_applicable_elements': cd_napp,
            'human_n_incomplete_rankings': cd_incomplete,
            'ai_n_applicable_elements': ai_napp,
            'ai_n_incomplete_rankings': ai_incomplete,
            'coded_by_both': (cd_napp >= 1) and (ai_napp >= 1),
        }
        for l in LOGICS:
            rec[f'human_points_{l}'] = cd_pts[l]
            rec[f'ai_points_{l}'] = ai_pts[l]
        cd_total, ai_total = sum(cd_pts.values()), sum(ai_pts.values())
        for l in LOGICS:
            rec[f'human_prop_{l}'] = (cd_pts[l] / cd_total) if cd_total > 0 else np.nan
            rec[f'ai_prop_{l}'] = (ai_pts[l] / ai_total) if ai_total > 0 else np.nan
        records.append(rec)

    chunk_df = pd.DataFrame(records)
    chunk_df.to_csv(outdir / 'chunk_level.csv', index=False)
    print(f"Wrote {outdir / 'chunk_level.csv'}  ({len(chunk_df)} rows)")

    # --- 3. Document-level aggregate (chunks coded by both only) ---
    included = chunk_df[chunk_df['coded_by_both']]
    print(f"Chunks coded by both coders (>=1 applicable element each): {len(included)} / {len(chunk_df)} matched")

    doc_human_pts = {l: int(included[f'human_points_{l}'].sum()) for l in LOGICS}
    doc_ai_pts = {l: int(included[f'ai_points_{l}'].sum()) for l in LOGICS}
    doc_human_total, doc_ai_total = sum(doc_human_pts.values()), sum(doc_ai_pts.values())
    doc_human_prop = {l: doc_human_pts[l] / doc_human_total for l in LOGICS}
    doc_ai_prop = {l: doc_ai_pts[l] / doc_ai_total for l in LOGICS}

    print("\nDocument-level HUMAN proportions:", {l: round(v, 4) for l, v in doc_human_prop.items()})
    print("Document-level AI proportions:   ", {l: round(v, 4) for l, v in doc_ai_prop.items()})

    with open(outdir / 'document_aggregate.json', 'w') as f:
        json.dump({
            'n_included': len(included),
            'human_points': doc_human_pts, 'ai_points': doc_ai_pts,
            'human_proportions': doc_human_prop, 'ai_proportions': doc_ai_prop,
        }, f, indent=2)

    # --- 4. Supplementary: plain top-1 exact-match rate (nominal, element-level) ---
    exact, comparable = 0, 0
    for cd_i, ai_i, _ in matches:
        cd_ranks = human_applicable_and_ranks(cd.iloc[cd_i])
        ai_ranks = ai_applicable_and_ranks(ai.iloc[ai_i])
        for el in ELEMENTS:
            h_t1, _, h_app = cd_ranks[el]
            a_t1, _, a_app = ai_ranks[el]
            if h_app and a_app and h_t1 and a_t1:
                comparable += 1
                if h_t1 == a_t1:
                    exact += 1
    top1_pct = exact / comparable if comparable else float('nan')
    print(f"\nSupplementary top-1 exact agreement (elements both coders ranked): {exact}/{comparable} = {top1_pct:.1%}")

    summary = {
        'n_human_rows': len(cd), 'n_ai_rows': len(ai),
        'n_matched_chunks': len(matches),
        'n_unmatched_human': n_unmatched_cd, 'n_unmatched_ai': n_unmatched_ai,
        'n_included_in_aggregate': len(included),
        'doc_human_points': doc_human_pts, 'doc_ai_points': doc_ai_pts,
        'doc_human_proportions': doc_human_prop, 'doc_ai_proportions': doc_ai_prop,
        'top1_pct_agreement': top1_pct, 'top1_n_comparable': comparable, 'top1_n_matches': exact,
    }
    with open(outdir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {outdir / 'summary.json'} and {outdir / 'document_aggregate.json'}")


if __name__ == '__main__':
    main()
