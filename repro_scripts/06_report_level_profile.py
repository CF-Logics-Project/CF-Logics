#!/usr/bin/env python3
"""
06_report_level_profile.py

Step 6: report-level ("document profile") analysis, independent of chunk
matching.

Why this exists
----------------
01_match_and_score.py / 02_chunk_kappa.py measure agreement at the level of
individual MATCHED chunks: a chunk is only scored if both coders coded it
(coded_by_both), and the agreement statistic is about whether the two
coders pick the SAME chunk's SAME dominant logic.

That is a different, and narrower, question from: "considered as two
separate coding passes over the whole report, how much of each logic did
the human coder find overall, how much did the AI pipeline find overall,
and do those two overall profiles look similar?" This script answers the
second question. It:

  1. Scores EVERY row in the human coder's raw file and EVERY row in the
     AI pipeline's raw file independently -- no matching, no coded_by_both
     filter, no requirement that the two files have the same chunks. Human
     rows that have nothing to do with any AI row still count toward the
     human profile, and vice versa.
  2. Sums each coder's own points into a report-level logic-frequency
     profile (proportions over the 4 logics) and reports the two coders'
     profiles side by side, plus a profile-distance metric between them
     (NOT a chunk-matched kappa -- a totally different, document-level
     notion of "how similar are the two coders' overall pictures of this
     report").
  3. Builds a smoothed logic-prevalence-by-document-position curve for
     each coder (normalized chunk position 0..1, binned), so the dashboard
     can show where in the report each logic concentrates, for each coder,
     even where the exact chunks don't align 1:1.

This script re-implements the per-row scoring logic from
01_match_and_score.py (human_applicable_and_ranks / ai_applicable_and_ranks
/ points_from_ranks) rather than importing it, so it has no dependency on
that script's matching step.

Usage
-----
    python3 06_report_level_profile.py \\
        --human-csv path/to/human_coded_export.csv \\
        --ai-csv path/to/ai_pipeline_output.csv \\
        --report-id 25-1650208_2021 \\
        --outdir multi_report/25-1650208_2021
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ELEMENTS = ['element_community', 'element_donor', 'element_activity', 'element_board', 'element_grantee']
LOGICS = ['PS', 'DS', 'PPG', 'SC']
DIGIT_TO_LOGIC = {'1': 'PS', '2': 'DS', '3': 'PPG', '4': 'SC'}
N_BINS = 20  # document-position bins for the smoothed prevalence curve


def human_applicable_and_ranks(row: pd.Series) -> dict:
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
    if not val or '/' not in str(val):
        return (None, None)
    parts = str(val).strip().split('/')
    if len(parts) < 2:
        return (None, None)
    d1, d2 = parts[0].strip(), parts[1].strip()
    return (DIGIT_TO_LOGIC.get(d1), DIGIT_TO_LOGIC.get(d2))


def ai_applicable_and_ranks(row: pd.Series) -> dict:
    out = {}
    for el in ELEMENTS:
        applicable = (row.get(f'{el}_answer') == 'yes')
        if applicable:
            t1, t2 = parse_contrastive(row.get(f'{el}_contrastive_logic_answer'))
        else:
            t1, t2 = None, None
        out[el] = (t1, t2, applicable)
    return out


def points_from_ranks(ranks_dict: dict) -> dict:
    pts = {l: 0 for l in LOGICS}
    for el, (t1, t2, applicable) in ranks_dict.items():
        if not applicable:
            continue
        if t1:
            pts[t1] += 2
        if t2:
            pts[t2] += 1
    return pts


def score_every_row(df: pd.DataFrame, ranks_fn) -> pd.DataFrame:
    """Score every row of a raw coder file independently. Returns one row
    per input row: points_<logic> (int), prop_<logic> (NaN if the row had
    no applicable elements at all), total_points, position (0..1, by row
    order in the file as given -- the file's own chunk order)."""
    recs = []
    n = len(df)
    for i in range(n):
        row = df.iloc[i]
        ranks = ranks_fn(row)
        pts = points_from_ranks(ranks)
        total = sum(pts.values())
        rec = {'row_index': i, 'position': (i / (n - 1)) if n > 1 else 0.0, 'total_points': total}
        for l in LOGICS:
            rec[f'points_{l}'] = pts[l]
            rec[f'prop_{l}'] = (pts[l] / total) if total > 0 else np.nan
        recs.append(rec)
    return pd.DataFrame(recs)


def profile_distance(p: dict, q: dict) -> dict:
    """Two distance/similarity metrics between two 4-category proportion
    vectors (dicts over LOGICS, each summing to ~1): total variation
    distance (0 = identical, 1 = disjoint support) and cosine similarity
    (1 = identical direction)."""
    pv = np.array([p[l] for l in LOGICS])
    qv = np.array([q[l] for l in LOGICS])
    tvd = float(0.5 * np.abs(pv - qv).sum())
    denom = (np.linalg.norm(pv) * np.linalg.norm(qv))
    cos = float(np.dot(pv, qv) / denom) if denom > 0 else float('nan')
    return {'total_variation_distance': tvd, 'cosine_similarity': cos}


def binned_prevalence(scored: pd.DataFrame, n_bins: int = N_BINS) -> dict:
    """Mean per-logic proportion within each of n_bins equal-width bins of
    normalized document position, skipping rows with no applicable
    elements (prop_* = NaN for those). Returns {logic: [n_bins values]},
    plus the bin centers, suitable for a smoothed line/area chart."""
    edges = np.linspace(0, 1, n_bins + 1)
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()
    out = {l: [] for l in LOGICS}
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == n_bins - 1:
            mask = (scored['position'] >= lo) & (scored['position'] <= hi)
        else:
            mask = (scored['position'] >= lo) & (scored['position'] < hi)
        sub = scored[mask]
        for l in LOGICS:
            vals = sub[f'prop_{l}'].dropna()
            out[l].append(float(vals.mean()) if len(vals) else None)
    return {'bin_centers': centers, 'prevalence': out}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--human-csv', required=True)
    ap.add_argument('--ai-csv', required=True)
    ap.add_argument('--report-id', required=True)
    ap.add_argument('--outdir', default='.')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cd = pd.read_csv(args.human_csv, dtype=str, keep_default_na=False, na_values=['', 'nan', 'NaN'])
    ai = pd.read_csv(args.ai_csv, dtype=str, keep_default_na=False, na_values=['', 'nan', 'NaN'], low_memory=False)

    human_scored = score_every_row(cd, human_applicable_and_ranks)
    ai_scored = score_every_row(ai, ai_applicable_and_ranks)

    human_scored.to_csv(outdir / 'report_level_human_rows.csv', index=False)
    ai_scored.to_csv(outdir / 'report_level_ai_rows.csv', index=False)

    human_pts = {l: int(human_scored[f'points_{l}'].sum()) for l in LOGICS}
    ai_pts = {l: int(ai_scored[f'points_{l}'].sum()) for l in LOGICS}
    human_total, ai_total = sum(human_pts.values()), sum(ai_pts.values())
    human_prop = {l: (human_pts[l] / human_total if human_total else 0.0) for l in LOGICS}
    ai_prop = {l: (ai_pts[l] / ai_total if ai_total else 0.0) for l in LOGICS}

    n_human_coded = int((human_scored['total_points'] > 0).sum())
    n_ai_coded = int((ai_scored['total_points'] > 0).sum())

    dist = profile_distance(human_prop, ai_prop)
    human_curve = binned_prevalence(human_scored)
    ai_curve = binned_prevalence(ai_scored)

    result = {
        'report_id': args.report_id,
        'n_human_rows_total': len(cd), 'n_ai_rows_total': len(ai),
        'n_human_rows_coded': n_human_coded, 'n_ai_rows_coded': n_ai_coded,
        'human_points': human_pts, 'ai_points': ai_pts,
        'human_proportions': human_prop, 'ai_proportions': ai_prop,
        'profile_distance': dist,
        'human_position_curve': human_curve,
        'ai_position_curve': ai_curve,
    }
    with open(outdir / 'report_level_profile.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f"[{args.report_id}] human: {n_human_coded}/{len(cd)} rows coded, "
          f"proportions {({l: round(v, 3) for l, v in human_prop.items()})}")
    print(f"[{args.report_id}] ai:    {n_ai_coded}/{len(ai)} rows coded, "
          f"proportions {({l: round(v, 3) for l, v in ai_prop.items()})}")
    print(f"[{args.report_id}] profile distance: TVD={dist['total_variation_distance']:.3f}  "
          f"cosine={dist['cosine_similarity']:.3f}")
    print(f"Wrote {outdir / 'report_level_profile.json'}")


if __name__ == '__main__':
    main()
