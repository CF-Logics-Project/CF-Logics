#!/usr/bin/env python3
"""
02_chunk_kappa.py

Step 2 of the Human-vs-AI institutional-logics agreement analysis.

Consumes chunk_level.csv (produced by 01_match_and_score.py) and computes
chunk-level inter-coder agreement.

Unit of observation
--------------------
ONE observation per chunk (not per chunk x element x logic). For each
coder, a chunk's "dominant logic" is whichever of PS/DS/PPG/SC holds that
coder's highest weighted-point total for that chunk (from the points
computed in step 1). Only chunks coded by both coders (coded_by_both ==
True in chunk_level.csv) are used; this is the N reported everywhere.

Ordinal weighting
------------------
PS, DS, PPG, SC are treated as an ORDINAL sequence (index 0,1,2,3),
following the coding instrument's own numbering (1=PS, 2=DS, 3=PPG,
4=SC). Weighted Cohen's kappa is computed with:
    - quadratic weights  (i-j)^2   <- PRIMARY, reported figure
    - linear weights     |i-j|     <- reference
    - unweighted/nominal  0/1      <- reference (ignores order)
A quadratic/linear weighting penalizes a disagreement between adjacent
logics (e.g. PPG vs SC) much less than one between opposite ends of the
sequence (PS vs SC); the unweighted kappa penalizes every mismatch the
same regardless of distance.

Ties
----
A coder's own points can tie across >=2 logics for a chunk, leaving
"dominant logic" undefined. Ties are broken by a fixed, disclosed
priority order PS > DS > PPG > SC. A sensitivity kappa is also reported
with every tied chunk (either coder) excluded, to confirm the headline
number is not an artifact of the tie-break rule.

Outputs
-------
    chunk_dominant.csv        one row per included chunk: dominant logic,
                               tie flag, match flag, per coder
    summary_chunk_kappa.json  confusion matrices + all three kappa
                               variants, for the full sample and the
                               ties-excluded sensitivity check

Usage
-----
    python3 02_chunk_kappa.py --chunk-level-csv ./out/chunk_level.csv --outdir ./out
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Ordinal order: PS(0) < DS(1) < PPG(2) < SC(3), per the coding instrument's
# own 1/2/3/4 numbering. Edit this if your instrument's logics differ or are
# not meant to be ordinal (in which case use weights=None / unweighted only).
LOGICS = ['PS', 'DS', 'PPG', 'SC']
TIE_BREAK_PRIORITY = ['PS', 'DS', 'PPG', 'SC']  # arbitrary, disclosed


def dominant_logic(points: dict, priority=TIE_BREAK_PRIORITY):
    """Return (label, is_tie) for one coder's points dict on one chunk."""
    mx = max(points.values())
    winners = [l for l in priority if points[l] == mx]
    is_tie = len(winners) > 1
    return winners[0], is_tie


def confusion_matrix(human_labels, ai_labels, labels=LOGICS) -> pd.DataFrame:
    cm = pd.crosstab(pd.Series(human_labels, name='human'), pd.Series(ai_labels, name='ai'))
    return cm.reindex(index=labels, columns=labels, fill_value=0)


def compute_kappas(human_labels, ai_labels, labels=LOGICS) -> dict:
    return {
        'kappa_unweighted': float(cohen_kappa_score(human_labels, ai_labels, labels=labels)),
        'kappa_linear': float(cohen_kappa_score(human_labels, ai_labels, labels=labels, weights='linear')),
        'kappa_quadratic': float(cohen_kappa_score(human_labels, ai_labels, labels=labels, weights='quadratic')),
        'raw_agreement': float((pd.Series(human_labels) == pd.Series(ai_labels)).mean()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--chunk-level-csv', required=True, help='chunk_level.csv from 01_match_and_score.py')
    ap.add_argument('--outdir', default='.', help='Directory to write outputs into')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    chunk_df = pd.read_csv(args.chunk_level_csv)
    included = chunk_df[chunk_df['coded_by_both'] == True].copy()  # noqa: E712
    print(f"Chunks coded by both coders: {len(included)} (out of {len(chunk_df)} matched)")

    # --- dominant logic + tie flag, per coder, per chunk ---
    rows = []
    for _, r in included.iterrows():
        h_pts = {l: r[f'human_points_{l}'] for l in LOGICS}
        a_pts = {l: r[f'ai_points_{l}'] for l in LOGICS}
        h_label, h_tie = dominant_logic(h_pts)
        a_label, a_tie = dominant_logic(a_pts)
        rows.append({
            'chunk_id': r['chunk_id_cd'],
            'human_dominant': h_label, 'human_tie': h_tie,
            'ai_dominant': a_label, 'ai_tie': a_tie,
            'match': h_label == a_label,
        })
    dom_df = pd.DataFrame(rows)
    dom_df.to_csv(outdir / 'chunk_dominant.csv', index=False)
    print(f"Wrote {outdir / 'chunk_dominant.csv'}  ({len(dom_df)} rows)")

    n_human_ties = int(dom_df['human_tie'].sum())
    n_ai_ties = int(dom_df['ai_tie'].sum())
    n_overlapping_ties = int((dom_df['human_tie'] & dom_df['ai_tie']).sum())
    print(f"Ties: {n_human_ties} human, {n_ai_ties} AI, {n_overlapping_ties} overlapping (tie-break rule: {'>'.join(TIE_BREAK_PRIORITY)})")

    # --- A. full sample ---
    cm_full = confusion_matrix(dom_df['human_dominant'], dom_df['ai_dominant'])
    kappas_full = compute_kappas(dom_df['human_dominant'].tolist(), dom_df['ai_dominant'].tolist())
    print("\n--- A. FULL SAMPLE ---")
    print(cm_full)
    print(kappas_full)

    # --- B. sensitivity: exclude any chunk tied for either coder ---
    untied = dom_df[~(dom_df['human_tie'] | dom_df['ai_tie'])]
    cm_untied = confusion_matrix(untied['human_dominant'], untied['ai_dominant'])
    kappas_untied = compute_kappas(untied['human_dominant'].tolist(), untied['ai_dominant'].tolist())
    print(f"\n--- B. SENSITIVITY: ties excluded (N={len(untied)}) ---")
    print(cm_untied)
    print(kappas_untied)

    summary = {
        'n_chunks_coded_by_both': int(len(dom_df)),
        'ordinal_order': LOGICS,
        'ordinal_rationale': ("PS(0) < DS(1) < PPG(2) < SC(3), following the coding instrument's own "
                               "numbering (1=PS,2=DS,3=PPG,4=SC), treated as an ordinal continuum so that "
                               "weighted kappa penalizes disagreement between non-adjacent classes more "
                               "than between neighboring ones."),
        'tie_break_rule': f"Ties broken by fixed priority order {' > '.join(TIE_BREAK_PRIORITY)} (arbitrary, disclosed).",
        'n_human_ties': n_human_ties, 'n_ai_ties': n_ai_ties, 'n_overlapping_ties': n_overlapping_ties,
        'full_sample': {
            'n': int(len(dom_df)),
            'confusion_matrix': cm_full.values.tolist(), 'labels_order': LOGICS,
            **kappas_full,
        },
        'tie_excluded_sensitivity': {
            'n': int(len(untied)),
            'confusion_matrix': cm_untied.values.tolist(), 'labels_order': LOGICS,
            **kappas_untied,
        },
    }
    with open(outdir / 'summary_chunk_kappa.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {outdir / 'summary_chunk_kappa.json'}")
    print(f"\nPRIMARY result: quadratic-weighted kappa = {kappas_full['kappa_quadratic']:.4f} "
          f"(ties-excluded sensitivity: {kappas_untied['kappa_quadratic']:.4f})")


if __name__ == '__main__':
    main()
