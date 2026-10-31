#!/usr/bin/env python3
"""
04_aggregate_multi_report.py

Step 4: pools the per-report outputs of 01/02 across multiple reports into
(a) a per-report summary table and (b) a single pooled (study-level)
confusion matrix and kappa.

Pooling is done at the confusion-matrix level (summing counts across
reports' 4x4 human-x-AI matrices), not by averaging each report's kappa,
because kappa is not a linear statistic and averaging per-report kappas
would not equal the kappa of the combined sample.

Usage
-----
    python3 04_aggregate_multi_report.py \
        --report 363466802_2021=multi_report/363466802_2021 \
        --report 382023395_2022=multi_report/382023395_2022 \
        --report 25-1650208_2021=multi_report/25-1650208_2021 \
        --outdir multi_report
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

LOGICS = ['PS', 'DS', 'PPG', 'SC']


def kappa_from_confusion(cm: np.ndarray, weights=None) -> float:
    """Cohen's kappa computed directly from an already-pooled confusion
    matrix (rows=human, cols=AI), reconstructing the implied label lists
    so it matches sklearn's weighting conventions exactly."""
    h_labels, a_labels = [], []
    for i, hv in enumerate(LOGICS):
        for j, av in enumerate(LOGICS):
            n = int(cm[i, j])
            h_labels += [hv] * n
            a_labels += [av] * n
    return float(cohen_kappa_score(h_labels, a_labels, labels=LOGICS, weights=weights))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--report', action='append', required=True,
                     help='report_id=path_to_report_outdir (repeatable)')
    ap.add_argument('--outdir', default='.')
    args = ap.parse_args()

    reports = {}
    for spec in args.report:
        rid, path = spec.split('=', 1)
        reports[rid] = Path(path)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    per_report_rows = []
    pooled_cm_full = np.zeros((4, 4), dtype=int)
    pooled_cm_untied = np.zeros((4, 4), dtype=int)
    pooled_doc_human_pts = {l: 0 for l in LOGICS}
    pooled_doc_ai_pts = {l: 0 for l in LOGICS}
    pooled_n_full = 0
    pooled_n_untied = 0

    for rid, path in reports.items():
        with open(path / 'summary.json') as f:
            summary = json.load(f)
        with open(path / 'summary_chunk_kappa.json') as f:
            ksum = json.load(f)
        dom = pd.read_csv(path / 'chunk_dominant.csv')

        cm_full = np.array(ksum['full_sample']['confusion_matrix'])
        cm_untied = np.array(ksum['tie_excluded_sensitivity']['confusion_matrix'])
        pooled_cm_full += cm_full
        pooled_cm_untied += cm_untied
        pooled_n_full += ksum['full_sample']['n']
        pooled_n_untied += ksum['tie_excluded_sensitivity']['n']

        for l in LOGICS:
            pooled_doc_human_pts[l] += summary['doc_human_points'][l]
            pooled_doc_ai_pts[l] += summary['doc_ai_points'][l]

        per_report_rows.append({
            'report_id': rid,
            'n_human_rows': summary['n_human_rows'],
            'n_ai_rows': summary['n_ai_rows'],
            'n_matched_chunks': summary['n_matched_chunks'],
            'n_coded_by_both': summary['n_included_in_aggregate'],
            'human_prop_PS': summary['doc_human_proportions']['PS'],
            'human_prop_DS': summary['doc_human_proportions']['DS'],
            'human_prop_PPG': summary['doc_human_proportions']['PPG'],
            'human_prop_SC': summary['doc_human_proportions']['SC'],
            'ai_prop_PS': summary['doc_ai_proportions']['PS'],
            'ai_prop_DS': summary['doc_ai_proportions']['DS'],
            'ai_prop_PPG': summary['doc_ai_proportions']['PPG'],
            'ai_prop_SC': summary['doc_ai_proportions']['SC'],
            'kappa_quadratic': ksum['full_sample']['kappa_quadratic'],
            'kappa_linear': ksum['full_sample']['kappa_linear'],
            'kappa_unweighted': ksum['full_sample']['kappa_unweighted'],
            'raw_agreement': ksum['full_sample']['raw_agreement'],
            'kappa_quadratic_ties_excluded': ksum['tie_excluded_sensitivity']['kappa_quadratic'],
            'n_human_ties': ksum['n_human_ties'],
            'n_ai_ties': ksum['n_ai_ties'],
            'top1_pct_agreement': summary['top1_pct_agreement'],
        })

    per_report_df = pd.DataFrame(per_report_rows)
    per_report_df.to_csv(outdir / 'per_report_summary.csv', index=False)
    print("Per-report summary:")
    print(per_report_df.to_string(index=False))

    # pooled kappa, recomputed from the pooled confusion matrix (not averaged)
    pooled_kappas_full = {
        'kappa_unweighted': kappa_from_confusion(pooled_cm_full, weights=None),
        'kappa_linear': kappa_from_confusion(pooled_cm_full, weights='linear'),
        'kappa_quadratic': kappa_from_confusion(pooled_cm_full, weights='quadratic'),
        'raw_agreement': float(np.trace(pooled_cm_full) / pooled_cm_full.sum()),
    }
    pooled_kappas_untied = {
        'kappa_unweighted': kappa_from_confusion(pooled_cm_untied, weights=None),
        'kappa_linear': kappa_from_confusion(pooled_cm_untied, weights='linear'),
        'kappa_quadratic': kappa_from_confusion(pooled_cm_untied, weights='quadratic'),
        'raw_agreement': float(np.trace(pooled_cm_untied) / pooled_cm_untied.sum()),
    }

    pooled_human_total = sum(pooled_doc_human_pts.values())
    pooled_ai_total = sum(pooled_doc_ai_pts.values())
    pooled_human_prop = {l: pooled_doc_human_pts[l] / pooled_human_total for l in LOGICS}
    pooled_ai_prop = {l: pooled_doc_ai_pts[l] / pooled_ai_total for l in LOGICS}

    print("\n--- POOLED (all reports) ---")
    print(pd.DataFrame(pooled_cm_full, index=[f'human_{l}' for l in LOGICS], columns=[f'ai_{l}' for l in LOGICS]))
    print(pooled_kappas_full)
    print("Pooled human proportions:", {l: round(v, 4) for l, v in pooled_human_prop.items()})
    print("Pooled AI proportions:   ", {l: round(v, 4) for l, v in pooled_ai_prop.items()})

    pooled_summary = {
        'n_reports': len(reports),
        'report_ids': list(reports.keys()),
        'pooled_n_coded_by_both_full': int(pooled_n_full),
        'pooled_n_coded_by_both_ties_excluded': int(pooled_n_untied),
        'full_sample': {
            'n': int(pooled_n_full),
            'confusion_matrix': pooled_cm_full.tolist(), 'labels_order': LOGICS,
            **pooled_kappas_full,
        },
        'tie_excluded_sensitivity': {
            'n': int(pooled_n_untied),
            'confusion_matrix': pooled_cm_untied.tolist(), 'labels_order': LOGICS,
            **pooled_kappas_untied,
        },
        'pooled_doc_human_points': pooled_doc_human_pts,
        'pooled_doc_ai_points': pooled_doc_ai_pts,
        'pooled_doc_human_proportions': pooled_human_prop,
        'pooled_doc_ai_proportions': pooled_ai_prop,
    }
    with open(outdir / 'pooled_summary.json', 'w') as f:
        json.dump(pooled_summary, f, indent=2)
    print(f"\nWrote {outdir / 'per_report_summary.csv'} and {outdir / 'pooled_summary.json'}")


if __name__ == '__main__':
    main()
