#!/usr/bin/env python3
"""
07_build_dashboard_data.py

Step 7: assembles the single JSON data bundle consumed by the dashboard
artifact, from the outputs of steps 01/02 (per-chunk, matched-pair
agreement) and step 06 (report-level, all-chunks document profile).

For each report this merges:
  - chunk_level.csv       (per-chunk human/ai point vectors, coded_by_both)
  - chunk_dominant.csv    (per-chunk dominant logic + tie flags)
  - summary.json / summary_chunk_kappa.json (report-level agreement stats)
  - report_level_profile.json (06's output: whole-file, unmatched profile)

into one `reports.<code>` entry, plus a `pooled` entry (confusion-matrix
summation across reports, as in 04_aggregate_multi_report.py, and a
pooled report-level profile by summing raw points across reports).

Per chunk, beyond the raw human/ai proportion vectors, this also computes
a "margin" for each coder: the point gap between that coder's dominant
logic and its runner-up on that same chunk. A small margin means the
coder's own call was close (e.g. a near-tie that didn't quite trigger the
tie-break rule); a large margin means the coder's call was clear-cut. This
lets the dashboard sort/filter disagreements by how decisive each side's
own coding was, which is more informative than just "agree / disagree".

Usage
-----
    python3 07_build_dashboard_data.py \\
        --report 363466802_2021=multi_report/363466802_2021 \\
        --report 382023395_2022=multi_report/382023395_2022 \\
        --report 25-1650208_2021=multi_report/25-1650208_2021 \\
        --codes R1,R2,R3 \\
        --out multi_report/dashboard_bundle_v2.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LOGICS = ['PS', 'DS', 'PPG', 'SC']


def second_highest(points: dict, winner: str) -> int:
    return max(v for l, v in points.items() if l != winner)


def build_report(path: Path, report_id: str) -> dict:
    chunk_level = pd.read_csv(path / 'chunk_level.csv')
    chunk_dom = pd.read_csv(path / 'chunk_dominant.csv')
    with open(path / 'summary.json') as f:
        summary = json.load(f)
    with open(path / 'summary_chunk_kappa.json') as f:
        ksum = json.load(f)
    with open(path / 'report_level_profile.json') as f:
        profile = json.load(f)
    with open(path / 'flexible_agreement.json') as f:
        flex = json.load(f)
    flex_by_chunk = {c['chunk_id']: c for c in flex['chunk_detail']}

    included = chunk_level[chunk_level['coded_by_both']].copy()
    included = included.rename(columns={'chunk_id_cd': 'chunk_id'})
    merged = included.merge(chunk_dom, on='chunk_id', how='inner')

    chunks = []
    for _, r in merged.iterrows():
        h_pts = {l: int(r[f'human_points_{l}']) for l in LOGICS}
        a_pts = {l: int(r[f'ai_points_{l}']) for l in LOGICS}
        h_tot, a_tot = sum(h_pts.values()), sum(a_pts.values())
        h_prop = [round(h_pts[l] / h_tot, 4) if h_tot else 0.0 for l in LOGICS]
        a_prop = [round(a_pts[l] / a_tot, 4) if a_tot else 0.0 for l in LOGICS]
        h_margin = h_pts[r['human_dominant']] - second_highest(h_pts, r['human_dominant'])
        a_margin = a_pts[r['ai_dominant']] - second_highest(a_pts, r['ai_dominant'])
        fdet = flex_by_chunk.get(int(r['chunk_id']), {})
        chunks.append({
            'id': r['chunk_id'], 'text': r['text'],
            'h': h_prop, 'a': a_prop,
            'hd': r['human_dominant'], 'ad': r['ai_dominant'],
            'ht': bool(r['human_tie']), 'at': bool(r['ai_tie']),
            'agree': bool(r['match']),
            'hm': int(h_margin), 'am': int(a_margin),
            'flexOnly': bool(fdet.get('flexible_only', False)),
        })

    cm = np.array(ksum['full_sample']['confusion_matrix'])

    return {
        'report_id': report_id,
        'n_human_rows': summary['n_human_rows'], 'n_ai_rows': summary['n_ai_rows'],
        'n_matched': summary['n_matched_chunks'], 'n_included': summary['n_included_in_aggregate'],
        'doc_human_prop': [round(summary['doc_human_proportions'][l], 4) for l in LOGICS],
        'doc_ai_prop': [round(summary['doc_ai_proportions'][l], 4) for l in LOGICS],
        'confusion': cm.tolist(),
        # Primary statistic: standard (nominal, unweighted) Cohen's kappa --
        # PS/DS/PPG/SC are categories with no established ordering, so no
        # ordinal weighting is applied. kappa_quad/kappa_lin are kept only
        # as a labeled, clearly-flagged exploratory comparison.
        'kappa_standard': ksum['full_sample']['kappa_unweighted'],
        'kappa_quad': ksum['full_sample']['kappa_quadratic'],
        'kappa_lin': ksum['full_sample']['kappa_linear'],
        'kappa_unw': ksum['full_sample']['kappa_unweighted'],
        'raw_agree': ksum['full_sample']['raw_agreement'],
        'n_human_ties': ksum['n_human_ties'], 'n_ai_ties': ksum['n_ai_ties'],
        'kappa_quad_sens': ksum['tie_excluded_sensitivity']['kappa_quadratic'],
        'kappa_standard_sens': ksum['tie_excluded_sensitivity']['kappa_unweighted'],
        'n_sens': ksum['tie_excluded_sensitivity']['n'],
        'chunks': chunks,
        'flex': {
            'n': flex['n'], 'n_permutations': flex['n_permutations'],
            'observed_rate': flex['flexible_agreement']['observed_rate'],
            'chance_rate': flex['flexible_agreement']['permutation_chance_rate'],
            'kappa_flexible': flex['flexible_agreement']['kappa_flexible'],
            'n_flexible_only': flex['flexible_agreement']['n_flexible_only_chunks'],
            'kappa_standard_perm_check': flex['permutation_sanity_check']['kappa_strict_via_permutation'],
        },
        # report-level, all-coded-chunks profile (06's output) -- independent
        # of chunk matching; NOT restricted to coded_by_both.
        'profile': {
            'n_human_rows_total': profile['n_human_rows_total'], 'n_ai_rows_total': profile['n_ai_rows_total'],
            'n_human_rows_coded': profile['n_human_rows_coded'], 'n_ai_rows_coded': profile['n_ai_rows_coded'],
            'human_prop': [round(profile['human_proportions'][l], 4) for l in LOGICS],
            'ai_prop': [round(profile['ai_proportions'][l], 4) for l in LOGICS],
            'tvd': round(profile['profile_distance']['total_variation_distance'], 4),
            'cosine': round(profile['profile_distance']['cosine_similarity'], 4),
            'human_curve': profile['human_position_curve'],
            'ai_curve': profile['ai_position_curve'],
        },
    }


def kappa_from_confusion(cm: np.ndarray, weights=None) -> float:
    from sklearn.metrics import cohen_kappa_score
    h, a = [], []
    for i, hv in enumerate(LOGICS):
        for j, av in enumerate(LOGICS):
            n = int(cm[i, j])
            h += [hv] * n
            a += [av] * n
    return float(cohen_kappa_score(h, a, labels=LOGICS, weights=weights))


def build_pooled(report_entries: dict) -> dict:
    cm_sum = np.zeros((4, 4), dtype=int)
    all_chunks = []
    n_human_total = n_ai_total = n_matched = n_included = 0
    n_human_ties = n_ai_ties = 0

    for code, rep in report_entries.items():
        cm_sum += np.array(rep['confusion'])
        n_human_total += rep['n_human_rows']; n_ai_total += rep['n_ai_rows']
        n_matched += rep['n_matched']; n_included += rep['n_included']
        n_human_ties += rep['n_human_ties']; n_ai_ties += rep['n_ai_ties']
        for c in rep['chunks']:
            c2 = dict(c); c2['r'] = code
            all_chunks.append(c2)

    # Pooled doc_human_prop / doc_ai_prop and the pooled report-level profile
    # are computed separately in main(), from exact point sums across each
    # report's own summary/profile files (not reconstructed from the rounded
    # per-report proportions here).
    return {
        'n_reports': len(report_entries),
        'report_ids': list(report_entries.keys()),
        'confusion': cm_sum.tolist(),
        'kappa_standard': kappa_from_confusion(cm_sum, None),
        'kappa_quad': kappa_from_confusion(cm_sum, 'quadratic'),
        'kappa_lin': kappa_from_confusion(cm_sum, 'linear'),
        'kappa_unw': kappa_from_confusion(cm_sum, None),
        'raw_agree': float(np.trace(cm_sum) / cm_sum.sum()),
        'n_included': int(cm_sum.sum()),
        'n_human_ties': n_human_ties, 'n_ai_ties': n_ai_ties,
        'chunks': all_chunks,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--report', action='append', required=True, help='report_id=path (repeatable)')
    ap.add_argument('--codes', required=True, help='comma-separated short codes, same order as --report')
    ap.add_argument('--flex-pooled', required=True, help='flexible_agreement_pooled.json from 09_pool_flexible_agreement.py')
    ap.add_argument('--pooled-summary-json', required=True, help='pooled_summary.json from 04_aggregate_multi_report.py (for the pooled ties-excluded sensitivity check)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    specs = [s.split('=', 1) for s in args.report]
    codes = args.codes.split(',')
    assert len(codes) == len(specs)

    reports = {}
    for code, (rid, path) in zip(codes, specs):
        reports[code] = build_report(Path(path), rid)

    pooled = build_pooled(reports)

    # pooled document-level profile: sum raw points across reports' own
    # report_level_profile.json files (exact, not reconstructed from rounded bundle values)
    pooled_human_pts = {l: 0 for l in LOGICS}
    pooled_ai_pts = {l: 0 for l in LOGICS}
    n_h_coded = n_a_coded = n_h_total = n_a_total = 0
    for code, (rid, path) in zip(codes, specs):
        with open(Path(path) / 'report_level_profile.json') as f:
            prof = json.load(f)
        for l in LOGICS:
            pooled_human_pts[l] += prof['human_points'][l]
            pooled_ai_pts[l] += prof['ai_points'][l]
        n_h_coded += prof['n_human_rows_coded']; n_a_coded += prof['n_ai_rows_coded']
        n_h_total += prof['n_human_rows_total']; n_a_total += prof['n_ai_rows_total']
    h_tot = sum(pooled_human_pts.values()); a_tot = sum(pooled_ai_pts.values())
    pooled_human_prop = {l: (pooled_human_pts[l] / h_tot if h_tot else 0.0) for l in LOGICS}
    pooled_ai_prop = {l: (pooled_ai_pts[l] / a_tot if a_tot else 0.0) for l in LOGICS}
    from sklearn.metrics import cohen_kappa_score  # noqa
    pv = np.array([pooled_human_prop[l] for l in LOGICS]); qv = np.array([pooled_ai_prop[l] for l in LOGICS])
    tvd = float(0.5 * np.abs(pv - qv).sum())
    denom = np.linalg.norm(pv) * np.linalg.norm(qv)
    cos = float(np.dot(pv, qv) / denom) if denom else float('nan')

    # Pooled position curves: mean of the 3 reports' per-bin prevalence
    # (bins are already normalized 0..1 position, so bin b is comparable
    # across reports of different length), skipping reports with no value
    # in that bin for that logic.
    def pooled_curve(curve_key):
        bin_centers = None
        prevalence = {l: [] for l in LOGICS}
        n_bins = None
        for code, (rid, path) in zip(codes, specs):
            with open(Path(path) / 'report_level_profile.json') as f:
                prof = json.load(f)
            curve = prof[curve_key]
            bin_centers = curve['bin_centers']
            n_bins = len(bin_centers)
            for l in LOGICS:
                prevalence[l].append(curve['prevalence'][l])
        out = {l: [] for l in LOGICS}
        for b in range(n_bins):
            for l in LOGICS:
                vals = [series[b] for series in prevalence[l] if series[b] is not None]
                out[l].append(round(float(np.mean(vals)), 4) if vals else None)
        return {'bin_centers': bin_centers, 'prevalence': out}

    with open(Path(args.pooled_summary_json)) as f:
        pooled_summary = json.load(f)
    pooled['kappa_standard_sens'] = pooled_summary['tie_excluded_sensitivity']['kappa_unweighted']
    pooled['kappa_quad_sens'] = pooled_summary['tie_excluded_sensitivity']['kappa_quadratic']
    pooled['n_sens'] = pooled_summary['tie_excluded_sensitivity']['n']

    with open(Path(args.flex_pooled)) as f:
        flex_pooled = json.load(f)
    pooled['flex'] = {
        'n': flex_pooled['n'], 'n_permutations': flex_pooled['n_permutations'],
        'observed_rate': flex_pooled['flexible_agreement']['observed_rate'],
        'chance_rate': flex_pooled['flexible_agreement']['permutation_chance_rate'],
        'kappa_flexible': flex_pooled['flexible_agreement']['kappa_flexible'],
        'n_flexible_only': flex_pooled['flexible_agreement']['n_flexible_only_chunks'],
        'kappa_standard_perm_check': flex_pooled['permutation_sanity_check']['kappa_strict_via_permutation'],
    }

    pooled['profile'] = {
        'n_human_rows_total': n_h_total, 'n_ai_rows_total': n_a_total,
        'n_human_rows_coded': n_h_coded, 'n_ai_rows_coded': n_a_coded,
        'human_prop': [round(pooled_human_prop[l], 4) for l in LOGICS],
        'ai_prop': [round(pooled_ai_prop[l], 4) for l in LOGICS],
        'tvd': round(tvd, 4), 'cosine': round(cos, 4),
        'human_curve': pooled_curve('human_position_curve'),
        'ai_curve': pooled_curve('ai_position_curve'),
    }

    bundle = {'reports': reports, 'pooled': pooled}
    with open(args.out, 'w') as f:
        json.dump(bundle, f, separators=(',', ':'), ensure_ascii=False)
    print(f"Wrote {args.out} ({Path(args.out).stat().st_size} bytes)")


if __name__ == '__main__':
    main()
