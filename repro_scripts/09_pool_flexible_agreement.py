#!/usr/bin/env python3
"""
09_pool_flexible_agreement.py

Pools 08_flexible_agreement.py's per-chunk inputs across the 3 reports
into one study-level (N=306) standard kappa and flexible-agreement
permutation test, by combining all reports' coded_by_both chunks into a
single pool and permuting across the full pooled set (consistent with how
04_aggregate_multi_report.py pools the confusion matrix by summing counts
rather than averaging per-report kappas: here too, pooling is done on the
combined chunk set, not by averaging the three reports' statistics).

Usage
-----
    python3 09_pool_flexible_agreement.py \\
        --chunk-level-csv multi_report/363466802_2021/chunk_level.csv --chunk-dominant-csv multi_report/363466802_2021/chunk_dominant.csv \\
        --chunk-level-csv multi_report/382023395_2022/chunk_level.csv --chunk-dominant-csv multi_report/382023395_2022/chunk_dominant.csv \\
        --chunk-level-csv multi_report/25-1650208_2021/chunk_level.csv --chunk-dominant-csv multi_report/25-1650208_2021/chunk_dominant.csv \\
        --n-perm 3000 --seed 0 --outdir multi_report
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

LOGICS = ['PS', 'DS', 'PPG', 'SC']


def top2_set(points: dict) -> set:
    ordered = sorted(LOGICS, key=lambda l: -points[l])
    s = {ordered[0]} if points[ordered[0]] > 0 else set()
    if len(ordered) > 1 and points[ordered[1]] > 0:
        s.add(ordered[1])
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--chunk-level-csv', action='append', required=True)
    ap.add_argument('--chunk-dominant-csv', action='append', required=True)
    ap.add_argument('--n-perm', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--outdir', default='.')
    args = ap.parse_args()

    assert len(args.chunk_level_csv) == len(args.chunk_dominant_csv)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    h_pts_list, a_pts_list, h_dom_list, a_dom_list = [], [], [], []
    for cl_path, cd_path in zip(args.chunk_level_csv, args.chunk_dominant_csv):
        chunk_level = pd.read_csv(cl_path)
        included = chunk_level[chunk_level['coded_by_both']].rename(columns={'chunk_id_cd': 'chunk_id'})
        dom = pd.read_csv(cd_path)
        merged = included.merge(dom, on='chunk_id', how='inner').reset_index(drop=True)
        for _, r in merged.iterrows():
            h_pts_list.append({l: int(r[f'human_points_{l}']) for l in LOGICS})
            a_pts_list.append({l: int(r[f'ai_points_{l}']) for l in LOGICS})
            h_dom_list.append(r['human_dominant']); a_dom_list.append(r['ai_dominant'])

    n = len(h_dom_list)
    kappa_standard = float(cohen_kappa_score(h_dom_list, a_dom_list, labels=LOGICS))
    raw_strict = float(np.mean([h == a for h, a in zip(h_dom_list, a_dom_list)]))

    h_top2_list = [top2_set(p) for p in h_pts_list]
    a_top2_list = [top2_set(p) for p in a_pts_list]

    flex_flags = []
    for i in range(n):
        h_label, a_label = h_dom_list[i], a_dom_list[i]
        flex_flags.append(h_label == a_label or a_label in h_top2_list[i] or h_label in a_top2_list[i])
    po_flex = float(np.mean(flex_flags))

    h_dom_arr = np.array(h_dom_list); a_dom_arr = np.array(a_dom_list)
    perm_flex_rates = np.empty(args.n_perm)
    perm_strict_rates = np.empty(args.n_perm)
    idx = np.arange(n)
    for p in range(args.n_perm):
        perm = rng.permutation(idx)
        strict_hits = flex_hits = 0
        for i in range(n):
            j = perm[i]
            h_label, a_label = h_dom_arr[i], a_dom_arr[j]
            if h_label == a_label:
                strict_hits += 1; flex_hits += 1
            elif (a_label in h_top2_list[i]) or (h_label in a_top2_list[j]):
                flex_hits += 1
        perm_strict_rates[p] = strict_hits / n
        perm_flex_rates[p] = flex_hits / n

    pe_flex = float(perm_flex_rates.mean())
    pe_strict_perm = float(perm_strict_rates.mean())
    kappa_flexible = (po_flex - pe_flex) / (1 - pe_flex)
    kappa_strict_perm = (raw_strict - pe_strict_perm) / (1 - pe_strict_perm)

    chunk_detail = []
    for i in range(n):
        strict = h_dom_list[i] == a_dom_list[i]
        chunk_detail.append({'strict_match': bool(strict), 'flexible_match': bool(flex_flags[i]),
                              'flexible_only': bool(flex_flags[i] and not strict)})

    result = {
        'report_id': 'pooled', 'n': n, 'n_permutations': args.n_perm, 'seed': args.seed,
        'standard_kappa': {'kappa': kappa_standard, 'raw_agreement': raw_strict},
        'flexible_agreement': {
            'observed_rate': po_flex, 'permutation_chance_rate': pe_flex,
            'kappa_flexible': kappa_flexible,
            'n_flexible_only_chunks': int(sum(c['flexible_only'] for c in chunk_detail)),
        },
        'permutation_sanity_check': {
            'strict_rate_permutation_chance': pe_strict_perm,
            'kappa_strict_via_permutation': kappa_strict_perm,
            'kappa_strict_closed_form': kappa_standard,
        },
    }
    with open(outdir / 'flexible_agreement_pooled.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f"[pooled] N={n}")
    print(f"  standard kappa = {kappa_standard:.4f}   raw agreement = {raw_strict:.4f}")
    print(f"  flexible rate = {po_flex:.4f}  chance = {pe_flex:.4f}  kappa_flexible = {kappa_flexible:.4f}")
    print(f"  sanity: permutation strict kappa = {kappa_strict_perm:.4f} (closed-form {kappa_standard:.4f})")
    print(f"Wrote {outdir / 'flexible_agreement_pooled.json'}")


if __name__ == '__main__':
    main()
