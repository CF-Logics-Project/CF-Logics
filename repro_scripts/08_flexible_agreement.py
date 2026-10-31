#!/usr/bin/env python3
"""
08_flexible_agreement.py

Step 8: revises the primary agreement statistic and adds an alternative,
chunk-flexible agreement measure, in response to a methodological
correction: PS/DS/PPG/SC are NOMINAL categories, not an ordinal sequence.
02_chunk_kappa.py's quadratic/linear weighting assumed a fixed category
order (PS<DS<PPG<SC) that has no theoretical basis -- there is no reason
a PPG/SC disagreement is intrinsically "smaller" than a PS/SC one. That
assumption is retired.

Two analyses, now produced by this script instead:

(A) STANDARD (nominal, unweighted) Cohen's kappa -- the primary statistic.
    This was already being computed correctly as kappa_unweighted in
    summary_chunk_kappa.json's cohen_kappa_score(weights=None) call, which
    does not use the disputed ordering at all; this script re-derives it
    directly from chunk_dominant.csv for a self-contained, independently
    checkable primary result and drops the quadratic/linear framing.

(B) AN ALTERNATIVE "chunk-flexible vicinity" agreement measure, for cases
    where the two coders' single dominant picks differ but their decisions
    were still "in the same vicinity" FOR THAT CHUNK specifically -- with
    no assumption about any global relationship between logic categories.
    Each coder's own chunk-level points (already computed in
    chunk_level.csv, by summing each sentence/element's top-1=2pt /
    top-2=1pt ranks) imply, for that one chunk, which logic that coder
    emphasized most (top-1) and, if any, a secondary logic that also
    picked up points (top-2) -- directly extending the project's own
    sentence-level top-1/top-2 ensemble design up to the chunk level,
    rather than importing an external ordinal scale between categories.

    A chunk counts as a FLEXIBLE match if either coder's dominant pick
    equals the other coder's secondary pick (or they strictly agree).
    This is NOT a standard two-label comparison (it uses both coders'
    richer per-chunk rankings, not just one label each), so a closed-form
    chance correction does not apply. Instead this script estimates the
    chance level by permutation: within each report, AI chunks are
    randomly re-paired against human chunks (breaking the actual
    chunk-to-chunk correspondence while holding each side's own points
    fixed) many times, and the flexible-match rate is recomputed each
    time. The permutation mean is the Monte Carlo chance baseline Pe_flex;
    kappa_flexible = (Po_flex - Pe_flex) / (1 - Pe_flex), structurally the
    same form as Cohen's kappa but with a simulated Pe instead of the
    closed-form marginal-product one (which assumes plain single-label
    matching). The same permutation is also run for strict (standard)
    agreement as a sanity check: the permutation-based strict kappa should
    closely match the closed-form cohen_kappa_score result.

Usage
-----
    python3 08_flexible_agreement.py \\
        --chunk-level-csv multi_report/<report>/chunk_level.csv \\
        --chunk-dominant-csv multi_report/<report>/chunk_dominant.csv \\
        --report-id <report_id> --n-perm 5000 --seed 0 \\
        --outdir multi_report/<report>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

LOGICS = ['PS', 'DS', 'PPG', 'SC']


def top2_set(points: dict) -> set:
    """The coder's own chunk-level top-1 (and, if present, top-2) logic by
    points -- i.e. which logic(s) this coder emphasized for this one
    chunk, independent of any other chunk or any assumed category order."""
    ordered = sorted(LOGICS, key=lambda l: -points[l])
    s = {ordered[0]} if points[ordered[0]] > 0 else set()
    if len(ordered) > 1 and points[ordered[1]] > 0:
        s.add(ordered[1])
    return s


def flexible_match(h_pts: dict, a_pts: dict, h_dom: str, a_dom: str) -> bool:
    if h_dom == a_dom:
        return True
    h_top2, a_top2 = top2_set(h_pts), top2_set(a_pts)
    return (a_dom in h_top2) or (h_dom in a_top2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--chunk-level-csv', required=True)
    ap.add_argument('--chunk-dominant-csv', required=True)
    ap.add_argument('--report-id', required=True)
    ap.add_argument('--n-perm', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--outdir', default='.')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    chunk_level = pd.read_csv(args.chunk_level_csv)
    included = chunk_level[chunk_level['coded_by_both']].rename(columns={'chunk_id_cd': 'chunk_id'})
    dom = pd.read_csv(args.chunk_dominant_csv)
    merged = included.merge(dom, on='chunk_id', how='inner').reset_index(drop=True)
    n = len(merged)

    h_pts_list = [{l: int(r[f'human_points_{l}']) for l in LOGICS} for _, r in merged.iterrows()]
    a_pts_list = [{l: int(r[f'ai_points_{l}']) for l in LOGICS} for _, r in merged.iterrows()]
    h_dom_list = merged['human_dominant'].tolist()
    a_dom_list = merged['ai_dominant'].tolist()

    # --- (A) standard (nominal, unweighted) Cohen's kappa -- primary statistic ---
    kappa_standard = float(cohen_kappa_score(h_dom_list, a_dom_list, labels=LOGICS))
    raw_strict = float(np.mean([h == a for h, a in zip(h_dom_list, a_dom_list)]))

    # --- (B) observed flexible-match rate ---
    flex_flags = [flexible_match(h_pts_list[i], a_pts_list[i], h_dom_list[i], a_dom_list[i]) for i in range(n)]
    po_flex = float(np.mean(flex_flags))

    # --- permutation null: re-pair AI chunks against human chunks at random,
    #     within this report, holding each side's own per-chunk data fixed.
    #     Each side's own top-2 set depends only on that side's own chunk, not
    #     on the pairing, so it is precomputed once outside the permutation loop. ---
    h_top2_list = [top2_set(p) for p in h_pts_list]
    a_top2_list = [top2_set(p) for p in a_pts_list]
    h_dom_arr = np.array(h_dom_list)
    a_dom_arr = np.array(a_dom_list)

    perm_flex_rates = np.empty(args.n_perm)
    perm_strict_rates = np.empty(args.n_perm)
    idx = np.arange(n)
    for p in range(args.n_perm):
        perm = rng.permutation(idx)
        strict_hits = 0
        flex_hits = 0
        for i in range(n):
            j = perm[i]
            h_label, a_label = h_dom_arr[i], a_dom_arr[j]
            if h_label == a_label:
                strict_hits += 1
                flex_hits += 1
            elif (a_label in h_top2_list[i]) or (h_label in a_top2_list[j]):
                flex_hits += 1
        perm_strict_rates[p] = strict_hits / n
        perm_flex_rates[p] = flex_hits / n

    pe_flex = float(perm_flex_rates.mean())
    pe_strict_perm = float(perm_strict_rates.mean())
    kappa_flexible = (po_flex - pe_flex) / (1 - pe_flex) if pe_flex < 1 else float('nan')
    kappa_strict_perm = (raw_strict - pe_strict_perm) / (1 - pe_strict_perm) if pe_strict_perm < 1 else float('nan')

    # per-chunk detail for the dashboard: which chunks are flexible-only matches
    # (disagree on the dominant label, but still "in vicinity")
    chunk_detail = []
    for i in range(n):
        h_label, a_label = h_dom_list[i], a_dom_list[i]
        strict = (h_label == a_label)
        flex = flex_flags[i]
        chunk_detail.append({
            'chunk_id': int(merged.loc[i, 'chunk_id']),
            'strict_match': bool(strict),
            'flexible_match': bool(flex),
            'flexible_only': bool(flex and not strict),
        })

    result = {
        'report_id': args.report_id,
        'n': n,
        'n_permutations': args.n_perm, 'seed': args.seed,
        'standard_kappa': {
            'kappa': kappa_standard,
            'raw_agreement': raw_strict,
        },
        'flexible_agreement': {
            'observed_rate': po_flex,
            'permutation_chance_rate': pe_flex,
            'kappa_flexible': kappa_flexible,
            'n_flexible_only_chunks': int(sum(c['flexible_only'] for c in chunk_detail)),
        },
        'permutation_sanity_check': {
            'strict_rate_closed_form_chance': None,  # filled by caller from the cohen_kappa_score Pe if desired
            'strict_rate_permutation_chance': pe_strict_perm,
            'kappa_strict_via_permutation': kappa_strict_perm,
            'kappa_strict_closed_form': kappa_standard,
        },
        'chunk_detail': chunk_detail,
    }
    with open(outdir / 'flexible_agreement.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f"[{args.report_id}] N={n}")
    print(f"  standard (nominal) kappa = {kappa_standard:.4f}   raw agreement = {raw_strict:.4f}")
    print(f"  flexible match rate      = {po_flex:.4f}   permutation chance = {pe_flex:.4f}   "
          f"kappa_flexible = {kappa_flexible:.4f}")
    print(f"  sanity check: permutation-based strict kappa = {kappa_strict_perm:.4f} "
          f"(closed-form = {kappa_standard:.4f})")
    print(f"  {result['flexible_agreement']['n_flexible_only_chunks']} chunks matched only under the flexible criterion")
    print(f"Wrote {outdir / 'flexible_agreement.json'}")


if __name__ == '__main__':
    main()
