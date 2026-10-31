#!/usr/bin/env python3
"""
agreement_analysis.py

ONE script that runs the full Human-vs-AI institutional-logics agreement
analysis, for as many annual reports as you list in a manifest file, and
pools the results across all of them. It replaces the earlier 9-script
pipeline (01_match_and_score.py ... 09_pool_flexible_agreement.py) with a
single, readable run.

What it does, per report
-------------------------
1. MATCH   — match the human coder's chunks to the AI pipeline's chunks by
             exact text (an occurrence index handles repeated text).
2. SCORE   — reduce each coder's 5 elements (community, donor, activity,
             board, grantee) to a weighted 4-logic points vector per chunk:
             that element's primary (top-1) logic scores 2 points, its
             secondary (top-2) logic scores 1 point, summed across the 5
             elements. A chunk counts as "coded by both" if each coder
             marked >=1 element applicable.
3. KAPPA   — one observation per chunk coded by both: each coder's
             "dominant logic" is whichever of PS/DS/PPG/SC holds their
             highest point total for that chunk (ties broken by the fixed,
             disclosed priority PS>DS>PPG>SC — mechanical, not theoretical).
             PRIMARY statistic: standard (nominal, unweighted) Cohen's
             kappa, because PS/DS/PPG/SC have no established ordering.
             Quadratic/linear ORDINAL-weighted kappa (which assumes
             PS<DS<PPG<SC) is computed too but only as a RETIRED,
             clearly-flagged comparison to an earlier version of this
             analysis — never read as primary.
4. FLEXIBLE — an alternative, chunk-specific "vicinity" measure: a chunk
             counts as a flexible match if either coder's dominant pick is
             at least the OTHER coder's own secondary (runner-up) emphasis
             for that one chunk — no global category order assumed. Chance
             is estimated by permutation (re-pairing AI chunks against
             human chunks at random, many times), since this criterion
             uses both coders' per-chunk rankings jointly and has no
             closed-form chance formula. The same permutation applied to
             strict matching is a built-in sanity check against the
             closed-form standard kappa.
5. PROFILE — scores EVERY row in each coder's own raw file independently
             (no matching, no coded-by-both filter) to get each coder's
             own overall logic-frequency profile for the whole report,
             plus a smoothed "where in the document" curve per logic.

Then, across all reports in the manifest
-----------------------------------------
  - the confusion matrix is POOLED BY SUMMATION (never by averaging
    per-report kappas, since kappa is not linear) and the pooled kappa is
    recomputed from that summed matrix;
  - the flexible-agreement permutation test is re-run on the full pooled
    set of coded-by-both chunks from every report;
  - the report-level profile is pooled by summing raw points across
    reports before renormalizing, and the position curves are pooled by
    averaging each report's own per-bin value;
  - the document-level proportions used by the headline bar chart
    (doc_human_prop / doc_ai_prop — restricted to chunks coded by both,
    unlike the all-chunks PROFILE above) are pooled the same way: sum each
    report's own points, then renormalize. (An earlier version of the
    dashboard build omitted this pooled figure, which is what made its
    "Pooled" tab render blank — see 03_build_dashboard.py's docstring.)

Outputs
-------
For each report, a folder <outdir>/<report_id>/ with the same
chunk_level.csv / chunk_dominant.csv / summary.json /
summary_chunk_kappa.json / flexible_agreement.json /
report_level_profile.json files as the old pipeline (so the xlsx builder
can still show full per-chunk tables) — plus one file that matters most to
a human reader:

    <outdir>/agreement_bundle.json

a single JSON with a `reports` entry per report code and one `pooled`
entry, structured exactly as 02_build_xlsx.py and 03_build_dashboard.py
expect it. That one file is the complete, reproducible result of this
analysis; the per-report folders are detail underneath it.

Usage
-----
    python3 agreement_analysis.py [--manifest out/out_coding/manifest.csv] \\
        [--outdir out/agreement] [--n-perm 3000] [--seed 0]

manifest.csv columns: code,report_id,human_csv,ai_csv (see
out/out_coding/manifest.csv for the 3 reports this project currently covers).
Relative paths, both the defaults and those inside the manifest, are resolved
from the repository root, where this script lives.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

REPO_ROOT = Path(__file__).resolve().parent

ELEMENTS = ['element_community', 'element_donor', 'element_activity', 'element_board', 'element_grantee']
LOGICS = ['PS', 'DS', 'PPG', 'SC']
LOGIC_NAMES = {'PS': 'Philanthropic Stewardship', 'DS': 'Donor Services',
               'PPG': 'Public-Private Governance', 'SC': 'Social Change'}
DIGIT_TO_LOGIC = {'1': 'PS', '2': 'DS', '3': 'PPG', '4': 'SC'}  # decoding only, not an ordering claim
TIE_BREAK_PRIORITY = ['PS', 'DS', 'PPG', 'SC']  # mechanical tie-break only; see module docstring
N_BINS = 20  # document-position bins for the smoothed profile curves

quad4 = [[0, 1, 4, 9], [1, 0, 1, 4], [4, 1, 0, 1], [9, 4, 1, 0]]
lin4 = [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]


# ======================================================================
# 1-2. Matching, scoring, dominant-logic kappa
# ======================================================================

def build_occurrence_index(df: pd.DataFrame, text_col: str = 'text') -> dict:
    counts = defaultdict(int)
    idx_map = {}
    for i, t in enumerate(df[text_col].astype(str)):
        counts[t] += 1
        idx_map[(t, counts[t])] = i
    return idx_map


def human_applicable_and_ranks(row: pd.Series) -> dict:
    skipped = row.get('human_skip') == 'yes'
    out = {}
    for el in ELEMENTS:
        if skipped:
            out[el] = (None, None, False)
            continue
        applicable = (row.get(f'human_{el}_answer') == 'yes')
        t1 = row.get(f'human_{el}_top1_logic') if applicable else None
        t2 = row.get(f'human_{el}_top2_logic') if applicable else None
        out[el] = (t1 if t1 in LOGICS else None, t2 if t2 in LOGICS else None, applicable)
    return out


def parse_contrastive(val) -> tuple:
    if not val or '/' not in str(val):
        return (None, None)
    parts = str(val).strip().split('/')
    if len(parts) < 2:
        return (None, None)
    return (DIGIT_TO_LOGIC.get(parts[0].strip()), DIGIT_TO_LOGIC.get(parts[1].strip()))


def ai_applicable_and_ranks(row: pd.Series) -> dict:
    out = {}
    for el in ELEMENTS:
        applicable = (row.get(f'{el}_answer') == 'yes')
        t1, t2 = parse_contrastive(row.get(f'{el}_contrastive_logic_answer')) if applicable else (None, None)
        out[el] = (t1, t2, applicable)
    return out


def points_from_ranks(ranks_dict: dict) -> tuple:
    pts = {l: 0 for l in LOGICS}
    n_applicable = n_incomplete = 0
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


def dominant_logic(points: dict, priority=TIE_BREAK_PRIORITY):
    mx = max(points.values())
    winners = [l for l in priority if points[l] == mx]
    return winners[0], len(winners) > 1


def confusion_matrix(human_labels, ai_labels) -> np.ndarray:
    cm = pd.crosstab(pd.Series(human_labels), pd.Series(ai_labels))
    return cm.reindex(index=LOGICS, columns=LOGICS, fill_value=0).values


def compute_kappas(human_labels, ai_labels) -> dict:
    return {
        'kappa_unweighted': float(cohen_kappa_score(human_labels, ai_labels, labels=LOGICS)),
        'kappa_linear': float(cohen_kappa_score(human_labels, ai_labels, labels=LOGICS, weights='linear')),
        'kappa_quadratic': float(cohen_kappa_score(human_labels, ai_labels, labels=LOGICS, weights='quadratic')),
        'raw_agreement': float((pd.Series(human_labels) == pd.Series(ai_labels)).mean()),
    }


def kappa_from_confusion(cm: np.ndarray, weights=None) -> float:
    h, a = [], []
    for i, hv in enumerate(LOGICS):
        for j, av in enumerate(LOGICS):
            n = int(cm[i, j])
            h += [hv] * n
            a += [av] * n
    return float(cohen_kappa_score(h, a, labels=LOGICS, weights=weights))


def match_and_score(human_csv, ai_csv):
    cd = pd.read_csv(human_csv, dtype=str, keep_default_na=False, na_values=['', 'nan', 'NaN'])
    ai = pd.read_csv(ai_csv, dtype=str, keep_default_na=False, na_values=['', 'nan', 'NaN'], low_memory=False)

    ai_idx = build_occurrence_index(ai)
    matches = []
    used_ai = set()
    cd_occurrence = defaultdict(int)
    for i, t in enumerate(cd['text'].astype(str)):
        cd_occurrence[t] += 1
        key = (t, cd_occurrence[t])
        if key in ai_idx:
            matches.append((i, ai_idx[key], t))
            used_ai.add(ai_idx[key])

    records = []
    for cd_i, ai_i, text in matches:
        cd_row, ai_row = cd.iloc[cd_i], ai.iloc[ai_i]
        cd_pts, cd_napp, cd_incomplete = points_from_ranks(human_applicable_and_ranks(cd_row))
        ai_pts, ai_napp, ai_incomplete = points_from_ranks(ai_applicable_and_ranks(ai_row))
        rec = {
            'chunk_id': int(cd_row.get('chunk_id')), 'text': text,
            'human_n_applicable_elements': cd_napp, 'human_n_incomplete_rankings': cd_incomplete,
            'ai_n_applicable_elements': ai_napp, 'ai_n_incomplete_rankings': ai_incomplete,
            'coded_by_both': (cd_napp >= 1) and (ai_napp >= 1),
        }
        for l in LOGICS:
            rec[f'human_points_{l}'] = cd_pts[l]
            rec[f'ai_points_{l}'] = ai_pts[l]
        records.append(rec)
    chunk_df = pd.DataFrame(records)

    summary = {
        'n_human_rows': len(cd), 'n_ai_rows': len(ai), 'n_matched_chunks': len(matches),
        'n_unmatched_human': len(cd) - len(matches), 'n_unmatched_ai': len(ai) - len(used_ai),
    }
    return chunk_df, summary


def chunk_kappa(chunk_df: pd.DataFrame) -> tuple:
    included = chunk_df[chunk_df['coded_by_both']].copy()
    rows = []
    for _, r in included.iterrows():
        h_pts = {l: r[f'human_points_{l}'] for l in LOGICS}
        a_pts = {l: r[f'ai_points_{l}'] for l in LOGICS}
        h_label, h_tie = dominant_logic(h_pts)
        a_label, a_tie = dominant_logic(a_pts)
        rows.append({'chunk_id': int(r['chunk_id']), 'human_dominant': h_label, 'human_tie': bool(h_tie),
                     'ai_dominant': a_label, 'ai_tie': bool(a_tie), 'match': h_label == a_label})
    dom_df = pd.DataFrame(rows)

    def block(df):
        cm = confusion_matrix(df['human_dominant'], df['ai_dominant'])
        return {'n': int(len(df)), 'confusion_matrix': cm.tolist(), 'labels_order': LOGICS,
                **compute_kappas(df['human_dominant'].tolist(), df['ai_dominant'].tolist())}

    untied = dom_df[~(dom_df['human_tie'] | dom_df['ai_tie'])]
    ksum = {
        'n_human_ties': int(dom_df['human_tie'].sum()), 'n_ai_ties': int(dom_df['ai_tie'].sum()),
        'tie_break_rule': f"Mechanical only — fixed priority {' > '.join(TIE_BREAK_PRIORITY)} (no theoretical ordering implied).",
        'full_sample': block(dom_df), 'tie_excluded_sensitivity': block(untied),
    }
    return dom_df, ksum


# ======================================================================
# 3. Flexible chunk-vicinity agreement (permutation-based)
# ======================================================================

def top2_set(points: dict) -> set:
    ordered = sorted(LOGICS, key=lambda l: -points[l])
    s = {ordered[0]} if points[ordered[0]] > 0 else set()
    if points[ordered[1]] > 0:
        s.add(ordered[1])
    return s


def flexible_agreement(chunk_df: pd.DataFrame, dom_df: pd.DataFrame, n_perm: int, seed: int) -> dict:
    merged = chunk_df[chunk_df['coded_by_both']].merge(dom_df, on='chunk_id', how='inner').reset_index(drop=True)
    n = len(merged)
    h_pts_list = [{l: int(r[f'human_points_{l}']) for l in LOGICS} for _, r in merged.iterrows()]
    a_pts_list = [{l: int(r[f'ai_points_{l}']) for l in LOGICS} for _, r in merged.iterrows()]
    h_dom = merged['human_dominant'].tolist()
    a_dom = merged['ai_dominant'].tolist()

    kappa_standard = float(cohen_kappa_score(h_dom, a_dom, labels=LOGICS))
    raw_strict = float(np.mean([h == a for h, a in zip(h_dom, a_dom)]))

    h_top2 = [top2_set(p) for p in h_pts_list]
    a_top2 = [top2_set(p) for p in a_pts_list]
    flex_flags = [h_dom[i] == a_dom[i] or a_dom[i] in h_top2[i] or h_dom[i] in a_top2[i] for i in range(n)]
    po_flex = float(np.mean(flex_flags))

    rng = np.random.default_rng(seed)
    h_dom_arr, a_dom_arr = np.array(h_dom), np.array(a_dom)
    perm_flex = np.empty(n_perm)
    perm_strict = np.empty(n_perm)
    for p in range(n_perm):
        perm = rng.permutation(n)
        strict_hits = flex_hits = 0
        for i in range(n):
            j = perm[i]
            if h_dom_arr[i] == a_dom_arr[j]:
                strict_hits += 1
                flex_hits += 1
            elif a_dom_arr[j] in h_top2[i] or h_dom_arr[i] in a_top2[j]:
                flex_hits += 1
        perm_strict[p] = strict_hits / n
        perm_flex[p] = flex_hits / n

    pe_flex, pe_strict = float(perm_flex.mean()), float(perm_strict.mean())
    chunk_detail = [{'chunk_id': int(merged.loc[i, 'chunk_id']), 'strict_match': bool(h_dom[i] == a_dom[i]),
                      'flexible_match': bool(flex_flags[i]),
                      'flexible_only': bool(flex_flags[i] and h_dom[i] != a_dom[i])} for i in range(n)]

    return {
        'n': n, 'n_permutations': n_perm, 'seed': seed,
        'standard_kappa': {'kappa': kappa_standard, 'raw_agreement': raw_strict},
        'flexible_agreement': {
            'observed_rate': po_flex, 'permutation_chance_rate': pe_flex,
            'kappa_flexible': (po_flex - pe_flex) / (1 - pe_flex) if pe_flex < 1 else float('nan'),
            'n_flexible_only_chunks': int(sum(c['flexible_only'] for c in chunk_detail)),
        },
        'permutation_sanity_check': {
            'strict_rate_permutation_chance': pe_strict,
            'kappa_strict_via_permutation': (raw_strict - pe_strict) / (1 - pe_strict) if pe_strict < 1 else float('nan'),
            'kappa_strict_closed_form': kappa_standard,
        },
        'chunk_detail': chunk_detail,
        '_h_top2': h_top2, '_a_top2': a_top2,  # kept for pooling; stripped before writing to disk
    }


# ======================================================================
# 4. Report-level profile (every coded row, independent of chunk matching)
# ======================================================================

def score_every_row(df: pd.DataFrame, ranks_fn) -> pd.DataFrame:
    recs = []
    n = len(df)
    for i in range(n):
        pts, _, _ = points_from_ranks(ranks_fn(df.iloc[i]))
        total = sum(pts.values())
        rec = {'position': (i / (n - 1)) if n > 1 else 0.0, 'total_points': total}
        for l in LOGICS:
            rec[f'points_{l}'] = pts[l]
            rec[f'prop_{l}'] = (pts[l] / total) if total > 0 else np.nan
        recs.append(rec)
    return pd.DataFrame(recs)


def profile_distance(p: dict, q: dict) -> dict:
    pv, qv = np.array([p[l] for l in LOGICS]), np.array([q[l] for l in LOGICS])
    tvd = float(0.5 * np.abs(pv - qv).sum())
    denom = np.linalg.norm(pv) * np.linalg.norm(qv)
    return {'total_variation_distance': tvd, 'cosine_similarity': float(np.dot(pv, qv) / denom) if denom else float('nan')}


def binned_prevalence(scored: pd.DataFrame, n_bins: int = N_BINS) -> dict:
    edges = np.linspace(0, 1, n_bins + 1)
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()
    out = {l: [] for l in LOGICS}
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (scored['position'] >= lo) & ((scored['position'] < hi) if b < n_bins - 1 else (scored['position'] <= hi))
        sub = scored[mask]
        for l in LOGICS:
            vals = sub[f'prop_{l}'].dropna()
            out[l].append(float(vals.mean()) if len(vals) else None)
    return {'bin_centers': centers, 'prevalence': out}


def report_level_profile(human_csv, ai_csv, report_id):
    cd = pd.read_csv(human_csv, dtype=str, keep_default_na=False, na_values=['', 'nan', 'NaN'])
    ai = pd.read_csv(ai_csv, dtype=str, keep_default_na=False, na_values=['', 'nan', 'NaN'], low_memory=False)
    h_scored = score_every_row(cd, human_applicable_and_ranks)
    a_scored = score_every_row(ai, ai_applicable_and_ranks)

    h_pts = {l: int(h_scored[f'points_{l}'].sum()) for l in LOGICS}
    a_pts = {l: int(a_scored[f'points_{l}'].sum()) for l in LOGICS}
    h_tot, a_tot = sum(h_pts.values()), sum(a_pts.values())
    h_prop = {l: (h_pts[l] / h_tot if h_tot else 0.0) for l in LOGICS}
    a_prop = {l: (a_pts[l] / a_tot if a_tot else 0.0) for l in LOGICS}

    return {
        'report_id': report_id, 'n_human_rows_total': len(cd), 'n_ai_rows_total': len(ai),
        'n_human_rows_coded': int((h_scored['total_points'] > 0).sum()),
        'n_ai_rows_coded': int((a_scored['total_points'] > 0).sum()),
        'human_points': h_pts, 'ai_points': a_pts, 'human_proportions': h_prop, 'ai_proportions': a_prop,
        'profile_distance': profile_distance(h_prop, a_prop),
        'human_position_curve': binned_prevalence(h_scored), 'ai_position_curve': binned_prevalence(a_scored),
    }


# ======================================================================
# Per-report bundle entry (what the xlsx/dashboard builders consume)
# ======================================================================

def second_highest(points: dict, winner: str) -> int:
    return max(v for l, v in points.items() if l != winner)


def build_report_bundle_entry(report_id, chunk_df, summary, ksum, dom_df, flex, profile):
    merged = chunk_df[chunk_df['coded_by_both']].merge(dom_df, on='chunk_id', how='inner')
    flex_by_chunk = {c['chunk_id']: c for c in flex['chunk_detail']}

    chunks = []
    for _, r in merged.iterrows():
        h_pts = {l: int(r[f'human_points_{l}']) for l in LOGICS}
        a_pts = {l: int(r[f'ai_points_{l}']) for l in LOGICS}
        h_tot, a_tot = sum(h_pts.values()), sum(a_pts.values())
        fdet = flex_by_chunk.get(int(r['chunk_id']), {})
        chunks.append({
            'id': int(r['chunk_id']), 'text': r['text'],
            'h': [round(h_pts[l] / h_tot, 4) if h_tot else 0.0 for l in LOGICS],
            'a': [round(a_pts[l] / a_tot, 4) if a_tot else 0.0 for l in LOGICS],
            'hd': r['human_dominant'], 'ad': r['ai_dominant'],
            'ht': bool(r['human_tie']), 'at': bool(r['ai_tie']), 'agree': bool(r['match']),
            'hm': int(h_pts[r['human_dominant']] - second_highest(h_pts, r['human_dominant'])),
            'am': int(a_pts[r['ai_dominant']] - second_highest(a_pts, r['ai_dominant'])),
            'flexOnly': bool(fdet.get('flexible_only', False)),
        })

    doc_h_pts = {l: int(chunk_df.loc[chunk_df['coded_by_both'], f'human_points_{l}'].sum()) for l in LOGICS}
    doc_a_pts = {l: int(chunk_df.loc[chunk_df['coded_by_both'], f'ai_points_{l}'].sum()) for l in LOGICS}
    doc_h_tot, doc_a_tot = sum(doc_h_pts.values()), sum(doc_a_pts.values())

    cm = np.array(ksum['full_sample']['confusion_matrix'])
    return {
        'report_id': report_id,
        'n_human_rows': summary['n_human_rows'], 'n_ai_rows': summary['n_ai_rows'],
        'n_matched': summary['n_matched_chunks'], 'n_included': int(cm.sum()),
        'doc_human_prop': [round(doc_h_pts[l] / doc_h_tot, 4) if doc_h_tot else 0.0 for l in LOGICS],
        'doc_ai_prop': [round(doc_a_pts[l] / doc_a_tot, 4) if doc_a_tot else 0.0 for l in LOGICS],
        'confusion': cm.tolist(),
        'kappa_standard': ksum['full_sample']['kappa_unweighted'],
        'kappa_quad': ksum['full_sample']['kappa_quadratic'],
        'kappa_lin': ksum['full_sample']['kappa_linear'],
        'kappa_unw': ksum['full_sample']['kappa_unweighted'],
        'raw_agree': ksum['full_sample']['raw_agreement'],
        'n_human_ties': ksum['n_human_ties'], 'n_ai_ties': ksum['n_ai_ties'],
        'kappa_standard_sens': ksum['tie_excluded_sensitivity']['kappa_unweighted'],
        'kappa_quad_sens': ksum['tie_excluded_sensitivity']['kappa_quadratic'],
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
        'profile': {
            'n_human_rows_total': profile['n_human_rows_total'], 'n_ai_rows_total': profile['n_ai_rows_total'],
            'n_human_rows_coded': profile['n_human_rows_coded'], 'n_ai_rows_coded': profile['n_ai_rows_coded'],
            'human_prop': [round(profile['human_proportions'][l], 4) for l in LOGICS],
            'ai_prop': [round(profile['ai_proportions'][l], 4) for l in LOGICS],
            'tvd': round(profile['profile_distance']['total_variation_distance'], 4),
            'cosine': round(profile['profile_distance']['cosine_similarity'], 4),
            'human_curve': profile['human_position_curve'], 'ai_curve': profile['ai_position_curve'],
        },
    }


# ======================================================================
# Pooling across reports
# ======================================================================

def pool_everything(entries: dict, raw: dict, n_perm: int, seed: int) -> dict:
    """entries: {code: bundle_entry}. raw: {code: {'chunk_df','dom_df','profile'}}."""
    cm_sum = np.zeros((4, 4), dtype=int)
    all_chunks = []
    n_human_total = n_ai_total = n_matched = n_human_ties = n_ai_ties = 0
    doc_h_pts = {l: 0 for l in LOGICS}
    doc_a_pts = {l: 0 for l in LOGICS}
    prof_h_pts = {l: 0 for l in LOGICS}
    prof_a_pts = {l: 0 for l in LOGICS}
    n_h_coded = n_a_coded = n_h_total = n_a_total = 0
    all_untied_h, all_untied_a = [], []  # for pooled tie-excluded sensitivity

    pooled_curve_raw = {'human_position_curve': {l: [] for l in LOGICS}, 'ai_position_curve': {l: [] for l in LOGICS}}
    bin_centers = None

    for code, entry in entries.items():
        cm_sum += np.array(entry['confusion'])
        n_human_total += entry['n_human_rows']; n_ai_total += entry['n_ai_rows']; n_matched += entry['n_matched']
        n_human_ties += entry['n_human_ties']; n_ai_ties += entry['n_ai_ties']
        for c in entry['chunks']:
            c2 = dict(c); c2['r'] = code
            all_chunks.append(c2)

        chunk_df = raw[code]['chunk_df']
        for l in LOGICS:
            doc_h_pts[l] += int(chunk_df.loc[chunk_df['coded_by_both'], f'human_points_{l}'].sum())
            doc_a_pts[l] += int(chunk_df.loc[chunk_df['coded_by_both'], f'ai_points_{l}'].sum())

        prof = raw[code]['profile']
        for l in LOGICS:
            prof_h_pts[l] += prof['human_points'][l]
            prof_a_pts[l] += prof['ai_points'][l]
        n_h_coded += prof['n_human_rows_coded']; n_a_coded += prof['n_ai_rows_coded']
        n_h_total += prof['n_human_rows_total']; n_a_total += prof['n_ai_rows_total']
        bin_centers = prof['human_position_curve']['bin_centers']
        for l in LOGICS:
            pooled_curve_raw['human_position_curve'][l].append(prof['human_position_curve']['prevalence'][l])
            pooled_curve_raw['ai_position_curve'][l].append(prof['ai_position_curve']['prevalence'][l])

        dom_df = raw[code]['dom_df']
        untied = dom_df[~(dom_df['human_tie'] | dom_df['ai_tie'])]
        all_untied_h += untied['human_dominant'].tolist()
        all_untied_a += untied['ai_dominant'].tolist()

    doc_h_tot, doc_a_tot = sum(doc_h_pts.values()), sum(doc_a_pts.values())
    prof_h_tot, prof_a_tot = sum(prof_h_pts.values()), sum(prof_a_pts.values())
    prof_h_prop = {l: (prof_h_pts[l] / prof_h_tot if prof_h_tot else 0.0) for l in LOGICS}
    prof_a_prop = {l: (prof_a_pts[l] / prof_a_tot if prof_a_tot else 0.0) for l in LOGICS}
    dist = profile_distance(prof_h_prop, prof_a_prop)

    def pooled_curve(key):
        out = {}
        for l in LOGICS:
            series = pooled_curve_raw[key][l]
            out[l] = [round(float(np.mean(vals)), 4) if (vals := [s[b] for s in series if s[b] is not None]) else None
                      for b in range(len(bin_centers))]
        return {'bin_centers': bin_centers, 'prevalence': out}

    # Pooled flexible-agreement permutation: combine every report's coded-by-both
    # chunks into one pool and permute across the full pool (same philosophy as
    # the confusion-matrix sum: pool the data, don't average the statistics).
    h_pts_list, a_pts_list, h_dom_list, a_dom_list = [], [], [], []
    for code in entries:
        chunk_df, dom_df = raw[code]['chunk_df'], raw[code]['dom_df']
        merged = chunk_df[chunk_df['coded_by_both']].merge(dom_df, on='chunk_id', how='inner')
        for _, r in merged.iterrows():
            h_pts_list.append({l: int(r[f'human_points_{l}']) for l in LOGICS})
            a_pts_list.append({l: int(r[f'ai_points_{l}']) for l in LOGICS})
            h_dom_list.append(r['human_dominant']); a_dom_list.append(r['ai_dominant'])
    n = len(h_dom_list)
    kappa_standard_pooled = float(cohen_kappa_score(h_dom_list, a_dom_list, labels=LOGICS))
    raw_strict_pooled = float(np.mean([h == a for h, a in zip(h_dom_list, a_dom_list)]))
    h_top2 = [top2_set(p) for p in h_pts_list]
    a_top2 = [top2_set(p) for p in a_pts_list]
    flex_flags = [h_dom_list[i] == a_dom_list[i] or a_dom_list[i] in h_top2[i] or h_dom_list[i] in a_top2[i] for i in range(n)]
    po_flex = float(np.mean(flex_flags))

    rng = np.random.default_rng(seed)
    h_dom_arr, a_dom_arr = np.array(h_dom_list), np.array(a_dom_list)
    perm_flex, perm_strict = np.empty(n_perm), np.empty(n_perm)
    for p in range(n_perm):
        perm = rng.permutation(n)
        strict_hits = flex_hits = 0
        for i in range(n):
            j = perm[i]
            if h_dom_arr[i] == a_dom_arr[j]:
                strict_hits += 1; flex_hits += 1
            elif a_dom_arr[j] in h_top2[i] or h_dom_arr[i] in a_top2[j]:
                flex_hits += 1
        perm_strict[p] = strict_hits / n
        perm_flex[p] = flex_hits / n
    pe_flex, pe_strict = float(perm_flex.mean()), float(perm_strict.mean())
    n_flex_only = int(sum(1 for i in range(n) if flex_flags[i] and h_dom_list[i] != a_dom_list[i]))

    sens_kappas = compute_kappas(all_untied_h, all_untied_a) if all_untied_h else {'kappa_unweighted': float('nan'), 'kappa_quadratic': float('nan')}

    return {
        'n_reports': len(entries), 'report_ids': [e['report_id'] for e in entries.values()],
        'confusion': cm_sum.tolist(),
        'kappa_standard': kappa_from_confusion(cm_sum, None), 'kappa_quad': kappa_from_confusion(cm_sum, 'quadratic'),
        'kappa_lin': kappa_from_confusion(cm_sum, 'linear'), 'kappa_unw': kappa_from_confusion(cm_sum, None),
        'raw_agree': float(np.trace(cm_sum) / cm_sum.sum()),
        'n_included': int(cm_sum.sum()), 'n_human_ties': n_human_ties, 'n_ai_ties': n_ai_ties,
        'doc_human_prop': [round(doc_h_pts[l] / doc_h_tot, 4) if doc_h_tot else 0.0 for l in LOGICS],
        'doc_ai_prop': [round(doc_a_pts[l] / doc_a_tot, 4) if doc_a_tot else 0.0 for l in LOGICS],
        'kappa_standard_sens': sens_kappas['kappa_unweighted'], 'kappa_quad_sens': sens_kappas['kappa_quadratic'],
        'n_sens': len(all_untied_h),
        'chunks': all_chunks,
        'flex': {
            'n': n, 'n_permutations': n_perm, 'observed_rate': po_flex, 'chance_rate': pe_flex,
            'kappa_flexible': (po_flex - pe_flex) / (1 - pe_flex) if pe_flex < 1 else float('nan'),
            'n_flexible_only': n_flex_only,
            'kappa_standard_perm_check': (raw_strict_pooled - pe_strict) / (1 - pe_strict) if pe_strict < 1 else float('nan'),
        },
        'profile': {
            'n_human_rows_total': n_h_total, 'n_ai_rows_total': n_a_total,
            'n_human_rows_coded': n_h_coded, 'n_ai_rows_coded': n_a_coded,
            'human_prop': [round(prof_h_prop[l], 4) for l in LOGICS], 'ai_prop': [round(prof_a_prop[l], 4) for l in LOGICS],
            'tvd': round(dist['total_variation_distance'], 4), 'cosine': round(dist['cosine_similarity'], 4),
            'human_curve': pooled_curve('human_position_curve'), 'ai_curve': pooled_curve('ai_position_curve'),
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', default='out/out_coding/manifest.csv',
                    help='CSV with columns: code,report_id,human_csv,ai_csv')
    ap.add_argument('--outdir', default='out/agreement')
    ap.add_argument('--n-perm', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    outdir = REPO_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    with open(REPO_ROOT / args.manifest, newline='') as f:
        manifest = list(csv.DictReader(f))
    for row in manifest:
        row['human_csv'] = REPO_ROOT / row['human_csv']
        row['ai_csv'] = REPO_ROOT / row['ai_csv']

    entries, raw = {}, {}
    for row in manifest:
        code, report_id = row['code'], row['report_id']
        rdir = outdir / report_id
        rdir.mkdir(parents=True, exist_ok=True)
        print(f"=== {code}: {report_id} ===")

        chunk_df, summary = match_and_score(row['human_csv'], row['ai_csv'])
        chunk_df.to_csv(rdir / 'chunk_level.csv', index=False)
        print(f"  matched {summary['n_matched_chunks']} chunks "
              f"(human rows {summary['n_human_rows']}, AI rows {summary['n_ai_rows']})")

        dom_df, ksum = chunk_kappa(chunk_df)
        dom_df.to_csv(rdir / 'chunk_dominant.csv', index=False)
        with open(rdir / 'summary_chunk_kappa.json', 'w') as f:
            json.dump(ksum, f, indent=2)
        print(f"  N coded by both = {ksum['full_sample']['n']}   "
              f"standard kappa = {ksum['full_sample']['kappa_unweighted']:.4f}")

        flex = flexible_agreement(chunk_df, dom_df, args.n_perm, args.seed)
        flex_to_write = {k: v for k, v in flex.items() if not k.startswith('_')}
        with open(rdir / 'flexible_agreement.json', 'w') as f:
            json.dump(flex_to_write, f, indent=2)
        print(f"  flexible match rate = {flex['flexible_agreement']['observed_rate']:.4f}   "
              f"kappa_flexible = {flex['flexible_agreement']['kappa_flexible']:.4f}")

        profile = report_level_profile(row['human_csv'], row['ai_csv'], report_id)
        with open(rdir / 'report_level_profile.json', 'w') as f:
            json.dump(profile, f, indent=2)

        with open(rdir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        entries[code] = build_report_bundle_entry(report_id, chunk_df, summary, ksum, dom_df, flex, profile)
        raw[code] = {'chunk_df': chunk_df, 'dom_df': dom_df, 'profile': profile}

    print("=== Pooling across all reports ===")
    pooled = pool_everything(entries, raw, args.n_perm, args.seed)
    print(f"  pooled N = {pooled['n_included']}   standard kappa = {pooled['kappa_standard']:.4f}   "
          f"kappa_flexible = {pooled['flex']['kappa_flexible']:.4f}")

    bundle = {'reports': entries, 'pooled': pooled}
    bundle_path = outdir / 'agreement_bundle.json'
    with open(bundle_path, 'w') as f:
        json.dump(bundle, f, separators=(',', ':'), ensure_ascii=False)
    print(f"\nWrote {bundle_path} ({bundle_path.stat().st_size} bytes)")


if __name__ == '__main__':
    main()
