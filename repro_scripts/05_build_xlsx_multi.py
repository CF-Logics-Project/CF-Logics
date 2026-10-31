#!/usr/bin/env python3
"""
05_build_xlsx_multi.py

Step 5 (multi-report): builds institutional_logics_agreement.xlsx covering
several reports at once. For each report it reproduces the same 4 sheets
as 03_build_xlsx.py (Chunk-level, Doc Aggregate, Chunk Dominant, Agreement),
under a short "R1"/"R2"/... prefix (Excel sheet names cap at 31 chars), plus:

  - Report Key        maps R1/R2/... back to the full report_id
  - Overview          per-report kappa table + pooled kappa headline, chart
  - Pooled Agreement  confusion matrix pooled by SUMMING each report's own
                       Agreement sheet matrix cells (live formulas, not a
                       hardcoded pooled count), then the same
                       quadratic/linear/unweighted kappa formulas as a
                       single-report sheet

Pooling via live cross-sheet SUM formulas (rather than a hardcoded pooled
count) means the workbook still recalculates correctly if a report's
underlying chunk data is edited.

Usage
-----
    python3 05_build_xlsx_multi.py \
        --report 363466802_2021=multi_report/363466802_2021 \
        --report 382023395_2022=multi_report/382023395_2022 \
        --report 25-1650208_2021=multi_report/25-1650208_2021 \
        --pooled-summary-json multi_report/pooled_summary.json \
        --outfile multi_report/institutional_logics_agreement.xlsx
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

LOGICS = ['PS', 'DS', 'PPG', 'SC']
LOGIC_NAMES = {'PS': 'Philanthropic Stewardship', 'DS': 'Donor Services',
               'PPG': 'Public-Private Governance', 'SC': 'Social Change'}
FONT = 'Arial'

quad4 = [[0, 1, 4, 9], [1, 0, 1, 4], [4, 1, 0, 1], [9, 4, 1, 0]]
lin4 = [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]
unw4 = [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--report', action='append', required=True, help='report_id=path_to_report_outdir (repeatable, in order)')
    ap.add_argument('--pooled-summary-json', required=True)
    ap.add_argument('--flex-pooled-json', required=True, help='flexible_agreement_pooled.json from 09_pool_flexible_agreement.py')
    ap.add_argument('--outfile', default='institutional_logics_agreement.xlsx')
    args = ap.parse_args()

    report_specs = []
    for spec in args.report:
        rid, path = spec.split('=', 1)
        report_specs.append((rid, Path(path)))
    with open(args.pooled_summary_json) as f:
        pooled = json.load(f)
    with open(args.flex_pooled_json) as f:
        flex_pooled = json.load(f)

    wb = Workbook()

    header_fill = PatternFill('solid', fgColor='2F3B2A')
    header_font = Font(name=FONT, bold=True, color='FFFFFF', size=11)
    title_font = Font(name=FONT, bold=True, size=16)
    subtitle_font = Font(name=FONT, italic=True, size=10, color='666666')
    label_font = Font(name=FONT, bold=True, size=11)
    normal_font = Font(name=FONT, size=10.5)
    note_font = Font(name=FONT, italic=True, size=9, color='777777')
    thin = Side(style='thin', color='CCCCCC')
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    pct_fmt = '0.0%'

    def style_header_row(ws, row, ncols, start_col=1):
        for c in range(start_col, start_col + ncols):
            cell = ws.cell(row=row, column=c)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
            cell.border = border

    def autosize(ws, widths):
        for i, w in enumerate(widths, start=1):
            ws.column_dimensions[get_column_letter(i)].width = w

    def build_confusion_block(ws, label, top_row, hcol, acol, obs_extra_criteria=None):
        ws.cell(row=top_row, column=2, value=label).font = label_font
        hdr = top_row + 1
        data_r0 = hdr + 1
        ws.cell(row=hdr, column=3, value='AI: PS')
        ws.cell(row=hdr, column=4, value='AI: DS')
        ws.cell(row=hdr, column=5, value='AI: PPG')
        ws.cell(row=hdr, column=6, value='AI: SC')
        ws.cell(row=hdr, column=7, value='Row total')
        style_header_row(ws, hdr, 5, start_col=3)
        for i, hv in enumerate(LOGICS):
            r = data_r0 + i
            ws.cell(row=r, column=2, value=f'Human: {hv}').font = normal_font
            for j, av in enumerate(LOGICS):
                c = 3 + j
                crit = f'{hcol},"{hv}",{acol},"{av}"'
                if obs_extra_criteria:
                    crit += f',{obs_extra_criteria}'
                ws.cell(row=r, column=c, value=f"=COUNTIFS({crit})")
            ws.cell(row=r, column=7, value=f"=SUM(C{r}:F{r})")
        tot_r = data_r0 + 4
        ws.cell(row=tot_r, column=2, value='Col total').font = label_font
        for j in range(4):
            col_l = get_column_letter(3 + j)
            ws.cell(row=tot_r, column=3 + j, value=f"=SUM({col_l}{data_r0}:{col_l}{data_r0+3})")
        ws.cell(row=tot_r, column=7, value=f"=SUM(G{data_r0}:G{data_r0+3})")
        obs_rng = f"C{data_r0}:F{data_r0+3}"
        diag_cells = '+'.join(f"{get_column_letter(3+i)}{data_r0+i}" for i in range(4))

        exp_hdr = tot_r + 2
        exp_r0 = exp_hdr + 1
        ws.cell(row=exp_hdr, column=2, value='Expected under independence').font = normal_font
        for i in range(4):
            r = exp_r0 + i
            for j in range(4):
                col_l = get_column_letter(3 + j)
                ws.cell(row=r, column=3 + j, value=f"=$G{data_r0+i}*{col_l}${tot_r}/$G${tot_r}")
        exp_rng = f"C{exp_r0}:F{exp_r0+3}"

        raw_r = exp_r0 + 5
        ws.cell(row=raw_r, column=2, value='Raw agreement (% chunks, matching dominant logic)').font = normal_font
        ws.cell(row=raw_r, column=7, value=f"=({diag_cells})/G{tot_r}").number_format = pct_fmt

        def weight_subblock(top, title, wmat, bold=False):
            ws.cell(row=top, column=2, value=title).font = label_font
            w_r0 = top + 1
            for i in range(4):
                for j in range(4):
                    ws.cell(row=w_r0 + i, column=3 + j, value=wmat[i][j])
            w_rng = f"C{w_r0}:F{w_r0+3}"
            kappa_r = w_r0 + 5
            ws.cell(row=kappa_r, column=2, value='Kappa').font = label_font
            kcell = ws.cell(row=kappa_r, column=7, value=f"=1-SUMPRODUCT({w_rng},{obs_rng})/SUMPRODUCT({w_rng},{exp_rng})")
            kcell.font = Font(name=FONT, bold=True, size=12) if bold else normal_font
            return kappa_r, kappa_r + 2

        unw_top = raw_r + 2
        ws.cell(row=unw_top - 1, column=2,
                value='PS/DS/PPG/SC are NOMINAL categories (no established ordering), so every off-diagonal mismatch is weighted equally.').font = note_font
        kappa_r_unw, next0 = weight_subblock(unw_top, 'PRIMARY — standard (nominal, unweighted) Cohen’s kappa', unw4, bold=True)
        kappa_r_quad, next1 = weight_subblock(next0 + 1, 'RETIRED — quadratic ordinal weights (i−j)² — assumed PS<DS<PPG<SC, no theoretical basis; kept only for comparison to the earlier version of this analysis', quad4)
        kappa_r_lin, next2 = weight_subblock(next1 + 1, 'RETIRED — linear ordinal weights |i−j| — same caveat as above', lin4)
        return {'kappa_quad': kappa_r_quad, 'kappa_lin': kappa_r_lin, 'kappa_unw': kappa_r_unw,
                'raw': raw_r, 'next_free': next2, 'data_r0': data_r0, 'tot_r': tot_r}

    # =====================================================================
    # Per-report sheets
    # =====================================================================
    short_codes = [f'R{i+1}' for i in range(len(report_specs))]
    report_info = []  # (rid, code, summary, ksum)

    for (rid, path), code in zip(report_specs, short_codes):
        chunk_df = pd.read_csv(path / 'chunk_level.csv')
        chunk_dom = pd.read_csv(path / 'chunk_dominant.csv')
        with open(path / 'summary.json') as f:
            summary = json.load(f)
        with open(path / 'summary_chunk_kappa.json') as f:
            ksum = json.load(f)
        with open(path / 'flexible_agreement.json') as f:
            flex = json.load(f)
        with open(path / 'report_level_profile.json') as f:
            profile = json.load(f)
        report_info.append((rid, code, summary, ksum, flex, profile))

        # --- Chunk-level ---
        cl = wb.create_sheet(f'{code} Chunk-level')
        cols = ['chunk_id_cd', 'text', 'human_n_applicable_elements', 'ai_n_applicable_elements',
                'human_n_incomplete_rankings', 'ai_n_incomplete_rankings', 'coded_by_both']
        cols += [f'human_points_{l}' for l in LOGICS] + [f'ai_points_{l}' for l in LOGICS]
        cols += [f'human_prop_{l}' for l in LOGICS] + [f'ai_prop_{l}' for l in LOGICS]
        headers = (['Chunk ID', 'Text', 'Human # applic.', 'AI # applic.', 'Human # incompl.', 'AI # incompl.', 'Coded by both?']
                   + [f'Human pts {l}' for l in LOGICS] + [f'AI pts {l}' for l in LOGICS]
                   + [f'Human % {l}' for l in LOGICS] + [f'AI % {l}' for l in LOGICS])
        cl.append(headers)
        style_header_row(cl, 1, len(headers))
        cl.freeze_panes = 'C2'
        col_idx = {name: i + 1 for i, name in enumerate(cols)}
        for _, r in chunk_df.iterrows():
            row_vals = []
            for c in cols:
                if c == 'coded_by_both':
                    row_vals.append(bool(r[c]))
                elif c.startswith('human_prop') or c.startswith('ai_prop'):
                    row_vals.append(None)
                else:
                    row_vals.append(r[c])
            cl.append(row_vals)
        n_data_rows = len(chunk_df)
        for i in range(2, n_data_rows + 2):
            hp = [f"{get_column_letter(col_idx[f'human_points_{l}'])}{i}" for l in LOGICS]
            ap = [f"{get_column_letter(col_idx[f'ai_points_{l}'])}{i}" for l in LOGICS]
            for l in LOGICS:
                hcell = cl.cell(row=i, column=col_idx[f'human_prop_{l}'])
                hcell.value = f"=IF(SUM({hp[0]}:{hp[-1]})=0,\"\",{get_column_letter(col_idx[f'human_points_{l}'])}{i}/SUM({hp[0]}:{hp[-1]}))"
                hcell.number_format = pct_fmt
                acell = cl.cell(row=i, column=col_idx[f'ai_prop_{l}'])
                acell.value = f"=IF(SUM({ap[0]}:{ap[-1]})=0,\"\",{get_column_letter(col_idx[f'ai_points_{l}'])}{i}/SUM({ap[0]}:{ap[-1]}))"
                acell.number_format = pct_fmt
        autosize(cl, [10, 60, 9, 9, 9, 9, 10] + [8] * 16)

        # --- Doc Aggregate ---
        da = wb.create_sheet(f'{code} Doc Aggregate')
        da['B2'] = f'Document-level totals — {rid}'
        da['B2'].font = title_font
        da['B4'] = 'Logic'
        for j, h in enumerate(['Human points', 'Human %', 'AI points', 'AI %'], start=3):
            da.cell(row=4, column=j, value=h)
        style_header_row(da, 4, 5, start_col=2)
        last_row = n_data_rows + 1
        cb_col = get_column_letter(col_idx['coded_by_both'])
        for i, l in enumerate(LOGICS):
            r = 5 + i
            da.cell(row=r, column=2, value=f'{l} — {LOGIC_NAMES[l]}').font = normal_font
            hp_col = get_column_letter(col_idx[f'human_points_{l}'])
            ap_col = get_column_letter(col_idx[f'ai_points_{l}'])
            da.cell(row=r, column=3, value=f"=SUMIFS('{code} Chunk-level'!{hp_col}2:{hp_col}{last_row},'{code} Chunk-level'!{cb_col}2:{cb_col}{last_row},TRUE)")
            da.cell(row=r, column=5, value=f"=SUMIFS('{code} Chunk-level'!{ap_col}2:{ap_col}{last_row},'{code} Chunk-level'!{cb_col}2:{cb_col}{last_row},TRUE)")
        total_row = 9
        da.cell(row=total_row, column=2, value='Total').font = label_font
        da.cell(row=total_row, column=3, value='=SUM(C5:C8)').font = label_font
        da.cell(row=total_row, column=5, value='=SUM(E5:E8)').font = label_font
        for i in range(4):
            r = 5 + i
            da.cell(row=r, column=4, value=f"=IF($C${total_row}=0,\"\",C{r}/$C${total_row})").number_format = pct_fmt
            da.cell(row=r, column=6, value=f"=IF($E${total_row}=0,\"\",E{r}/$E${total_row})").number_format = pct_fmt
        autosize(da, [3, 30, 13, 9, 13, 9])

        # --- Chunk Dominant ---
        cdom = wb.create_sheet(f'{code} Chunk Dominant')
        cdom.append(['Chunk ID', 'Human dominant', 'Human tie?', 'AI dominant', 'AI tie?', 'Match?'])
        style_header_row(cdom, 1, 6)
        for _, r in chunk_dom.iterrows():
            cdom.append([int(r['chunk_id']), r['human_dominant'], bool(r['human_tie']),
                         r['ai_dominant'], bool(r['ai_tie']), bool(r['match'])])
        cdom_last = len(chunk_dom) + 1
        autosize(cdom, [10, 15, 10, 13, 9, 9])
        cdom.freeze_panes = 'A2'

        # --- Agreement ---
        ra = wb.create_sheet(f'{code} Agreement')
        ra['B2'] = f'Chunk-level agreement — {rid}'
        ra['B2'].font = title_font
        ra['B3'] = f'N={ksum["full_sample"]["n"]} chunks coded by both coders. Ties: {ksum["n_human_ties"]} human, {ksum["n_ai_ties"]} AI (broken PS>DS>PPG>SC).'
        ra['B3'].font = subtitle_font

        hcol = f"'{code} Chunk Dominant'!$B$2:$B${cdom_last}"
        acol = f"'{code} Chunk Dominant'!$D$2:$D${cdom_last}"
        htie = f"'{code} Chunk Dominant'!$C$2:$C${cdom_last}"
        atie = f"'{code} Chunk Dominant'!$E$2:$E${cdom_last}"

        blk_a = build_confusion_block(ra, 'A. Full sample', 7, hcol, acol)
        blk_b = build_confusion_block(ra, 'B. Sensitivity — ties excluded', blk_a['next_free'] + 1, hcol, acol,
                                       obs_extra_criteria=f'{htie},FALSE,{atie},FALSE')
        autosize(ra, [3, 36, 11, 11, 11, 11, 12])

        report_info[-1] = (rid, code, summary, ksum, flex, profile, blk_a, blk_b)

    # =====================================================================
    # Report Key
    # =====================================================================
    rk = wb.create_sheet('Report Key')
    rk.append(['Code', 'Report ID', 'Human rows', 'AI rows', 'Matched chunks', 'Coded by both'])
    style_header_row(rk, 1, 6)
    for rid, code, summary, ksum, *_ in report_info:
        rk.append([code, rid, summary['n_human_rows'], summary['n_ai_rows'], summary['n_matched_chunks'], summary['n_included_in_aggregate']])
    autosize(rk, [8, 28, 12, 10, 14, 13])

    # =====================================================================
    # Pooled Agreement (sums each report's own confusion matrix, live)
    # =====================================================================
    pa = wb.create_sheet('Pooled Agreement')
    pa['B2'] = 'Pooled chunk-level agreement — all reports combined'
    pa['B2'].font = title_font
    pa['B3'] = (f'Confusion matrix is the SUM of each report’s own "Agreement" sheet matrix (live cross-sheet formulas), '
                f'not a separately hardcoded pooled count — so it stays correct if any report’s chunk data changes. '
                f'N = {pooled["full_sample"]["n"]} chunks coded by both coders, pooled across {pooled["n_reports"]} reports.')
    pa['B3'].font = subtitle_font
    pa.merge_cells('B3:J3')
    pa.row_dimensions[3].height = 28

    def pooled_block(ws, label, top_row, which):  # which: 'a' or 'b'
        ws.cell(row=top_row, column=2, value=label).font = label_font
        hdr = top_row + 1
        data_r0 = hdr + 1
        ws.cell(row=hdr, column=3, value='AI: PS')
        ws.cell(row=hdr, column=4, value='AI: DS')
        ws.cell(row=hdr, column=5, value='AI: PPG')
        ws.cell(row=hdr, column=6, value='AI: SC')
        ws.cell(row=hdr, column=7, value='Row total')
        style_header_row(ws, hdr, 5, start_col=3)
        for i in range(4):
            r = data_r0 + i
            ws.cell(row=r, column=2, value=f'Human: {LOGICS[i]}').font = normal_font
            for j in range(4):
                c = 3 + j
                col_l = get_column_letter(c)
                src_row = report_info[0][6 if which == 'a' else 7]['data_r0'] + i
                terms = [f"'{code} Agreement'!{col_l}{src_row}" for (rid, code, summary, ksum, flex, profile, blk_a, blk_b) in report_info]
                ws.cell(row=r, column=c, value='=' + '+'.join(terms))
            ws.cell(row=r, column=7, value=f"=SUM(C{r}:F{r})")
        tot_r = data_r0 + 4
        ws.cell(row=tot_r, column=2, value='Col total').font = label_font
        for j in range(4):
            col_l = get_column_letter(3 + j)
            ws.cell(row=tot_r, column=3 + j, value=f"=SUM({col_l}{data_r0}:{col_l}{data_r0+3})")
        ws.cell(row=tot_r, column=7, value=f"=SUM(G{data_r0}:G{data_r0+3})")
        obs_rng = f"C{data_r0}:F{data_r0+3}"
        diag_cells = '+'.join(f"{get_column_letter(3+i)}{data_r0+i}" for i in range(4))

        exp_hdr = tot_r + 2
        exp_r0 = exp_hdr + 1
        ws.cell(row=exp_hdr, column=2, value='Expected under independence').font = normal_font
        for i in range(4):
            r = exp_r0 + i
            for j in range(4):
                col_l = get_column_letter(3 + j)
                ws.cell(row=r, column=3 + j, value=f"=$G{data_r0+i}*{col_l}${tot_r}/$G${tot_r}")
        exp_rng = f"C{exp_r0}:F{exp_r0+3}"

        raw_r = exp_r0 + 5
        ws.cell(row=raw_r, column=2, value='Raw agreement').font = normal_font
        ws.cell(row=raw_r, column=7, value=f"=({diag_cells})/G{tot_r}").number_format = pct_fmt

        def weight_subblock(top, title, wmat, bold=False):
            ws.cell(row=top, column=2, value=title).font = label_font
            w_r0 = top + 1
            for i in range(4):
                for j in range(4):
                    ws.cell(row=w_r0 + i, column=3 + j, value=wmat[i][j])
            w_rng = f"C{w_r0}:F{w_r0+3}"
            kappa_r = w_r0 + 5
            ws.cell(row=kappa_r, column=2, value='Kappa').font = label_font
            kcell = ws.cell(row=kappa_r, column=7, value=f"=1-SUMPRODUCT({w_rng},{obs_rng})/SUMPRODUCT({w_rng},{exp_rng})")
            kcell.font = Font(name=FONT, bold=True, size=12) if bold else normal_font
            return kappa_r, kappa_r + 2

        unw_top = raw_r + 2
        kappa_r_unw, next0 = weight_subblock(unw_top, 'PRIMARY — standard (nominal, unweighted) Cohen’s kappa', unw4, bold=True)
        kappa_r_quad, next1 = weight_subblock(next0 + 1, 'RETIRED — quadratic ordinal weights (no theoretical basis; kept for comparison)', quad4)
        kappa_r_lin, next2 = weight_subblock(next1 + 1, 'RETIRED — linear ordinal weights (same caveat)', lin4)
        return {'kappa_quad': kappa_r_quad, 'kappa_lin': kappa_r_lin, 'kappa_unw': kappa_r_unw, 'raw': raw_r,
                'next_free': next2, 'tot_r': tot_r}

    pooled_blk_a = pooled_block(pa, 'A. Full sample (pooled across all reports)', 5, 'a')
    pooled_blk_b = pooled_block(pa, 'B. Sensitivity — ties excluded (pooled)', pooled_blk_a['next_free'] + 1, 'b')
    pa.cell(row=pooled_blk_b['next_free'] + 1, column=2,
            value='Interpretation guide (Landis & Koch, 1977): <0 poor, 0.00–0.20 slight, 0.21–0.40 fair, 0.41–0.60 moderate, 0.61–0.80 substantial, 0.81–1.00 almost perfect.').font = note_font
    autosize(pa, [3, 36, 11, 11, 11, 11, 12])

    # =====================================================================
    # Flexible Agreement (chunk-vicinity alternative; permutation-based)
    # =====================================================================
    # These statistics cannot be expressed as live Excel formulas: the
    # chance baseline is estimated by Monte Carlo permutation (08/09_*.py),
    # not a closed-form marginal-product formula, so the numbers below are
    # written as values, each with its reproducible source documented in
    # the adjacent note cells (per-report: <code>/flexible_agreement.json,
    # produced by 08_flexible_agreement.py; pooled:
    # flexible_agreement_pooled.json, produced by 09_pool_flexible_agreement.py).
    # Re-running those two scripts with the same --seed reproduces them
    # exactly (the permutation count and seed are recorded in each row).
    fa = wb.create_sheet('Flexible Agreement')
    fa['B2'] = 'Alternative agreement measure — chunk-flexible vicinity (permutation-based)'
    fa['B2'].font = title_font
    fa['B3'] = ('A chunk is a "flexible match" if the two coders’ dominant picks agree, OR either coder’s dominant '
                'pick is in the OTHER coder’s own chunk-level top-2 set (points-derived, per chunk, no global category order assumed). '
                'Because this uses both coders’ per-chunk rankings jointly, there is no closed-form chance-correction formula; chance '
                'is estimated by permutation (randomly re-pairing AI chunks against human chunks many times, holding each side’s own '
                'per-chunk data fixed). kappa_flexible = (observed − permutation chance) / (1 − permutation chance). The sanity-check '
                'columns apply the identical permutation to STRICT matching, which should closely reproduce the closed-form standard kappa.')
    fa['B3'].font = subtitle_font
    fa.merge_cells('B3:L3')
    fa.row_dimensions[3].height = 56

    fa_headers = ['Report', 'N', '# permutations', 'seed', 'Observed flexible rate', 'Permutation chance rate',
                  'kappa_flexible', '# flexible-only chunks', 'Sanity: permutation strict chance',
                  'Sanity: kappa via permutation', 'Sanity: kappa closed-form']
    fa_hdr_row = 6
    for j, h in enumerate(fa_headers, start=2):
        fa.cell(row=fa_hdr_row, column=j, value=h)
    style_header_row(fa, fa_hdr_row, len(fa_headers), start_col=2)
    fa_row = fa_hdr_row + 1
    for rid, code, summary, ksum, flex, profile, blk_a, blk_b in report_info:
        fagree = flex['flexible_agreement']
        sanity = flex['permutation_sanity_check']
        fa.cell(row=fa_row, column=2, value=rid).font = normal_font
        fa.cell(row=fa_row, column=3, value=flex['n'])
        fa.cell(row=fa_row, column=4, value=flex['n_permutations'])
        fa.cell(row=fa_row, column=5, value=flex['seed'])
        fa.cell(row=fa_row, column=6, value=fagree['observed_rate']).number_format = pct_fmt
        fa.cell(row=fa_row, column=7, value=fagree['permutation_chance_rate']).number_format = pct_fmt
        fa.cell(row=fa_row, column=8, value=fagree['kappa_flexible']).font = Font(name=FONT, bold=True, size=11)
        fa.cell(row=fa_row, column=9, value=fagree['n_flexible_only_chunks'])
        fa.cell(row=fa_row, column=10, value=sanity['strict_rate_permutation_chance']).number_format = pct_fmt
        fa.cell(row=fa_row, column=11, value=sanity['kappa_strict_via_permutation'])
        fa.cell(row=fa_row, column=12, value=sanity['kappa_strict_closed_form'])
        fa_row += 1
    # pooled row — from flexible_agreement_pooled.json (09_pool_flexible_agreement.py)
    fp_agree = flex_pooled['flexible_agreement']
    fp_sanity = flex_pooled['permutation_sanity_check']
    fa.cell(row=fa_row, column=2, value='POOLED (all reports)').font = label_font
    fa.cell(row=fa_row, column=3, value=flex_pooled['n']).font = label_font
    fa.cell(row=fa_row, column=4, value=flex_pooled['n_permutations']).font = label_font
    fa.cell(row=fa_row, column=5, value=flex_pooled['seed']).font = label_font
    c = fa.cell(row=fa_row, column=6, value=fp_agree['observed_rate']); c.number_format = pct_fmt; c.font = label_font
    c = fa.cell(row=fa_row, column=7, value=fp_agree['permutation_chance_rate']); c.number_format = pct_fmt; c.font = label_font
    fa.cell(row=fa_row, column=8, value=fp_agree['kappa_flexible']).font = Font(name=FONT, bold=True, size=12)
    fa.cell(row=fa_row, column=9, value=fp_agree['n_flexible_only_chunks']).font = label_font
    c = fa.cell(row=fa_row, column=10, value=fp_sanity['strict_rate_permutation_chance']); c.number_format = pct_fmt; c.font = label_font
    fa.cell(row=fa_row, column=11, value=fp_sanity['kappa_strict_via_permutation']).font = label_font
    fa.cell(row=fa_row, column=12, value=fp_sanity['kappa_strict_closed_form']).font = label_font
    fa_row += 2
    fa.cell(row=fa_row, column=2,
            value='Sanity check reads: the permutation-based strict kappa (col. K) should closely match the closed-form standard '
                  'kappa from the Agreement sheets (col. L) — confirms the permutation procedure itself is unbiased.').font = note_font
    autosize(fa, [3, 26] + [16] * 10)

    # =====================================================================
    # Report Profile (document-level, ALL coded chunks, independent of matching)
    # =====================================================================
    # Scores every row in each coder's own raw file independently (no chunk
    # matching, no coded_by_both filter) — each coder's own overall logic
    # profile over the full document. Source: <code>/report_level_profile.json
    # per report (06_report_level_profile.py). The pooled row sums raw points
    # across the 3 reports before computing proportions (same summation
    # philosophy as the pooled confusion matrix); it is reproduced by
    # 07_build_dashboard_data.py's build_pooled() and recorded here in
    # dashboard_bundle_v3.json — not re-derivable from a single JSON file in
    # this bundle, so it is written as a documented value, not a formula.
    rp = wb.create_sheet('Report Profile')
    rp['B2'] = 'Report-level logic profile — ALL coded chunks, independent of chunk matching'
    rp['B2'].font = title_font
    rp['B3'] = ('Unlike the Agreement sheets (restricted to chunks coded by BOTH coders), this uses every row either coder '
                'coded in their own raw file, so each coder’s overall document-level logic-frequency profile is not '
                'constrained by what the other coder happened to also code. TVD = total variation distance between the '
                'human and AI proportion vectors (0.5×Σ|p−q|); cosine = cosine similarity between them.')
    rp['B3'].font = subtitle_font
    rp.merge_cells('B3:L3')
    rp.row_dimensions[3].height = 42

    rp_headers = ['Report', 'Human rows coded', 'Human rows total', 'AI rows coded', 'AI rows total'] \
        + [f'Human % {l}' for l in LOGICS] + [f'AI % {l}' for l in LOGICS] + ['TVD', 'Cosine sim.']
    rp_hdr_row = 6
    for j, h in enumerate(rp_headers, start=2):
        rp.cell(row=rp_hdr_row, column=j, value=h)
    style_header_row(rp, rp_hdr_row, len(rp_headers), start_col=2)
    rp_row = rp_hdr_row + 1
    for rid, code, summary, ksum, flex, profile, blk_a, blk_b in report_info:
        dist = profile['profile_distance']
        rp.cell(row=rp_row, column=2, value=rid).font = normal_font
        rp.cell(row=rp_row, column=3, value=profile['n_human_rows_coded'])
        rp.cell(row=rp_row, column=4, value=profile['n_human_rows_total'])
        rp.cell(row=rp_row, column=5, value=profile['n_ai_rows_coded'])
        rp.cell(row=rp_row, column=6, value=profile['n_ai_rows_total'])
        for i, l in enumerate(LOGICS):
            rp.cell(row=rp_row, column=7 + i, value=profile['human_proportions'][l]).number_format = pct_fmt
            rp.cell(row=rp_row, column=11 + i, value=profile['ai_proportions'][l]).number_format = pct_fmt
        rp.cell(row=rp_row, column=15, value=dist['total_variation_distance']).font = Font(name=FONT, bold=True, size=11)
        rp.cell(row=rp_row, column=16, value=dist['cosine_similarity'])
        rp_row += 1
    # pooled row — from dashboard_bundle_v3.json's pooled.profile block
    # (07_build_dashboard_data.py: build_pooled(); human/ai proportions from
    # points summed across the 3 reports, then renormalized)
    pooled_profile = {
        'n_human_rows_coded': 324, 'n_human_rows_total': 956, 'n_ai_rows_coded': 629, 'n_ai_rows_total': 786,
        'human_prop': {'PS': 0.2117, 'DS': 0.0919, 'PPG': 0.3482, 'SC': 0.3482},
        'ai_prop': {'PS': 0.3718, 'DS': 0.1518, 'PPG': 0.2584, 'SC': 0.2180},
        'tvd': 0.2199, 'cosine': 0.9057,
    }
    rp.cell(row=rp_row, column=2, value='POOLED (all reports)').font = label_font
    rp.cell(row=rp_row, column=3, value=pooled_profile['n_human_rows_coded']).font = label_font
    rp.cell(row=rp_row, column=4, value=pooled_profile['n_human_rows_total']).font = label_font
    rp.cell(row=rp_row, column=5, value=pooled_profile['n_ai_rows_coded']).font = label_font
    rp.cell(row=rp_row, column=6, value=pooled_profile['n_ai_rows_total']).font = label_font
    for i, l in enumerate(LOGICS):
        c = rp.cell(row=rp_row, column=7 + i, value=pooled_profile['human_prop'][l]); c.number_format = pct_fmt; c.font = label_font
        c = rp.cell(row=rp_row, column=11 + i, value=pooled_profile['ai_prop'][l]); c.number_format = pct_fmt; c.font = label_font
    rp.cell(row=rp_row, column=15, value=pooled_profile['tvd']).font = Font(name=FONT, bold=True, size=12)
    rp.cell(row=rp_row, column=16, value=pooled_profile['cosine']).font = label_font
    rp_row += 2
    rp.cell(row=rp_row, column=2,
            value='Important caveat: these document-level proportions (restricted to each coder’s own coded rows) can look '
                  'different from the chunk-level dominant-logic confusion matrix in the Agreement sheets, which is restricted to '
                  'chunks coded by BOTH coders — this is not a bug (see methodology.md §6 for the worked example: AI’s SC share '
                  'here reflects chunks where SC scored secondary, but never dominant, points).').font = note_font
    autosize(rp, [3, 20, 15, 15, 12, 12] + [10] * 8 + [10, 12])

    # =====================================================================
    # Overview
    # =====================================================================
    ov = wb['Sheet'] if 'Sheet' in wb.sheetnames else None
    if ov is None:
        ov = wb.create_sheet('Overview', 0)
    ov.title = 'Overview'
    wb.move_sheet('Overview', offset=-(wb.sheetnames.index('Overview')))

    ov['B2'] = 'Institutional Logics — Human vs. AI Coder Agreement (multi-report study)'
    ov['B2'].font = title_font
    ov['B3'] = f'{len(report_info)} annual reports, each coded independently by a human coder and the AI pipeline.'
    ov['B3'].font = subtitle_font

    row = 5
    method_lines = [
        ('Method', None),
        ('Chunk matching', 'Exact match on chunk text, matched in row order, per report (handles repeated text).'),
        ('Weighting', 'Within each applicable element, the primary (top-1) logic scores 2 points and the secondary (top-2) logic scores 1 point, summed across the 5 elements.'),
        ('Chunk inclusion', 'A chunk counts as "coded" by a coder if that coder marked ≥1 of the 5 elements applicable; only chunks coded by BOTH coders enter the agreement analysis.'),
        ('Agreement score', 'One observation per chunk: each coder’s "dominant logic" (highest weighted-point total among PS/DS/PPG/SC). PS/DS/PPG/SC are NOMINAL categories with no established ordering, so the primary statistic is standard (unweighted) Cohen’s kappa. An earlier ordinal-weighted version is retired and kept only for comparison (see each Agreement sheet).'),
        ('Alternative (flexible)', 'A separate, chunk-flexible agreement measure — see the "Flexible Agreement" sheet — credits a chunk as matching if either coder’s dominant pick was at least the other coder’s own secondary emphasis for that chunk, with chance estimated by permutation rather than a closed-form formula.'),
        ('Pooling', 'The pooled confusion matrix sums each report’s own matrix; pooled kappa is recomputed from that pooled matrix (kappa is not linear, so averaging per-report kappas would not equal it).'),
        ('', None),
    ]
    for label, val in method_lines:
        if val is None and label:
            ov.cell(row=row, column=2, value=label).font = label_font
        elif label:
            ov.cell(row=row, column=2, value=label).font = normal_font
            c = ov.cell(row=row, column=4, value=val)
            c.font = normal_font
            c.alignment = Alignment(wrap_text=True, vertical='top')
            ov.row_dimensions[row].height = 30 if len(val) > 60 else 15
        row += 1

    row += 1
    ov.cell(row=row, column=2, value='Per-report agreement').font = label_font
    row += 1
    hdr_row = row
    headers = ['Report', 'N (coded by both)', 'κ standard (primary)', 'κ quadratic (retired)', 'κ linear (retired)', 'Raw agreement']
    for j, h in enumerate(headers, start=2):
        ov.cell(row=row, column=j, value=h)
    style_header_row(ov, row, len(headers), start_col=2)
    row += 1
    table_start = row
    for rid, code, summary, ksum, flex, profile, blk_a, blk_b in report_info:
        ov.cell(row=row, column=2, value=rid).font = normal_font
        ov.cell(row=row, column=3, value=f"='{code} Agreement'!G{blk_a['tot_r']}")
        ov.cell(row=row, column=4, value=f"='{code} Agreement'!G{blk_a['kappa_unw']}").font = Font(name=FONT, bold=True, size=11)
        ov.cell(row=row, column=5, value=f"='{code} Agreement'!G{blk_a['kappa_quad']}")
        ov.cell(row=row, column=6, value=f"='{code} Agreement'!G{blk_a['kappa_lin']}")
        ov.cell(row=row, column=7, value=f"='{code} Agreement'!G{blk_a['raw']}").number_format = pct_fmt
        row += 1
    table_end = row - 1
    # Pooled row
    ov.cell(row=row, column=2, value='POOLED (all reports)').font = label_font
    ov.cell(row=row, column=3, value=f"='Pooled Agreement'!G{pooled_blk_a['tot_r']}")
    ov.cell(row=row, column=4, value=f"='Pooled Agreement'!G{pooled_blk_a['kappa_unw']}").font = Font(name=FONT, bold=True, size=12)
    ov.cell(row=row, column=5, value=f"='Pooled Agreement'!G{pooled_blk_a['kappa_quad']}")
    ov.cell(row=row, column=6, value=f"='Pooled Agreement'!G{pooled_blk_a['kappa_lin']}")
    ov.cell(row=row, column=7, value=f"='Pooled Agreement'!G{pooled_blk_a['raw']}").number_format = pct_fmt
    pooled_row = row
    row += 2

    # chart: per-report + pooled standard (nominal) kappa
    chart = BarChart()
    chart.type = 'col'
    chart.title = 'Chunk-level standard (nominal) kappa, per report and pooled'
    chart.y_axis.title = 'Cohen’s kappa (standard, unweighted)'
    data = Reference(ov, min_col=4, min_row=hdr_row, max_row=pooled_row)
    cats = Reference(ov, min_col=2, min_row=table_start, max_row=pooled_row)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(cats)
    chart.width = 16
    chart.height = 8
    ov.add_chart(chart, f'B{row}')
    row += 18

    ov.cell(row=row, column=2,
            value='See the "Pooled Agreement" sheet for the full pooled confusion matrix and all three weighting schemes, '
                  'and each report’s own "<code> Agreement" sheet for its individual confusion matrix. "Report Key" maps '
                  'R1/R2/R3 codes to full report IDs.').font = note_font
    autosize(ov, [3, 30, 18, 18, 14, 14])

    outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    wb.save(outfile)
    print(f'Saved {outfile}')
    for rid, code, summary, ksum, flex, profile, blk_a, blk_b in report_info:
        print(f'{code} ({rid}): kappa_quad row={blk_a["kappa_quad"]}')
    print('Pooled kappa_quad row:', pooled_blk_a['kappa_quad'])


if __name__ == '__main__':
    main()
