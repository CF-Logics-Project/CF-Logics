#!/usr/bin/env python3
"""
agreement_xlsx.py

Builds institutional_logics_agreement.xlsx from the single
agreement_bundle.json produced by agreement_analysis.py (plus that
same run's per-report chunk_level.csv / chunk_dominant.csv files, for the
full per-chunk tables).

One sheet group per report ("<code> Chunk-level", "<code> Doc Aggregate",
"<code> Chunk Dominant", "<code> Agreement"), a "Report Key" sheet mapping
codes to full report IDs, a "Pooled Agreement" sheet (confusion matrix
pooled by LIVE cross-sheet SUM formulas over each report's own Agreement
sheet, so it recalculates correctly if a report's chunk data changes), a
"Flexible Agreement" sheet and a "Report Profile" sheet (both necessarily
written as sourced values, not formulas — see the note in each sheet; the
underlying statistics involve a Monte Carlo permutation and a document-wide
rescoring that aren't expressible as Excel formulas over this workbook's
own cells), and an "Overview" sheet with the headline kappa table + chart.

PS/DS/PPG/SC are NOMINAL categories (no established ordering), so standard
(unweighted) Cohen's kappa is PRIMARY everywhere in this workbook. The
earlier ordinal-weighted (quadratic/linear) kappa is kept only as a
RETIRED, clearly labeled comparison.

Usage
-----
    python3 agreement_xlsx.py [--bundle out/agreement/agreement_bundle.json] \\
        [--outdir out/agreement] [--outfile out/agreement/institutional_logics_agreement.xlsx]

Relative paths are resolved from the repository root, where this script lives.
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
REPO_ROOT = Path(__file__).resolve().parent

unw4 = [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bundle', default='out/agreement/agreement_bundle.json',
                    help='agreement_bundle.json from agreement_analysis.py')
    ap.add_argument('--outdir', default='out/agreement',
                    help='Folder holding <report_id>/chunk_level.csv + chunk_dominant.csv')
    ap.add_argument('--outfile', default='out/agreement/institutional_logics_agreement.xlsx')
    args = ap.parse_args()

    with open(REPO_ROOT / args.bundle) as f:
        bundle = json.load(f)
    reports = bundle['reports']
    pooled = bundle['pooled']
    codes = list(reports.keys())
    outdir = REPO_ROOT / args.outdir

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
            cell.fill, cell.font = header_fill, header_font
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
            cell.border = border

    def autosize(ws, widths):
        for i, w in enumerate(widths, start=1):
            ws.column_dimensions[get_column_letter(i)].width = w

    def build_confusion_block(ws, label, top_row, hcol, acol, obs_extra_criteria=None):
        ws.cell(row=top_row, column=2, value=label).font = label_font
        hdr = top_row + 1
        data_r0 = hdr + 1
        for j, l in enumerate(LOGICS):
            ws.cell(row=hdr, column=3 + j, value=f'AI: {l}')
        ws.cell(row=hdr, column=7, value='Row total')
        style_header_row(ws, hdr, 5, start_col=3)
        for i, hv in enumerate(LOGICS):
            r = data_r0 + i
            ws.cell(row=r, column=2, value=f'Human: {hv}').font = normal_font
            for j, av in enumerate(LOGICS):
                crit = f'{hcol},"{hv}",{acol},"{av}"'
                if obs_extra_criteria:
                    crit += f',{obs_extra_criteria}'
                ws.cell(row=r, column=3 + j, value=f"=COUNTIFS({crit})")
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
            for j in range(4):
                col_l = get_column_letter(3 + j)
                ws.cell(row=exp_r0 + i, column=3 + j, value=f"=$G{data_r0+i}*{col_l}${tot_r}/$G${tot_r}")
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

    # =================================================================
    # Per-report sheets
    # =================================================================
    report_info = []  # (rid, code, entry, blk_a, blk_b)
    for code in codes:
        entry = reports[code]
        rid = entry['report_id']
        rdir = outdir / rid
        chunk_df = pd.read_csv(rdir / 'chunk_level.csv')
        chunk_dom = pd.read_csv(rdir / 'chunk_dominant.csv')

        cl = wb.create_sheet(f'{code} Chunk-level')
        cols = ['chunk_id', 'text', 'human_n_applicable_elements', 'ai_n_applicable_elements',
                'human_n_incomplete_rankings', 'ai_n_incomplete_rankings', 'coded_by_both']
        cols += [f'human_points_{l}' for l in LOGICS] + [f'ai_points_{l}' for l in LOGICS]
        headers = (['Chunk ID', 'Text', 'Human # applic.', 'AI # applic.', 'Human # incompl.', 'AI # incompl.', 'Coded by both?']
                   + [f'Human pts {l}' for l in LOGICS] + [f'AI pts {l}' for l in LOGICS]
                   + [f'Human % {l}' for l in LOGICS] + [f'AI % {l}' for l in LOGICS])
        cl.append(headers)
        style_header_row(cl, 1, len(headers))
        cl.freeze_panes = 'C2'
        col_idx = {name: i + 1 for i, name in enumerate(cols)}
        for _, r in chunk_df.iterrows():
            cl.append([bool(r[c]) if c == 'coded_by_both' else r[c] for c in cols] + [None] * 8)
        n_data_rows = len(chunk_df)
        prop_base = len(cols) + 1
        for i, l in enumerate(LOGICS):
            cl.cell(row=1, column=prop_base + i, value=f'Human % {l}')
            cl.cell(row=1, column=prop_base + 4 + i, value=f'AI % {l}')
        style_header_row(cl, 1, 8, start_col=prop_base)
        for i in range(2, n_data_rows + 2):
            hp = [f"{get_column_letter(col_idx[f'human_points_{l}'])}{i}" for l in LOGICS]
            ap = [f"{get_column_letter(col_idx[f'ai_points_{l}'])}{i}" for l in LOGICS]
            for j, l in enumerate(LOGICS):
                hc = cl.cell(row=i, column=prop_base + j)
                hc.value = f"=IF(SUM({hp[0]}:{hp[-1]})=0,\"\",{hp[j]}/SUM({hp[0]}:{hp[-1]}))"
                hc.number_format = pct_fmt
                ac = cl.cell(row=i, column=prop_base + 4 + j)
                ac.value = f"=IF(SUM({ap[0]}:{ap[-1]})=0,\"\",{ap[j]}/SUM({ap[0]}:{ap[-1]}))"
                ac.number_format = pct_fmt
        autosize(cl, [10, 60, 9, 9, 9, 9, 10] + [8] * 16)

        da = wb.create_sheet(f'{code} Doc Aggregate')
        da['B2'] = f'Document-level totals (chunks coded by both only) — {rid}'
        da['B2'].font = title_font
        for j, h in enumerate(['Logic', 'Human points', 'Human %', 'AI points', 'AI %'], start=2):
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

        cdom = wb.create_sheet(f'{code} Chunk Dominant')
        cdom.append(['Chunk ID', 'Human dominant', 'Human tie?', 'AI dominant', 'AI tie?', 'Match?'])
        style_header_row(cdom, 1, 6)
        for _, r in chunk_dom.iterrows():
            cdom.append([int(r['chunk_id']), r['human_dominant'], bool(r['human_tie']),
                         r['ai_dominant'], bool(r['ai_tie']), bool(r['match'])])
        cdom_last = len(chunk_dom) + 1
        autosize(cdom, [10, 15, 10, 13, 9, 9])
        cdom.freeze_panes = 'A2'

        ra = wb.create_sheet(f'{code} Agreement')
        ra['B2'] = f'Chunk-level agreement — {rid}'
        ra['B2'].font = title_font
        ra['B3'] = f'N={entry["n_included"]} chunks coded by both coders. Ties: {entry["n_human_ties"]} human, {entry["n_ai_ties"]} AI (broken PS>DS>PPG>SC, mechanical only).'
        ra['B3'].font = subtitle_font
        hcol = f"'{code} Chunk Dominant'!$B$2:$B${cdom_last}"
        acol = f"'{code} Chunk Dominant'!$D$2:$D${cdom_last}"
        htie = f"'{code} Chunk Dominant'!$C$2:$C${cdom_last}"
        atie = f"'{code} Chunk Dominant'!$E$2:$E${cdom_last}"
        blk_a = build_confusion_block(ra, 'A. Full sample', 7, hcol, acol)
        blk_b = build_confusion_block(ra, 'B. Sensitivity — ties excluded', blk_a['next_free'] + 1, hcol, acol,
                                       obs_extra_criteria=f'{htie},FALSE,{atie},FALSE')
        autosize(ra, [3, 36, 11, 11, 11, 11, 12])

        report_info.append((rid, code, entry, blk_a, blk_b))

    # =================================================================
    # Report Key
    # =================================================================
    rk = wb.create_sheet('Report Key')
    rk.append(['Code', 'Report ID', 'Human rows', 'AI rows', 'Matched chunks', 'Coded by both'])
    style_header_row(rk, 1, 6)
    for rid, code, entry, *_ in report_info:
        rk.append([code, rid, entry['n_human_rows'], entry['n_ai_rows'], entry['n_matched'], entry['n_included']])
    autosize(rk, [8, 28, 12, 10, 14, 13])

    # =================================================================
    # Pooled Agreement (sums each report's own confusion matrix, live)
    # =================================================================
    pa = wb.create_sheet('Pooled Agreement')
    pa['B2'] = 'Pooled chunk-level agreement — all reports combined'
    pa['B2'].font = title_font
    pa['B3'] = (f'Confusion matrix is the SUM of each report’s own "Agreement" sheet matrix (live cross-sheet formulas), '
                f'not a separately hardcoded pooled count — so it stays correct if any report’s chunk data changes. '
                f'N = {pooled["n_included"]} chunks coded by both coders, pooled across {pooled["n_reports"]} reports.')
    pa['B3'].font = subtitle_font
    pa.merge_cells('B3:J3')
    pa.row_dimensions[3].height = 28

    def pooled_block(ws, label, top_row, which):
        ws.cell(row=top_row, column=2, value=label).font = label_font
        hdr = top_row + 1
        data_r0 = hdr + 1
        for j, l in enumerate(LOGICS):
            ws.cell(row=hdr, column=3 + j, value=f'AI: {l}')
        ws.cell(row=hdr, column=7, value='Row total')
        style_header_row(ws, hdr, 5, start_col=3)
        for i in range(4):
            r = data_r0 + i
            ws.cell(row=r, column=2, value=f'Human: {LOGICS[i]}').font = normal_font
            for j in range(4):
                col_l = get_column_letter(3 + j)
                src_row = report_info[0][3 if which == 'a' else 4]['data_r0'] + i
                terms = [f"'{code} Agreement'!{col_l}{src_row}" for (_, code, *_rest) in report_info]
                ws.cell(row=r, column=3 + j, value='=' + '+'.join(terms))
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
            for j in range(4):
                col_l = get_column_letter(3 + j)
                ws.cell(row=exp_r0 + i, column=3 + j, value=f"=$G{data_r0+i}*{col_l}${tot_r}/$G${tot_r}")
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
                'next_free': next2, 'data_r0': data_r0, 'tot_r': tot_r}

    pooled_blk_a = pooled_block(pa, 'A. Full sample (pooled across all reports)', 5, 'a')
    pooled_blk_b = pooled_block(pa, 'B. Sensitivity — ties excluded (pooled)', pooled_blk_a['next_free'] + 1, 'b')
    pa.cell(row=pooled_blk_b['next_free'] + 1, column=2,
            value='Interpretation guide (Landis & Koch, 1977): <0 poor, 0.00–0.20 slight, 0.21–0.40 fair, 0.41–0.60 moderate, 0.61–0.80 substantial, 0.81–1.00 almost perfect.').font = note_font
    autosize(pa, [3, 36, 11, 11, 11, 11, 12])

    # =================================================================
    # Flexible Agreement (permutation-based; values, documented source)
    # =================================================================
    fa = wb.create_sheet('Flexible Agreement')
    fa['B2'] = 'Alternative agreement measure — chunk-flexible vicinity (permutation-based)'
    fa['B2'].font = title_font
    fa['B3'] = ('A chunk is a "flexible match" if the two coders’ dominant picks agree, OR either coder’s dominant pick is '
                'in the OTHER coder’s own chunk-level top-2 set. Chance is estimated by permutation (no closed-form formula '
                'applies to this joint-ranking criterion); kappa_flexible = (observed − permutation chance) / (1 − chance). '
                'These values cannot be Excel formulas (the chance baseline is a Monte Carlo simulation over the Python '
                'analysis, not a function of this workbook’s cells) — source: agreement_bundle.json’s "flex" block per '
                'report and pooled, from agreement_analysis.py (seed and permutation count recorded per row below).')
    fa['B3'].font = subtitle_font
    fa.merge_cells('B3:L3')
    fa.row_dimensions[3].height = 42

    fa_headers = ['Report', 'N', '# permutations', 'seed', 'Observed flexible rate', 'Permutation chance rate',
                  'kappa_flexible', '# flexible-only chunks', 'Sanity: kappa via permutation', 'Sanity: kappa closed-form']
    fa_hdr_row = 6
    for j, h in enumerate(fa_headers, start=2):
        fa.cell(row=fa_hdr_row, column=j, value=h)
    style_header_row(fa, fa_hdr_row, len(fa_headers), start_col=2)
    fa_row = fa_hdr_row + 1
    for rid, code, entry, *_ in report_info:
        flex = entry['flex']
        fa.cell(row=fa_row, column=2, value=rid).font = normal_font
        fa.cell(row=fa_row, column=3, value=flex['n'])
        fa.cell(row=fa_row, column=4, value=flex['n_permutations'])
        fa.cell(row=fa_row, column=5, value=0)
        fa.cell(row=fa_row, column=6, value=flex['observed_rate']).number_format = pct_fmt
        fa.cell(row=fa_row, column=7, value=flex['chance_rate']).number_format = pct_fmt
        fa.cell(row=fa_row, column=8, value=flex['kappa_flexible']).font = Font(name=FONT, bold=True, size=11)
        fa.cell(row=fa_row, column=9, value=flex['n_flexible_only'])
        fa.cell(row=fa_row, column=10, value=flex['kappa_standard_perm_check'])
        fa.cell(row=fa_row, column=11, value=entry['kappa_standard'])
        fa_row += 1
    pflex = pooled['flex']
    fa.cell(row=fa_row, column=2, value='POOLED (all reports)').font = label_font
    fa.cell(row=fa_row, column=3, value=pflex['n']).font = label_font
    fa.cell(row=fa_row, column=4, value=pflex['n_permutations']).font = label_font
    fa.cell(row=fa_row, column=5, value=0).font = label_font
    c = fa.cell(row=fa_row, column=6, value=pflex['observed_rate']); c.number_format = pct_fmt; c.font = label_font
    c = fa.cell(row=fa_row, column=7, value=pflex['chance_rate']); c.number_format = pct_fmt; c.font = label_font
    fa.cell(row=fa_row, column=8, value=pflex['kappa_flexible']).font = Font(name=FONT, bold=True, size=12)
    fa.cell(row=fa_row, column=9, value=pflex['n_flexible_only']).font = label_font
    fa.cell(row=fa_row, column=10, value=pflex['kappa_standard_perm_check']).font = label_font
    fa.cell(row=fa_row, column=11, value=pooled['kappa_standard']).font = label_font
    fa_row += 2
    fa.cell(row=fa_row, column=2,
            value='Sanity check: col. J (permutation-based strict kappa) should closely match col. K (closed-form standard kappa '
                  'from the Agreement sheets) — confirms the permutation procedure itself is unbiased.').font = note_font
    autosize(fa, [3, 26] + [15] * 9)

    # =================================================================
    # Report Profile (document-level, ALL coded chunks, independent of matching)
    # =================================================================
    rp = wb.create_sheet('Report Profile')
    rp['B2'] = 'Report-level logic profile — ALL coded chunks, independent of chunk matching'
    rp['B2'].font = title_font
    rp['B3'] = ('Unlike the Agreement sheets (chunks coded by BOTH coders only), this scores every row either coder coded in '
                'their own raw file, so each coder’s overall document-level logic-frequency profile is not constrained by '
                'what the other coder happened to also code. TVD = total variation distance between the human and AI '
                'proportion vectors; cosine = cosine similarity between them. Source: agreement_bundle.json’s "profile" '
                'block per report and pooled.')
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
    for rid, code, entry, *_ in report_info:
        p = entry['profile']
        rp.cell(row=rp_row, column=2, value=rid).font = normal_font
        rp.cell(row=rp_row, column=3, value=p['n_human_rows_coded'])
        rp.cell(row=rp_row, column=4, value=p['n_human_rows_total'])
        rp.cell(row=rp_row, column=5, value=p['n_ai_rows_coded'])
        rp.cell(row=rp_row, column=6, value=p['n_ai_rows_total'])
        for i in range(4):
            rp.cell(row=rp_row, column=7 + i, value=p['human_prop'][i]).number_format = pct_fmt
            rp.cell(row=rp_row, column=11 + i, value=p['ai_prop'][i]).number_format = pct_fmt
        rp.cell(row=rp_row, column=15, value=p['tvd']).font = Font(name=FONT, bold=True, size=11)
        rp.cell(row=rp_row, column=16, value=p['cosine'])
        rp_row += 1
    pp = pooled['profile']
    rp.cell(row=rp_row, column=2, value='POOLED (all reports)').font = label_font
    rp.cell(row=rp_row, column=3, value=pp['n_human_rows_coded']).font = label_font
    rp.cell(row=rp_row, column=4, value=pp['n_human_rows_total']).font = label_font
    rp.cell(row=rp_row, column=5, value=pp['n_ai_rows_coded']).font = label_font
    rp.cell(row=rp_row, column=6, value=pp['n_ai_rows_total']).font = label_font
    for i in range(4):
        c = rp.cell(row=rp_row, column=7 + i, value=pp['human_prop'][i]); c.number_format = pct_fmt; c.font = label_font
        c = rp.cell(row=rp_row, column=11 + i, value=pp['ai_prop'][i]); c.number_format = pct_fmt; c.font = label_font
    rp.cell(row=rp_row, column=15, value=pp['tvd']).font = Font(name=FONT, bold=True, size=12)
    rp.cell(row=rp_row, column=16, value=pp['cosine']).font = label_font
    rp_row += 2
    rp.cell(row=rp_row, column=2,
            value='Important caveat: these document-level proportions can look different from the chunk-level dominant-logic '
                  'confusion matrix in the Agreement sheets, which is restricted to chunks coded by BOTH coders — this is not '
                  'a bug (see methodology.md for the worked example).').font = note_font
    autosize(rp, [3, 20, 15, 15, 12, 12] + [10] * 8 + [10, 12])

    # =================================================================
    # Overview
    # =================================================================
    ov = wb['Sheet'] if 'Sheet' in wb.sheetnames else wb.create_sheet('Overview', 0)
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
    for rid, code, entry, blk_a, blk_b in report_info:
        ov.cell(row=row, column=2, value=rid).font = normal_font
        ov.cell(row=row, column=3, value=f"='{code} Agreement'!G{blk_a['tot_r']}")
        ov.cell(row=row, column=4, value=f"='{code} Agreement'!G{blk_a['kappa_unw']}").font = Font(name=FONT, bold=True, size=11)
        ov.cell(row=row, column=5, value=f"='{code} Agreement'!G{blk_a['kappa_quad']}")
        ov.cell(row=row, column=6, value=f"='{code} Agreement'!G{blk_a['kappa_lin']}")
        ov.cell(row=row, column=7, value=f"='{code} Agreement'!G{blk_a['raw']}").number_format = pct_fmt
        row += 1
    table_end = row - 1
    ov.cell(row=row, column=2, value='POOLED (all reports)').font = label_font
    ov.cell(row=row, column=3, value=f"='Pooled Agreement'!G{pooled_blk_a['tot_r']}")
    ov.cell(row=row, column=4, value=f"='Pooled Agreement'!G{pooled_blk_a['kappa_unw']}").font = Font(name=FONT, bold=True, size=12)
    ov.cell(row=row, column=5, value=f"='Pooled Agreement'!G{pooled_blk_a['kappa_quad']}")
    ov.cell(row=row, column=6, value=f"='Pooled Agreement'!G{pooled_blk_a['kappa_lin']}")
    ov.cell(row=row, column=7, value=f"='Pooled Agreement'!G{pooled_blk_a['raw']}").number_format = pct_fmt
    pooled_row = row
    row += 2

    chart = BarChart()
    chart.type = 'col'
    chart.title = 'Chunk-level standard (nominal) kappa, per report and pooled'
    chart.y_axis.title = 'Cohen’s kappa (standard, unweighted)'
    data = Reference(ov, min_col=4, min_row=hdr_row, max_row=pooled_row)
    cats = Reference(ov, min_col=2, min_row=table_start, max_row=pooled_row)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(cats)
    chart.width, chart.height = 16, 8
    ov.add_chart(chart, f'B{row}')
    row += 18

    ov.cell(row=row, column=2,
            value='See the "Pooled Agreement" sheet for the full pooled confusion matrix and all three weighting schemes, '
                  'the "Flexible Agreement" and "Report Profile" sheets for the two alternative analyses, and each report’s own '
                  '"<code> Agreement" sheet for its individual confusion matrix. "Report Key" maps codes to full report IDs.').font = note_font
    autosize(ov, [3, 30, 18, 18, 14, 14])

    outfile = REPO_ROOT / args.outfile
    outfile.parent.mkdir(parents=True, exist_ok=True)
    wb.save(outfile)
    print(f'Saved {outfile}')


if __name__ == '__main__':
    main()
