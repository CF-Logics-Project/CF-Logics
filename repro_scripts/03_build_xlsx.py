#!/usr/bin/env python3
"""
03_build_xlsx.py

Step 3 of the Human-vs-AI institutional-logics agreement analysis.

Builds institutional_logics_agreement.xlsx from the outputs of
01_match_and_score.py and 02_chunk_kappa.py. Every number in the workbook
is a LIVE formula (COUNTIFS/SUMIFS/SUMPRODUCT), not a hardcoded value, so
it recalculates if the underlying chunk data changes.

Sheets produced
----------------
    Overview              method summary, document-level proportions +
                           chart, headline kappa figures
    Chunk-level            one row per matched chunk (points & proportions)
    Document Aggregate     document-level point totals/proportions
    Chunk Dominant Logic   one row per chunk coded by both coders (N),
                           dominant logic + tie flags per coder
    Agreement (Kappa)      4x4 confusion matrix, expected-under-independence,
                           and quadratic/linear/unweighted Cohen's kappa,
                           for the full sample and a ties-excluded
                           sensitivity check

IMPORTANT — recalculation: openpyxl writes formulas as strings with no
cached values. Open the file in real Excel (which recalculates on open)
or run it through LibreOffice headless recalculation before trusting any
previewer that reads cached values, e.g.:
    soffice --headless --convert-to xlsx --outdir . institutional_logics_agreement.xlsx
(or the `recalc.py` helper from Anthropic's xlsx skill, if you have it).

Usage
-----
    python3 03_build_xlsx.py \\
        --chunk-level-csv ./out/chunk_level.csv \\
        --chunk-dominant-csv ./out/chunk_dominant.csv \\
        --summary-json ./out/summary.json \\
        --chunk-kappa-json ./out/summary_chunk_kappa.json \\
        --outfile ./out/institutional_logics_agreement.xlsx
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--chunk-level-csv', required=True)
    ap.add_argument('--chunk-dominant-csv', required=True)
    ap.add_argument('--summary-json', required=True)
    ap.add_argument('--chunk-kappa-json', required=True)
    ap.add_argument('--outfile', default='institutional_logics_agreement.xlsx')
    args = ap.parse_args()

    chunk_df = pd.read_csv(args.chunk_level_csv)
    chunk_dom = pd.read_csv(args.chunk_dominant_csv)
    with open(args.summary_json) as f:
        summary = json.load(f)
    with open(args.chunk_kappa_json) as f:
        ksum = json.load(f)

    wb = Workbook()

    # ---------- styles ----------
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

    # =====================================================================
    # Sheet 1: Overview
    # =====================================================================
    ov = wb.active
    ov.title = 'Overview'
    ov['B2'] = 'Institutional Logics — Human vs. AI Coder Agreement'
    ov['B2'].font = title_font
    ov['B3'] = 'Comparison of the human coder export against the AI pipeline run'
    ov['B3'].font = subtitle_font

    row = 5
    lines = [
        ('Method', None),
        ('Chunk matching', 'Exact match on chunk text, matched in row order (handles repeated text).'),
        ('Weighting', 'Within each applicable element, the primary (top-1) logic scores 2 points and the secondary (top-2) logic scores 1 point. Points are summed across the 5 elements, then each logic’s share of the chunk/document total is its proportion.'),
        ('AI top-1/top-2 source', 'The AI pipeline has no explicit ranking field, so top-1/top-2 are parsed from its contrastive-logic answer (e.g. "2/1"), using the question’s own numbering: 1=PS, 2=DS, 3=PPG, 4=SC.'),
        ('Chunk inclusion', 'A chunk counts as "coded" by a coder if that coder marked ≥1 of the 5 elements applicable. Only chunks coded by BOTH coders are included in the document-level aggregate and the agreement score.'),
        ('Agreement score', f'One observation per chunk (N={ksum["full_sample"]["n"]}), not per chunk×element×logic. For each coder, the chunk’s "dominant logic" is whichever of PS/DS/PPG/SC holds the most weighted points for that coder on that chunk. PS/DS/PPG/SC are treated as ordinal, so a quadratic-weighted Cohen’s kappa (penalizing neighbor mismatches less than opposite-ends mismatches) is the primary reported figure. See the "Chunk Dominant Logic" and "Agreement (Kappa)" sheets for the full calculation, including the tie-break rule and a sensitivity check that drops tied chunks.'),
        ('', None),
        ('Coverage', None),
        ('Human coder rows', str(summary['n_human_rows'])),
        ('AI pipeline rows', str(summary['n_ai_rows'])),
        ('Matched chunks (same text, both files)', str(summary['n_matched_chunks'])),
        ('Unmatched human rows (no AI counterpart)', str(summary['n_unmatched_human'])),
        ('Chunks coded by both — included in aggregate', str(summary['n_included_in_aggregate'])),
    ]
    for label, val in lines:
        if val is None and label:
            ov.cell(row=row, column=2, value=label).font = label_font
        elif label:
            ov.cell(row=row, column=2, value=label).font = normal_font
            ov.cell(row=row, column=4, value=val).font = normal_font
            ov.cell(row=row, column=4).alignment = Alignment(wrap_text=True, vertical='top')
            ov.row_dimensions[row].height = 30 if val and len(val) > 60 else 15
        row += 1

    row += 1
    ov.cell(row=row, column=2, value='Document-level logic proportions (chunks coded by both only)').font = label_font
    row += 1
    hdr_row = row
    ov.cell(row=row, column=2, value='Logic')
    ov.cell(row=row, column=3, value='Human %')
    ov.cell(row=row, column=4, value='AI %')
    style_header_row(ov, row, 3, start_col=2)
    row += 1
    doc_table_start = row
    for l in LOGICS:
        ov.cell(row=row, column=2, value=f'{l} — {LOGIC_NAMES[l]}').font = normal_font
        ov.cell(row=row, column=3, value=f"=SUMIFS('Document Aggregate'!$D$5:$D$8,'Document Aggregate'!$B$5:$B$8,$B{row})").number_format = pct_fmt
        ov.cell(row=row, column=4, value=f"=SUMIFS('Document Aggregate'!$F$5:$F$8,'Document Aggregate'!$B$5:$B$8,$B{row})").number_format = pct_fmt
        row += 1
    doc_table_end = row - 1

    kappa_summary_row_start = row + 1  # filled in below, after the Agreement (Kappa) sheet exists

    chart = BarChart()
    chart.type = 'col'
    chart.title = 'Document-level logic proportions'
    chart.y_axis.title = 'Share of weighted points'
    chart.y_axis.numFmt = '0%'
    data = Reference(ov, min_col=3, max_col=4, min_row=hdr_row, max_row=doc_table_end)
    cats = Reference(ov, min_col=2, min_row=doc_table_start, max_row=doc_table_end)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(cats)
    chart.width = 16
    chart.height = 8
    ov.add_chart(chart, f'F{hdr_row}')
    autosize(ov, [3, 45, 16, 50, 4])

    # =====================================================================
    # Sheet 2: Chunk-level
    # =====================================================================
    cl = wb.create_sheet('Chunk-level')
    cols = ['chunk_id_cd', 'text', 'human_n_applicable_elements', 'ai_n_applicable_elements',
            'human_n_incomplete_rankings', 'ai_n_incomplete_rankings', 'coded_by_both']
    cols += [f'human_points_{l}' for l in LOGICS]
    cols += [f'ai_points_{l}' for l in LOGICS]
    cols += [f'human_prop_{l}' for l in LOGICS]
    cols += [f'ai_prop_{l}' for l in LOGICS]

    headers = (['Chunk ID', 'Text', 'Human # applicable elements', 'AI # applicable elements',
                'Human # incomplete rankings', 'AI # incomplete rankings', 'Coded by both?']
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
                row_vals.append(None)  # filled with formula below
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

    widths = [10, 60, 10, 10, 10, 10, 11] + [9] * 4 + [9] * 4 + [9] * 4 + [9] * 4
    autosize(cl, widths)
    cl.cell(row=n_data_rows + 3, column=1,
            value='"Coded by both" = TRUE means both coders marked ≥1 of the 5 elements applicable for that chunk; '
                  'only TRUE rows feed the Document Aggregate sheet.').font = note_font

    # =====================================================================
    # Sheet 3: Document Aggregate
    # =====================================================================
    da = wb.create_sheet('Document Aggregate')
    da['B2'] = 'Document-level point totals and proportions'
    da['B2'].font = title_font
    da['B3'] = 'Sums weighted points (top-1=2, top-2=1) across chunks where "Coded by both" = TRUE on the Chunk-level sheet.'
    da['B3'].font = subtitle_font
    da['B4'] = 'Logic'
    for j, h in enumerate(['Human points', 'Human %', 'AI points', 'AI %'], start=3):
        da.cell(row=4, column=j, value=h)
    style_header_row(da, 4, 5, start_col=2)

    last_row = n_data_rows + 1
    for i, l in enumerate(LOGICS):
        r = 5 + i
        da.cell(row=r, column=2, value=f'{l} — {LOGIC_NAMES[l]}').font = normal_font
        hp_col = get_column_letter(col_idx[f'human_points_{l}'])
        ap_col = get_column_letter(col_idx[f'ai_points_{l}'])
        cb_col = get_column_letter(col_idx['coded_by_both'])
        da.cell(row=r, column=3, value=f"=SUMIFS('Chunk-level'!{hp_col}2:{hp_col}{last_row},'Chunk-level'!{cb_col}2:{cb_col}{last_row},TRUE)")
        da.cell(row=r, column=5, value=f"=SUMIFS('Chunk-level'!{ap_col}2:{ap_col}{last_row},'Chunk-level'!{cb_col}2:{cb_col}{last_row},TRUE)")

    total_row = 9
    da.cell(row=total_row, column=2, value='Total').font = label_font
    da.cell(row=total_row, column=3, value='=SUM(C5:C8)').font = label_font
    da.cell(row=total_row, column=5, value='=SUM(E5:E8)').font = label_font
    for i in range(4):
        r = 5 + i
        da.cell(row=r, column=4, value=f"=IF($C${total_row}=0,\"\",C{r}/$C${total_row})").number_format = pct_fmt
        da.cell(row=r, column=6, value=f"=IF($E${total_row}=0,\"\",E{r}/$E${total_row})").number_format = pct_fmt

    da.cell(row=11, column=2, value='Chunks included (coded by both)').font = normal_font
    cb_col = get_column_letter(col_idx['coded_by_both'])
    da.cell(row=11, column=4, value=f"=COUNTIF('Chunk-level'!{cb_col}2:{cb_col}{last_row},TRUE)")
    da.cell(row=12, column=2, value='Chunks matched (total rows, Chunk-level sheet)').font = normal_font
    da.cell(row=12, column=4, value=n_data_rows)
    autosize(da, [3, 32, 14, 10, 14, 10])

    # =====================================================================
    # Sheet 4: Chunk Dominant Logic (one row per chunk coded by both coders)
    # =====================================================================
    cdom = wb.create_sheet('Chunk Dominant Logic')
    cdom.append(['Chunk ID', 'Human dominant logic', 'Human tie?', 'AI dominant logic', 'AI tie?', 'Match?'])
    style_header_row(cdom, 1, 6)
    for _, r in chunk_dom.iterrows():
        cdom.append([int(r['chunk_id']), r['human_dominant'], bool(r['human_tie']),
                     r['ai_dominant'], bool(r['ai_tie']), bool(r['match'])])
    cdom_last = len(chunk_dom) + 1
    autosize(cdom, [10, 20, 11, 20, 9, 9])
    cdom.freeze_panes = 'A2'
    cdom.cell(row=cdom_last + 2, column=1,
              value='Dominant logic = whichever of PS/DS/PPG/SC holds that coder’s highest weighted-point total for the '
                    'chunk (see "Chunk-level" sheet). "Tie?" = that coder’s own top points were tied across ≥2 logics '
                    'for this chunk; ties are broken by a fixed priority order PS>DS>PPG>SC, disclosed and tested for '
                    'sensitivity on the "Agreement (Kappa)" sheet.').font = note_font

    # =====================================================================
    # Sheet 5: Agreement (Kappa) — chunk-level confusion matrix & Cohen's kappa
    # =====================================================================
    ra = wb.create_sheet('Agreement (Kappa)')
    ra['B2'] = 'Chunk-level agreement: confusion matrix & Cohen’s kappa'
    ra['B2'].font = title_font
    ra['B3'] = f'One observation per chunk (N={ksum["full_sample"]["n"]}) — each coder’s dominant logic for that chunk. Source: "Chunk Dominant Logic" sheet.'
    ra['B3'].font = subtitle_font
    ra['B5'] = (f'Tie-break rule: when a coder’s own points tie across ≥2 logics for a chunk, the tie is broken by a fixed, arbitrary priority '
                f'order PS > DS > PPG > SC. {ksum["n_human_ties"]} human chunks and {ksum["n_ai_ties"]} AI chunks were tied '
                f'({ksum["n_overlapping_ties"]} chunks tied for both coders at once). '
                f'Section B below recomputes kappa excluding every tied chunk, as a robustness check.')
    ra['B5'].font = note_font
    ra.merge_cells('B5:J5')
    ra.row_dimensions[5].height = 42

    hcol = f"'Chunk Dominant Logic'!$B$2:$B${cdom_last}"
    acol = f"'Chunk Dominant Logic'!$D$2:$D${cdom_last}"
    htie = f"'Chunk Dominant Logic'!$C$2:$C${cdom_last}"
    atie = f"'Chunk Dominant Logic'!$E$2:$E${cdom_last}"

    # PS/DS/PPG/SC are an ORDINAL sequence (index 0,1,2,3, per the coding
    # instrument's own numbering). These weight matrices penalize a
    # disagreement by ordinal distance, so neighboring logics (e.g. PPG vs
    # SC) cost less than opposite ends (PS vs SC).
    quad4 = [[0, 1, 4, 9], [1, 0, 1, 4], [4, 1, 0, 1], [9, 4, 1, 0]]  # (i-j)^2
    lin4 = [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]   # |i-j|
    unw4 = [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]   # nominal, reference only

    def build_confusion_block(ws, label, top_row, obs_extra_criteria=None):
        """Writes a 4x4 confusion matrix, expected-under-independence block,
        and three weighted-kappa sub-blocks (quadratic/linear/unweighted).
        Returns (kappa_row_quadratic, kappa_row_linear, kappa_row_unweighted,
        raw_agreement_row, next_free_row)."""
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

        quad_top = raw_r + 2
        ws.cell(row=quad_top - 1, column=2,
                value='PS/DS/PPG/SC are ordinal (index 0,1,2,3); weights below are squared ordinal distance (i-j)², so a '
                      'neighbor mismatch (e.g. PPG↔SC) is penalized far less than an opposite-ends mismatch (PS↔SC).').font = note_font
        kappa_r_quad, next1 = weight_subblock(quad_top, 'PRIMARY — Weights: quadratic ordinal distance (i−j)²', quad4, bold=True)
        kappa_r_lin, next2 = weight_subblock(next1 + 1, 'Reference — Weights: linear ordinal distance |i−j|', lin4)
        kappa_r_unw, next3 = weight_subblock(next2 + 1, 'Reference — Weights: unweighted/nominal (0 diagonal, 1 off-diagonal; ignores order)', unw4)
        return kappa_r_quad, kappa_r_lin, kappa_r_unw, raw_r, next3

    kappa_r_quad_a, kappa_r_lin_a, kappa_r_unw_a, raw_r_a, next_free = build_confusion_block(
        ra, 'A. Full sample — all chunks coded by both coders', 7)
    kappa_r_quad_b, kappa_r_lin_b, kappa_r_unw_b, raw_r_b, next_free2 = build_confusion_block(
        ra, 'B. Sensitivity — chunks with a tie (either coder) excluded', next_free + 1,
        obs_extra_criteria=f'{htie},FALSE,{atie},FALSE')

    ra.cell(row=next_free2 + 1, column=2,
            value='Interpretation guide (Landis & Koch, 1977): <0 poor, 0.00–0.20 slight, 0.21–0.40 fair, '
                  '0.41–0.60 moderate, 0.61–0.80 substantial, 0.81–1.00 almost perfect. Guide was developed for '
                  'unweighted kappa; weighted kappa values run somewhat higher for the same underlying agreement because '
                  'near-miss disagreements count less, so read the weighted figure against this scale with that in mind.').font = note_font
    autosize(ra, [3, 36, 11, 11, 11, 11, 12])

    # ---- Fill in the Overview sheet's deferred kappa summary block, now that
    # ---- the Agreement (Kappa) sheet's row layout is known. ----
    row = kappa_summary_row_start
    n_full = ksum['full_sample']['n']
    n_sens = ksum['tie_excluded_sensitivity']['n']
    ov.cell(row=row, column=2, value=f'Chunk-level weighted kappa — quadratic ordinal weights (PRIMARY, N={n_full})').font = label_font
    ov.cell(row=row, column=4, value=f"='Agreement (Kappa)'!G{kappa_r_quad_a}").font = Font(name=FONT, bold=True, size=12)
    row += 1
    ov.cell(row=row, column=2, value='Chunk-level kappa — linear ordinal weights (for reference)').font = normal_font
    ov.cell(row=row, column=4, value=f"='Agreement (Kappa)'!G{kappa_r_lin_a}")
    row += 1
    ov.cell(row=row, column=2, value='Chunk-level kappa — unweighted/nominal (for reference)').font = normal_font
    ov.cell(row=row, column=4, value=f"='Agreement (Kappa)'!G{kappa_r_unw_a}")
    row += 1
    ov.cell(row=row, column=2, value='Raw agreement (% chunks, same dominant logic)').font = normal_font
    ov.cell(row=row, column=4, value=f"='Agreement (Kappa)'!G{raw_r_a}").number_format = pct_fmt
    row += 1
    ov.cell(row=row, column=2, value=f'Sensitivity: quadratic-weighted kappa excluding tied chunks (N={n_sens})').font = normal_font
    ov.cell(row=row, column=4, value=f"='Agreement (Kappa)'!G{kappa_r_quad_b}")
    row += 2
    ov.cell(row=row, column=2,
            value=(f"Supplementary: top-1 logic exact agreement (nominal, elements both coders ranked) = "
                   f"{summary['top1_n_matches']}/{summary['top1_n_comparable']} = {summary['top1_pct_agreement']:.1%}")).font = note_font

    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
    wb.save(args.outfile)
    print(f'Saved {args.outfile}')
    print('kappa rows (quad/lin/unw, raw) A:', kappa_r_quad_a, kappa_r_lin_a, kappa_r_unw_a, raw_r_a)
    print('kappa rows (quad/lin/unw, raw) B:', kappa_r_quad_b, kappa_r_lin_b, kappa_r_unw_b, raw_r_b)


if __name__ == '__main__':
    main()
