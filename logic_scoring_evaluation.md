# Methodology — Human vs. AI Coder Agreement on Institutional Logics

This note documents every method, formula, and parameter used to compare the human coder's export against the AI pipeline's own run, across three nonprofit annual reports, as implemented in the numbered scripts in `repro_scripts/` and reproduced with live formulas in `institutional_logics_agreement.xlsx`.

**Revision history.** This is the third revision of this document. (1) The original single-report version used quadratic ordinal-weighted Cohen's kappa as the primary statistic. (2) That was extended to 3 reports plus a pooled combination, keeping the same ordinal weighting. (3) **This revision retires the ordinal weighting entirely**: PS, DS, PPG and SC are nominal categories with no established theoretical ordering between them, so there is no principled basis for treating a PPG↔SC disagreement as "smaller" than a PS↔SC one. Standard (nominal, unweighted) Cohen's kappa is now the sole primary statistic, and an alternative, chunk-flexible agreement measure (§8) is reported alongside it — not as a replacement, but as a different, independently-motivated way of giving partial credit for near misses without assuming any category order.

## 1. Sample

Three nonprofit annual reports, each independently coded in full by the human coder and by the AI pipeline, using the same codebook and the same chunk segmentation:

| Report | Human rows | AI rows | Matched chunks | Coded by both (N) |
|---|---|---|---|---|
| 363466802 (FY2021) | 403 | 337 | 337 | 161 |
| 382023395 (FY2022) | 393 | 335 | 333 | 113 |
| 25-1650208 (FY2021) | 160 | 114 | 114 | 32 |
| **Pooled** | 956 | 786 | 784 | **306** |

Report 25-1650208's small N is not a matching artifact: the human coder marked an applicable logic in only 32 of that report's 160 chunks in total (20%), against 103 of 114 (90%) for the AI pipeline (§9). Every coded-by-both chunk in that report happens to be matched, so the agreement analysis there is not losing data to mismatch — the human coder simply found far less to code in that report than the AI did.

## 2. Chunk matching

The human and AI files have different row counts and no shared ID column. Chunks are matched by exact text, in row order, using an occurrence index so repeated text values are matched one-to-one against their corresponding occurrence rather than collapsed:

```
build_occurrence_index(df, text_col='text'):
    for i, t in enumerate(df[text_col]):
        counts[t] += 1
        idx_map[(t, counts[t])] = i
```

(`01_match_and_score.py`)

## 3. Per-chunk, per-coder applicability and ranking

For each of the 5 elements (community, donor, activity, board, grantee), each coder's response is reduced to a triple `(top1, top2, applicable)`:

- **Human coder**: `applicable = (human_{element}_answer == "yes")`; `top1`/`top2` are read directly from `human_{element}_top1_logic` / `human_{element}_top2_logic`. A chunk marked `human_skip = "yes"` is not applicable for any element.
- **AI pipeline**: `applicable = ({element}_answer == "yes")`. The AI pipeline has no explicit top1/top2 field; these are parsed from its `{element}_contrastive_logic_answer` field (format `"X/Y"`), using the coding instrument's own digit mapping `{'1':'PS','2':'DS','3':'PPG','4':'SC'}`. This mapping is used only to decode the AI's raw output format — it is **not** treated as implying any ordering among the four logics (see §7).

## 4. Weighted points per chunk, per coder

Within each applicable element, the primary (top-1) logic scores **2 points** and the secondary (top-2) logic scores **1 point**, summed across the 5 elements, giving each coder a 4-vector of points over `{PS, DS, PPG, SC}` per chunk:

```
pts[logic] = Σ over applicable elements of:  2 if logic == top1  +  1 if logic == top2
prop[logic] = pts[logic] / Σ_l pts[l]      (undefined if the chunk has 0 points for that coder)
```

## 5. Chunk inclusion rule ("coded by both")

A chunk counts as coded by a coder if that coder marked ≥1 of the 5 elements applicable. A chunk enters the agreement calculation only if it was coded by **both** coders:

```
coded_by_both = (human_n_applicable_elements >= 1) AND (ai_n_applicable_elements >= 1)
```

This is the N in the table in §1, and is also the subsample used for the "document-level logic proportions" in §6 below — **not** the full set of chunks either coder coded on its own (that broader, unrestricted comparison is a separate analysis; see §9).

## 6. Document-level logic proportions (coded-by-both chunks only)

Summed per coder, over only the included (coded-by-both) chunks of a report:

```
doc_pts[logic]  = Σ over included chunks of that coder's pts[logic]
doc_prop[logic] = doc_pts[logic] / Σ_l doc_pts[l]
```

| Report | Human PS/DS/PPG/SC | AI PS/DS/PPG/SC |
|---|---|---|
| 363466802 | 20.5 / 0.6 / 23.7 / 55.3% | 23.8 / 5.6 / 29.6 / 41.0% |
| 382023395 | 18.1 / 16.1 / 50.2 / 15.6% | 33.0 / 17.4 / 32.4 / 17.2% |
| 25-1650208 | 25.4 / 26.8 / 47.9 / 0.0% | 47.1 / 20.0 / 28.1 / 4.8% |

**Important caveat, directly tested in this session:** a logic can have a nonzero share here while never once being a chunk's single *dominant* pick in §7's confusion matrix. For report 25-1650208, AI's 4.8% SC share comes entirely from 4 specific chunks where AI gave SC *secondary* (top-2) points, but SC was outscored outright in 3 of them and lost a 3-3 tie with PPG in the 4th (§7's tie-break rule) — so SC never appears as AI's dominant pick for that report, even though it is not a zero in the proportions table. Both numbers are computed correctly; they simply answer different questions (overall point share vs. single strongest pick per chunk), and this is flagged explicitly to avoid the two being misread as contradictory.

## 7. Chunk-level agreement — unit of observation, and the primary statistic

**Unit of observation: one per chunk.** For each coder, the chunk's dominant logic is whichever of PS/DS/PPG/SC holds that coder's own highest point total for that chunk (§4):

```
dominant(coder, chunk) = argmax_logic  pts[coder][chunk][logic]
```

**Tie-break rule (mechanical, not theoretical).** A coder's own points can tie across ≥2 logics for a chunk, leaving `argmax` undefined. Ties are broken by a fixed rule, priority order PS > DS > PPG > SC. This rule exists only to force a single label out of a tie so a confusion matrix can be built — **it carries no claim that the categories are ordered**, and does not affect whether the resulting label matches the other coder's; it only decides *which* label a tied chunk receives. (An earlier draft of this methodology used the same priority string to justify ordinal-weighted kappa in §7 as well; that reuse was the error this revision corrects — the tie-break convenience and the category-ordering claim are logically independent, and only the first is retained.)

**Primary statistic: standard (nominal, unweighted) Cohen's kappa.** PS, DS, PPG and SC are nominal categories: there is no theoretical basis for an ordering between them, so every off-diagonal disagreement is weighted equally.

```
κ = (p_o − p_e) / (1 − p_e)
```

computed via `sklearn.metrics.cohen_kappa_score(human_labels, ai_labels, labels=['PS','DS','PPG','SC'])` (no `weights` argument).

### Confusion matrices (human rows × AI columns)

**363466802 (N=161):**

| Human \ AI | PS | DS | PPG | SC |
|---|---|---|---|---|
| PS | 28 | 1 | 3 | 7 |
| DS | 0 | 1 | 0 | 0 |
| PPG | 13 | 0 | 14 | 2 |
| SC | 5 | 1 | 12 | 74 |

**382023395 (N=113):**

| Human \ AI | PS | DS | PPG | SC |
|---|---|---|---|---|
| PS | 14 | 4 | 5 | 0 |
| DS | 2 | 17 | 0 | 0 |
| PPG | 13 | 1 | 30 | 13 |
| SC | 2 | 0 | 1 | 11 |

**25-1650208 (N=32):**

| Human \ AI | PS | DS | PPG | SC |
|---|---|---|---|---|
| PS | 6 | 1 | 0 | 0 |
| DS | 4 | 5 | 0 | 0 |
| PPG | 7 | 0 | 9 | 0 |
| SC | 0 | 0 | 0 | 0 |

**Pooled (N=306; summed cell-by-cell, not averaged — see §9):**

| Human \ AI | PS | DS | PPG | SC |
|---|---|---|---|---|
| PS | 48 | 6 | 8 | 7 |
| DS | 6 | 23 | 0 | 0 |
| PPG | 33 | 1 | 53 | 15 |
| SC | 7 | 1 | 13 | 85 |

### Results

| Report | N | Raw agreement | **κ (standard)** |
|---|---|---|---|
| 363466802 | 161 | 72.7% | **0.547** |
| 382023395 | 113 | 63.7% | **0.499** |
| 25-1650208 | 32 | 62.5% | **0.457** |
| **Pooled** | **306** | **68.3%** | **0.559** |

Interpretation band (Landis & Koch, 1977): `<0` poor, `0.00–0.20` slight, `0.21–0.40` fair, `0.41–0.60` **moderate**, `0.61–0.80` substantial, `0.81–1.00` almost perfect. All four results fall in the moderate band.

## 8. Sensitivity check — tied chunks excluded

Ties (§7): 7 human / 3 AI for 363466802; 2 human / 3 AI for 382023395; 0 human / 4 AI for 25-1650208; 9 human / 10 AI pooled (19 total chunks with a tie on at least one side, pooled N drops from 306 to 287).

| Report | κ, full sample | κ, ties excluded |
|---|---|---|
| 363466802 (N=161→151) | 0.547 | 0.600 |
| 382023395 (N=113→108) | 0.499 | 0.498 |
| 25-1650208 (N=32→28) | 0.457 | 0.526 |
| Pooled (N=306→287) | 0.559 | 0.593 |

All four move the same direction or stay flat; the primary result is not an artifact of the tie-break rule.

## 9. Alternative analysis — chunk-flexible vicinity agreement

**Motivation.** Standard kappa (§7) treats any mismatch between dominant logics as a full disagreement. But each coder's own chunk-level points (§4) already imply, for that one chunk, a secondary ("runner-up") logic in addition to the dominant one — directly extending the project's own sentence-level top-1/top-2 ensemble design up to the chunk level. This section asks a different, chunk-specific question: even when the two coders' *dominant* picks differ, was one coder's pick at least the *other* coder's own secondary emphasis for that same chunk? This makes no assumption about any relationship between logic categories in general — the "vicinity" is defined independently, chunk by chunk, from each coder's own ranking on that one chunk.

**Definitions.** For a chunk, each coder's top-2 set is the one or two logics with that coder's highest point totals on that chunk (the second slot only if it has positive points):

```
top2_set(points) = {argmax logic} ∪ ({second-highest logic} if its points > 0)
```

A chunk is a **flexible match** if `human_dominant == ai_dominant`, OR `ai_dominant ∈ human_top2_set`, OR `human_dominant ∈ ai_top2_set`.

**Chance correction by permutation.** The flexible-match criterion uses both coders' per-chunk rankings jointly, not a single label each, so there is no closed-form chance baseline (unlike standard kappa's marginal-product formula). Chance is instead estimated by permutation: within a report (or, for the pooled figure, across the full 306-chunk pool), AI chunks are randomly re-paired against human chunks 3,000 times, holding each side's own per-chunk data fixed, and the flexible-match rate is recomputed each time. The permutation mean is the Monte Carlo chance baseline `Pe_flex`, and:

```
κ_flexible = (Po_flex − Pe_flex) / (1 − Pe_flex)
```

**Built-in sanity check.** The identical permutation is also run for strict (standard) matching; its result should closely reproduce §7's closed-form standard kappa. It does for every report, confirming the permutation procedure itself before trusting the flexible-criterion result next to it:

| Report | κ standard (closed-form) | κ standard (via permutation) |
|---|---|---|
| 363466802 | 0.5473 | 0.5466 |
| 382023395 | 0.4991 | 0.4988 |
| 25-1650208 | 0.4569 | 0.4592 |
| Pooled | 0.5594 | 0.5596 |

### Results

| Report | N | Observed flexible rate | Permutation chance | **κ (flexible)** | Flexible-only chunks |
|---|---|---|---|---|---|
| 363466802 | 161 | 95.7% | 65.7% | **0.873** | 37 |
| 382023395 | 113 | 96.5% | 60.1% | **0.911** | 37 |
| 25-1650208 | 32 | 96.9% | 63.6% | **0.914** | 11 |
| Pooled | 306 | 96.1% | 58.4% | **0.906** | 85 |

"Flexible-only chunks" are chunks where the two coders' dominant picks differ but each was still within the other's own top-2 emphasis for that chunk — the dashboard's "Flexible chunk-vicinity" tab lists these explicitly, per report and pooled, so individual disagreements can be inspected rather than only summarized.

**How to read the two statistics together.** Standard kappa (§7, moderate: 0.46–0.56) is the primary, conservative measure of whether the two coders reach the identical conclusion about a chunk's single strongest logic. The flexible measure (almost perfect: 0.87–0.91) answers a related but looser question: how often are the two coders' overall emphases for a chunk at least partially compatible. The large gap between the two is itself informative — it says that most chunk-level "disagreements" are not wild mismatches but near-misses where the two coders' second-choice and first-choice simply swapped.

## 10. Report-level profile — all coded chunks, independent of chunk matching

A separate question from §6–9: considered as two independent full passes over a report, how much of each logic did the human coder find overall, and how much did the AI pipeline find overall — using **every** chunk each coder coded on its own, not only the chunks coded by both (`report_level_profile.json`, via `06_report_level_profile.py`).

| Report | Human coded / total rows | AI coded / total rows | Human PS/DS/PPG/SC | AI PS/DS/PPG/SC | Profile TVD | Cosine |
|---|---|---|---|---|---|---|
| 363466802 | 174 / 403 | 254 / 337 | 23.1/0.5/22.1/54.4% | 29.5/5.8/27.7/37.1% | 0.173 | 0.951 |
| 382023395 | 118 / 393 | 272 / 335 | 17.1/17.5/50.6/14.8% | 36.7/18.6/28.2/16.5% | 0.224 | 0.860 |
| 25-1650208 | 32 / 160 | 103 / 114 | 25.4/26.8/47.9/0.0% | 53.8/25.6/16.5/4.1% | 0.326 | 0.758 |
| Pooled | 324 / 956 | 629 / 786 | 21.2/9.2/34.8/34.8% | 37.2/15.2/25.8/21.8% | 0.220 | 0.906 |

TVD = total variation distance between the two coders' 4-category proportion vectors (0 = identical overall mix, 1 = disjoint); cosine = cosine similarity of the same two vectors (1 = identical direction). Report 25-1650208 has both the lowest human coding coverage (20% of its chunks) and the largest profile distance between coders — worth flagging as a limitation, since a 32-chunk human sample is also the least stable basis for any proportion estimate.

The dashboard's "Report-level profile" tab also shows, for each coder, a smoothed curve of where each logic concentrates across the normalized 0–100% position of the document (20 bins), separately for the human's own chunk sequence and the AI's own chunk sequence.

## 11. Pooling procedure

A pooled kappa across the 3 reports cannot be obtained by averaging the three reports' kappa values, because kappa is a nonlinear function of a confusion matrix. Pooling is instead done by summing the three reports' 4×4 confusion matrices cell-by-cell (§7's pooled matrix) and recomputing kappa directly from that summed matrix. The same approach is used for the pooled flexible-agreement permutation test (§9) and the pooled report-level profile (§10, summing raw points across reports before taking proportions).

## Files

- `repro_scripts/run_agreement.sh` — runs steps 01–09 in order (03 is not needed for the 3-report study), reading `out/out_coding/` and writing every output below to `out/agreement/`.
- `repro_scripts/01_match_and_score.py` — chunk matching, points/proportions, per-report document aggregate.
- `repro_scripts/02_chunk_kappa.py` — chunk-level dominant logic, tie flags, confusion matrix, all kappa variants (standard is `kappa_unweighted`; quadratic/linear are retained in the output only as a labeled, retired comparison — see note in §7).
- `repro_scripts/03_build_xlsx.py` — single-report Excel workbook (superseded by `05_build_xlsx_multi.py` for this 3-report study; kept for the original single-report reproduction).
- `repro_scripts/04_aggregate_multi_report.py` — pools per-report confusion matrices by summation; recomputes pooled kappa.
- `repro_scripts/05_build_xlsx_multi.py` — multi-report Excel workbook (per-report + pooled sheets, live formulas).
- `repro_scripts/06_report_level_profile.py` — §10: whole-file, unmatched logic-frequency profile and document-position curves, per coder.
- `repro_scripts/07_build_dashboard_data.py` — assembles the single JSON bundle consumed by the dashboard, merging 01/02/04/06/08/09's outputs.
- `repro_scripts/08_flexible_agreement.py` — §9: per-report standard kappa (self-contained re-derivation) and the permutation-based flexible-agreement alternative.
- `repro_scripts/09_pool_flexible_agreement.py` — §9's pooled version (pools the chunk set, then permutes across the full pool).
- `out/agreement/institutional_logics_agreement.xlsx` — per-report + pooled sheets, live formulas, standard kappa as the primary figure, zero formula errors on recalculation.
- Dashboard (built from `out/agreement/dashboard_bundle.json`; not included in this repository) — "Chunk-matched agreement" (§7–8), "Flexible chunk-vicinity" (§9), and "Report-level profile" (§10) tabs, each switchable between the 3 reports and the pooled total.
