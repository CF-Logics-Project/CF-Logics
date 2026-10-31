#!/usr/bin/env bash
# Run the full human-AI agreement analysis (steps 01-09, see logic_scoring_evaluation.md).
# Reads the coding results in out/out_coding/ and writes everything to out/agreement/,
# paths relative to the repository root. Run from anywhere:
#     bash repro_scripts/run_agreement.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$REPO_ROOT/repro_scripts"
CODING="$REPO_ROOT/out/out_coding"
OUT="$REPO_ROOT/out/agreement"
REPORTS=(363466802_2021 382023395_2022 25-1650208_2021)
CODES=R1,R2,R3
N_PERM=3000

report_args=()
pool_args=()
for rid in "${REPORTS[@]}"; do
    human_csv=$(ls "$CODING"/human/*"$rid".csv)
    ai_csv="$CODING/ai/output_$rid.csv"
    rdir="$OUT/$rid"
    mkdir -p "$rdir"

    python3 "$SCRIPTS/01_match_and_score.py" --human-csv "$human_csv" --ai-csv "$ai_csv" --outdir "$rdir"
    python3 "$SCRIPTS/02_chunk_kappa.py" --chunk-level-csv "$rdir/chunk_level.csv" --outdir "$rdir"
    python3 "$SCRIPTS/06_report_level_profile.py" --human-csv "$human_csv" --ai-csv "$ai_csv" --report-id "$rid" --outdir "$rdir"
    python3 "$SCRIPTS/08_flexible_agreement.py" --chunk-level-csv "$rdir/chunk_level.csv" \
        --chunk-dominant-csv "$rdir/chunk_dominant.csv" --report-id "$rid" --n-perm "$N_PERM" --outdir "$rdir"

    report_args+=(--report "$rid=$rdir")
    pool_args+=(--chunk-level-csv "$rdir/chunk_level.csv" --chunk-dominant-csv "$rdir/chunk_dominant.csv")
done

python3 "$SCRIPTS/04_aggregate_multi_report.py" "${report_args[@]}" --outdir "$OUT"
python3 "$SCRIPTS/09_pool_flexible_agreement.py" "${pool_args[@]}" --n-perm "$N_PERM" --outdir "$OUT"
python3 "$SCRIPTS/05_build_xlsx_multi.py" "${report_args[@]}" --pooled-summary-json "$OUT/pooled_summary.json" \
    --flex-pooled-json "$OUT/flexible_agreement_pooled.json" --outfile "$OUT/institutional_logics_agreement.xlsx"
python3 "$SCRIPTS/07_build_dashboard_data.py" "${report_args[@]}" --codes "$CODES" \
    --flex-pooled "$OUT/flexible_agreement_pooled.json" --pooled-summary-json "$OUT/pooled_summary.json" \
    --out "$OUT/dashboard_bundle.json"
