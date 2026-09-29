#!/usr/bin/env bash
#
# Run each database size (1k, 5k, 10k, 65k, 1mil, 10mil) on the same fixed query
# vector, once per variant, and report setup, per-query, batching and
# communication numbers:
#   unbatched  -> main                   (one Piano query per candidate)
#   spread     -> partition_batch_main   (SimpleBatchPianoPIR, lists spread over partitions)
#   contiguous -> regular_batch_main  (SimpleBatchPianoPIR, original DB order)
#
# IMPORTANT: the file mapping below is a best guess from directory listings.
# Run with --dry-run first to print the resolved paths and commands.
#
# Paths: this script lives in the build directory (cmake-build-debug-remote-host-03
# or -05) and changes into it first, so all non-PIANO-RAG paths are relative to
# that build directory: binaries and parse_output.sh next to it, lists/centroids
# and the query vector in the batch_pir directory above it (".."). Only
# PIANO_RAG_DIR is absolute. --output-dir is relative to the build directory too.
#
# Usage:
#   ./run_db_size_sweep.sh --dry-run
#   ./run_db_size_sweep.sh                                        # unbatched only
#   ./run_db_size_sweep.sh --variants "unbatched spread contiguous"
#   ./run_db_size_sweep.sh --variants "spread contiguous" --batch-size 32 --sizes "1k 65k"
#   ./run_db_size_sweep.sh --repeat 20 --main-args "--threads 20"
#
# Outputs (in --output-dir):
#   timings_summary.csv      one row per size x variant
#   sweep_results.json       one record per size x variant, with the full parse_output.sh result
#   <label>_<variant>_raw.log / _raw.err.log   stdout / stderr of the binary
#
# Peak memory (max_rss_kb) is recorded when /usr/bin/time is available
# (GNU time on Linux: `sudo apt install time`; BSD time on macOS).
set -uo pipefail

SCRIPT_PATH=$(realpath "$0")
cd "$(dirname "$SCRIPT_PATH")" || { echo "cannot cd to $(dirname "$SCRIPT_PATH")" >&2; exit 1; }

# ---------------------------------------------------------------------------
# EDIT THESE to match your actual layout if any of the guesses below are wrong
# ---------------------------------------------------------------------------
PIANO_RAG_DIR="/home/ajanusze/PIANO-RAG"   # the only absolute path

# Relative to the build directory this script is in:
BATCH_PIR_DIR=".."                           # batch_pir: lists_*, centroids_*, query vector
MAIN_BIN="./main"
BATCH_MAIN_BIN="./partition_batch_main"         # spread variant
CONTIG_MAIN_BIN="./regular_batch_main"       # contiguous variant
PARSE_SCRIPT="./parse_output.sh"
RUN_CWD="."
QUERY_FILE="$BATCH_PIR_DIR/query_768d_from_k4096_centroid0.txt"

TOP_K=10

OUTPUT_DIR="db_size_sweep_results"

# label:db-size:embedding-db:centroid-mapping:centroids-file:text-db:centroid-top-k[:embedding-bin[:batch-size]]
#
# centroid-top-k is per-size (how many centroids are kept for PIR retrieval).
#
# Optional 8th field: a raw float32 embedding file (db-size x 768), passed as
# --embedding-bin instead of the JSON. Worth it for 1mil/10mil.
# Optional 9th field: PIR 1 batch size for this size (batched variants only);
# overrides --batch-size. Must be even. Leave the 8th field empty to set only
# the 9th, e.g. "...:43::64".
#
# 1k/5k/10k reuse the 65k-sized uniform_index_1024.txt; main truncates to the
# first --db-size entries automatically.
#
# 1mil/10mil centroids: both a plain and a "_new" file exist. The "_new" ones
# are used here with lists_*_new.json. Your older batched 10mil main instead
# paired centroids_10m.txt with $PIANO_RAG_DIR/prototype/data/10000000_lists.json;
# if that is the right pairing, swap it in on the 10mil line.
CONFIGS=(
  "1k:1000:$PIANO_RAG_DIR/modified_faiss_1000.json:$BATCH_PIR_DIR/lists_1k_new.json:$BATCH_PIR_DIR/centroids_1k_new.txt:$PIANO_RAG_DIR/uniform_index_1024.txt:2"
  "5k:5000:$PIANO_RAG_DIR/modified_faiss_5000.json:$BATCH_PIR_DIR/lists_5k_new.json:$BATCH_PIR_DIR/centroids_5k_new.txt:$PIANO_RAG_DIR/uniform_index_1024.txt:6"
  "10k:10000:$PIANO_RAG_DIR/modified_faiss_10000.json:$BATCH_PIR_DIR/lists_10k_new.json:$BATCH_PIR_DIR/centroids_10k_new.txt:$PIANO_RAG_DIR/uniform_index_1024.txt:6"
  "65k:65000:$PIANO_RAG_DIR/faiss.json:$BATCH_PIR_DIR/lists.json:$BATCH_PIR_DIR/centroids_65k_new.txt:$PIANO_RAG_DIR/uniform_index_1024.txt:43"
  "1mil:702873:$PIANO_RAG_DIR/modified_faiss_1000000.json:$BATCH_PIR_DIR/lists_1m_new.json:$BATCH_PIR_DIR/centroids_1m_new.txt:$PIANO_RAG_DIR/uniform_index_1000000_1024.txt:128"
  "10mil:1120486:$PIANO_RAG_DIR/modified_faiss_10000000.json:$BATCH_PIR_DIR/lists_10m_new.json:$BATCH_PIR_DIR/centroids_10m_new.txt:$PIANO_RAG_DIR/uniform_index_10000000_1024.txt:128"
)
# ---------------------------------------------------------------------------

DRY_RUN=0
SIZES_FILTER=""
VARIANTS="unbatched"
BATCH_SIZE=""
MIN_BATCHES=""
REPEAT=1
MAIN_ARGS=""
KEEP_WORK_DIR=0

usage() {
  cat >&2 <<EOF
Usage: $0 [options]
  --dry-run             print resolved paths and commands, don't run
  --sizes "L1 L2 ..."   only run these labels (1k 5k 10k 65k 1mil 10mil)
  --variants "..."      any of: unbatched spread contiguous (default: unbatched)
  --batch-size N        PIR 1 batch size for the batched variants (default: the
                        binary's own default; a 9th CONFIGS field overrides it)
  --min-batches N       pad every query to at least N PIR 1 batches (batched variants)
  --repeat N            run the query N times in one process (default: 1)
  --main-args "ARGS"    extra flags for every binary, e.g. "--threads 20"
  --output-dir DIR      where results go (default: $OUTPUT_DIR)
  --keep-work-dir       keep each run's keys/ciphertexts instead of deleting them
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --sizes) SIZES_FILTER="$2"; shift 2 ;;
    --variants) VARIANTS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --min-batches) MIN_BATCHES="$2"; shift 2 ;;
    --repeat) REPEAT="$2"; shift 2 ;;
    --main-args) MAIN_ARGS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --keep-work-dir) KEEP_WORK_DIR=1; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1" >&2; usage ;;
  esac
done

[[ "$REPEAT" =~ ^[1-9][0-9]*$ ]] || { echo "--repeat must be a positive integer" >&2; exit 1; }
[[ -z "$BATCH_SIZE" || "$BATCH_SIZE" =~ ^[0-9]*[02468]$ ]] || { echo "--batch-size must be an even number" >&2; exit 1; }
MAIN_EXTRA=()
[[ -n "$MAIN_ARGS" ]] && read -r -a MAIN_EXTRA <<< "$MAIN_ARGS"

variant_name() {
  case "$1" in
    unbatched|main) echo unbatched ;;
    spread|partition_batch_main|batched) echo spread ;;
    contiguous|regular_batch_main|regular) echo contiguous ;;
    *) return 1 ;;
  esac
}
variant_bin() {
  case "$1" in
    unbatched) echo "$MAIN_BIN" ;;
    spread) echo "$BATCH_MAIN_BIN" ;;
    contiguous) echo "$CONTIG_MAIN_BIN" ;;
  esac
}
VARIANT_LIST=()
for v in $VARIANTS; do
  n=$(variant_name "$v") || { echo "Unknown variant '$v' (use unbatched, spread, contiguous)" >&2; exit 1; }
  VARIANT_LIST+=("$n")
done
[[ ${#VARIANT_LIST[@]} -gt 0 ]] || { echo "--variants is empty" >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR_ABS=$(cd "$OUTPUT_DIR" && pwd)
SUMMARY_CSV="$OUTPUT_DIR_ABS/timings_summary.csv"
RESULTS_JSON="$OUTPUT_DIR_ABS/sweep_results.json"

# ------------------------------------------------------------------ helpers --

if [[ "$(date +%N)" =~ ^[0-9]+$ ]]; then
  now() { date +%s.%N; }
else
  now() { date +%s; }  # macOS date has no %N
fi

json_str() {
  local s=$1
  s=${s//\\/\\\\}; s=${s//\"/\\\"}; s=${s//$'\t'/ }; s=${s//$'\r'/}; s=${s//$'\n'/ }
  printf '"%s"' "$s"
}

# First numeric value for "key": in parse_output.sh's JSON. Every key read this
# way appears first at the top level or in the "breakdown" block.
get_num_field() {
  grep -m1 "\"$2\":" <<< "$1" | sed -nE "s/.*\"$2\": *([0-9.eE+-]+).*/\1/p"
}

# A value from parse_output.sh's "per_pir" block, e.g. get_pir_field "$json" pir1 client_preprocessing_time
get_pir_field() {
  awk -v p="\"$2\": {" -v k="\"$3\":" '
    index($0, p) { inblk = 1; next }
    inblk && index($0, k) { sub(/.*: */, ""); sub(/,[ \t]*$/, ""); print; exit }
    inblk && /^    }/ { inblk = 0 }
  ' <<< "$1"
}

# a / n, empty if either is missing.
avg() { awk -v a="$1" -v n="$2" 'BEGIN { if (a == "" || n == "" || n + 0 == 0) print ""; else printf "%.6g", a / n }'; }

TIME_MODE=none
if /usr/bin/time -v true > /dev/null 2>&1; then
  TIME_MODE=gnu
elif /usr/bin/time -l true > /dev/null 2>&1; then
  TIME_MODE=bsd
fi

CSV_HEADER="db_size_label,variant,db_size,centroid_top_k,batch_size,pir1_partitions,repeat,status,wall_seconds,max_rss_kb"
CSV_HEADER+=",one_time_setup_time,server_setup_time,client_setup_time"
CSV_HEADER+=",pir1_client_preprocessing_time,pir2_client_preprocessing_time"
CSV_HEADER+=",num_queries,avg_query_time,avg_centroid_time,avg_pir1_time,avg_pir2_time"
CSV_HEADER+=",avg_pir1_candidates,avg_pir1_batches,avg_pir1_dummy_batches,max_pir1_per_partition,pir1_retried,pir1_dropped"
CSV_HEADER+=",refresh_count,refresh_time"
CSV_HEADER+=",avg_pir_upload,avg_pir_download,avg_fhe_upload,avg_fhe_download"
CSV_HEADER+=",offline_initial_download,offline_refresh_download"
CSV_HEADER+=",reported_total_time,top1_index,top1_distance"
N_CSV_COLS=$(awk -F, '{ print NF }' <<< "$CSV_HEADER")

csv_short_row() {
  local row="$1" have
  have=$(awk -F, '{ print NF }' <<< "$row")
  while [[ $have -lt $N_CSV_COLS ]]; do row+=","; have=$((have + 1)); done
  echo "$row" >> "$SUMMARY_CSV"
}

json_records=()
add_json_record() {  # label variant dbsize ctopk batch status wall rss error result_json
  local batch=${5:-} rss=${8:-null} err="null" result=${10:-null}
  [[ -n "$batch" ]] || batch=null
  [[ -n "$rss" ]] || rss=null
  [[ -n "${9:-}" ]] && err=$(json_str "$9")
  json_records+=("$(printf '{\n  "label": %s,\n  "variant": %s,\n  "db_size": %s,\n  "centroid_top_k": %s,\n  "batch_size": %s,\n  "repeat": %s,\n  "status": %s,\n  "wall_seconds": %s,\n  "max_rss_kb": %s,\n  "error": %s,\n  "result": %s\n}' \
    "$(json_str "$1")" "$(json_str "$2")" "$3" "${4:-null}" "$batch" "$REPEAT" "$(json_str "$6")" "${7:-null}" "$rss" "$err" "$result")")
}

echo "$CSV_HEADER" > "$SUMMARY_CSV"
echo "Build directory: $(pwd)" >&2
[[ -f "$BATCH_PIR_DIR/piano_pir.h" ]] || echo "  ! warning: $BATCH_PIR_DIR/piano_pir.h not found - is this script in a build directory inside batch_pir?" >&2
[[ -d "$PIANO_RAG_DIR" ]] || echo "  ! warning: PIANO_RAG_DIR not found: $PIANO_RAG_DIR" >&2
echo "Query file: $QUERY_FILE (repeat $REPEAT); variants: ${VARIANT_LIST[*]}" >&2
[[ -f "$QUERY_FILE" ]] || echo "  ! warning: query file not found" >&2
[[ -f "$PARSE_SCRIPT" ]] || echo "  ! warning: parse script not found: $PARSE_SCRIPT" >&2
for n in "${VARIANT_LIST[@]}"; do
  [[ -f "$(variant_bin "$n")" ]] || echo "  ! warning: binary for $n not found: $(variant_bin "$n")" >&2
done
[[ $TIME_MODE == none ]] && echo "  (no /usr/bin/time found: max_rss_kb will be empty)" >&2
echo "" >&2

# -------------------------------------------------------------- the sweep --

for cfg in "${CONFIGS[@]}"; do
  IFS=':' read -r label dbsize embdb centmap centfile textdb ctopk embbin cfg_batch <<< "$cfg"
  embbin=${embbin:-}
  cfg_batch=${cfg_batch:-}

  if [[ -n "$SIZES_FILTER" ]] && [[ " $SIZES_FILTER " != *" $label "* ]]; then
    continue
  fi

  if ! [[ "$ctopk" =~ ^[1-9][0-9]*$ ]]; then
    echo "=== $label: skipped (centroid-top-k '$ctopk' is not set to a number - fill in CONFIGS) ===" >&2
    for n in "${VARIANT_LIST[@]}"; do
      csv_short_row "$label,$n,$dbsize,,,,$REPEAT,skipped_no_centroid_top_k"
      add_json_record "$label" "$n" "$dbsize" "" "" "skipped_no_centroid_top_k"
    done
    echo "" >&2
    continue
  fi

  echo "=== $label (db-size=$dbsize, centroid-top-k=$ctopk) ===" >&2
  if [[ -n "$embbin" ]]; then echo "  embedding-bin:    $embbin" >&2; else echo "  embedding-db:     $embdb" >&2; fi
  echo "  centroid-mapping: $centmap" >&2
  echo "  centroids-file:   $centfile" >&2
  echo "  text-db:          $textdb" >&2

  needed=("$centmap" "$centfile" "$textdb")
  if [[ -n "$embbin" ]]; then needed+=("$embbin"); else needed+=("$embdb"); fi
  any_missing=0
  for f in "${needed[@]}"; do
    if [[ ! -f "$f" ]]; then echo "  ! MISSING: $f" >&2; any_missing=1; fi
  done

  for variant in "${VARIANT_LIST[@]}"; do
    bin=$(variant_bin "$variant")
    batch=""
    if [[ $variant != unbatched ]]; then batch=${cfg_batch:-$BATCH_SIZE}; fi
    run_id="${label}_${variant}"
    work_dir="$OUTPUT_DIR_ABS/work_$run_id"
    raw_log="$OUTPUT_DIR_ABS/${run_id}_raw.log"
    err_log="$OUTPUT_DIR_ABS/${run_id}_raw.err.log"

    cmd=("$bin" --db-size "$dbsize" --centroid-mapping "$centmap" --centroids-file "$centfile"
         --text-db "$textdb" --top-k "$TOP_K" --centroid-top-k "$ctopk" --work-dir "$work_dir")
    if [[ -n "$embbin" ]]; then cmd+=(--embedding-bin "$embbin"); else cmd+=(--embedding-db "$embdb"); fi
    if [[ $REPEAT -gt 1 ]]; then
      query_list="$OUTPUT_DIR_ABS/${run_id}_query_list.txt"
      cmd+=(--query-list "$query_list")
    else
      cmd+=(--input-vector "$QUERY_FILE")
    fi
    if [[ $variant != unbatched ]]; then
      [[ -n "$batch" ]] && cmd+=(--batch-size "$batch")
      [[ -n "$MIN_BATCHES" ]] && cmd+=(--min-batches "$MIN_BATCHES")
    fi
    cmd+=(${MAIN_EXTRA[@]+"${MAIN_EXTRA[@]}"})

    echo "  --- variant $variant ($bin${batch:+, batch size $batch})" >&2
    if [[ $DRY_RUN -eq 1 ]]; then
      echo "    command (from $RUN_CWD):" >&2
      printf '     ' >&2
      printf ' %q' "${cmd[@]}" >&2
      echo "" >&2
      continue
    fi

    if [[ $any_missing -eq 1 ]]; then
      echo "    ! skipping: one or more input files are missing" >&2
      csv_short_row "$label,$variant,$dbsize,$ctopk,$batch,,$REPEAT,skipped_missing_files"
      add_json_record "$label" "$variant" "$dbsize" "$ctopk" "$batch" "skipped_missing_files"
      continue
    fi
    if [[ ! -f "$bin" ]]; then
      echo "    ! skipping: binary not found: $bin" >&2
      csv_short_row "$label,$variant,$dbsize,$ctopk,$batch,,$REPEAT,skipped_missing_binary"
      add_json_record "$label" "$variant" "$dbsize" "$ctopk" "$batch" "skipped_missing_binary" "" "" "binary not found: $bin"
      continue
    fi

    if [[ $REPEAT -gt 1 ]]; then
      : > "$query_list"
      for ((r = 0; r < REPEAT; r++)); do echo "$QUERY_FILE" >> "$query_list"; done
    fi

    rss_file="$OUTPUT_DIR_ABS/${run_id}_time.txt"
    rm -f "$rss_file"
    start=$(now)
    case $TIME_MODE in
      gnu) (cd "$RUN_CWD" && /usr/bin/time -v -o "$rss_file" "${cmd[@]}") > "$raw_log" 2> "$err_log" ;;
      bsd) (cd "$RUN_CWD" && /usr/bin/time -l "${cmd[@]}") > "$raw_log" 2> "$err_log" ;;
      *)   (cd "$RUN_CWD" && "${cmd[@]}") > "$raw_log" 2> "$err_log" ;;
    esac
    status=$?
    wall=$(awk -v s="$start" -v e="$(now)" 'BEGIN { printf "%.2f", e - s }')

    max_rss_kb=""
    if [[ $TIME_MODE == gnu && -f "$rss_file" ]]; then
      max_rss_kb=$(sed -nE 's/.*Maximum resident set size \(kbytes\): *([0-9]+).*/\1/p' "$rss_file")
    elif [[ $TIME_MODE == bsd ]]; then
      rss_bytes=$(sed -nE 's/^ *([0-9]+) +maximum resident set size.*/\1/p' "$err_log")
      [[ -n "$rss_bytes" ]] && max_rss_kb=$((rss_bytes / 1024))
    fi
    rm -f "$rss_file"
    if [[ $KEEP_WORK_DIR -eq 0 ]]; then rm -rf "$work_dir"; fi

    if [[ $status -ne 0 ]]; then
      err=$(grep -m1 '^fatal: ' "$err_log" || grep -v 'maximum resident\|^ *[0-9]\+  ' "$err_log" | tail -n 3 | tr '\n' ' ')
      reason="failed_exit_$status"
      if [[ $status -eq 137 || $status -eq 9 ]]; then
        reason="killed_sigkill"
        err="killed by SIGKILL (likely out of memory; peak RSS ${max_rss_kb:-unknown} kB). $err"
      fi
      echo "    ! exited $status after ${wall}s: $err" >&2
      echo "      see $raw_log and $err_log" >&2
      csv_short_row "$label,$variant,$dbsize,$ctopk,$batch,,$REPEAT,$reason,$wall,$max_rss_kb"
      parsed=$(bash "$PARSE_SCRIPT" "$raw_log" 2> /dev/null)
      add_json_record "$label" "$variant" "$dbsize" "$ctopk" "$batch" "$reason" "$wall" "$max_rss_kb" "$err" "${parsed:-null}"
      continue
    fi

    parsed=$(bash "$PARSE_SCRIPT" "$raw_log")
    if [[ -z "$parsed" ]]; then
      echo "    ! parse_output.sh produced no output - see $raw_log" >&2
      csv_short_row "$label,$variant,$dbsize,$ctopk,$batch,,$REPEAT,failed_parse,$wall,$max_rss_kb"
      add_json_record "$label" "$variant" "$dbsize" "$ctopk" "$batch" "failed_parse" "$wall" "$max_rss_kb" "parse_output.sh produced no output"
      continue
    fi

    run_status="ok"
    grep -q '"completed": true' <<< "$parsed" || run_status="incomplete"
    reported_variant=$(sed -nE 's/^  "variant": "([a-z_]+)".*/\1/p' <<< "$parsed" | head -1)
    case "$variant" in unbatched) expect=unbatched ;; spread) expect=batched_spread ;; contiguous) expect=batched_contiguous ;; esac
    if [[ -n "$reported_variant" && "$reported_variant" != "$expect" ]]; then
      echo "    ! WARNING: $(basename "$bin") reports variant $reported_variant, expected $expect" \
           "- check which source each binary was built from" >&2
    fi

    n=$(get_num_field "$parsed" "num_queries")
    used_batch=$(get_num_field "$parsed" "pir1_batch_size")
    partitions=$(get_num_field "$parsed" "pir1_partitions")
    one_time_setup_time=$(get_num_field "$parsed" "one_time_setup_time")
    server_setup_time=$(get_num_field "$parsed" "server_setup_time")
    client_setup_time=$(get_num_field "$parsed" "client_setup_time")
    pir1_prep=$(get_pir_field "$parsed" pir1 client_preprocessing_time)
    pir2_prep=$(get_pir_field "$parsed" pir2 client_preprocessing_time)
    avg_query_time=$(get_num_field "$parsed" "avg_query_time")
    avg_centroid_time=$(avg "$(get_num_field "$parsed" "centroid_time")" "$n")
    avg_pir1_time=$(avg "$(get_num_field "$parsed" "pir1_time")" "$n")
    avg_pir2_time=$(avg "$(get_num_field "$parsed" "pir2_time")" "$n")
    avg_cand=$(avg "$(get_num_field "$parsed" "pir1_candidates")" "$n")
    avg_batches=$(avg "$(get_num_field "$parsed" "pir1_batches")" "$n")
    avg_dummy=$(avg "$(get_num_field "$parsed" "pir1_dummy_batches")" "$n")
    max_pp=$(get_num_field "$parsed" "pir1_max_per_partition")
    retried=$(get_num_field "$parsed" "pir1_retried")
    dropped=$(get_num_field "$parsed" "pir1_dropped")
    refresh_count=$(get_num_field "$parsed" "refresh_count")
    refresh_time=$(get_num_field "$parsed" "refresh_time")
    avg_pir_upload=$(avg "$(get_num_field "$parsed" "pir_upload")" "$n")
    avg_pir_download=$(avg "$(get_num_field "$parsed" "pir_download")" "$n")
    avg_fhe_upload=$(avg "$(get_num_field "$parsed" "fhe_upload")" "$n")
    avg_fhe_download=$(avg "$(get_num_field "$parsed" "fhe_download")" "$n")
    offline_initial=$(get_num_field "$parsed" "pir_offline_initial_download")
    offline_refresh=$(get_num_field "$parsed" "pir_offline_refresh_download")
    reported_total_time=$(get_num_field "$parsed" "reported_total_time")
    top1=$(grep -m1 '"index":' <<< "$parsed")
    top1_index=$(sed -nE 's/.*"index": ([0-9]+).*/\1/p' <<< "$top1")
    top1_distance=$(sed -nE 's/.*"distance": ([0-9.eE+-]+).*/\1/p' <<< "$top1")

    echo "    $run_status  wall=${wall}s  peak_rss=${max_rss_kb:-?}kB  one_time_setup=${one_time_setup_time}s (server ${server_setup_time}s, client ${client_setup_time}s)" >&2
    echo "      avg over $n query(s): query=${avg_query_time}s centroid=${avg_centroid_time}s pir1=${avg_pir1_time}s pir2=${avg_pir2_time}s  up=${avg_pir_upload} down=${avg_pir_download}" >&2
    if [[ $variant != unbatched ]]; then
      echo "      batching: batch size ${used_batch}, ${partitions} partitions, avg ${avg_batches} batches (+${avg_dummy} dummy) for ${avg_cand} candidates, max ${max_pp} in one partition, ${retried} retried, ${dropped} dropped" >&2
    fi
    echo "      refreshes=${refresh_count}  top1=[${top1_index:-none} ${top1_distance}]" >&2
    grep -m2 '^Truncating ' "$raw_log" | sed 's/^/      /' >&2

    echo "$label,$variant,$dbsize,$ctopk,$used_batch,$partitions,$REPEAT,$run_status,$wall,$max_rss_kb,$one_time_setup_time,$server_setup_time,$client_setup_time,$pir1_prep,$pir2_prep,$n,$avg_query_time,$avg_centroid_time,$avg_pir1_time,$avg_pir2_time,$avg_cand,$avg_batches,$avg_dummy,$max_pp,$retried,$dropped,$refresh_count,$refresh_time,$avg_pir_upload,$avg_pir_download,$avg_fhe_upload,$avg_fhe_download,$offline_initial,$offline_refresh,$reported_total_time,$top1_index,$top1_distance" >> "$SUMMARY_CSV"
    add_json_record "$label" "$variant" "$dbsize" "$ctopk" "$used_batch" "$run_status" "$wall" "$max_rss_kb" "" "$parsed"
  done
  echo "" >&2
done

if [[ $DRY_RUN -eq 1 ]]; then
  echo "Dry run only - nothing was executed." >&2
  exit 0
fi

n_records=${#json_records[@]}
{
  printf '[\n'
  for ((idx = 0; idx < n_records; idx++)); do
    printf '%s' "${json_records[$idx]}"
    if [[ $idx -lt $((n_records - 1)) ]]; then printf ',\n'; else printf '\n'; fi
  done
  printf ']\n'
} > "$RESULTS_JSON"

echo "Done. Summary: $SUMMARY_CSV" >&2
echo "Full results: $RESULTS_JSON" >&2
echo "" >&2
column -s, -t "$SUMMARY_CSV" >&2 2>/dev/null || cat "$SUMMARY_CSV" >&2