#!/usr/bin/env bash
#
# Run main (or one of the batched builds) over every query file listed in a
# manifest (as produced by make_query_files.py) and produce:
#   - batched_output.json            one record per query: per-query metrics + top_k
#   - topk_summary.json              dataset, query id and top_k only
#   - batched_output.runs.json       one record per main invocation: setup times,
#                                    offline/online totals, refreshes (full
#                                    parse_output.sh result for that invocation)
#   - batched_output.progress.jsonl  live log, one line per finished query
#   - batched_output.main.log / .main.err.log   raw stdout / stderr of main
#   - batched_output.failures.json   only if something failed
#
# By default all queries go to ONE main process via --query-list, so database
# loading, Piano preprocessing and keygen happen once, and Piano refreshes are
# exercised across queries. Output is streamed: each query is parsed and logged
# as soon as main prints its "Total time" line.
#
# If main dies part-way, the query it was working on is recorded as failed and
# a new main process is started for the remaining queries. A failure during
# setup (before the first query) stops the batch, since it would repeat.
#
# --queries-per-run N starts a fresh main every N queries; N=1 reproduces the
# old one-process-per-query behaviour (full setup for every query).
#
# Usage (from the build directory; the defaults cover --manifest, --main,
# --parse-script, --cwd and --bin-dir):
#   ./run_batch.sh --top-k 10 --centroid-top-k 10 \
#       --main-args "--db-size 65000 --threads 20" \
#       --output results/batched_output.json --topk-output results/topk_summary.json
#
# Variants: --variants "unbatched spread contiguous" runs the whole manifest once
# per binary, found in --bin-dir (default: the directory of --main):
#   unbatched  -> main                   (one Piano query per candidate)
#   spread     -> partition_batch_main   (SimpleBatchPianoPIR, lists spread over partitions)
#   contiguous -> regular_batch_main  (SimpleBatchPianoPIR, original DB order)
# (binary names are set in UNBATCHED_BIN_NAME / SPREAD_BIN_NAME / CONTIG_BIN_NAME below)
# Each variant writes its own files, e.g. batched_output_spread.json,
# topk_summary_spread.json, batched_output_spread.runs.json ...
# --batch-size / --min-batches are passed to the batched binaries (main ignores them).
#
#   ./run_batch.sh --variants "unbatched spread contiguous" --batch-size 32 \
#       --main-args "--db-size 65000 --threads 20" --output results/out.json \
#       --topk-output results/topk.json
#
# Paths: this script lives in the build directory (cmake-build-debug-remote-host-03
# or -05) and changes into it first, so EVERY relative path - the defaults below
# and any you pass - is relative to that build directory, wherever you run it
# from. The binaries, parse_output.sh and queries/ are expected next to it, and
# main's own relative defaults (../lists.json, ../centroids_65k_new.txt) point
# into the batch_pir directory above it. Output directories are created.
#
# Needs bash 3.2+ (works with macOS /bin/bash) and the parse_output.sh that
# matches the current main.cpp output.
set -uo pipefail

ORIG_ARGS=("$@")
SCRIPT_PATH=$(realpath "$0")
cd "$(dirname "$SCRIPT_PATH")" || { echo "cannot cd to $(dirname "$SCRIPT_PATH")" >&2; exit 1; }

# Binary names for --variants, looked up in --bin-dir.
UNBATCHED_BIN_NAME="main"
SPREAD_BIN_NAME="partition_batch_main"
CONTIG_BIN_NAME="regular_batch_main"

MANIFEST="queries/manifest.jsonl"
MAIN_BIN="./main"
PARSE_SCRIPT="./parse_output.sh"
RUN_CWD="."
TOP_K="10"
CENTROID_TOP_K="10"
MAIN_ARGS=""
QUERIES_PER_RUN="0"
WORK_DIR=""
KEEP_WORK_DIR=0
OUTPUT="results/batched_output.json"
TOPK_OUTPUT="results/topk_summary.json"
PROGRESS_LOG=""
DATASETS_FILTER=""
VARIANTS=""
BIN_DIR=""
BATCH_SIZE=""
MIN_BATCHES=""
EXPECT_VARIANT=""

usage() {
  cat >&2 <<EOF
Usage: $0 [options]
  --manifest FILE          manifest.jsonl from make_query_files.py (default: $MANIFEST)
  --main PATH              path to the main binary (default: $MAIN_BIN)
  --parse-script PATH      path to parse_output.sh (default: $PARSE_SCRIPT)
  --cwd DIR                directory to run main from (default: $RUN_CWD)
  --top-k N                final top-k, passed to main as --top-k (default: $TOP_K)
  --centroid-top-k N       centroids to probe, passed as --centroid-top-k (default: $CENTROID_TOP_K)
  --main-args "ARGS"       extra flags for main, e.g. "--mac --db-size 200000"
                           (split on whitespace; no quoting inside)
  --queries-per-run N      start a fresh main every N queries; 0 = all in one run,
                           1 = one run per query (default: $QUERIES_PER_RUN)
  --work-dir DIR           main's --work-dir, relative to --cwd (default: a fresh
                           batch_work.<pid> directory, deleted at the end)
  --keep-work-dir          don't delete the default work dir at the end
  --output FILE            per-query results (default: $OUTPUT)
  --topk-output FILE       top-k summary (default: $TOPK_OUTPUT)
  --progress-log FILE      live JSONL log (default: "<output>.progress.jsonl")
  --datasets "d1 d2 ..."   only run these dataset names (default: all)
  --variants "v1 v2 ..."   run the manifest once per variant: unbatched, spread,
                           contiguous (binaries main, partition_batch_main,
                           regular_batch_main in --bin-dir)
  --bin-dir DIR            where the variant binaries are (default: dir of --main, i.e. .)
  All relative paths are relative to this script's (build) directory.
  --batch-size N           PIR 1 batch size for the batched binaries
  --min-batches N          pad each query to at least N PIR 1 batches (batched binaries)
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="$2"; shift 2 ;;
    --main) MAIN_BIN="$2"; shift 2 ;;
    --parse-script) PARSE_SCRIPT="$2"; shift 2 ;;
    --cwd) RUN_CWD="$2"; shift 2 ;;
    --top-k) TOP_K="$2"; shift 2 ;;
    --centroid-top-k) CENTROID_TOP_K="$2"; shift 2 ;;
    --main-args) MAIN_ARGS="$2"; shift 2 ;;
    --queries-per-run) QUERIES_PER_RUN="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --keep-work-dir) KEEP_WORK_DIR=1; shift ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --topk-output) TOPK_OUTPUT="$2"; shift 2 ;;
    --progress-log) PROGRESS_LOG="$2"; shift 2 ;;
    --datasets) DATASETS_FILTER="$2"; shift 2 ;;
    --variants) VARIANTS="$2"; shift 2 ;;
    --bin-dir) BIN_DIR="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --min-batches) MIN_BATCHES="$2"; shift 2 ;;
    --expect-variant) EXPECT_VARIANT="$2"; shift 2 ;;   # set internally by --variants
    -h|--help) usage ;;
    *) echo "Unknown option: $1" >&2; usage ;;
  esac
done

[[ -f "$MANIFEST" ]] || { echo "Manifest not found: $MANIFEST" >&2; exit 1; }

# ----------------------------------------------------- several variants --
# Re-runs this script once per variant with --main/--output/--topk-output
# (and --progress-log, if given) pointed at that variant.
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
    unbatched) echo "$BIN_DIR/$UNBATCHED_BIN_NAME" ;;
    spread) echo "$BIN_DIR/$SPREAD_BIN_NAME" ;;
    contiguous) echo "$BIN_DIR/$CONTIG_BIN_NAME" ;;
  esac
}

if [[ -n "$VARIANTS" ]]; then
  [[ -n "$BIN_DIR" ]] || BIN_DIR=$(dirname "$MAIN_BIN")
  names=()
  for v in $VARIANTS; do
    n=$(variant_name "$v") || { echo "Unknown variant '$v' (use unbatched, spread, contiguous)" >&2; exit 1; }
    b=$(variant_bin "$n")
    [[ -f "$b" ]] || { echo "binary for variant $n not found: $b (build it or set --bin-dir)" >&2; exit 1; }
    names+=("$n")
  done
  pass=()
  skip=0
  for a in ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"}; do
    if [[ $skip -eq 1 ]]; then skip=0; continue; fi
    case "$a" in --variants|--bin-dir|--expect-variant) skip=1; continue ;; esac
    pass+=("$a")
  done
  overall=0
  summary=()
  for n in "${names[@]}"; do
    vout="${OUTPUT%.json}_$n.json"
    case "$n" in unbatched) expect=unbatched ;; spread) expect=batched_spread ;; contiguous) expect=batched_contiguous ;; esac
    extra=(--main "$(variant_bin "$n")" --output "$vout" --topk-output "${TOPK_OUTPUT%.json}_$n.json"
           --expect-variant "$expect")
    [[ -n "$PROGRESS_LOG" ]] && extra+=(--progress-log "${PROGRESS_LOG%.jsonl}_$n.jsonl")
    echo "" >&2
    echo "========== variant: $n ($(variant_bin "$n")) ==========" >&2
    bash "$SCRIPT_PATH" ${pass[@]+"${pass[@]}"} "${extra[@]}"
    st=$?
    summary+=("  $n: exit $st, results in $vout")
    [[ $st -gt $overall ]] && overall=$st
  done
  echo "" >&2
  echo "Variants finished:" >&2
  printf '%s\n' "${summary[@]}" >&2
  exit $overall
fi

[[ -f "$MAIN_BIN" ]] || { echo "main binary not found: $MAIN_BIN" >&2; exit 1; }
[[ -f "$PARSE_SCRIPT" ]] || { echo "parse script not found: $PARSE_SCRIPT" >&2; exit 1; }
[[ -d "$RUN_CWD" ]] || { echo "--cwd directory not found: $RUN_CWD" >&2; exit 1; }
[[ "$QUERIES_PER_RUN" =~ ^[0-9]+$ ]] || { echo "--queries-per-run must be a non-negative integer" >&2; exit 1; }

MAIN_BIN_ABS=$(realpath "$MAIN_BIN")
PARSE_SCRIPT_ABS=$(realpath "$PARSE_SCRIPT")
BASE="${OUTPUT%.json}"
RUNS_OUTPUT="$BASE.runs.json"
FAILURES_OUTPUT="$BASE.failures.json"
MAIN_LOG="$BASE.main.log"
MAIN_ERR_LOG="$BASE.main.err.log"
[[ -n "$PROGRESS_LOG" ]] || PROGRESS_LOG="$BASE.progress.jsonl"
mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$TOPK_OUTPUT")" "$(dirname "$PROGRESS_LOG")" ||
  { echo "cannot create output directories" >&2; exit 1; }

MAIN_EXTRA=()
[[ -n "$MAIN_ARGS" ]] && read -r -a MAIN_EXTRA <<< "$MAIN_ARGS"
[[ -n "$BATCH_SIZE" ]] && MAIN_EXTRA+=(--batch-size "$BATCH_SIZE")
[[ -n "$MIN_BATCHES" ]] && MAIN_EXTRA+=(--min-batches "$MIN_BATCHES")
RUN_VARIANT="unknown"   # set from main's Config: line

DEFAULT_WORK_DIR=0
if [[ -z "$WORK_DIR" ]]; then
  WORK_DIR="batch_work.$$"
  DEFAULT_WORK_DIR=1
fi

TMP_DIR=$(mktemp -d)
main_pid=""
cleanup() {
  [[ -n "$main_pid" ]] && kill "$main_pid" 2>/dev/null
  rm -rf "$TMP_DIR"
  if [[ $DEFAULT_WORK_DIR -eq 1 && $KEEP_WORK_DIR -eq 0 && -n "$WORK_DIR" ]]; then
    rm -rf "${RUN_CWD:?}/$WORK_DIR"
  fi
}
trap cleanup EXIT
trap 'echo "Interrupted" >&2; exit 130' INT TERM

: > "$PROGRESS_LOG"
: > "$MAIN_LOG"
: > "$MAIN_ERR_LOG"
echo "Live progress: tail -f $PROGRESS_LOG" >&2

# ------------------------------------------------------------------ helpers --

# Seconds since the epoch, with fractions where date supports %N (not macOS).
if [[ "$(date +%N)" =~ ^[0-9]+$ ]]; then
  now() { date +%s.%N; }
else
  now() { date +%s; }
fi
elapsed_since() { awk -v s="$1" -v e="$(now)" 'BEGIN { printf "%.2f", e - s }'; }

json_str() {
  local s=$1
  s=${s//\\/\\\\}
  s=${s//\"/\\\"}
  s=${s//$'\t'/ }
  s=${s//$'\r'/}
  s=${s//$'\n'/ }
  printf '"%s"' "$s"
}

# First numeric value for "key": in a block of JSON text.
get_num_field() {
  grep -m1 "\"$2\":" <<< "$1" | sed -E "s/.*\"$2\": *([0-9.eE+-]+).*/\1/"
}

write_json_array() {
  local outfile="$1"; shift
  local n=$# k=0 rec
  {
    printf '[\n'
    for rec in "$@"; do
      k=$((k + 1))
      printf '%s' "$rec"
      if [[ $k -lt $n ]]; then printf ',\n'; else printf '\n'; fi
    done
    printf ']\n'
  } > "$outfile"
}

combined_records=()
topk_records=()
run_records=()
failure_records=()
completed=0
failed_count=0
total_refreshes=0

record_failure() {  # dataset id file error [wall] [run]
  local dataset=$1 qid=$2 qfile=$3 err=$4 wall=${5:-null} run=${6:-null}
  err=$(cut -c1-500 <<< "$err")
  echo "  ! $dataset id=$qid: $err" >&2
  failure_records+=("$(printf '  { "dataset": %s, "id": %s, "file": %s, "run": %s, "error": %s }' \
    "$(json_str "$dataset")" "${qid:-null}" "$(json_str "$qfile")" "$run" "$(json_str "$err")")")
  printf '{"dataset": %s, "id": %s, "status": "failed", "run": %s, "wall_seconds": %s, "error": %s}\n' \
    "$(json_str "$dataset")" "${qid:-null}" "$run" "$wall" "$(json_str "$err")" >> "$PROGRESS_LOG"
  failed_count=$((failed_count + 1))
}

# -------------------------------------------------------- build job list --

job_dataset=()
job_id=()
job_file=()
job_abs=()

while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" ]] && continue
  dataset=$(sed -n 's/.*"dataset": *"\([^"]*\)".*/\1/p' <<< "$line")
  qid=$(sed -n 's/.*"id": *\([0-9][0-9]*\).*/\1/p' <<< "$line")
  qfile=$(sed -n 's/.*"file": *"\([^"]*\)".*/\1/p' <<< "$line")
  if [[ -n "$DATASETS_FILTER" ]] && [[ " $DATASETS_FILTER " != *" $dataset "* ]]; then
    continue
  fi
  if [[ ! -f "$qfile" ]]; then
    record_failure "$dataset" "$qid" "$qfile" "query file missing"
    continue
  fi
  job_dataset+=("$dataset")
  job_id+=("$qid")
  job_file+=("$qfile")
  job_abs+=("$(realpath "$qfile")")
done < "$MANIFEST"

njobs=${#job_abs[@]}
grand_total=$((njobs + failed_count))
echo "$njobs queries to run ($failed_count missing), work dir $RUN_CWD/$WORK_DIR" >&2
run_start=$(now)

# ------------------------------------------------- per-query finalisation --

# Parses one query's block of main output (from its "=== Query" header to its
# "Total time" line) with parse_output.sh and records the result.
finalize_query() {  # job_index block start_time run
  local j=$1 block=$2 qstart=$3 run=$4
  local wall parsed perr qobj fields entries topk_json
  wall=$(elapsed_since "$qstart")
  perr="$TMP_DIR/parse.err"

  parsed=$(bash "$PARSE_SCRIPT_ABS" 2> "$perr" <<< "$block")
  if [[ $? -ne 0 || -z "$parsed" ]]; then
    record_failure "${job_dataset[$j]}" "${job_id[$j]}" "${job_file[$j]}" \
      "parse_output.sh failed: $(tr '\n' ' ' < "$perr")" "$wall" "$run"
    return
  fi

  # queries[0] of the parse output: the lines between "    {" and "    }".
  qobj=$(sed -n '/^  "queries": \[/,/^  \],*$/p' <<< "$parsed" | sed '1d;$d')
  fields=$(sed '1d;$d' <<< "$qobj")
  if [[ -z "$fields" ]]; then
    record_failure "${job_dataset[$j]}" "${job_id[$j]}" "${job_file[$j]}" \
      "parse_output.sh returned no per-query metrics" "$wall" "$run"
    return
  fi

  entries=$(sed -n '/"top_k": \[/,$p' <<< "$parsed" | grep '"index":' |
            sed -E 's/^[[:space:]]*//; s/,[[:space:]]*$//')
  topk_json=""
  if [[ -n "$entries" ]]; then
    topk_json=$(awk '{ printf "%s    %s", (NR > 1 ? ",\n" : ""), $0 } END { printf "\n" }' <<< "$entries")
  fi

  local ds; ds=$(json_str "${job_dataset[$j]}")
  local qid=${job_id[$j]:-null}

  combined_records+=("$(printf '{\n  "dataset": %s,\n  "id": %s,\n  "query_file": %s,\n  "variant": "%s",\n  "status": "ok",\n  "run": %s,\n  "wall_seconds": %s,\n  "metrics": {\n%s\n  },\n  "top_k": [\n%s  ]\n}' \
    "$ds" "$qid" "$(json_str "${job_file[$j]}")" "$RUN_VARIANT" "$run" "$wall" "$fields" "$topk_json")")
  topk_records+=("$(printf '{\n  "dataset": %s,\n  "id": %s,\n  "variant": "%s",\n  "top_k": [\n%s  ]\n}' "$ds" "$qid" "$RUN_VARIANT" "$topk_json")")

  local total_time query_time centroid_time pir1_time pir2_time pir_upload pir_download refreshes top1 top1_index top1_distance
  local pir1_batches pir1_maxpp
  total_time=$(get_num_field "$fields" "total_time")
  query_time=$(get_num_field "$fields" "query_time")
  centroid_time=$(get_num_field "$fields" "centroid_time")
  pir1_time=$(get_num_field "$fields" "pir1_time")
  pir2_time=$(get_num_field "$fields" "pir2_time")
  pir_upload=$(get_num_field "$fields" "pir_upload")
  pir_download=$(get_num_field "$fields" "pir_download")
  refreshes=$(get_num_field "$fields" "refresh_count")
  pir1_batches=$(get_num_field "$fields" "pir1_batches")
  pir1_maxpp=$(get_num_field "$fields" "pir1_max_per_partition")
  top1=$(head -1 <<< "$entries")
  top1_index=$(sed -nE 's/.*"index": ([0-9]+).*/\1/p' <<< "$top1")
  top1_distance=$(sed -nE 's/.*"distance": ([0-9.eE+-]+).*/\1/p' <<< "$top1")

  completed=$((completed + 1))
  total_refreshes=$((total_refreshes + ${refreshes:-0}))
  echo "  ok  wall=${wall}s total_time=${total_time} centroid=${centroid_time} pir1=${pir1_time} pir2=${pir2_time} up=${pir_upload} down=${pir_download} batches=${pir1_batches:-0} max/part=${pir1_maxpp:-0} refreshes=${refreshes:-0} top1=[${top1_index:-none} ${top1_distance:-}]  (ok=${completed} failed=${failed_count} of ${grand_total}, elapsed=$(elapsed_since "$run_start")s)" >&2

  printf '{"dataset": %s, "id": %s, "variant": "%s", "status": "ok", "run": %s, "wall_seconds": %s, "total_time": %s, "query_time": %s, "centroid_time": %s, "pir1_time": %s, "pir2_time": %s, "pir_upload": %s, "pir_download": %s, "pir1_batches": %s, "pir1_max_per_partition": %s, "refresh_count": %s, "top1_index": %s, "top1_distance": %s}\n' \
    "$ds" "$qid" "$RUN_VARIANT" "$run" "$wall" "${total_time:-null}" "${query_time:-null}" "${centroid_time:-null}" \
    "${pir1_time:-null}" "${pir2_time:-null}" "${pir_upload:-null}" "${pir_download:-null}" \
    "${pir1_batches:-0}" "${pir1_maxpp:-0}" "${refreshes:-0}" "${top1_index:-null}" "${top1_distance:-null}" >> "$PROGRESS_LOG"
}

# ------------------------------------------------------------ main runs --

re_header='^=== Query ([0-9]+)/([0-9]+): '
fifo="$TMP_DIR/main.fifo"
mkfifo "$fifo"
run=0
next=0
stop=0

while [[ $next -lt $njobs && $stop -eq 0 ]]; do
  run=$((run + 1))
  end=$njobs
  if [[ $QUERIES_PER_RUN -gt 0 && $((next + QUERIES_PER_RUN)) -lt $njobs ]]; then
    end=$((next + QUERIES_PER_RUN))
  fi

  list_file="$TMP_DIR/query_list.$run.txt"
  : > "$list_file"
  for ((j = next; j < end; j++)); do printf '%s\n' "${job_abs[$j]}" >> "$list_file"; done

  seg_log="$TMP_DIR/run.$run.log"
  err_log="$TMP_DIR/run.$run.err"
  : > "$seg_log"
  echo "[run $run] queries $((next + 1))-$end of $njobs" >&2
  printf '\n##### run %s: queries %s-%s #####\n' "$run" "$((next + 1))" "$end" >> "$MAIN_LOG"

  ( cd "$RUN_CWD" && exec "$MAIN_BIN_ABS" --query-list "$list_file" --work-dir "$WORK_DIR" \
      --top-k "$TOP_K" --centroid-top-k "$CENTROID_TOP_K" ${MAIN_EXTRA[@]+"${MAIN_EXTRA[@]}"} ) \
      > "$fifo" 2> "$err_log" &
  main_pid=$!

  cur=-1          # job index of the query currently being printed, -1 if none
  done_upto=$next # first job index not yet finished in this run
  saw_query=0
  block=""
  q_start=""
  exec 5>> "$seg_log"
  while IFS= read -r ml || [[ -n "$ml" ]]; do
    printf '%s\n' "$ml" >&5
    if [[ $ml =~ $re_header ]]; then
      cur=$((next + BASH_REMATCH[1] - 1))
      saw_query=1
      block="$ml"$'\n'
      q_start=$(now)
      echo "[$((cur + 1))/$njobs] ${job_dataset[$cur]} id=${job_id[$cur]} (${job_file[$cur]})" >&2
      continue
    fi
    case "$ml" in
      "[refresh]"*" completed in "*) echo "  $ml" >&2 ;;
      "One-time Setup time "*) echo "  [run $run] setup done in ${ml#One-time Setup time }s" >&2 ;;
      "Config: "*)
        RUN_VARIANT="unbatched"
        if [[ $ml == *"PIR 1 batched:"* ]]; then
          if [[ $ml == *"(contiguous)"* || $ml == *"spread-clusters off"* ]]; then
            RUN_VARIANT="batched_contiguous"
          else
            RUN_VARIANT="batched_spread"
          fi
        fi
        echo "  [run $run] variant $RUN_VARIANT: ${ml#Config: }" >&2
        if [[ -n "$EXPECT_VARIANT" && "$RUN_VARIANT" != "$EXPECT_VARIANT" ]]; then
          echo "  ! WARNING: $(basename "$MAIN_BIN") reports variant $RUN_VARIANT, expected $EXPECT_VARIANT" \
               "- check which source each binary was built from" >&2
        fi ;;
    esac
    if [[ $cur -ge 0 ]]; then
      block+="$ml"$'\n'
      if [[ $ml == "Total time "* ]]; then
        finalize_query "$cur" "$block" "$q_start" "$run"
        done_upto=$((cur + 1))
        cur=-1
        block=""
      fi
    fi
  done < "$fifo"
  exec 5>&-
  wait "$main_pid"
  main_status=$?
  main_pid=""

  cat "$seg_log" >> "$MAIN_LOG"
  { printf '\n##### run %s (exit %s) #####\n' "$run" "$main_status"; cat "$err_log"; } >> "$MAIN_ERR_LOG"
  err_tail=$(grep -m1 '^fatal: ' "$err_log" || tail -n 3 "$err_log" | tr '\n' ' ')

  # Run-level record: the whole invocation parsed as one log.
  run_parsed=$(bash "$PARSE_SCRIPT_ABS" < "$seg_log" 2>/dev/null)
  if [[ -n "$run_parsed" ]]; then
    run_records+=("$(printf '{\n  "run": %s,\n  "first_query": %s,\n  "last_query": %s,\n  "exit_status": %s,\n%s\n}' \
      "$run" "$((next + 1))" "$end" "$main_status" "$(sed '1d;$d' <<< "$run_parsed")")")
  fi

  if [[ $cur -ge 0 ]]; then
    # main stopped in the middle of a query: that query failed, go on after it.
    record_failure "${job_dataset[$cur]}" "${job_id[$cur]}" "${job_file[$cur]}" \
      "main exited $main_status during this query: $err_tail" "$(elapsed_since "$q_start")" "$run"
    done_upto=$((cur + 1))
  elif [[ $saw_query -eq 0 && $main_status -ne 0 ]]; then
    echo "  ! [run $run] main failed during setup (exit $main_status): $err_tail" >&2
    echo "  ! stopping: setup failures repeat for every run" >&2
    for ((j = next; j < njobs; j++)); do
      record_failure "${job_dataset[$j]}" "${job_id[$j]}" "${job_file[$j]}" \
        "not run: main failed during setup (exit $main_status): $err_tail" null "$run"
    done
    done_upto=$njobs
    stop=1
  elif [[ $main_status -ne 0 ]]; then
    echo "  ! [run $run] main exited $main_status after its last completed query: $err_tail" >&2
  fi

  # Queries main never reached (e.g. it exited 0 early): mark them and move on.
  if [[ $stop -eq 0 && $main_status -eq 0 && $done_upto -lt $end ]]; then
    for ((j = done_upto; j < end; j++)); do
      record_failure "${job_dataset[$j]}" "${job_id[$j]}" "${job_file[$j]}" \
        "main exited 0 without producing output for this query" null "$run"
    done
    done_upto=$end
  fi

  next=$done_upto
done

# --------------------------------------------------------------- outputs --

write_json_array "$OUTPUT" ${combined_records[@]+"${combined_records[@]}"}
write_json_array "$TOPK_OUTPUT" ${topk_records[@]+"${topk_records[@]}"}
write_json_array "$RUNS_OUTPUT" ${run_records[@]+"${run_records[@]}"}

echo "" >&2
echo "Wrote ${#combined_records[@]} per-query results to $OUTPUT" >&2
echo "Wrote top-k summary to $TOPK_OUTPUT" >&2
echo "Wrote $run run summaries (setup, offline/online totals) to $RUNS_OUTPUT" >&2
echo "Progress log: $PROGRESS_LOG; raw main output: $MAIN_LOG, $MAIN_ERR_LOG" >&2
echo "Total wall time: $(elapsed_since "$run_start")s (${completed} ok, ${failed_count} failed, ${run} main run(s), ${total_refreshes} Piano refresh(es))" >&2

if [[ ${#failure_records[@]} -gt 0 ]]; then
  write_json_array "$FAILURES_OUTPUT" "${failure_records[@]}"
  echo "${#failure_records[@]} queries failed; details in $FAILURES_OUTPUT" >&2
  exit 2
fi
exit 0