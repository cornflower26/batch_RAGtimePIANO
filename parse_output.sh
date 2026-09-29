#!/usr/bin/env bash
# Summarises a PIANO-RAG run log as JSON. Works for all three binaries:
#   main                   -> "variant": "unbatched"
#   partition_batch_main      -> "variant": "batched_spread"  ("batched_contiguous"
#                             if run with --spread-clusters false)
#   regular_batch_main     -> "variant": "batched_contiguous"
#
# Usage: ./parse_output.sh run.log     (or pipe the log in on stdin)
#
# Handles single- and multi-query (--query-list) runs, Piano refreshes,
# offline preprocessing communication, and PIR 1 batch statistics
# (batches, dummy batches, max candidates in one partition, retries).
#
# excluding_one_time_setup:
#   upload   = online PIR upload, summed over all queries
#   download = online PIR download + offline download of mid-query refreshes
#   time     = sum of per-query "Total time" (centroid step + PIR 1 + FAISS +
#              PIR 2 + any refresh)
# including_one_time_setup adds:
#   the FHE distance upload/download
#   the offline download of the initial Piano preprocessing
#   "One-time Setup time" (DB loading, server setup, preprocessing, keygen)
set -euo pipefail

input_file="${1:--}"
if [[ "$input_file" != "-" && ! -r "$input_file" ]]; then
  echo "usage: $0 <run.log>   (or pipe the log on stdin)" >&2
  exit 1
fi

awk '
# Value of the field right after the first field equal to tok (0 if absent).
function after(tok,   i) {
  for (i = 1; i < NF; i++) if ($i == tok) return $(i + 1) + 0;
  return 0;
}
function jstr(s) {
  gsub(/\\/, "\\\\", s); gsub(/"/, "\\\"", s); gsub(/\t/, "\\t", s);
  return "\"" s "\"";
}

BEGIN {
  q = 0; completed = 0; fatal_msg = "";
  fhe_upload = 0; fhe_download = 0;
  one_time_setup = 0; server_setup = 0; client_setup = 0;
  topk_batches = 0;
  variant = "unbatched"; batch_size = ""; partitions = ""; min_batches = "";
  for (p = 1; p <= 2; p++) {
    up[p] = 0; dn[p] = 0; srv[p] = 0; prep[p] = 0;
    off_init[p] = 0; off_ref[p] = 0; ref_n[p] = 0; ref_t[p] = 0;
  }
}

# ---- one-time setup ------------------------------------------------------
/^Config: / {
  if (index($0, "PIR 1 batched:") > 0) {
    variant = (index($0, "(contiguous)") > 0 || index($0, "spread-clusters off") > 0) \
              ? "batched_contiguous" : "batched_spread";
    batch_size = after("size");
    min_batches = after("min-batches");
    for (i = 2; i <= NF; i++) if ($i == "partitions") { partitions = $(i - 1) + 0; break }
  }
}
/^One-time Setup time /                  { one_time_setup = $4 }
/^Server Setup time: /                   { server_setup = $4 }
/^Client Setup time: /                   { client_setup = $4 }
/^PIR [0-9]+ \(.*\) server setup time /  { srv[$2] = after("time") }
/^PIR [0-9]+ \(.*\) client preprocessing time / {
  prep[$2] = after("time"); off_init[$2] = after("download");
}

# ---- per query -----------------------------------------------------------
/^=== Query [0-9]+\/[0-9]+: / {
  q++;
  path = $0;
  sub(/^=== Query [0-9]+\/[0-9]+: /, "", path);
  sub(/ ===[ \t]*$/, "", path);
  q_path[q] = path;
  q_centroid[q] = 0; q_pir1[q] = 0; q_pir2[q] = 0; q_time[q] = 0; q_total[q] = 0;
  q_up[q] = 0; q_dn[q] = 0; q_ref_n[q] = 0; q_ref_t[q] = 0; q_ref_b[q] = 0;
  q_fhe_up[q] = 0; q_fhe_dn[q] = 0;
  q_cand[q] = 0; q_batches[q] = 0; q_dummy[q] = 0; q_maxpp[q] = 0; q_retried[q] = 0; q_dropped[q] = 0;
}

/^Queries to batch :/   { q_cand[q] = NF - 4 }
# PIR 1 batching: C candidates, max M in one partition, R batch(es) + D dummy, X retried, Y dropped
/^PIR 1 batching: / {
  q_cand[q] = $4 + 0; q_maxpp[q] = $7 + 0; q_batches[q] = $11 + 0;
  q_dummy[q] = $14 + 0; q_retried[q] = $16 + 0; q_dropped[q] = $18 + 0;
}

/FHE Distance upload/   { fhe_upload += $4;   q_fhe_up[q] += $4 }
/FHE Distance download/ { fhe_download += $4; q_fhe_dn[q] += $4 }

/^Centroid time /       { q_centroid[q] = $3 }
/^PIR 1 time /          { q_pir1[q] = $4 }
/^PIR 2 time /          { q_pir2[q] = $4 }
/^PIR [0-9]+ Upload /   { up[$2] += $4; q_up[q] += $4 }
/^PIR [0-9]+ Download / { dn[$2] += $4; q_dn[q] += $4 }
/^Query time /          { q_time[q] = $3 }
/^Total time /          { q_total[q] = $3 }

# Only the completion line of a refresh carries its time and bytes.
/^\[refresh\] PIR [0-9]+ .* completed in / {
  p = $3; t = after("in"); b = after("download");
  ref_n[p]++; ref_t[p] += t; off_ref[p] += b;
  q_ref_n[q]++; q_ref_t[q] += t; q_ref_b[q] += b;
}

/^Top-K:/ {
  topk_batches++;
  line = $0;
  sub(/^Top-K:[ \t]*/, "", line);
  cnt = 0;
  while (match(line, /\[[^]]*\]/)) {
    item = substr(line, RSTART + 1, RLENGTH - 2);
    gsub(/^[ \t]+|[ \t]+$/, "", item);
    split(item, parts, /[ \t]+/);
    cnt++;
    topk_idx[topk_batches, cnt] = parts[1];
    topk_val[topk_batches, cnt] = parts[2];
    line = substr(line, RSTART + RLENGTH);
  }
  topk_count[topk_batches] = cnt;
}

/^Successful RAG completion!/ { completed = 1 }
/^fatal: /                    { fatal_msg = substr($0, 8) }

END {
  pir_upload = up[1] + up[2];
  pir_download = dn[1] + dn[2];
  offline_initial = off_init[1] + off_init[2];
  offline_refresh = off_ref[1] + off_ref[2];
  refresh_count = ref_n[1] + ref_n[2];
  refresh_time = ref_t[1] + ref_t[2];

  centroid_time = 0; pir1_time = 0; pir2_time = 0; query_time = 0; total_time = 0;
  for (i = 1; i <= q; i++) {
    centroid_time += q_centroid[i]; pir1_time += q_pir1[i]; pir2_time += q_pir2[i];
    query_time += q_time[i]; total_time += q_total[i];
    t_batches += q_batches[i]; t_dummy += q_dummy[i]; t_retried += q_retried[i]; t_dropped += q_dropped[i];
    t_cand += q_cand[i];
    if (q_maxpp[i] > t_maxpp) t_maxpp = q_maxpp[i];
  }

  upload_excl   = pir_upload;
  download_excl = pir_download + offline_refresh;
  time_excl     = (total_time > 0) ? total_time : query_time;

  upload_incl   = upload_excl + fhe_upload;
  download_incl = download_excl + fhe_download + offline_initial;
  time_incl     = one_time_setup + time_excl;

  printf "{\n";
  printf "  \"completed\": %s,\n", completed ? "true" : "false";
  if (fatal_msg != "") printf "  \"error\": %s,\n", jstr(fatal_msg);
  printf "  \"variant\": \"%s\",\n", variant;
  printf "  \"pir1_batch_size\": %s,\n", (batch_size == "") ? "null" : batch_size;
  printf "  \"pir1_partitions\": %s,\n", (partitions == "") ? "null" : partitions;
  printf "  \"pir1_min_batches\": %s,\n", (min_batches == "") ? "null" : min_batches;
  printf "  \"num_queries\": %d,\n", q;
  printf "  \"avg_query_time\": %.15g,\n", (q > 0) ? time_excl / q : 0;

  printf "  \"excluding_one_time_setup\": {\n";
  printf "    \"upload\": %.15g,\n", upload_excl;
  printf "    \"download\": %.15g,\n", download_excl;
  printf "    \"time\": %.15g\n", time_excl;
  printf "  },\n";
  printf "  \"including_one_time_setup\": {\n";
  printf "    \"upload\": %.15g,\n", upload_incl;
  printf "    \"download\": %.15g,\n", download_incl;
  printf "    \"time\": %.15g\n", time_incl;
  printf "  },\n";

  printf "  \"breakdown\": {\n";
  printf "    \"fhe_upload\": %.15g,\n", fhe_upload;
  printf "    \"fhe_download\": %.15g,\n", fhe_download;
  printf "    \"pir_upload\": %.15g,\n", pir_upload;
  printf "    \"pir_download\": %.15g,\n", pir_download;
  printf "    \"pir_offline_initial_download\": %.15g,\n", offline_initial;
  printf "    \"pir_offline_refresh_download\": %.15g,\n", offline_refresh;
  printf "    \"one_time_setup_time\": %.15g,\n", one_time_setup;
  printf "    \"server_setup_time\": %.15g,\n", server_setup;
  printf "    \"client_setup_time\": %.15g,\n", client_setup;
  printf "    \"centroid_time\": %.15g,\n", centroid_time;
  printf "    \"pir1_time\": %.15g,\n", pir1_time;
  printf "    \"pir2_time\": %.15g,\n", pir2_time;
  printf "    \"refresh_count\": %d,\n", refresh_count;
  printf "    \"refresh_time\": %.15g,\n", refresh_time;
  printf "    \"query_time\": %.15g,\n", query_time;
  printf "    \"pir1_candidates\": %d,\n", t_cand;
  printf "    \"pir1_batches\": %d,\n", t_batches;
  printf "    \"pir1_dummy_batches\": %d,\n", t_dummy;
  printf "    \"pir1_retried\": %d,\n", t_retried;
  printf "    \"pir1_dropped\": %d,\n", t_dropped;
  printf "    \"pir1_max_per_partition\": %d,\n", t_maxpp;
  printf "    \"reported_total_time\": %.15g\n", total_time;
  printf "  },\n";

  printf "  \"per_pir\": {\n";
  for (p = 1; p <= 2; p++) {
    printf "    \"pir%d\": {\n", p;
    printf "      \"online_upload\": %.15g,\n", up[p];
    printf "      \"online_download\": %.15g,\n", dn[p];
    printf "      \"offline_initial_download\": %.15g,\n", off_init[p];
    printf "      \"offline_refresh_download\": %.15g,\n", off_ref[p];
    printf "      \"server_setup_time\": %.15g,\n", srv[p];
    printf "      \"client_preprocessing_time\": %.15g,\n", prep[p];
    printf "      \"refresh_count\": %d,\n", ref_n[p];
    printf "      \"refresh_time\": %.15g\n", ref_t[p];
    printf "    }%s\n", (p < 2) ? "," : "";
  }
  printf "  },\n";

  printf "  \"queries\": [\n";
  for (i = 1; i <= q; i++) {
    printf "    {\n";
    printf "      \"input_vector\": %s,\n", jstr(q_path[i]);
    printf "      \"centroid_time\": %.15g,\n", q_centroid[i];
    printf "      \"pir1_time\": %.15g,\n", q_pir1[i];
    printf "      \"pir2_time\": %.15g,\n", q_pir2[i];
    printf "      \"query_time\": %.15g,\n", q_time[i];
    printf "      \"total_time\": %.15g,\n", q_total[i];
    printf "      \"pir_upload\": %.15g,\n", q_up[i];
    printf "      \"pir_download\": %.15g,\n", q_dn[i];
    printf "      \"fhe_upload\": %.15g,\n", q_fhe_up[i];
    printf "      \"fhe_download\": %.15g,\n", q_fhe_dn[i];
    printf "      \"refresh_count\": %d,\n", q_ref_n[i];
    printf "      \"refresh_time\": %.15g,\n", q_ref_t[i];
    printf "      \"refresh_offline_download\": %.15g,\n", q_ref_b[i];
    printf "      \"pir1_candidates\": %d,\n", q_cand[i];
    printf "      \"pir1_batches\": %d,\n", q_batches[i];
    printf "      \"pir1_dummy_batches\": %d,\n", q_dummy[i];
    printf "      \"pir1_max_per_partition\": %d,\n", q_maxpp[i];
    printf "      \"pir1_retried\": %d,\n", q_retried[i];
    printf "      \"pir1_dropped\": %d\n", q_dropped[i];
    printf "    }%s\n", (i < q) ? "," : "";
  }
  printf "  ],\n";

  # top_k: one array per "Top-K:" line (one per query), {index, distance} in order.
  printf "  \"top_k\": [\n";
  for (b = 1; b <= topk_batches; b++) {
    printf "    [\n";
    for (i = 1; i <= topk_count[b]; i++) {
      printf "      { \"index\": %s, \"distance\": %s }", topk_idx[b, i], topk_val[b, i];
      if (i < topk_count[b]) printf ",";
      printf "\n";
    }
    printf "    ]";
    if (b < topk_batches) printf ",";
    printf "\n";
  }
  printf "  ]\n";
  printf "}\n";
}
' "$input_file"