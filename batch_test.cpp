//
// Created by Antonia Januszewicz on 4/14/26.
//
/*
 * piano_batch_test.cpp
 *
 * Correctness + throughput test for SimpleBatchPianoPIR.
 *
 * Build (together with existing sources):
 *   g++ -O2 -std=c++17 -march=native -maes -msse4.1 \
 *       piano_batch_test.cpp piano_pir.cpp piano_prf.cpp piano_batch.cpp \
 *       -lsodium -o piano_batch_test
 */

#include "piano_pir.h"

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
static double elapsedMs(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

int main() {
    // -----------------------------------------------------------------------
    // Parameters
    // -----------------------------------------------------------------------
    constexpr uint64_t DBSize          = 10000;
    constexpr uint64_t EntryBytes      = 128;   // 16 uint64 per entry
    constexpr uint64_t BatchSize       = 10;    // 10 queries per batch
    constexpr uint64_t FailureProbLog2 = 40;
    constexpr int      NumBatches      = 5;

    printf("=== SimpleBatchPianoPIR C++ test ===\n");
    printf("DBSize=%llu  EntryBytes=%llu  BatchSize=%llu\n\n",
           (unsigned long long)DBSize,
           (unsigned long long)EntryBytes,
           (unsigned long long)BatchSize);

    // -----------------------------------------------------------------------
    // Build random database
    // -----------------------------------------------------------------------
    const uint64_t DBEntrySize = EntryBytes / 8;
    std::mt19937_64 rng(42);
    std::vector<uint64_t> rawDB(DBSize * DBEntrySize);
    for (auto& v : rawDB) v = rng();

    // -----------------------------------------------------------------------
    // Construct and preprocess
    // -----------------------------------------------------------------------
    SimpleBatchPianoPIR bpir(DBSize, EntryBytes, BatchSize, rawDB, FailureProbLog2);

    auto t0 = Clock::now();
    bpir.Preprocessing();
    printf("Preprocessing wall time: %.2f ms\n\n", elapsedMs(t0));

    // -----------------------------------------------------------------------
    // Correctness: issue NumBatches batches, check every answered entry
    // -----------------------------------------------------------------------
    printf("Running %d batches of %llu queries each...\n", NumBatches, (unsigned long long)BatchSize);

    std::uniform_int_distribution<uint64_t> idxDist(0, DBSize - 1);
    int passed = 0, failed = 0, unanswered = 0;

    for (int b = 0; b < NumBatches; ++b) {
        std::vector<uint64_t> batchIdx(BatchSize);
        for (auto& v : batchIdx) v = idxDist(rng);

        auto responses = bpir.Query(batchIdx);

        for (size_t i = 0; i < batchIdx.size(); ++i) {
            uint64_t idx = batchIdx[i];

            // Ground truth
            std::vector<uint64_t> expected(DBEntrySize);
            for (uint64_t w = 0; w < DBEntrySize; ++w)
                expected[w] = rawDB[idx * DBEntrySize + w];

            const auto& got = responses[i];

            // Check if it was a zero (unanswered) response
            bool isZero = true;
            for (uint64_t w = 0; w < DBEntrySize; ++w)
                if (got[w] != 0) { isZero = false; break; }

            if (isZero) {
                ++unanswered;
                continue; // Unanswered slots are OK — partition overflow
            }

            if (got == expected) {
                ++passed;
            } else {
                ++failed;
                printf("  [FAIL] batch=%d i=%zu idx=%llu expected[0]=%016llx got[0]=%016llx\n",
                       b, i, (unsigned long long)idx,
                       (unsigned long long)expected[0],
                       (unsigned long long)got[0]);
            }
        }
    }

    printf("Correctness: %d passed, %d failed, %d unanswered (partition overflow)\n\n",
           passed, failed, unanswered);

    // -----------------------------------------------------------------------
    // Throughput benchmark (dummy preprocessing — skip hint build)
    // -----------------------------------------------------------------------
    printf("Benchmarking dummy batch queries (no preprocessing)...\n");
    SimpleBatchPianoPIR bpir2(DBSize, EntryBytes, BatchSize, rawDB, FailureProbLog2);
    bpir2.DummyPreprocessing();

    constexpr int BenchBatches = 20;
    std::vector<uint64_t> benchIdx(BatchSize);
    for (auto& v : benchIdx) v = idxDist(rng);

    auto t1 = Clock::now();
    for (int b = 0; b < BenchBatches; ++b) {
        bpir2.Query(benchIdx);
    }
    double ms = elapsedMs(t1);
    printf("  %d batches (each=%llu queries) in %.2f ms  (%.2f ms/batch)\n",
           BenchBatches, (unsigned long long)BatchSize, ms, ms / BenchBatches);

    printf("\nStorage: %.2f MB  |  Online comm per batch: %.2f KB\n",
           bpir2.LocalStorageSize() / 1024.0 / 1024.0,
           static_cast<double>(bpir2.CommCostPerBatchOnline()) / 1024.0);

    return (failed == 0) ? 0 : 1;
}