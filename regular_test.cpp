/*
 * piano_pir_test.cpp
 *
 * C++ port of the Go test file (pianopir/piano_test.go).
 * Reproduces all four test functions faithfully:
 *
 *   TestPIRBasic        — correctness over full MaxQueryNum queries
 *   TestBatchPIRBasic   — batch correctness with three query patterns
 *   TestBatchPIRPerf    — throughput benchmark with ANN latency estimate
 *   TestXORPerf         — EntryXor throughput vs naive loop
 *   TestAESPerf         — PRFEvalWithLongKeyAndTag throughput
 *
 * Build (minimum flags — works on x86 and Apple Silicon):
 *   g++ -O2 -std=c++17 \
 *       piano_pir_test.cpp piano_pir.cpp piano_prf.cpp piano_batch.cpp \
 *       -lsodium -o piano_pir_test
 *
 * Run all:   ./piano_pir_test
 * Run one:   ./piano_pir_test TestPIRBasic
 */

#include "piano_pir.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdarg>
#include <functional>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

// ============================================================================
// Minimal test harness  (mirrors Go's *testing.T)
// ============================================================================

struct T {
    const char* name;
    int         failed  = 0;
    int         logged  = 0;

    void Logf(const char* fmt, ...) {
        printf("    [LOG] ");
        va_list ap; va_start(ap, fmt); vprintf(fmt, ap); va_end(ap);
        if (fmt[strlen(fmt)-1] != '\n') printf("\n");
        ++logged;
    }
    void Errorf(const char* fmt, ...) {
        printf("    [FAIL] ");
        va_list ap; va_start(ap, fmt); vprintf(fmt, ap); va_end(ap);
        if (fmt[strlen(fmt)-1] != '\n') printf("\n");
        ++failed;
    }
    bool Failed() const { return failed > 0; }
};

using TestFn = std::function<void(T&)>;

static int RunTest(const char* name, TestFn fn, const char* filter = nullptr) {
    if (filter && std::string(name).find(filter) == std::string::npos) return -1;
    printf("\n=== RUN   %s ===\n", name);
    T t; t.name = name;
    auto t0 = std::chrono::steady_clock::now();
    fn(t);
    double ms = std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now() - t0).count();
    if (t.Failed())
        printf("--- FAIL: %s (%.2fs)\n", name, ms/1000.0);
    else
        printf("--- PASS: %s (%.2fs)\n", name, ms/1000.0);
    return t.Failed() ? 1 : 0;
}

using Clock = std::chrono::steady_clock;
static double elapsedMs(Clock::time_point t0) {
    return std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
}
static double elapsedSec(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now()-t0).count();
}

// ============================================================================
// TestPIRBasic
//
// Go: DBSize=18750, DBEntrySize=4 uint64 (32 bytes), FailureProbLog2=40.
// Queries every index up to MaxQueryNum; verifies each response exactly.
// ============================================================================
static void TestPIRBasic(T& t) {
    constexpr uint64_t DBSize        = 18750;
    constexpr uint64_t DBEntrySize   = 4;       // uint64s per entry
    constexpr uint64_t DBEntryBytes  = DBEntrySize * 8;
    constexpr uint64_t FailProb      = 40;

    std::mt19937_64 rng(
            (uint64_t)std::chrono::high_resolution_clock::now()
                    .time_since_epoch().count());

    std::vector<uint64_t> rawDB(DBEntrySize * DBSize);
    for (auto& v : rawDB) v = rng();

    PianoPIR pir(DBSize, DBEntryBytes, rawDB, FailProb);

    const auto& cfg = pir.Config();
    t.Logf("PIR config: DBSize=%llu ChunkSize=%llu SetSize=%llu",
           (unsigned long long)cfg.DBSize,
           (unsigned long long)cfg.ChunkSize,
           (unsigned long long)cfg.SetSize);
    t.Logf("hint num: %llu",      (unsigned long long)pir.client_PrimaryHintNum());
    t.Logf("max query num: %llu", (unsigned long long)pir.client_MaxQueryNum());

    const uint64_t maxQueryNum = pir.client_MaxQueryNum();

    pir.Preprocessing();

    std::uniform_int_distribution<uint64_t> idxDist(0, DBSize - 1);

    for (uint64_t i = 0; i < maxQueryNum; ++i) {
        uint64_t idx = idxDist(rng);

        std::vector<uint64_t> query;
        try {
            query = pir.Query(idx, /*realQuery=*/true);
        } catch (const std::exception& e) {
            t.Errorf("PIR.Query(%llu) failed: %s",
                     (unsigned long long)idx, e.what());
            continue;
        }

        for (uint64_t j = 0; j < DBEntrySize; ++j) {
            if (query[j] != rawDB[idx * DBEntrySize + j]) {
                t.Errorf("query[%llu][%llu] = %llu; want %llu",
                         (unsigned long long)idx, (unsigned long long)j,
                         (unsigned long long)query[j],
                         (unsigned long long)rawDB[idx*DBEntrySize+j]);
            }
        }

        if (i == 0) {
            t.Logf("response[0] = %016llx ...", (unsigned long long)query[0]);
        }
    }
}

// ============================================================================
// TestBatchPIRBasic
//
// Go: DBSize=1,000,000, DBEntrySize=16 uint64 (128 bytes), BatchSize=32.
// rawDB[i][j] = i  (deterministic, easy to verify).
// Three query patterns:
//   1. One query per partition (QueryPerPartition-1 per part → all answered)
//   2. Four queries per partition (overflow → only first 2 per part answered)
//   3. BatchSize queries all in partition 0 → only first QueryPerPartition correct
// ============================================================================
static void TestBatchPIRBasic(T& t) {
    constexpr uint64_t DBSize       = 1000000;
    constexpr uint64_t DBEntrySize  = 16;
    constexpr uint64_t DBEntryBytes = DBEntrySize * 8;
    constexpr uint64_t BatchSize    = 32;
    constexpr uint64_t FailProb     = 20;

    std::vector<uint64_t> rawDB(DBEntrySize * DBSize);
    for (uint64_t i = 0; i < DBSize; ++i)
        for (uint64_t j = 0; j < DBEntrySize; ++j)
            rawDB[i * DBEntrySize + j] = i;   // deterministic: rawDB[i][*] = i

    SimpleBatchPianoPIR pir(DBSize, DBEntryBytes, BatchSize, rawDB, FailProb);
    const auto& cfg = pir.Config();
    t.Logf("Batch PIR config: DBSize=%llu PartitionNum=%llu PartitionSize=%llu BatchSize=%llu",
           (unsigned long long)cfg.DBSize,
           (unsigned long long)cfg.PartitionNum,
           (unsigned long long)cfg.PartitionSize,
           (unsigned long long)cfg.BatchSize);

    pir.Preprocessing();

    std::mt19937_64 rng(
            (uint64_t)std::chrono::high_resolution_clock::now()
                    .time_since_epoch().count());

    // ------------------------------------------------------------------
    // Pattern 1: QueryPerPartition-1 queries per partition
    // Go: for each partition pick (QueryPerPartition-1) random offsets
    // All should be answered correctly (under the per-partition budget).
    // ------------------------------------------------------------------
    {
        std::vector<uint64_t> batchQuery;
        batchQuery.reserve(BatchSize);

        for (uint64_t i = 0; i < cfg.PartitionNum; ++i) {
            uint64_t start = i * cfg.PartitionSize;
            uint64_t end   = std::min((i+1)*cfg.PartitionSize, cfg.DBSize);
            std::uniform_int_distribution<uint64_t> d(0, end - start - 1);
            for (uint64_t j = 0; j < kQueryPerPartition - 1; ++j)
                batchQuery.push_back(start + d(rng));
        }

        auto responses = pir.Query(batchQuery);

        for (size_t i = 0; i < batchQuery.size(); ++i) {
            uint64_t idx = batchQuery[i];
            for (uint64_t j = 0; j < DBEntrySize; ++j) {
                if (responses[i][j] != rawDB[idx * DBEntrySize + j]) {
                    t.Errorf("Pattern1: query[%llu][%llu] = %llu; want %llu",
                             (unsigned long long)idx, (unsigned long long)j,
                             (unsigned long long)responses[i][j],
                             (unsigned long long)rawDB[idx*DBEntrySize+j]);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Pattern 2: 4 queries per partition (overflow → only first 2 answered)
    // Go verifies ALL responses but marks overflowed ones as zero.
    // The test body checks every response unconditionally.
    // ------------------------------------------------------------------
    {
        std::vector<uint64_t> batchQuery(4 * cfg.PartitionNum);
        for (uint64_t i = 0; i < cfg.PartitionNum; ++i) {
            uint64_t start = i * cfg.PartitionSize;
            uint64_t end   = std::min((i+1)*cfg.PartitionSize, cfg.DBSize);
            std::uniform_int_distribution<uint64_t> d(0, end - start - 1);
            for (int j = 0; j < 4; ++j)
                batchQuery[i*4 + j] = start + d(rng);
        }

        auto responses = pir.Query(batchQuery);

        // Go checks ALL of these — overflow slots are zeros, not errors.
        // We mirror that: for each response, if non-zero it must be correct.
        for (size_t i = 0; i < batchQuery.size(); ++i) {
            uint64_t idx = batchQuery[i];
            bool isZero = true;
            for (uint64_t j = 0; j < DBEntrySize; ++j)
                if (responses[i][j] != 0) { isZero = false; break; }

            if (!isZero) {
                for (uint64_t j = 0; j < DBEntrySize; ++j) {
                    if (responses[i][j] != rawDB[idx * DBEntrySize + j]) {
                        t.Errorf("Pattern2: query[%llu][%llu] = %llu; want %llu",
                                 (unsigned long long)idx, (unsigned long long)j,
                                 (unsigned long long)responses[i][j],
                                 (unsigned long long)rawDB[idx*DBEntrySize+j]);
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Pattern 3: BatchSize queries all inside partition 0.
    // Go: only first QueryPerPartition responses are correct; rest are zeros.
    // ------------------------------------------------------------------
    {
        // Sample BatchSize distinct indices from partition 0 (no repeats)
        std::unordered_set<uint64_t> querySet;
        std::vector<uint64_t> batchQuery;
        batchQuery.reserve(BatchSize);

        std::uniform_int_distribution<uint64_t> d(0, cfg.PartitionSize - 1);
        while (batchQuery.size() < BatchSize) {
            uint64_t idx = d(rng);
            if (querySet.insert(idx).second)
                batchQuery.push_back(idx);
        }

        auto responses = pir.Query(batchQuery);

        for (uint64_t i = 0; i < BatchSize; ++i) {
            uint64_t idx = batchQuery[i];
            if (i < kQueryPerPartition) {
                // Must be correct
                for (uint64_t j = 0; j < DBEntrySize; ++j) {
                    if (responses[i][j] != rawDB[idx * DBEntrySize + j]) {
                        t.Errorf("Pattern3[answered]: query[%llu][%llu] = %llu; want %llu",
                                 (unsigned long long)idx, (unsigned long long)j,
                                 (unsigned long long)responses[i][j],
                                 (unsigned long long)rawDB[idx*DBEntrySize+j]);
                    }
                }
            } else {
                // Must be zero (overflow)
                for (uint64_t j = 0; j < DBEntrySize; ++j) {
                    if (responses[i][j] != 0) {
                        t.Errorf("Pattern3[overflow]: query[%llu][%llu] = %llu; want 0",
                                 (unsigned long long)idx, (unsigned long long)j,
                                 (unsigned long long)responses[i][j]);
                    }
                }
            }
        }
    }
}

// ============================================================================
// TestBatchPIRPerf
//
// Go: DBSize=3,201,821, DBEntrySize=112 uint64 (896 bytes), BatchSize=32.
// Runs 300 random batch queries; reports throughput and ANN latency estimate.
// ============================================================================
static void TestBatchPIRPerf(T& t) {
    // Go original: DBSize=3201821, DBEntrySize=112, BatchSize=32, FailProb=8
    // (~2.9 GB rawDB — requires significant RAM).
    // Default here is a scaled-down version that fits in ~256 MB.
    // Build with -DPIANO_PERF_FULL to restore the original Go parameters.
#ifdef PIANO_PERF_FULL
    constexpr uint64_t DBSize       = 3201821;
    constexpr uint64_t DBEntrySize  = 112;
    constexpr uint64_t FailProb     = 8;
#else
    constexpr uint64_t DBSize       = 200000;
    constexpr uint64_t DBEntrySize  = 112;
    constexpr uint64_t FailProb     = 20;  // bumped from 8 for small DB
#endif
    constexpr uint64_t DBEntryBytes = DBEntrySize * 8;
    constexpr uint64_t BatchSize    = 32;

    std::mt19937_64 rng(
            (uint64_t)std::chrono::high_resolution_clock::now()
                    .time_since_epoch().count());

    std::vector<uint64_t> rawDB(DBEntrySize * DBSize);
    for (auto& v : rawDB) v = rng();

    SimpleBatchPianoPIR pir(DBSize, DBEntryBytes, BatchSize, rawDB, FailProb);
    const auto& cfg = pir.Config();

    t.Logf("Batch PIR config: DBSize=%llu PartitionNum=%llu PartitionSize=%llu BatchSize=%llu",
           (unsigned long long)cfg.DBSize,
           (unsigned long long)cfg.PartitionNum,
           (unsigned long long)cfg.PartitionSize,
           (unsigned long long)cfg.BatchSize);
    t.Logf("Batch PIR storage %.2f MB", pir.LocalStorageSize() / 1024.0 / 1024.0);
    t.Logf("Batch PIR max query num %llu", (unsigned long long)pir.subPIR_MaxQueryNum(0));
    t.Logf("Sub PIR primary hint num: %llu", (unsigned long long)pir.subPIR_PrimaryHintNum(0));
    t.Logf("Sub PIR storage %.2f MB", pir.subPIR_LocalStorageSize(0) / 1024.0 / 1024.0);
    pir.PrintSubPIRStorageBreakdown(0);

    auto prepStart = Clock::now();
    pir.Preprocessing();
    t.Logf("Preprocessing time = %.3f s", elapsedSec(prepStart));

    // 300 random batch queries
    constexpr int QueryNum = 300;
    std::uniform_int_distribution<uint64_t> idxDist(0, DBSize - 1);

    auto queryStart = Clock::now();
    for (int i = 0; i < QueryNum; ++i) {
        std::vector<uint64_t> batch(BatchSize);
        for (auto& v : batch) v = idxDist(rng);

        auto response = pir.Query(batch);

        // Go: check response[0] — either all-zero (overflow) or correct
        for (uint64_t j = 0; j < DBEntrySize; ++j) {
            if (response[0][j] != 0 &&
                response[0][j] != rawDB[batch[0] * DBEntrySize + j]) {
                t.Errorf("Perf: response[0][%llu] = %llu; want %llu",
                         (unsigned long long)j,
                         (unsigned long long)response[0][j],
                         (unsigned long long)rawDB[batch[0]*DBEntrySize+j]);
            }
        }
    }
    double totalSec    = elapsedSec(queryStart);
    double avgBatchMs  = totalSec * 1000.0 / QueryNum;

    t.Logf("Total query time = %.3f s", totalSec);
    t.Logf("Average query time per batch = %.3f ms", avgBatchMs);

    // ANN latency estimate: (avgBatchTime * parallel + rtt) * step
    // Go: rtt=50ms, parallel=2, step=15
    constexpr double rttMs     = 50.0;
    constexpr int    parallel  = 2;
    constexpr int    step      = 15;
    double annLatencyMs = (avgBatchMs * parallel + rttMs) * step;
    t.Logf("Estimated ANN latency = %.1f ms", annLatencyMs);
}

// ============================================================================
// TestXORPerf
//
// Go: verifies xorSlices correctness then benchmarks naive loop vs EntryXor.
// n=1,000,000 entries of l=112 uint64s each.
// ============================================================================
static void TestXORPerf(T& t) {
    // Correctness: small sanity check (mirrors Go's first few lines)
    {
        std::vector<uint64_t> p = {12312312,12312312,12312312,12312312,
                                   12312312,12312312,12312312,12312312};
        std::vector<uint64_t> q = {12312,12312,12312,12312,
                                   12312,12312,12312,12312};
        EntryXor(p.data(), q.data(), 8);
        for (int i = 0; i < 8; ++i) {
            if (p[i] != (12312312ULL ^ 12312ULL)) {
                t.Errorf("EntryXor sanity: p[%d] = %llu; want %llu",
                         i, (unsigned long long)p[i],
                         (unsigned long long)(12312312ULL ^ 12312ULL));
            }
        }
    }

    constexpr int n = 1000000;
    constexpr int l = 112;

    std::vector<uint64_t> a(l * n, 12312312ULL);
    std::vector<uint64_t> b(l * n, 12312ULL);

    // Naive XOR
    auto t0 = Clock::now();
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < l; ++j)
            a[i*l+j] ^= b[i*l+j];
    t.Logf("Naive XOR time = %.3f ms", elapsedMs(t0));

    // Reset
    std::fill(a.begin(), a.end(), 12312312ULL);
    std::fill(b.begin(), b.end(), 12312ULL);

    // EntryXor (hardware-accelerated)
    auto t1 = Clock::now();
    for (int i = 0; i < n; ++i)
        EntryXor(a.data() + i*l, b.data() + i*l, l);
    t.Logf("EntryXor time  = %.3f ms", elapsedMs(t1));

    // Verify
    for (int i = 0; i < l * n; ++i) {
        if (a[i] != (12312312ULL ^ 12312ULL)) {
            t.Errorf("EntryXor result: a[%d] = %llu; want %llu",
                     i, (unsigned long long)a[i],
                     (unsigned long long)(12312312ULL ^ 12312ULL));
            break; // report first failure only
        }
    }
}

// ============================================================================
// TestAESPerf
//
// Go: benchmarks PRFEvalWithLongKeyAndTag for n=1,000,000 calls,
//     then benchmarks EntryXor for n batches of l=112 uint64s.
// ============================================================================
static void TestAESPerf(T& t) {
    std::mt19937_64 rng(
            (uint64_t)std::chrono::high_resolution_clock::now()
                    .time_since_epoch().count());

    PrfKey masterKey = RandKey();
    std::vector<uint32_t> longKey = GetLongKey(*reinterpret_cast<const PrfKey128*>(&masterKey));

    constexpr int n = 1000000;

    std::vector<uint64_t> tag(n), results(n, 0);
    for (auto& v : tag) v = rng();

    auto t0 = Clock::now();
    for (int i = 0; i < n; ++i)
        results[i] = PRFEvalWithLongKeyAndTag(longKey, tag[i], (uint64_t)i);
    double prfMs = elapsedMs(t0);
    t.Logf("PRFEvalWithLongKeyAndTag time = %.3f ms", prfMs);
    t.Logf("average time = %.1f ns/call", prfMs * 1e6 / n);

    constexpr int l = 112;
    std::vector<uint64_t> a(l * n, 12312312ULL);
    std::vector<uint64_t> b(l * n, 12312ULL);

    auto t1 = Clock::now();
    for (int i = 0; i < n; ++i)
        EntryXor(a.data() + i*l, b.data() + i*l, l);
    double xorMs = elapsedMs(t1);
    t.Logf("EntryXor time  = %.3f ms", xorMs);
    t.Logf("average time = %.1f ns/call", xorMs * 1e6 / n);
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char* argv[]) {
    const char* filter = (argc > 1) ? argv[1] : nullptr;

    int total = 0, failed = 0;
    auto run = [&](const char* name, TestFn fn) {
        int r = RunTest(name, fn, filter);
        if (r < 0) return;   // filtered out
        ++total;
        if (r > 0) ++failed;
    };

    run("TestPIRBasic",      TestPIRBasic);
    run("TestBatchPIRBasic", TestBatchPIRBasic);
    run("TestBatchPIRPerf",  TestBatchPIRPerf);
    run("TestXORPerf",       TestXORPerf);
    run("TestAESPerf",       TestAESPerf);

    printf("\n");
    if (total == 0) {
        printf("no tests matched filter '%s'\n", filter ? filter : "");
        return 1;
    }
    if (failed == 0)
        printf("ok  — %d/%d tests passed\n", total, total);
    else
        printf("FAIL — %d/%d tests failed\n", failed, total);

    return failed > 0 ? 1 : 0;
}