#include "piano_pir.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

SimpleBatchPianoPIR::SimpleBatchPianoPIR(uint64_t DBSize,
                                         uint64_t DBEntryByteNum,
                                         uint64_t BatchSize,
                                         std::vector<uint64_t> rawDB,
                                         uint64_t FailureProbLog2) {
    const uint64_t DBEntrySize = DBEntryByteNum / 8;

    if (rawDB.size() != DBSize * DBEntrySize) {
        throw std::invalid_argument(
                "BatchPIR: len(rawDB) = " + std::to_string(rawDB.size())
                + "; want " + std::to_string(DBSize * DBEntrySize));
    }

    // PartitionNum = BatchSize / RealQueryPerPartition
    const uint64_t PartitionNum  = BatchSize / kRealQueryPerPartition;
    // PartitionSize = ceil(DBSize / PartitionNum)
    const uint64_t PartitionSize = (DBSize + PartitionNum - 1) / PartitionNum;

    config = SimpleBatchPianoPIRConfig{
            .DBEntryByteNum  = DBEntryByteNum,
            .DBEntrySize     = DBEntrySize,
            .DBSize          = DBSize,
            .BatchSize       = BatchSize,
            .PartitionNum    = PartitionNum,
            .PartitionSize   = PartitionSize,
            .ThreadNum       = kBatchThreadNum,
            .FailureProbLog2 = FailureProbLog2,
    };

    subPIR.reserve(PartitionNum);
    for (uint64_t i = 0; i < PartitionNum; ++i) {
        uint64_t start      = i * PartitionSize;
        uint64_t end        = std::min((i + 1) * PartitionSize, DBSize); // mirrors Go's min()
        uint64_t partDBSize = end - start;

        std::vector<uint64_t> partDB(
                rawDB.begin() + static_cast<ptrdiff_t>(start * DBEntrySize),
                rawDB.begin() + static_cast<ptrdiff_t>(end   * DBEntrySize));

        subPIR.push_back(std::make_unique<PianoPIR>(
                partDBSize, DBEntryByteNum, std::move(partDB), FailureProbLog2));
    }
}

void SimpleBatchPianoPIR::PrintInfo() const {
    const double DBSizeInBytes =
            static_cast<double>(config.DBSize) * static_cast<double>(config.DBEntryByteNum);

    printf("-----------BatchPIR config --------\n");
    printf("DB size in MB = %g\n", DBSizeInBytes / 1024.0 / 1024.0);

    // Go prints all fields on one line with fmt.Printf
    printf("DBSize: %llu, DBEntryByteNum: %llu, BatchSize: %llu, "
           "PartitionNum: %llu, PartitionSize: %llu, "
           "ThreadNum: %llu, FailureProbLog2: %llu\n",
           (unsigned long long)config.DBSize,
           (unsigned long long)config.DBEntryByteNum,
           (unsigned long long)config.BatchSize,
           (unsigned long long)config.PartitionNum,
           (unsigned long long)config.PartitionSize,
           (unsigned long long)config.ThreadNum,
           (unsigned long long)config.FailureProbLog2);

    // Go: maxQuery := p.subPIR[0].client.MaxQueryNum / QueryPerPartition
    const uint64_t maxQuery = subPIR[0]->client_MaxQueryNum() / kQueryPerPartition;
    printf("max query num = %llu\n",       (unsigned long long)maxQuery);

    // Go: p.subPIR[0].client.maxQueryPerChunk
    printf("max query per chunk = %llu\n", (unsigned long long)subPIR[0]->client_MaxQueryPerChunk());

    printf("total storage = %g MB\n",         LocalStorageSize() / 1024.0 / 1024.0);
    printf("comm cost per batch = %g KB\n",   static_cast<double>(CommCostPerBatchOnline()) / 1024.0);

    if (maxQuery > 0) {
        const double amortPrep = DBSizeInBytes / static_cast<double>(maxQuery) / 1024.0;
        printf("amortized preprocessing comm cost = %g KB\n", amortPrep);
        printf("total amortized comm cost = %g KB\n",
               amortPrep + static_cast<double>(CommCostPerBatchOnline()) / 1024.0);
    }
    printf("-----------------------------\n");
}

void SimpleBatchPianoPIR::RecordStats(double prepTimeSecs) {
    preprocessingTime      = prepTimeSecs;
    localStorage           = static_cast<uint64_t>(LocalStorageSize());
    commCostPerBatchOnline = CommCostPerBatchOnline();

    // Go: p.SupportBatchNum = p.subPIR[0].client.MaxQueryNum / QueryPerPartition
    SupportBatchNum = subPIR[0]->client_MaxQueryNum() / kQueryPerPartition;

    const double DBSizeInBytes =
            static_cast<double>(config.DBSize) * static_cast<double>(config.DBEntryByteNum);

    // Go: p.commCostPerBatchOffline = uint64(float64(DBSizeInBytes) / float64(p.SupportBatchNum))
    commCostPerBatchOffline =
            SupportBatchNum > 0
            ? static_cast<uint64_t>(DBSizeInBytes / static_cast<double>(SupportBatchNum))
            : 0;
}

void SimpleBatchPianoPIR::Preprocessing() {
    PrintInfo();

    // Go resets both counters at the top of Preprocessing()
    FinishedBatchNum       = 0;
    QueriesMadeInPartition = 0;

    const uint64_t threadNum    = config.ThreadNum;
    const uint64_t partitionNum = config.PartitionNum;

    const auto startTime = std::chrono::steady_clock::now();

    // perThreadPartitionNum = ceil(PartitionNum / ThreadNum)  — matches Go
    const uint64_t perThread = (partitionNum + threadNum - 1) / threadNum;

    if (threadNum <= 1) {
        // Fast single-threaded path
        for (uint64_t i = 0; i < partitionNum; ++i) {
            subPIR[i]->Preprocessing();
        }
    } else {
        std::vector<std::thread> threads;
        threads.reserve(threadNum);

        for (uint64_t tid = 0; tid < threadNum; ++tid) {
            const uint64_t start = tid * perThread;
            const uint64_t end   = std::min((tid + 1) * perThread, partitionNum);

            threads.emplace_back([this, start, end]() {
                for (uint64_t i = start; i < end; ++i) {
                    subPIR[i]->Preprocessing();
                }
            });
        }
        for (auto& t : threads) t.join();
    }

    const double elapsed =
            std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - startTime).count();

    // Go: log.Printf("Preprocessing time = %v\n", endTime.Sub(startTime))
    printf("Preprocessing time = %.6f s\n", elapsed);

    RecordStats(elapsed);
}

void SimpleBatchPianoPIR::DummyPreprocessing() {
    PrintInfo();

    for (uint64_t i = 0; i < config.PartitionNum; ++i) {
        subPIR[i]->DummyPreprocessing();
    }

    // Go: log.Printf("Skipping Prep")
    printf("Skipping Prep\n");

    RecordStats(0.0);
}

std::vector<std::vector<uint64_t>>
SimpleBatchPianoPIR::Query(const std::vector<uint64_t>& idx) {
    const uint64_t partitionNum  = config.PartitionNum;
    const uint64_t partitionSize = config.PartitionSize;
    const uint64_t dbEntrySize   = config.DBEntrySize;

    // Go: queryNumToMake := len(idx) / int(p.config.PartitionNum)
    const int queryNumToMake =
            static_cast<int>(idx.size()) / static_cast<int>(partitionNum);

    // ---- Step 3a: arrange queries into partitions (first-come-first-served) ----
    // Go: partitionIdx := idx[i] / p.config.PartitionSize  (no clamping)
    std::vector<std::vector<uint64_t>> partitionQueries(partitionNum);
    for (size_t i = 0; i < idx.size(); ++i) {
        const uint64_t pId = idx[i] / partitionSize;   // no clamp — matches Go
        partitionQueries[pId].push_back(idx[i]);
    }

    // ---- Step 3b: query each sub-PIR ----------------------------------------
    // Map: global index → DB entry
    std::unordered_map<uint64_t, std::vector<uint64_t>> responses;
    responses.reserve(idx.size());

    for (uint64_t i = 0; i < partitionNum; ++i) {
        // Pad under-filled partition with DefaultValue sentinels
        // Go: if len(partitionQueries[i]) < queryNumToMake { append DefaultValue }
        while (static_cast<int>(partitionQueries[i].size()) < queryNumToMake) {
            partitionQueries[i].push_back(kDefaultValue);
        }

        // Issue exactly queryNumToMake queries
        for (int j = 0; j < queryNumToMake; ++j) {
            const uint64_t globalIdx = partitionQueries[i][j];

            if (globalIdx == kDefaultValue) {
                // Go: _, _ = p.subPIR[i].Query(0, false)
                subPIR[i]->Query(0, /*realQuery=*/false);
            } else {
                // Go: query, _ := p.subPIR[i].Query(partitionQueries[i][j]-i*p.config.PartitionSize, true)
                const uint64_t localIdx = globalIdx - i * partitionSize;
                responses[globalIdx] =
                        subPIR[i]->Query(localIdx, /*realQuery=*/true);
            }
        }
    }

    // ---- Step 4: assemble output in original query order --------------------
    std::vector<std::vector<uint64_t>> ret(idx.size());
    for (size_t i = 0; i < idx.size(); ++i) {
        auto it = responses.find(idx[i]);
        if (it != responses.end()) {
            ret[i] = std::move(it->second);
        } else {
            // Go: ret[i] = make([]uint64, p.config.DBEntrySize)  (zero-filled)
            ret[i].assign(dbEntrySize, 0);
        }
    }

    // ---- Repreprocessing guard ----------------------------------------------
    // Go: if p.QueriesMadeInPartition >= p.subPIR[0].client.MaxQueryNum-2
    const uint64_t maxQ = subPIR[0]->client_MaxQueryNum();
    if (QueriesMadeInPartition >= maxQ - 2) {
        printf("Redo preprocessing. Made %llu batches (%llu queries in a partition), "
               "redo the preprocessing\n",
               (unsigned long long)FinishedBatchNum,
               (unsigned long long)QueriesMadeInPartition);
        Preprocessing();
    } else {
        // Go: p.FinishedBatchNum += uint64(len(idx) / int(p.config.BatchSize))
        FinishedBatchNum       += static_cast<uint64_t>(
                static_cast<int>(idx.size())
                / static_cast<int>(config.BatchSize));
        // Go: p.QueriesMadeInPartition += uint64(queryNumToMake)
        QueriesMadeInPartition += static_cast<uint64_t>(queryNumToMake);
    }

    return ret;
}

double SimpleBatchPianoPIR::LocalStorageSize() const {
    double total = 0.0;
    for (const auto& p : subPIR) total += p->LocalStorageSize();
    return total;
}

uint64_t SimpleBatchPianoPIR::CommCostPerBatchOnline() const {
    double total = 0.0;
    for (const auto& p : subPIR) {
        // Go: p.subPIR[i].CommCostPerQuery() * float64(QueryPerPartition)
        total += p->CommCostPerQuery() * static_cast<double>(kQueryPerPartition);
    }
    return static_cast<uint64_t>(total);
}

uint64_t SimpleBatchPianoPIR::UploadCostPerBatchOnline() const {
    double total = 0.0;
    for (const auto& p: subPIR){
        total += p->UploadCostPerQuery() * static_cast<double>(kQueryPerPartition);
    }
    return static_cast<uint64_t>(total);
}

uint64_t SimpleBatchPianoPIR::DownloadCostPerBatchOnline() const {
    double total = 0.0;
    for (const auto& p: subPIR){
        total += p->DownloadCostPerQuery() * static_cast<double>(kQueryPerPartition);
    }
    return static_cast<uint64_t>(total);
}