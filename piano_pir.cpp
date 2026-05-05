//
// Created by Antonia Januszewicz on 4/8/26.
//

#include "piano_pir.h"
#include <cassert>
#include <random>
#include <thread>
static constexpr uint64_t DefaultProgramPoint = 0x7fffffffULL;


PianoPIRServer::PianoPIRServer(const PianoPIRConfig& config,
                               std::vector<uint64_t> rawDB)
        : config(config), rawDB(std::move(rawDB)) {}

std::vector<uint64_t> PianoPIRServer::NonePrivateQuery(uint64_t idx) const {
    std::vector<uint64_t> ret(config.DBEntrySize, 0);

    if (idx >= config.DBSize) {
        // Check padding region
        if (idx < config.ChunkSize * config.SetSize) {
            return ret; // zero entry for padded region
        }
        throw std::out_of_range("NonePrivateQuery: idx out of range");
    }

    const uint64_t base = idx * config.DBEntrySize;
    std::copy(rawDB.begin() + base,
              rawDB.begin() + base + config.DBEntrySize,
              ret.begin());
    return ret;
}

std::vector<uint64_t> PianoPIRServer::PrivateQuery(
        const std::vector<uint32_t>& offsets) const {

    assert(offsets.size() == config.SetSize);
    std::vector<uint64_t> ret(config.DBEntrySize, 0);

    for (uint64_t i = 0; i < config.SetSize; ++i) {
        uint64_t idx = static_cast<uint64_t>(offsets[i]) + i * config.ChunkSize;
        if (idx >= config.DBSize) continue;

        const uint64_t base = idx * config.DBEntrySize;
        EntryXor(ret.data(),
                 rawDB.data() + base,
                 config.DBEntrySize);
    }
    return ret;
}

PianoPIRClient::PianoPIRClient(const PianoPIRConfig& config)
        : config(config)
        , masterKey(RandKey())
        , longKey(GetLongKey(masterKey))
        , finishedQueryNum(0)
{
    double dbSizeF = static_cast<double>(config.DBSize);
    maxQueryNum = static_cast<uint64_t>(
            std::sqrt(dbSizeF) * std::log(dbSizeF));

    primaryHintNum = primaryNumParam(
            static_cast<double>(maxQueryNum),
            static_cast<double>(config.ChunkSize),
            config.FailureProbLog2 + 1);
    // Round up to nearest multiple of ThreadNum
    primaryHintNum =
            (primaryHintNum + config.ThreadNum - 1) / config.ThreadNum * config.ThreadNum;

    maxQueryPerChunk =
            3 * maxQueryNum / config.SetSize;
    maxQueryPerChunk =
            (maxQueryPerChunk + config.ThreadNum - 1) / config.ThreadNum * config.ThreadNum;

    // Allocate all client-side state
    queryHistogram.assign(config.SetSize, 0);

    primaryShortTag.resize(primaryHintNum);
    primaryParity.resize(primaryHintNum * config.DBEntrySize, 0);
    primaryProgramPoint.resize(primaryHintNum);

    replacementIdx.resize(config.SetSize);
    replacementVal.resize(config.SetSize);
    backupShortTag.resize(config.SetSize);
    backupParity.resize(config.SetSize);

    for (uint64_t i = 0; i < config.SetSize; ++i) {
        replacementIdx[i].resize(maxQueryPerChunk);
        replacementVal[i].resize(maxQueryPerChunk * config.DBEntrySize, 0);
        backupShortTag[i].resize(maxQueryPerChunk);
        backupParity[i].resize(maxQueryPerChunk * config.DBEntrySize, 0);
    }
}

void PianoPIRClient::Initialization() {
    finishedQueryNum = 0;

    // Resample master key and expand long key
    masterKey = RandKey();
    longKey  = GetLongKey(masterKey);

    std::fill(queryHistogram.begin(), queryHistogram.end(), 0);

    // Re-initialize primary hints
    primaryShortTag.resize(primaryHintNum);
    primaryParity.assign(primaryHintNum * config.DBEntrySize, 0);
    primaryProgramPoint.resize(primaryHintNum);

    uint64_t shortTagCount = 0;
    for (uint64_t i = 0; i < primaryHintNum; ++i) {
        primaryShortTag[i]    = shortTagCount;
        primaryProgramPoint[i] = DefaultProgramPoint;
        shortTagCount++;
    }
    // primaryParity already zeroed above

    // Re-initialize backup hints and replacement store
    for (uint64_t i = 0; i < config.SetSize; ++i) {
        replacementIdx[i].resize(maxQueryPerChunk);
        replacementVal[i].assign(maxQueryPerChunk * config.DBEntrySize, 0);
        backupShortTag[i].resize(maxQueryPerChunk);
        backupParity[i].assign(maxQueryPerChunk * config.DBEntrySize, 0);

        for (uint64_t j = 0; j < maxQueryPerChunk; ++j) {
            replacementIdx[i][j]  = DefaultProgramPoint;
            backupShortTag[i][j]  = shortTagCount;
            shortTagCount++;
        }
    }

    localCache.clear();
}

void PianoPIRClient::Preprocessing(const std::vector<uint64_t>& rawDB) {
    Initialization(); // clean state first

    if (skipPrep) return;

    for (uint64_t i = 0; i < config.SetSize; ++i) {
        uint64_t start = i * config.ChunkSize;
        uint64_t end   = (i + 1) * config.ChunkSize;
        uint64_t endElem = end * config.DBEntrySize;

        if (endElem > rawDB.size()) {
            // Pad the last chunk with zeros
            std::vector<uint64_t> tmpChunk(config.ChunkSize * config.DBEntrySize, 0);
            uint64_t startElem = start * config.DBEntrySize;
            uint64_t available = rawDB.size() > startElem ? rawDB.size() - startElem : 0;
            std::copy(rawDB.data() + startElem,
                      rawDB.data() + startElem + available,
                      tmpChunk.data());
            UpdatePreprocessing(i, tmpChunk.data(), tmpChunk.size());
        } else {
            uint64_t startElem = start * config.DBEntrySize;
            UpdatePreprocessing(i,
                                rawDB.data() + startElem,
                                config.ChunkSize * config.DBEntrySize);
        }
    }
}

void PianoPIRClient::UpdatePreprocessing(uint64_t chunkId,
                                         const uint64_t* chunk,
                                         size_t chunkElems) {
    // Local RNG for replacement sampling
    auto seed = static_cast<uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::mt19937_64 rng(seed);

    assert(chunkElems >= config.ChunkSize * config.DBEntrySize);
    (void) chunkElems;

    const uint64_t chunkMask = config.ChunkSize - 1; // ChunkSize is a power of 2

    // ---- 1. Update primary hints ----------------------------------------
    for (uint64_t i = 0; i < primaryHintNum; ++i) {
        uint64_t offset =
                PRFEvalWithLongKeyAndTag(longKey, primaryShortTag[i], chunkId)
                & chunkMask;

        EntryXor(primaryParity.data() + i * config.DBEntrySize,
                 chunk + offset * config.DBEntrySize,
                 config.DBEntrySize);
    }

    // ---- 2. Update backup hints -----------------------------------------
    for (uint64_t i = 0; i < config.SetSize; ++i) {
        if (i == chunkId) continue; // skip own chunk
        for (uint64_t j = 0; j < maxQueryPerChunk; ++j) {
            uint64_t offset =
                    PRFEvalWithLongKeyAndTag(longKey, backupShortTag[i][j], chunkId)
                    & chunkMask;

            EntryXor(backupParity[i].data() + j * config.DBEntrySize,
                     chunk + offset * config.DBEntrySize,
                     config.DBEntrySize);
        }
    }

    // ---- 3. Store replacements ------------------------------------------
    for (uint64_t j = 0; j < maxQueryPerChunk; ++j) {
        uint64_t offset = rng() & chunkMask;
        replacementIdx[chunkId][j] = offset + chunkId * config.ChunkSize;
        std::copy(chunk + offset * config.DBEntrySize,
                  chunk + (offset + 1) * config.DBEntrySize,
                  replacementVal[chunkId].data() + j * config.DBEntrySize);
    }


}

std::vector<uint64_t> PianoPIRClient::Query(uint64_t idx,
                                            PianoPIRServer& server,
                                            bool realQuery) {
    std::vector<uint64_t> ret(config.DBEntrySize, 0);

    // Dummy query path: just send random offsets for cover traffic
    if (!realQuery) {
        std::mt19937_64 rng(
                static_cast<uint64_t>(
                        std::chrono::high_resolution_clock::now().time_since_epoch().count()));
        std::vector<uint32_t> offsets(config.SetSize);
        for (uint64_t i = 0; i < config.SetSize; ++i) {
            offsets[i] = static_cast<uint32_t>(rng() & (config.ChunkSize - 1));
        }
        server.PrivateQuery(offsets); // result discarded
        return ret;
    }

    if (idx >= config.DBSize) {
        throw std::out_of_range("Query: idx out of range");
    }

    // Local cache hit
    auto cit = localCache.find(idx);
    if (cit != localCache.end()) {
        return cit->second;
    }

    if (finishedQueryNum >= maxQueryNum) {
        throw std::runtime_error("Query: exceeded maximum number of queries");
    }

    uint64_t chunkId = idx / config.ChunkSize;
    uint64_t offset  = idx % config.ChunkSize;

    if (queryHistogram[chunkId] >= maxQueryPerChunk) {
        throw std::runtime_error("Query: too many queries in chunk "
                                 + std::to_string(chunkId));
    }

    const uint64_t chunkMask = config.ChunkSize - 1;

    // ---- Find a hit hint in the primary table ---------------------------
    uint64_t hitId = DefaultProgramPoint;
    for (uint64_t i = 0; i < primaryHintNum; ++i) {
        uint64_t hintOffset =
                PRFEvalWithLongKeyAndTag(longKey, primaryShortTag[i], chunkId)
                & chunkMask;

        if (hintOffset == offset) {
            // If this hint was programmed in a *different* chunk, it still
            // covers chunkId naturally — accept it.
            // If it was programmed in chunkId itself, skip (stale).
            bool programmmedInSameChunk =
                    (primaryProgramPoint[i] != DefaultProgramPoint) &&
                    (primaryProgramPoint[i] / config.ChunkSize == chunkId);

            if (!programmmedInSameChunk) {
                hitId = i;
                break;
            }
        }
    }

    if (hitId == DefaultProgramPoint) {
        throw std::runtime_error("Query: no hit hint in primary table for idx="
                                 + std::to_string(idx));
    }

    // ---- Expand the hit hint into a full query set ----------------------
    std::vector<uint64_t> querySet(config.SetSize);
    for (uint64_t i = 0; i < config.SetSize; ++i) {
        uint64_t hintOffset =
                PRFEvalWithLongKeyAndTag(longKey, primaryShortTag[hitId], i)
                & chunkMask;
        querySet[i] = i * config.ChunkSize + hintOffset;
    }

    // If this hint was programmed, enforce the programmed position
    if (primaryProgramPoint[hitId] != DefaultProgramPoint) {
        uint64_t progChunk = primaryProgramPoint[hitId] / config.ChunkSize;
        querySet[progChunk] = primaryProgramPoint[hitId];
    }

    // Replace the target chunk's element with a random replacement index
    uint64_t inGroupIdx = queryHistogram[chunkId];
    uint64_t replIdx    = replacementIdx[chunkId][inGroupIdx];
    const uint64_t* replVal =
            replacementVal[chunkId].data() + inGroupIdx * config.DBEntrySize;

    querySet[chunkId] = replIdx;

    // ---- Send private query to server -----------------------------------
    std::vector<uint32_t> querySetOffset(config.SetSize);
    for (uint64_t i = 0; i < config.SetSize; ++i) {
        querySetOffset[i] = static_cast<uint32_t>(querySet[i] & chunkMask);
    }

    std::vector<uint64_t> response = server.PrivateQuery(querySetOffset);

    // ---- Reconstruct the answer -----------------------------------------
    // XOR out replacement value
    EntryXor(response.data(), replVal, config.DBEntrySize);
    // XOR in original hint parity
    EntryXor(response.data(),
             primaryParity.data() + hitId * config.DBEntrySize,
             config.DBEntrySize);
    // response now holds DB[idx]

    // ---- Refresh: replace consumed hint with a backup hint --------------
    primaryShortTag[hitId] = backupShortTag[chunkId][inGroupIdx];

    std::copy(backupParity[chunkId].data() + inGroupIdx * config.DBEntrySize,
              backupParity[chunkId].data() + (inGroupIdx + 1) * config.DBEntrySize,
              primaryParity.data() + hitId * config.DBEntrySize);

    primaryProgramPoint[hitId] = idx;

    // Also XOR in the current response to the new hint's parity
    EntryXor(primaryParity.data() + hitId * config.DBEntrySize,
             response.data(),
             config.DBEntrySize);

    // ---- Update bookkeeping ---------------------------------------------
    ++finishedQueryNum;
    ++queryHistogram[chunkId];
    localCache[idx] = response;

    return response;
}

double PianoPIRClient::LocalStorageSize() const {
    double sz = 0.0;
    sz += static_cast<double>(primaryHintNum) * 8;                                          // primaryShortTag
    sz += static_cast<double>(primaryHintNum) * static_cast<double>(config.DBEntryByteNum); // primaryParity
    sz += static_cast<double>(primaryHintNum) * 8;                                          // primaryProgramPoint
    double totalBackup = static_cast<double>(config.SetSize) * static_cast<double>(maxQueryPerChunk);
    sz += totalBackup * 8;                                          // replacementIdx
    sz += totalBackup * static_cast<double>(config.DBEntryByteNum); // replacementVal
    sz += totalBackup * 8;                                          // backupShortTag
    sz += totalBackup * static_cast<double>(config.DBEntryByteNum); // backupParity
    return sz;
}

void PianoPIRClient::PrintStorageBreakdown() const {
    printf("primary hint short tag   = %llu bytes\n",
           (unsigned long long)(primaryHintNum * 4));
    printf("primary parity           = %llu bytes\n",
           (unsigned long long)(primaryHintNum * config.DBEntryByteNum));
    printf("primary program point    = %llu bytes\n",
           (unsigned long long)(primaryHintNum * 4));
    uint64_t totalBackup = config.SetSize * maxQueryPerChunk;
    printf("replacement indices      = %llu bytes\n",
           (unsigned long long)(totalBackup * 4));
    printf("replacement values       = %llu bytes\n",
           (unsigned long long)(totalBackup * config.DBEntryByteNum));
    printf("backup short tag         = %llu bytes\n",
           (unsigned long long)(totalBackup * 4));
    printf("backup parities          = %llu bytes\n",
           (unsigned long long)(totalBackup * config.DBEntryByteNum));
}

PianoPIR::PianoPIR(uint64_t DBSize,
                   uint64_t DBEntryByteNum,
                   std::vector<uint64_t> rawDB,
                   uint64_t FailureProbLog2) {
    uint64_t DBEntrySize = DBEntryByteNum / 8;
    if (rawDB.size() != DBSize * DBEntrySize) {
        throw std::invalid_argument(
                "PianoPIR: rawDB size mismatch: expected "
                + std::to_string(DBSize * DBEntrySize)
                + " got " + std::to_string(rawDB.size()));
    }

    // ChunkSize = smallest power of 2 >= 2*sqrt(DBSize)
    uint64_t targetChunkSize = static_cast<uint64_t>(2.0 * std::sqrt(static_cast<double>(DBSize)));
    uint64_t ChunkSize = 1;
    while (ChunkSize < targetChunkSize) ChunkSize *= 2;

    uint64_t SetSize = static_cast<uint64_t>(
            std::ceil(static_cast<double>(DBSize) / static_cast<double>(ChunkSize)));
    // Round up to next multiple of 4
    SetSize = (SetSize + 3) / 4 * 4;

    config = PianoPIRConfig{
            .DBEntryByteNum  = DBEntryByteNum,
            .DBEntrySize     = DBEntrySize,
            .DBSize          = DBSize,
            .ChunkSize       = ChunkSize,
            .SetSize         = SetSize,
            .ThreadNum       = 8,
            .FailureProbLog2 = FailureProbLog2,
    };

    server = std::make_unique<PianoPIRServer>(config, std::move(rawDB));
    client = std::make_unique<PianoPIRClient>(config);
}

void PianoPIR::Preprocessing() {
    client->Preprocessing(server->RawDB());
}

void PianoPIR::DummyPreprocessing() {
    client->Initialization();
    client->skipPrep = true;
}

std::vector<uint64_t> PianoPIR::Query(uint64_t idx, bool realQuery) {
    // Auto-repreprocess if we've exhausted the query budget
    if (client->FinishedQueryNum() == client->MaxQueryNum()) {
        printf("exceeded maximum queries (%llu), redoing preprocessing\n",
               (unsigned long long)client->MaxQueryNum());
        client->Preprocessing(server->RawDB());
    }
    return client->Query(idx, *server, realQuery);
}

double PianoPIR::LocalStorageSize() const {
    return client->LocalStorageSize();
}

double PianoPIR::CommCostPerQuery() const {
    // Upload: SetSize uint32 (offsets)
    // Download: DBEntrySize uint64 (XOR response)
    return static_cast<double>(config.SetSize * 4 + config.DBEntrySize * 8);
}
/***
SimpleBatchPianoPIR::SimpleBatchPianoPIR(uint64_t DBSize,
                                         uint64_t DBEntryByteNum,
                                         uint64_t BatchSize,
                                         std::vector<uint64_t> rawDB,
                                         uint64_t FailureProbLog2) {
    const uint64_t DBEntrySize = DBEntryByteNum / 8;

    if (rawDB.size() != DBSize * DBEntrySize) {
        throw std::invalid_argument(
                "SimpleBatchPianoPIR: rawDB size mismatch: expected "
                + std::to_string(DBSize * DBEntrySize)
                + " got " + std::to_string(rawDB.size()));
    }

    const uint64_t PartitionNum  = BatchSize / kRealQueryPerPartition;
    // Round up: PartitionSize = ceil(DBSize / PartitionNum)
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
        uint64_t start = i * PartitionSize;
        uint64_t end   = std::min((i + 1) * PartitionSize, DBSize);
        uint64_t partDBSize = end - start;

        // Slice the raw DB for this partition
        std::vector<uint64_t> partDB(
                rawDB.begin() + start * DBEntrySize,
                rawDB.begin() + end   * DBEntrySize);

        subPIR.push_back(std::make_unique<PianoPIR>(
                partDBSize, DBEntryByteNum, std::move(partDB), FailureProbLog2));
    }
}

void SimpleBatchPianoPIR::PrintInfo() const {
    const double DBSizeInBytes =
            static_cast<double>(config.DBSize) * static_cast<double>(config.DBEntryByteNum);

    printf("-----------BatchPIR config --------\n");
    printf("DB size in MB = %.2f\n", DBSizeInBytes / 1024.0 / 1024.0);
    printf("DBSize: %llu  DBEntryByteNum: %llu  BatchSize: %llu  "
           "PartitionNum: %llu  PartitionSize: %llu  "
           "ThreadNum: %llu  FailureProbLog2: %llu\n",
           (unsigned long long)config.DBSize,
           (unsigned long long)config.DBEntryByteNum,
           (unsigned long long)config.BatchSize,
           (unsigned long long)config.PartitionNum,
           (unsigned long long)config.PartitionSize,
           (unsigned long long)config.ThreadNum,
           (unsigned long long)config.FailureProbLog2);

    const uint64_t maxQuery =
            subPIR[0]->Config().DBSize > 0
            ? subPIR[0]->client_MaxQueryNum() / kQueryPerPartition
            : 0;

    printf("max query num            = %llu\n", (unsigned long long)maxQuery);
    printf("total storage            = %.2f MB\n", LocalStorageSize() / 1024.0 / 1024.0);
    printf("comm cost per batch      = %.2f KB\n",
           static_cast<double>(CommCostPerBatchOnline()) / 1024.0);

    if (maxQuery > 0) {
        double amortPrep = DBSizeInBytes / static_cast<double>(maxQuery) / 1024.0;
        printf("amortized prep comm cost = %.2f KB\n", amortPrep);
        printf("total amortized comm     = %.2f KB\n",
               amortPrep + static_cast<double>(CommCostPerBatchOnline()) / 1024.0);
    }
    printf("-----------------------------\n");
}

void SimpleBatchPianoPIR::RecordStats(double prepTimeSecs) {
    preprocessingTime      = prepTimeSecs;
    localStorage           = static_cast<uint64_t>(LocalStorageSize());
    commCostPerBatchOnline = CommCostPerBatchOnline();
    SupportBatchNum         = subPIR[0]->client_MaxQueryNum() / kQueryPerPartition;

    const double DBSizeInBytes =
            static_cast<double>(config.DBSize) * static_cast<double>(config.DBEntryByteNum);
    commCostPerBatchOffline =
            SupportBatchNum > 0
            ? static_cast<uint64_t>(DBSizeInBytes / static_cast<double>(SupportBatchNum))
            : 0;
}

void SimpleBatchPianoPIR::Preprocessing() {
    PrintInfo();

    FinishedBatchNum       = 0;
    QueriesMadeInPartition = 0;

    const uint64_t threadNum    = config.ThreadNum;
    const uint64_t partitionNum = config.PartitionNum;

    auto t0 = std::chrono::high_resolution_clock::now();

    if (threadNum <= 1) {
        // Single-threaded path (default: kBatchThreadNum = 1)
        for (uint64_t i = 0; i < partitionNum; ++i) {
            subPIR[i]->Preprocessing();
        }
    } else {
        // Multi-threaded path: distribute partitions across threads
        const uint64_t perThread =
                (partitionNum + threadNum - 1) / threadNum;

        std::vector<std::thread> threads;
        threads.reserve(threadNum);

        for (uint64_t tid = 0; tid < threadNum; ++tid) {
            uint64_t start = tid * perThread;
            uint64_t end   = std::min((tid + 1) * perThread, partitionNum);

            threads.emplace_back([this, start, end]() {
                for (uint64_t i = start; i < end; ++i) {
                    subPIR[i]->Preprocessing();
                }
            });
        }
        for (auto& t : threads) t.join();
    }

    double elapsed =
            std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - t0).count();

    printf("Preprocessing time = %.3f s\n", elapsed);
    RecordStats(elapsed);
}

void SimpleBatchPianoPIR::DummyPreprocessing() {
    PrintInfo();
    for (uint64_t i = 0; i < config.PartitionNum; ++i) {
        subPIR[i]->DummyPreprocessing();
    }
    printf("Skipping Prep\n");
    RecordStats(0.0);
}

std::vector<std::vector<uint64_t>>
SimpleBatchPianoPIR::Query(const std::vector<uint64_t>& idx) {
    const uint64_t partitionNum    = config.PartitionNum;
    const uint64_t partitionSize   = config.PartitionSize;
    const uint64_t dbEntrySize     = config.DBEntrySize;

    // How many real queries to issue per partition
    // (same logic as Go: queryNumToMake = len(idx) / PartitionNum)
    const int queryNumToMake = static_cast<int>(idx.size()) / static_cast<int>(partitionNum);

    // ---- 1. Arrange queries into partitions (first-come-first-served) ----
    std::vector<std::vector<uint64_t>> partitionQueries(partitionNum);
    for (uint64_t globalIdx : idx) {
        uint64_t pId = globalIdx / partitionSize;
        if (pId >= partitionNum) pId = partitionNum - 1; // clamp last partition
        partitionQueries[pId].push_back(globalIdx);
    }

    // ---- 2. Issue queries to each sub-PIR --------------------------------
    // Map: global index → response entry
    std::unordered_map<uint64_t, std::vector<uint64_t>> responses;
    responses.reserve(idx.size());

    for (uint64_t i = 0; i < partitionNum; ++i) {
        // Pad with sentinel if this partition received fewer than queryNumToMake queries
        while (static_cast<int>(partitionQueries[i].size()) < queryNumToMake) {
            partitionQueries[i].push_back(kDefaultValue);
        }

        // Issue exactly queryNumToMake queries to subPIR[i]
        for (int j = 0; j < queryNumToMake; ++j) {
            uint64_t globalIdx = partitionQueries[i][j];

            if (globalIdx == kDefaultValue) {
                // Dummy cover query — result discarded
                subPIR[i]->Query(0, false);
            } else {
                // Real query: translate global index to partition-local index
                uint64_t localIdx = globalIdx - i * partitionSize;
                std::vector<uint64_t> entry = subPIR[i]->Query(localIdx, true);
                responses[globalIdx] = std::move(entry);
            }
        }
    }

    // ---- 3. Assemble output in original query order ----------------------
    std::vector<std::vector<uint64_t>> ret(idx.size());
    for (size_t i = 0; i < idx.size(); ++i) {
        auto it = responses.find(idx[i]);
        if (it != responses.end()) {
            ret[i] = it->second;
        } else {
            // Unanswered (partition overflow or dummy): return zero entry
            ret[i].assign(dbEntrySize, 0);
        }
    }

    // ---- 4. Check if sub-PIR needs to be re-preprocessed ----------------
    // Trigger 2 batches before exhaustion (-2 matches Go's MaxQueryNum-2 guard)
    const uint64_t maxQ = subPIR[0]->client_MaxQueryNum();
    if (QueriesMadeInPartition + 2 >= maxQ) {
        printf("Redo preprocessing. Made %llu batches (%llu queries in partition).\n",
               (unsigned long long)FinishedBatchNum,
               (unsigned long long)QueriesMadeInPartition);
        Preprocessing();
    } else {
        const uint64_t batchSize = config.BatchSize;
        FinishedBatchNum       += idx.size() / batchSize;
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
        total += p->CommCostPerQuery() * static_cast<double>(kQueryPerPartition);
    }
    return static_cast<uint64_t>(total);
}
***/