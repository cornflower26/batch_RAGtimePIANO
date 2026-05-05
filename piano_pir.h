//
// Created by Antonia Januszewicz on 4/8/26.
//

#ifndef BATCH_PIR_PIANO_PIR_H
#define BATCH_PIR_PIANO_PIR_H

#include "util.cpp"
#include <unordered_map>
#include <vector>
#include <memory>

struct PianoPIRConfig {
    uint64_t DBEntryByteNum; // bytes per DB entry
    uint64_t DBEntrySize;    // uint64s per DB entry  (= DBEntryByteNum / 8)
    uint64_t DBSize;         // number of DB entries
    uint64_t ChunkSize;      // entries per chunk (power of 2)
    uint64_t SetSize;        // number of chunks (rounded up, multiple of 4)
    uint64_t ThreadNum;      // parallelism hint (currently informational)
    uint64_t FailureProbLog2;
};

class PianoPIRServer {
public:
    PianoPIRServer(const PianoPIRConfig& config, std::vector<uint64_t> rawDB);

    // Non-private point query (for debugging / correctness checks)
    std::vector<uint64_t> NonePrivateQuery(uint64_t idx) const;

    // Private query: XOR of one element per chunk selected by offsets[i]
    // offsets has SetSize entries; offsets[i] is the intra-chunk offset for chunk i
    std::vector<uint64_t> PrivateQuery(const std::vector<uint32_t>& offsets) const;

    const PianoPIRConfig& Config() const { return config; }
    const std::vector<uint64_t>& RawDB() const { return rawDB; }

private:
    PianoPIRConfig config;
    std::vector<uint64_t> rawDB;
};

class PianoPIRClient {
public:
    explicit PianoPIRClient(const PianoPIRConfig& config);

    // Run full preprocessing over the raw DB
    void Preprocessing(const std::vector<uint64_t>& rawDB);

    // Process one chunk during (re)preprocessing
    void UpdatePreprocessing(uint64_t chunkId, const uint64_t* chunk, size_t chunkElems);

    // Issue a private query for DB index idx against server.
    // realQuery=false sends a dummy (random) query for bandwidth obfuscation.
    std::vector<uint64_t> Query(uint64_t idx, PianoPIRServer& server, bool realQuery);

    // Approximate local storage in bytes
    double LocalStorageSize() const;
    void   PrintStorageBreakdown() const;

    uint64_t MaxQueryNum()      const { return maxQueryNum; }
    uint64_t FinishedQueryNum() const { return finishedQueryNum; }
    uint64_t MaxQueryPerChunk()  const { return maxQueryPerChunk; }
    uint64_t PrimaryHintNum() const {return primaryHintNum;}

    // Skip actual preprocessing (for benchmarking bandwidth only)
    bool skipPrep = false;

    void Initialization();

private:

    PianoPIRConfig config;

    PrfKey              masterKey;
    std::vector<uint32_t> longKey;

    uint64_t maxQueryNum;
    uint64_t finishedQueryNum;
    uint64_t primaryHintNum;
    uint64_t maxQueryPerChunk;

    std::vector<uint64_t> queryHistogram; // per-chunk query count

    // Primary hint table
    std::vector<uint64_t> primaryShortTag;      // [primaryHintNum]
    std::vector<uint64_t> primaryParity;         // [primaryHintNum * DBEntrySize]
    std::vector<uint64_t> primaryProgramPoint;   // [primaryHintNum]

    // Replacement store (one slot per chunk per in-chunk query)
    std::vector<std::vector<uint64_t>> replacementIdx; // [SetSize][maxQueryPerChunk]
    std::vector<std::vector<uint64_t>> replacementVal; // [SetSize][maxQueryPerChunk * DBEntrySize]

    // Backup hint table
    std::vector<std::vector<uint64_t>> backupShortTag; // [SetSize][maxQueryPerChunk]
    std::vector<std::vector<uint64_t>> backupParity;   // [SetSize][maxQueryPerChunk * DBEntrySize]

    // Local cache: idx -> entry
    std::unordered_map<uint64_t, std::vector<uint64_t>> localCache;
};

class PianoPIR {
public:
    // Construct from raw DB. DBEntryByteNum must be a multiple of 8.
    PianoPIR(uint64_t DBSize,
             uint64_t DBEntryByteNum,
             std::vector<uint64_t> rawDB,
             uint64_t FailureProbLog2 = 40);

    void Preprocessing();
    void DummyPreprocessing(); // skip for benchmarking

    // Query DB index idx. realQuery=false sends a cover query.
    std::vector<uint64_t> Query(uint64_t idx, bool realQuery = true);

    double   LocalStorageSize()  const;
    double   CommCostPerQuery()  const;
    double   UploadCostPerQuery() const;
    double   DownloadCostPerQuery() const;
    const PianoPIRConfig& Config() const { return config; }

    // Accessors into the client (used by SimpleBatchPianoPIR)
    uint64_t client_MaxQueryNum()     const { return client->MaxQueryNum(); }
    uint64_t client_MaxQueryPerChunk()const { return client->MaxQueryPerChunk(); }
    uint64_t client_PrimaryHintNum() const { return client->PrimaryHintNum();}
    uint64_t client_LocalStorageSize() const {return client->LocalStorageSize();}
    void client_PrintStorageBreakdown() const {client->PrintStorageBreakdown();}


private:
    PianoPIRConfig                  config;
    std::unique_ptr<PianoPIRClient> client;
    std::unique_ptr<PianoPIRServer> server;
};

static constexpr uint64_t kRealQueryPerPartition = 2; // mirrors Go RealQueryPerPartition
static constexpr uint64_t kQueryPerPartition      = 2; // mirrors Go QueryPerPartition
static constexpr uint64_t kBatchThreadNum         = 1; // mirrors Go ThreadNum const
static constexpr uint64_t kDefaultValue = 0xdeadbeef;

struct SimpleBatchPianoPIRConfig {
    uint64_t DBEntryByteNum;
    uint64_t DBEntrySize;
    uint64_t DBSize;
    uint64_t BatchSize;
    uint64_t PartitionNum;
    uint64_t PartitionSize;
    uint64_t ThreadNum;
    uint64_t FailureProbLog2;
};

class SimpleBatchPianoPIR {
public:
    SimpleBatchPianoPIR(uint64_t DBSize,
                        uint64_t DBEntryByteNum,
                        uint64_t BatchSize,
                        std::vector<uint64_t> rawDB,
                        uint64_t FailureProbLog2 = 40);

    // Full preprocessing of all partitions (multi-threaded)
    void Preprocessing();

    // Skip preprocessing — for bandwidth benchmarking only
    void DummyPreprocessing();

    // Batch query: idx is a list of global DB indices.
    // Returns one entry vector per input index (zero entry if unanswered).
    std::vector<std::vector<uint64_t>> Query(const std::vector<uint64_t>& idx);

    void   PrintInfo()   const;

    double LocalStorageSize()        const;
    uint64_t CommCostPerBatchOnline() const;
    uint64_t CommCostPerBatchOffline() const { return commCostPerBatchOffline; }
    double PreprocessingTime()        const { return preprocessingTime; }

    uint64_t subPIR_MaxQueryNum(int index) const {return subPIR[index]->client_MaxQueryNum();}
    uint64_t subPIR_PrimaryHintNum(int index) const {return subPIR[index]->client_PrimaryHintNum();}
    uint64_t subPIR_LocalStorageSize(int index) const {return subPIR[index]->client_LocalStorageSize();}

    void PrintSubPIRStorageBreakdown(int index) const {return subPIR[index]->client_PrintStorageBreakdown();}


    const SimpleBatchPianoPIRConfig& Config() const { return config; }

    // Public stats (mirrors Go exported fields)
    uint64_t FinishedBatchNum        = 0;
    uint64_t QueriesMadeInPartition  = 0;
    uint64_t SupportBatchNum         = 0;

private:
    void RecordStats(double prepTimeSecs);

    SimpleBatchPianoPIRConfig        config;
    std::vector<std::unique_ptr<PianoPIR>> subPIR;

    uint64_t localStorage            = 0;
    double   preprocessingTime       = 0.0;
    uint64_t commCostPerBatchOnline  = 0;
    uint64_t commCostPerBatchOffline = 0;
};


#endif //BATCH_PIR_PIANO_PIR_H
