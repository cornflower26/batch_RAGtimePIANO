// PIANO-RAG driver, BATCHED variant: encrypted centroid selection ->
// SimpleBatchPianoPIR over embeddings -> local FAISS re-rank -> Piano PIR over text.
//
// Identical to main.cpp (same flags, output format, truncation, refresh and
// cost reporting) except for PIR 1, which uses SimpleBatchPianoPIR:
//   * --batch-size B (default 64, even): the embedding DB is split into B/2
//     partitions and each batch sends 2 queries to every partition.
//   * --spread-clusters (default true): documents are laid out so each IVF list
//     is dealt round-robin across the partitions, so a query's candidates fit in
//     few batches. Results are always reported with ORIGINAL document ids, and
//     the text DB (PIR 2) keeps the original order.
//   * --min-batches N (default 0): pad every query to at least N batches with
//     dummy batches, so the batch count doesn't depend on the query.
//
// Changes vs. the original main:
//   * Embeddings are stored as float32, two per uint64_t (3072 B/entry instead of
//     6144), and every staging copy of a database is released as soon as it is
//     packed. The packed DBs are moved (not copied) into PianoPIR.
//   * Optional --embedding-bin: stream a raw float32 file instead of parsing JSON.
//   * Piano query budget (total and per-chunk) is tracked; when it would be
//     exceeded the client re-runs preprocessing and logs a "[refresh]" line.
//     Nothing is silently dropped any more.
//   * --query-list runs many queries in one process with setup done once.
//   * --work-dir isolates each process's keys / ciphertexts / top-k file, so
//     parallel runs no longer clobber each other. Stale top-k files are removed
//     before every query.
//   * Local CKKS path: centroids loaded once, each batch decrypted inside the
//     loop (no vector of thousands of ciphertexts), slot count checked,
//     nested OpenMP disabled, thread count configurable.
//   * Cost/timing reports use the number of queries actually issued, and the
//     centroid step is reported as per-query work.
//   * Input validation, stricter --mac parsing, and a top-level try/catch.
//   * PIR setup (server build, client preprocessing, refreshes) is timed per
//     PIR, and online plus offline communication is tracked per query and in
//     run totals.
//   * Auto truncation: if the database files hold more entries than --db-size,
//     only the first --db-size entries are packed into the PIR databases, and
//     the effect on the centroid lists is reported.

#include "piano_pir.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cctype>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <faiss/IndexFlat.h>
#include "openfhe.h"
#include "cryptocontext-ser.h"
#include "ciphertext-ser.h"
#include "key/key-ser.h"
#include "scheme/bfvrns/bfvrns-ser.h"
#include "scheme/bgvrns/bgvrns-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"
// NOTE: including a .cpp works only as long as this file is NOT also compiled as
// its own translation unit (same for util.cpp inside piano_pir.h). Otherwise you
// get multiple-definition link errors; the proper fix is a header + one .cpp.
#include "openfhe_batched_workload.cpp"

using namespace lbcrypto;
namespace fs = std::filesystem;

namespace {

    using Clock = std::chrono::steady_clock;

    double SecondsSince(Clock::time_point start) {
        return std::chrono::duration<double>(Clock::now() - start).count();
    }

    // ------------------------------------------------------------------ CLI --

    using CliOptions = std::unordered_map<std::string, std::string>;

    // --flag [value]. A flag followed by another --flag (or at the end) is "true".
    CliOptions ParseCliOptions(int argc, char **argv) {
        CliOptions opts;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a.rfind("--", 0) != 0) {
                continue;
            }
            if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                opts[a] = argv[++i];
            } else {
                opts[a] = "true";
            }
        }
        return opts;
    }

    std::string GetOpt(const CliOptions &opts, const std::string &key, const std::string &defaultValue) {
        auto it = opts.find(key);
        return it != opts.end() ? it->second : defaultValue;
    }

    bool GetBoolOpt(const CliOptions &opts, const std::string &key, bool defaultValue) {
        auto it = opts.find(key);
        if (it == opts.end()) {
            return defaultValue;
        }
        std::string v = it->second;
        std::transform(v.begin(), v.end(), v.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (v == "true" || v == "1" || v == "yes" || v == "on") return true;
        if (v == "false" || v == "0" || v == "no" || v == "off") return false;
        throw std::invalid_argument(key + " expects true/false, got '" + it->second + "'");
    }

    uint64_t GetPositiveU64Opt(const CliOptions &opts, const std::string &key, uint64_t defaultValue) {
        auto it = opts.find(key);
        if (it == opts.end()) {
            return defaultValue;
        }
        const std::string &s = it->second;
        unsigned long long v = 0;
        size_t pos = 0;
        try {
            if (s.empty() || s[0] == '-') throw std::invalid_argument("negative");
            v = std::stoull(s, &pos);
        } catch (const std::exception &) {
            pos = 0;
        }
        if (pos == 0 || pos != s.size() || v == 0) {
            throw std::invalid_argument(key + " must be a positive integer, got '" + s + "'");
        }
        return v;
    }

    int GetPositiveIntOpt(const CliOptions &opts, const std::string &key, int defaultValue) {
        const uint64_t v = GetPositiveU64Opt(opts, key, static_cast<uint64_t>(defaultValue));
        if (v > static_cast<uint64_t>(INT_MAX)) {
            throw std::invalid_argument(key + " is too large");
        }
        return static_cast<int>(v);
    }

    // One query-vector path per line; blank lines and lines starting with '#' are skipped.
    std::vector<std::string> ReadQueryList(const std::string &path) {
        std::ifstream in(path);
        if (!in) {
            throw std::runtime_error("Cannot open --query-list file " + path);
        }
        std::vector<std::string> out;
        std::string line;
        while (std::getline(in, line)) {
            const auto b = line.find_first_not_of(" \t\r");
            if (b == std::string::npos || line[b] == '#') {
                continue;
            }
            const auto e = line.find_last_not_of(" \t\r");
            out.push_back(line.substr(b, e - b + 1));
        }
        if (out.empty()) {
            throw std::runtime_error("--query-list file " + path + " contains no query paths");
        }
        return out;
    }

    // ---------------------------------------------------- DB size handling --

    // A packed database plus the number of entries the source file actually had.
    // If the file is larger than --db-size, only the first dbSize entries are
    // packed; the rest are never copied into the PIR database.
    struct PackedDB {
        std::vector<uint64_t> raw;
        uint64_t sourceEntries = 0;
    };

    void RequireEnoughEntries(const std::string &what, uint64_t sourceEntries, uint64_t dbSize) {
        if (sourceEntries < dbSize) {
            throw std::runtime_error(what + " has " + std::to_string(sourceEntries) +
                                     " entries, fewer than --db-size " + std::to_string(dbSize));
        }
    }

    void ReportTruncation(const std::string &what, uint64_t sourceEntries, uint64_t dbSize) {
        if (sourceEntries > dbSize) {
            std::cout << "Truncating " << what << ": using the first " << dbSize << " of "
                      << sourceEntries << " entries (--db-size " << dbSize << ")" << std::endl;
        } else {
            std::cout << what << ": using all " << sourceEntries << " entries" << std::endl;
        }
    }

    // Lists entries equal to 2^64-1 are the -1 padding in lists.json.
    constexpr uint64_t kPaddingId = UINT64_MAX;

    // Summarises how truncation affects the IVF lists: centroids whose documents
    // all lie at or beyond --db-size will retrieve nothing if selected.
    void ReportMappingCoverage(const std::unordered_map<uint64_t, std::vector<uint64_t>> &mapping,
                               uint64_t dbSize, uint64_t sourceEntries) {
        uint64_t truncatedIds = 0, invalidIds = 0, emptyCentroids = 0, partialCentroids = 0;
        for (const auto &[centroid, ids]: mapping) {
            (void) centroid;
            uint64_t inRange = 0, cutHere = 0;
            for (uint64_t id: ids) {
                if (id == kPaddingId) continue;
                if (id >= sourceEntries) {
                    ++invalidIds;
                } else if (id >= dbSize) {
                    ++cutHere;
                } else {
                    ++inRange;
                }
            }
            truncatedIds += cutHere;
            if (inRange == 0) {
                ++emptyCentroids;
            } else if (cutHere > 0) {
                ++partialCentroids;
            }
        }
        if (truncatedIds > 0) {
            std::cout << "Centroid mapping under truncation: " << truncatedIds << " listed ids are >= --db-size; "
                      << emptyCentroids << " of " << mapping.size() << " centroids have no documents left and "
                      << partialCentroids << " lose some. Queries selecting them retrieve fewer (or no) "
                      << "documents; those slots are sent as dummy PIR queries." << std::endl;
        }
        if (invalidIds > 0) {
            std::cerr << "warning: centroid mapping lists " << invalidIds << " ids beyond the end of the "
                      << "database files (" << sourceEntries << " entries); check that lists.json matches "
                      << "the embedding/text files" << std::endl;
        }
    }

    // ------------------------------------------------------ Text DB packing --

    constexpr uint64_t kTextEntryChars = 1024;  // max bytes of text per entry
    static_assert(kTextEntryChars % 8 == 0, "entry chars must be a multiple of 8");
    constexpr uint64_t kTextWords = kTextEntryChars / 8;       // 128 packed words
    constexpr uint64_t kTextEntryBytes = kTextWords * 8;       // 1024 bytes

    // Packs up to nWords*8 bytes of s into words (byte j -> bits 8*(j%8) of word j/8).
    // `words` must be zero-initialized. Truncation backs off so a UTF-8 multibyte
    // character isn't split. Returns the number of bytes stored.
    size_t PackTextEntry(const std::string &s, uint64_t *words, size_t nWords) {
        const size_t maxBytes = nWords * 8;
        size_t len = std::min(s.size(), maxBytes);
        if (len < s.size()) {
            while (len > 0 && (static_cast<unsigned char>(s[len]) & 0xC0) == 0x80) {
                --len;
            }
        }
        for (size_t j = 0; j < len; ++j) {
            words[j / 8] |= static_cast<uint64_t>(static_cast<unsigned char>(s[j])) << (8 * (j % 8));
        }
        return len;
    }

    // Inverse of PackTextEntry: unpacks the words and strips the trailing zero padding.
    std::string UnpackTextEntry(const std::vector<uint64_t> &words) {
        std::string s;
        s.reserve(words.size() * 8);
        for (uint64_t w: words) {
            for (int b = 0; b < 8; ++b) {
                s.push_back(static_cast<char>((w >> (8 * b)) & 0xFF));
            }
        }
        const auto end = s.find_last_not_of('\0');
        s.erase(end == std::string::npos ? 0 : end + 1);
        return s;
    }

    PackedDB LoadTextRawDB(const std::string &path, uint64_t dbSize) {
        std::vector<std::string> docs = load_text_database(path);
        const uint64_t sourceEntries = docs.size();
        RequireEnoughEntries("Text DB", sourceEntries, dbSize);
        ReportTruncation("text database", sourceEntries, dbSize);
        docs.resize(dbSize);  // drop entries beyond --db-size before packing
        docs.shrink_to_fit();
        std::vector<uint64_t> raw(dbSize * kTextWords, 0);
        uint64_t truncated = 0;
        for (uint64_t i = 0; i < dbSize; ++i) {
            const size_t stored = PackTextEntry(docs[i], &raw[i * kTextWords], kTextWords);
            if (stored < docs[i].size()) {
                ++truncated;
            }
            std::string().swap(docs[i]);  // release as we go to keep peak memory down
        }
        if (truncated > 0) {
            std::cerr << "warning: " << truncated << " text entries exceeded " << kTextEntryChars
                      << " bytes and were truncated" << std::endl;
        }
        return {std::move(raw), sourceEntries};
    }

    // ------------------------------------------------- Embedding DB packing --

    constexpr uint64_t kEmbeddingDim = 768;
    static_assert(kEmbeddingDim % 2 == 0, "embedding dim must be even to pack two float32 per word");
    constexpr uint64_t kEmbeddingWords = kEmbeddingDim / 2;         // 384 words
    constexpr uint64_t kEmbeddingEntryBytes = kEmbeddingWords * 8;  // 3072 bytes (was 6144)

    void PackEmbeddingRow(const float *v, uint64_t *out) {
        for (uint64_t w = 0; w < kEmbeddingWords; ++w) {
            const uint64_t lo = std::bit_cast<uint32_t>(v[2 * w]);
            const uint64_t hi = std::bit_cast<uint32_t>(v[2 * w + 1]);
            out[w] = lo | (hi << 32);
        }
    }

    std::vector<float> UnpackEmbeddingRow(const std::vector<uint64_t> &words) {
        if (words.size() != kEmbeddingWords) {
            throw std::runtime_error("PIR embedding response has " + std::to_string(words.size()) +
                                     " words, expected " + std::to_string(kEmbeddingWords));
        }
        std::vector<float> v(kEmbeddingDim);
        for (uint64_t w = 0; w < kEmbeddingWords; ++w) {
            v[2 * w] = std::bit_cast<float>(static_cast<uint32_t>(words[w]));
            v[2 * w + 1] = std::bit_cast<float>(static_cast<uint32_t>(words[w] >> 32));
        }
        return v;
    }

    // Raw little-endian float32 file, row-major, kEmbeddingDim floats per row.
    // Only the first dbSize rows are read, so a larger file costs no extra memory.
    PackedDB LoadEmbeddingRawDBFromBinary(const std::string &path, uint64_t dbSize) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Cannot open --embedding-bin file " + path);
        }
        const uint64_t rowBytes = kEmbeddingDim * sizeof(float);
        const uint64_t fileBytes = fs::file_size(path);
        if (fileBytes % rowBytes != 0) {
            throw std::runtime_error(path + " is not a whole number of " + std::to_string(kEmbeddingDim) +
                                     "-dim float32 rows (" + std::to_string(fileBytes) + " bytes)");
        }
        const uint64_t sourceEntries = fileBytes / rowBytes;
        RequireEnoughEntries("Embedding file", sourceEntries, dbSize);
        ReportTruncation("embedding database", sourceEntries, dbSize);
        std::vector<uint64_t> raw(dbSize * kEmbeddingWords);
        std::vector<float> row(kEmbeddingDim);
        for (uint64_t i = 0; i < dbSize; ++i) {
            in.read(reinterpret_cast<char *>(row.data()), static_cast<std::streamsize>(rowBytes));
            if (!in) {
                throw std::runtime_error("Short read in " + path + " at row " + std::to_string(i));
            }
            PackEmbeddingRow(row.data(), &raw[i * kEmbeddingWords]);
        }
        return {std::move(raw), sourceEntries};
    }

    // The JSON loader parses the whole file; rows beyond dbSize are freed
    // immediately, before validation and packing.
    PackedDB LoadEmbeddingRawDBFromJson(const std::string &path, uint64_t dbSize) {
        std::vector<std::vector<uint64_t>> rows = load_json_distances(path);
        const uint64_t sourceEntries = rows.size();
        RequireEnoughEntries("Embedding DB", sourceEntries, dbSize);
        ReportTruncation("embedding database", sourceEntries, dbSize);
        rows.resize(dbSize);
        rows.shrink_to_fit();

        // Validate before packing: a row of the wrong length means the loader is
        // misreading the file (e.g. including the "id" field).
        uint64_t badRows = 0, firstBad = 0;
        size_t minLen = SIZE_MAX, maxLen = 0;
        for (uint64_t i = 0; i < dbSize; ++i) {
            const size_t len = rows[i].size();
            minLen = std::min(minLen, len);
            maxLen = std::max(maxLen, len);
            if (len != kEmbeddingDim) {
                if (badRows == 0) firstBad = i;
                ++badRows;
            }
        }
        if (badRows > 0) {
            const auto &row = rows[firstBad];
            std::ostringstream msg;
            msg << badRows << " of " << dbSize << " embedding rows are not " << kEmbeddingDim
                << " values long (min " << minLen << ", max " << maxLen << "). First bad row "
                << firstBad << " has " << row.size() << " values; first entries:";
            for (size_t k = 0; k < std::min<size_t>(3, row.size()); ++k) {
                msg << " [" << k << "] raw=" << row[k] << " as double=" << std::bit_cast<double>(row[k]);
            }
            msg << ". If rows are " << kEmbeddingDim + 1 << " long and [0] is the record id, "
                << "the loader is including \"id\" from {\"id\", \"vector\"} records.";
            throw std::runtime_error(msg.str());
        }

        std::vector<uint64_t> raw(dbSize * kEmbeddingWords);
        std::vector<float> row(kEmbeddingDim);
        for (uint64_t i = 0; i < dbSize; ++i) {
            for (uint64_t d = 0; d < kEmbeddingDim; ++d) {
                row[d] = static_cast<float>(std::bit_cast<double>(rows[i][d]));
            }
            PackEmbeddingRow(row.data(), &raw[i * kEmbeddingWords]);
            std::vector<uint64_t>().swap(rows[i]);  // release as we go
        }
        return {std::move(raw), sourceEntries};
    }

    // ------------------------------------------- Piano PIR with budget/refresh --

    // Wraps PianoPIR and refreshes the client (re-runs preprocessing) before the
    // query budget runs out. Two limits are tracked:
    //   - total queries since the last preprocessing (client_MaxQueryNum), and
    //   - real queries per chunk (client_MaxQueryPerChunk), since each real query
    //     consumes one backup hint / replacement slot in the chunk of its index.
    // Counting is conservative: queries that Piano might serve from its local
    // cache are still counted, so a refresh may happen slightly early, never late.
    class BudgetedPianoPIR {
    public:
        // Construction builds the server-side database; it is timed as "server setup".
        BudgetedPianoPIR(std::string name, uint64_t dbSize, uint64_t entryBytes,
                         std::vector<uint64_t> rawDB, uint64_t failProbLog2)
                : name_(std::move(name)) {
            const auto start = Clock::now();
            pir_ = std::make_unique<PianoPIR>(dbSize, entryBytes, std::move(rawDB), failProbLog2);
            serverSetupSeconds_ = SecondsSince(start);
            std::cout << name_ << " server setup time " << serverSetupSeconds_ << " s" << std::endl;
        }

        // Initial client preprocessing. Timed and charged as offline communication.
        void Preprocess() {
            const auto start = Clock::now();
            pir_->Preprocessing();
            ResetBudget();
            initialPrepSeconds_ = SecondsSince(start);
            initialOfflineBytes_ = OfflineBytesPerPreprocessing();
            std::cout << name_ << " client preprocessing time " << initialPrepSeconds_ << " s, offline download "
                      << FormatBytes(initialOfflineBytes_) << std::endl;
        }

        std::vector<uint64_t> Query(uint64_t idx, bool realQuery) {
            if (!prepared_) {
                throw std::logic_error(name_ + ": Query() called before Preprocess()");
            }
            if (usedSinceRefresh_ >= maxQueryNum_) {
                Refresh("total query budget reached");
            } else if (realQuery && chunkUsed_[ChunkOf(idx)] >= maxPerChunk_) {
                Refresh("per-chunk budget reached for chunk " + std::to_string(ChunkOf(idx)));
            }
            std::vector<uint64_t> result = pir_->Query(idx, realQuery);
            ++usedSinceRefresh_;
            onlineUploadBytes_ += pir_->UploadCostPerQuery();
            onlineDownloadBytes_ += pir_->DownloadCostPerQuery();
            if (realQuery) {
                ++chunkUsed_[ChunkOf(idx)];
                ++realIssued_;
            } else {
                ++dummyIssued_;
            }
            return result;
        }

        void PrintBudget() const {
            std::cout << name_ << ": max " << maxQueryNum_ << " queries per preprocessing, max "
                      << maxPerChunk_ << " per chunk (" << pir_->Config().SetSize << " chunks of "
                      << pir_->Config().ChunkSize << "), client storage "
                      << FormatBytes(static_cast<double>(pir_->client_LocalStorageSize())) << std::endl;
        }

        // Snapshot of the cumulative counters, used to report per-query deltas.
        struct Totals {
            double onlineUploadBytes = 0, onlineDownloadBytes = 0, refreshOfflineBytes = 0, refreshSeconds = 0;
            uint64_t queries = 0, refreshes = 0;
        };

        Totals Snapshot() const {
            return {onlineUploadBytes_, onlineDownloadBytes_, refreshOfflineBytes_, refreshSeconds_,
                    realIssued_ + dummyIssued_, refreshCount_};
        }

        // Prints this PIR's online and refresh costs since `before`.
        void PrintQueryCosts(const std::string &label, const Totals &before) const {
            const Totals now = Snapshot();
            // Raw numbers kept on their own lines so existing log parsers still work.
            std::cout << label << " Upload " << std::llround(now.onlineUploadBytes - before.onlineUploadBytes) << std::endl;
            std::cout << label << " Download " << std::llround(now.onlineDownloadBytes - before.onlineDownloadBytes) << std::endl;
            std::cout << label << " queries " << now.queries - before.queries << ", online "
                      << FormatBytes(now.onlineUploadBytes - before.onlineUploadBytes) << " up / "
                      << FormatBytes(now.onlineDownloadBytes - before.onlineDownloadBytes) << " down";
            if (now.refreshes > before.refreshes) {
                std::cout << "; " << (now.refreshes - before.refreshes) << " refresh(es): offline "
                          << FormatBytes(now.refreshOfflineBytes - before.refreshOfflineBytes) << " down, "
                          << now.refreshSeconds - before.refreshSeconds << " s";
            }
            std::cout << std::endl;
        }

        void PrintSummary() const {
            const double offlineTotal = initialOfflineBytes_ + refreshOfflineBytes_;
            std::cout << name_ << ":\n"
                      << "  setup time:          server " << serverSetupSeconds_ << " s, client preprocessing "
                      << initialPrepSeconds_ << " s\n"
                      << "  queries:             " << realIssued_ << " real + " << dummyIssued_ << " dummy\n"
                      << "  refreshes:           " << refreshCount_ << " (" << refreshSeconds_ << " s)\n"
                      << "  online upload:       " << FormatBytes(onlineUploadBytes_) << "\n"
                      << "  online download:     " << FormatBytes(onlineDownloadBytes_) << "\n"
                      << "  offline (initial):   " << FormatBytes(initialOfflineBytes_) << "\n"
                      << "  offline (refreshes): " << FormatBytes(refreshOfflineBytes_) << "\n"
                      << "  total communication: "
                      << FormatBytes(onlineUploadBytes_ + onlineDownloadBytes_ + offlineTotal) << "\n"
                      << "  client storage:      "
                      << FormatBytes(static_cast<double>(pir_->client_LocalStorageSize())) << std::endl;
        }

        double ServerSetupSeconds() const { return serverSetupSeconds_; }
        double InitialPrepSeconds() const { return initialPrepSeconds_; }
        double OnlineBytes() const { return onlineUploadBytes_ + onlineDownloadBytes_; }
        double OfflineBytes() const { return initialOfflineBytes_ + refreshOfflineBytes_; }
        uint64_t RefreshCount() const { return refreshCount_; }
        double RefreshSeconds() const { return refreshSeconds_; }
        const std::string &Name() const { return name_; }

        static std::string FormatBytes(double bytes) {
            static const char *units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
            double v = bytes;
            int u = 0;
            while (v >= 1024.0 && u < 4) {
                v /= 1024.0;
                ++u;
            }
            std::ostringstream os;
            os << static_cast<uint64_t>(bytes) << " B";
            if (u > 0) {
                os.setf(std::ios::fixed);
                os.precision(2);
                os << " (" << v << " " << units[u] << ")";
            }
            return os.str();
        }

    private:
        // Piano's offline phase streams the whole database to the client once
        // per preprocessing run: DBSize entries of DBEntryByteNum bytes.
        double OfflineBytesPerPreprocessing() const {
            return static_cast<double>(pir_->Config().DBSize) * static_cast<double>(pir_->Config().DBEntryByteNum);
        }

        uint64_t ChunkOf(uint64_t idx) const {
            const uint64_t chunk = idx / chunkSize_;
            if (chunk >= chunkUsed_.size()) {
                throw std::out_of_range(name_ + ": index " + std::to_string(idx) + " is outside the DB");
            }
            return chunk;
        }

        void ResetBudget() {
            maxQueryNum_ = pir_->client_MaxQueryNum();
            maxPerChunk_ = pir_->client_MaxQueryPerChunk();
            chunkSize_ = pir_->Config().ChunkSize;
            if (maxQueryNum_ == 0 || maxPerChunk_ == 0 || chunkSize_ == 0) {
                throw std::runtime_error(name_ + ": Piano reports a zero query budget or chunk size "
                                                 "after preprocessing");
            }
            chunkUsed_.assign(pir_->Config().SetSize, 0);
            usedSinceRefresh_ = 0;
            prepared_ = true;
        }

        void Refresh(const std::string &reason) {
            std::cout << "[refresh] " << name_ << ": " << reason << " (" << usedSinceRefresh_ << "/"
                      << maxQueryNum_ << " queries used since last preprocessing); "
                      << "re-running Piano preprocessing ..." << std::endl;
            const auto start = Clock::now();
            pir_->Preprocessing();
            ResetBudget();
            const double sec = SecondsSince(start);
            const double bytes = OfflineBytesPerPreprocessing();
            ++refreshCount_;
            refreshSeconds_ += sec;
            refreshOfflineBytes_ += bytes;
            std::cout << "[refresh] " << name_ << ": refresh #" << refreshCount_ << " completed in "
                      << sec << " s, offline download " << FormatBytes(bytes) << "; budget reset to "
                      << maxQueryNum_ << " queries" << std::endl;
        }

        std::string name_;
        std::unique_ptr<PianoPIR> pir_;
        bool prepared_ = false;
        uint64_t maxQueryNum_ = 0;
        uint64_t maxPerChunk_ = 0;
        uint64_t chunkSize_ = 0;
        uint64_t usedSinceRefresh_ = 0;
        std::vector<uint64_t> chunkUsed_;

        double serverSetupSeconds_ = 0.0;
        double initialPrepSeconds_ = 0.0;
        double initialOfflineBytes_ = 0.0;
        uint64_t refreshCount_ = 0;
        double refreshSeconds_ = 0.0;
        double refreshOfflineBytes_ = 0.0;
        double onlineUploadBytes_ = 0.0;
        double onlineDownloadBytes_ = 0.0;
        uint64_t realIssued_ = 0;
        uint64_t dummyIssued_ = 0;
    };

    // ---------------------------------- Batched PIR 1 (SimpleBatchPianoPIR) --

    constexpr uint64_t kUnplaced = UINT64_MAX;

    // Where each document sits in the PIR 1 database. SimpleBatchPianoPIR splits
    // the database into BatchSize/2 contiguous partitions and answers at most
    // 2 queries per partition per batch, so a query's candidates should be spread
    // over as many partitions as possible. With spreading on, every IVF list is
    // dealt round-robin across the partitions (the job repartitionEqualSize did
    // in the old batched main); with it off, the original order is kept.
    struct DocLayout {
        std::vector<uint64_t> posOfDoc;  // original document id -> PIR 1 position
        std::vector<uint64_t> docAtPos;  // PIR 1 position -> original document id
    };

    DocLayout BuildDocLayout(const std::unordered_map<uint64_t, std::vector<uint64_t>> &mapping,
                             uint64_t dbSize, uint64_t partitionNum, uint64_t partitionSize, bool spread) {
        DocLayout layout;
        layout.posOfDoc.assign(dbSize, kUnplaced);
        layout.docAtPos.assign(dbSize, kUnplaced);
        if (!spread) {
            std::iota(layout.posOfDoc.begin(), layout.posOfDoc.end(), 0);
            std::iota(layout.docAtPos.begin(), layout.docAtPos.end(), 0);
            return layout;
        }
        std::vector<uint64_t> capacity(partitionNum), filled(partitionNum, 0);
        for (uint64_t p = 0; p < partitionNum; ++p) {
            const uint64_t start = p * partitionSize;
            const uint64_t end = std::min((p + 1) * partitionSize, dbSize);
            capacity[p] = end > start ? end - start : 0;
        }
        uint64_t next = 0;
        auto place = [&](uint64_t doc) {
            for (uint64_t tries = 0; tries < partitionNum; ++tries) {
                const uint64_t p = next++ % partitionNum;
                if (filled[p] < capacity[p]) {
                    const uint64_t pos = p * partitionSize + filled[p]++;
                    layout.posOfDoc[doc] = pos;
                    layout.docAtPos[pos] = doc;
                    return;
                }
            }
            throw std::logic_error("BuildDocLayout: no free slot left");
        };
        std::vector<uint64_t> centroidIds;
        centroidIds.reserve(mapping.size());
        for (const auto &kv: mapping) centroidIds.push_back(kv.first);
        std::sort(centroidIds.begin(), centroidIds.end());
        for (uint64_t c: centroidIds) {
            for (uint64_t doc: mapping.at(c)) {
                if (doc < dbSize && layout.posOfDoc[doc] == kUnplaced) {
                    place(doc);
                }
            }
        }
        for (uint64_t doc = 0; doc < dbSize; ++doc) {  // documents in no list
            if (layout.posOfDoc[doc] == kUnplaced) {
                place(doc);
            }
        }
        return layout;
    }

    // Reorders the packed rows in place so row i ends up at posOfDoc[i]
    // (cycle-following; one extra row of memory instead of a second DB copy).
    void PermuteRows(std::vector<uint64_t> &raw, const std::vector<uint64_t> &posOfDoc, uint64_t wordsPerRow) {
        std::vector<uint64_t> dest = posOfDoc;
        for (uint64_t i = 0; i < dest.size(); ++i) {
            while (dest[i] != i) {
                const uint64_t j = dest[i];
                std::swap_ranges(raw.begin() + static_cast<std::ptrdiff_t>(i * wordsPerRow),
                                 raw.begin() + static_cast<std::ptrdiff_t>((i + 1) * wordsPerRow),
                                 raw.begin() + static_cast<std::ptrdiff_t>(j * wordsPerRow));
                std::swap(dest[i], dest[j]);
            }
        }
    }

    // Wraps SimpleBatchPianoPIR with the same timing / communication / refresh
    // reporting as BudgetedPianoPIR. Refreshes are done here, before a batch
    // would exceed the smallest partition's Piano budget, so the class's own
    // "Redo preprocessing" guard (which fires after answering) never has to.
    class BatchedEmbeddingPIR {
    public:
        BatchedEmbeddingPIR(std::string name, uint64_t dbSize, uint64_t entryBytes, uint64_t batchSize,
                            std::vector<uint64_t> rawDB, uint64_t failProbLog2)
                : name_(std::move(name)), dbSize_(dbSize), entryBytes_(entryBytes), batchSize_(batchSize) {
            const auto start = Clock::now();
            pir_ = std::make_unique<SimpleBatchPianoPIR>(dbSize, entryBytes, batchSize, std::move(rawDB),
                                                         failProbLog2);
            serverSetupSeconds_ = SecondsSince(start);
            partitionNum_ = pir_->Config().PartitionNum;
            partitionSize_ = pir_->Config().PartitionSize;
            slotsPerPartition_ = batchSize_ / partitionNum_;
            std::cout << name_ << " server setup time " << serverSetupSeconds_ << " s (" << partitionNum_
                      << " partitions of " << partitionSize_ << ", batch size " << batchSize_ << ")" << std::endl;
        }

        void Preprocess() {
            const auto start = Clock::now();
            pir_->Preprocessing();
            ResetBudget();
            initialPrepSeconds_ = SecondsSince(start);
            initialOfflineBytes_ = OfflineBytesPerPreprocessing();
            std::cout << name_ << " client preprocessing time " << initialPrepSeconds_ << " s, offline download "
                      << BudgetedPianoPIR::FormatBytes(initialOfflineBytes_) << std::endl;
        }

        uint64_t PartitionNum() const { return partitionNum_; }
        uint64_t SlotsPerPartition() const { return slotsPerPartition_; }
        uint64_t BatchSize() const { return batchSize_; }
        uint64_t PartitionOf(uint64_t pos) const { return pos / partitionSize_; }

        // Answers one batch; `batch` must hold exactly BatchSize positions.
        std::vector<std::vector<uint64_t>> QueryBatch(const std::vector<uint64_t> &batch) {
            if (!prepared_) {
                throw std::logic_error(name_ + ": QueryBatch() called before Preprocess()");
            }
            if (batch.size() != batchSize_) {
                throw std::logic_error(name_ + ": batch has " + std::to_string(batch.size()) +
                                       " entries, expected " + std::to_string(batchSize_));
            }
            if (pir_->QueriesMadeInPartition + slotsPerPartition_ + 2 > maxQueryNum_) {
                Refresh("per-partition query budget reached");
            }
            const uint64_t madeBefore = pir_->QueriesMadeInPartition;
            std::vector<std::vector<uint64_t>> result = pir_->Query(batch);
            if (madeBefore > 0 && pir_->QueriesMadeInPartition == 0) {
                // The class re-ran preprocessing itself (should not happen, see above).
                ++refreshCount_;
                refreshOfflineBytes_ += OfflineBytesPerPreprocessing();
                std::cout << "[refresh] " << name_ << ": refresh #" << refreshCount_
                          << " (triggered inside SimpleBatchPianoPIR) completed in 0 s, offline download "
                          << BudgetedPianoPIR::FormatBytes(OfflineBytesPerPreprocessing()) << std::endl;
                ResetBudget();
            }
            onlineUploadBytes_ += static_cast<double>(pir_->UploadCostPerBatchOnline());
            onlineDownloadBytes_ += static_cast<double>(pir_->DownloadCostPerBatchOnline());
            ++batches_;
            return result;
        }

        void PrintBudget() const {
            std::cout << name_ << ": " << partitionNum_ << " partitions of " << partitionSize_ << " entries, batch size "
                      << batchSize_ << " (" << slotsPerPartition_ << " queries per partition per batch), max "
                      << BatchesPerPreprocessing() << " batches per preprocessing, client storage "
                      << BudgetedPianoPIR::FormatBytes(pir_->LocalStorageSize()) << std::endl;
        }

        struct Totals {
            double onlineUploadBytes = 0, onlineDownloadBytes = 0, refreshOfflineBytes = 0, refreshSeconds = 0;
            uint64_t batches = 0, refreshes = 0;
        };

        Totals Snapshot() const {
            return {onlineUploadBytes_, onlineDownloadBytes_, refreshOfflineBytes_, refreshSeconds_, batches_,
                    refreshCount_};
        }

        void PrintQueryCosts(const std::string &label, const Totals &before) const {
            const Totals now = Snapshot();
            std::cout << label << " Upload " << std::llround(now.onlineUploadBytes - before.onlineUploadBytes) << std::endl;
            std::cout << label << " Download " << std::llround(now.onlineDownloadBytes - before.onlineDownloadBytes) << std::endl;
            std::cout << label << " batches " << now.batches - before.batches << " (batch size " << batchSize_
                      << "), online " << BudgetedPianoPIR::FormatBytes(now.onlineUploadBytes - before.onlineUploadBytes)
                      << " up / " << BudgetedPianoPIR::FormatBytes(now.onlineDownloadBytes - before.onlineDownloadBytes)
                      << " down";
            if (now.refreshes > before.refreshes) {
                std::cout << "; " << (now.refreshes - before.refreshes) << " refresh(es): offline "
                          << BudgetedPianoPIR::FormatBytes(now.refreshOfflineBytes - before.refreshOfflineBytes)
                          << " down, " << now.refreshSeconds - before.refreshSeconds << " s";
            }
            std::cout << std::endl;
        }

        void PrintSummary() const {
            const double offlineTotal = initialOfflineBytes_ + refreshOfflineBytes_;
            std::cout << name_ << ":\n"
                      << "  setup time:          server " << serverSetupSeconds_ << " s, client preprocessing "
                      << initialPrepSeconds_ << " s\n"
                      << "  batches:             " << batches_ << " of " << batchSize_ << " (" << partitionNum_
                      << " partitions)\n"
                      << "  refreshes:           " << refreshCount_ << " (" << refreshSeconds_ << " s)\n"
                      << "  online upload:       " << BudgetedPianoPIR::FormatBytes(onlineUploadBytes_) << "\n"
                      << "  online download:     " << BudgetedPianoPIR::FormatBytes(onlineDownloadBytes_) << "\n"
                      << "  offline (initial):   " << BudgetedPianoPIR::FormatBytes(initialOfflineBytes_) << "\n"
                      << "  offline (refreshes): " << BudgetedPianoPIR::FormatBytes(refreshOfflineBytes_) << "\n"
                      << "  total communication: "
                      << BudgetedPianoPIR::FormatBytes(onlineUploadBytes_ + onlineDownloadBytes_ + offlineTotal) << "\n"
                      << "  client storage:      " << BudgetedPianoPIR::FormatBytes(pir_->LocalStorageSize())
                      << std::endl;
        }

        double ServerSetupSeconds() const { return serverSetupSeconds_; }
        double InitialPrepSeconds() const { return initialPrepSeconds_; }
        double OnlineBytes() const { return onlineUploadBytes_ + onlineDownloadBytes_; }
        double OfflineBytes() const { return initialOfflineBytes_ + refreshOfflineBytes_; }
        uint64_t RefreshCount() const { return refreshCount_; }
        double RefreshSeconds() const { return refreshSeconds_; }
        const std::string &Name() const { return name_; }

    private:
        double OfflineBytesPerPreprocessing() const {
            return static_cast<double>(dbSize_) * static_cast<double>(entryBytes_);
        }

        uint64_t BatchesPerPreprocessing() const {
            return maxQueryNum_ >= slotsPerPartition_ + 2 ? (maxQueryNum_ - 2) / slotsPerPartition_ : 0;
        }

        void ResetBudget() {
            // The last partition can be smaller than the rest and so have a
            // smaller Piano budget; use the smallest one.
            maxQueryNum_ = UINT64_MAX;
            for (uint64_t i = 0; i < partitionNum_; ++i) {
                maxQueryNum_ = std::min<uint64_t>(maxQueryNum_, pir_->subPIR_MaxQueryNum(static_cast<int>(i)));
            }
            if (maxQueryNum_ < slotsPerPartition_ + 2) {
                throw std::runtime_error(name_ + ": Piano allows only " + std::to_string(maxQueryNum_) +
                                         " queries per partition; too few for one batch. Use a smaller "
                                         "--batch-size (bigger partitions).");
            }
            prepared_ = true;
        }

        void Refresh(const std::string &reason) {
            std::cout << "[refresh] " << name_ << ": " << reason << " (" << pir_->QueriesMadeInPartition << "/"
                      << maxQueryNum_ << " queries per partition used since last preprocessing); "
                      << "re-running Piano preprocessing ..." << std::endl;
            const auto start = Clock::now();
            pir_->Preprocessing();
            ResetBudget();
            const double sec = SecondsSince(start);
            const double bytes = OfflineBytesPerPreprocessing();
            ++refreshCount_;
            refreshSeconds_ += sec;
            refreshOfflineBytes_ += bytes;
            std::cout << "[refresh] " << name_ << ": refresh #" << refreshCount_ << " completed in " << sec
                      << " s, offline download " << BudgetedPianoPIR::FormatBytes(bytes) << "; budget reset to "
                      << BatchesPerPreprocessing() << " batches" << std::endl;
        }

        std::string name_;
        uint64_t dbSize_, entryBytes_, batchSize_;
        std::unique_ptr<SimpleBatchPianoPIR> pir_;
        uint64_t partitionNum_ = 0, partitionSize_ = 0, slotsPerPartition_ = 0;
        bool prepared_ = false;
        uint64_t maxQueryNum_ = 0;

        double serverSetupSeconds_ = 0.0;
        double initialPrepSeconds_ = 0.0;
        double initialOfflineBytes_ = 0.0;
        uint64_t refreshCount_ = 0;
        double refreshSeconds_ = 0.0;
        double refreshOfflineBytes_ = 0.0;
        double onlineUploadBytes_ = 0.0;
        double onlineDownloadBytes_ = 0.0;
        uint64_t batches_ = 0;
    };

    struct BatchRetrieval {
        std::vector<std::vector<float>> vectors;  // retrieved embeddings
        std::vector<uint64_t> docIds;             // their ORIGINAL document ids
        uint64_t realBatches = 0, dummyBatches = 0, maxPerPartition = 0, retried = 0, dropped = 0;
    };

    // Retrieves the candidates' embeddings in as few batches as possible: batch b
    // takes the (b*S)-th..((b+1)*S-1)-th candidate of every partition, S = queries
    // per partition per batch, so nothing is left unanswered. Unused slots are
    // filled with a repeat of an index from a full partition (it lands past that
    // partition's S slots and costs no extra sub-query); failing that, a repeat
    // of the batch's first index. Answers that still come back empty (all-zero)
    // are retried in follow-up rounds. --min-batches pads with whole dummy batches
    // so every query can show the server the same number of batches.
    BatchRetrieval RetrieveBatched(BatchedEmbeddingPIR &pir, const DocLayout &layout,
                                   const std::vector<uint64_t> &docs, uint64_t minBatches) {
        constexpr int kMaxRounds = 3;
        BatchRetrieval out;
        const uint64_t P = pir.PartitionNum(), S = pir.SlotsPerPartition(), B = pir.BatchSize();

        std::vector<uint64_t> pending;
        pending.reserve(docs.size());
        for (uint64_t doc: docs) pending.push_back(layout.posOfDoc[doc]);
        out.vectors.reserve(docs.size());
        out.docIds.reserve(docs.size());

        for (int round = 0; round < kMaxRounds && !pending.empty(); ++round) {
            std::vector<std::vector<uint64_t>> byPartition(P);
            for (uint64_t pos: pending) byPartition[pir.PartitionOf(pos)].push_back(pos);
            uint64_t maxPer = 0;
            for (const auto &v: byPartition) maxPer = std::max<uint64_t>(maxPer, v.size());
            if (round == 0) out.maxPerPartition = maxPer;
            const uint64_t nBatches = (maxPer + S - 1) / S;

            std::vector<uint64_t> retry;
            for (uint64_t b = 0; b < nBatches; ++b) {
                std::vector<uint64_t> batch;
                batch.reserve(B);
                uint64_t filler = kUnplaced;
                for (uint64_t p = 0; p < P; ++p) {
                    const auto &v = byPartition[p];
                    const uint64_t from = b * S, to = std::min<uint64_t>((b + 1) * S, v.size());
                    for (uint64_t k = from; k < to; ++k) batch.push_back(v[k]);
                    if (to > from && to - from == S && filler == kUnplaced) filler = v[from];
                }
                const size_t realInBatch = batch.size();
                if (filler == kUnplaced) filler = batch.front();
                while (batch.size() < B) batch.push_back(filler);

                const auto answers = pir.QueryBatch(batch);
                ++out.realBatches;
                for (size_t i = 0; i < realInBatch; ++i) {
                    const auto &a = answers[i];
                    const bool empty = std::all_of(a.begin(), a.end(), [](uint64_t w) { return w == 0; });
                    if (empty) {
                        retry.push_back(batch[i]);
                    } else {
                        out.vectors.push_back(UnpackEmbeddingRow(a));
                        out.docIds.push_back(layout.docAtPos[batch[i]]);
                    }
                }
            }
            out.retried += retry.size();
            pending.swap(retry);
        }
        if (!pending.empty()) {
            out.dropped = pending.size();
            std::cerr << "warning: " << out.dropped << " candidates were still unanswered after " << kMaxRounds
                      << " batch rounds and were dropped" << std::endl;
        }
        while (out.realBatches + out.dummyBatches < minBatches) {
            pir.QueryBatch(std::vector<uint64_t>(B, 0));
            ++out.dummyBatches;
        }
        return out;
    }

    // ------------------------------------------ Local (--mac) centroid search --

    constexpr int kPaddedDim = 1024;
    // Must match the rotation keys made by RunKeygen("keygen-centroid"), which the
    // sum below uses at offsets 8, 16, ..., 4096. Changing this needs new keys.
    constexpr int kCentroidsPerCiphertext = 8;
    constexpr int kSlotsUsed = kPaddedDim * kCentroidsPerCiphertext;

    struct LocalCentroidSearcher {
        CryptoContext<DCRTPoly> cc;
        PublicKey<DCRTPoly> pk;
        PrivateKey<DCRTPoly> sk;
        std::vector<std::vector<double>> centroidMatrix;
        std::vector<double> centroidNorms;
        int centroidDim = 0;
        int numThreads = 1;
    };

    // Loaded once per process, not once per query.
    LocalCentroidSearcher MakeLocalCentroidSearcher(const std::string &contextDir,
                                                    const std::string &centroidsFile, int numThreads) {
        LocalCentroidSearcher s;
        s.pk = LoadPublicKey(contextDir + "/public_key.bin");
        s.cc = s.pk->GetCryptoContext();
        s.sk = LoadSecretKey(contextDir + "/secret_key.bin");
        s.numThreads = numThreads;

        const uint32_t slots = s.cc->GetRingDimension() / 2;
        if (slots < static_cast<uint32_t>(kSlotsUsed)) {
            throw std::runtime_error("CKKS context has " + std::to_string(slots) + " slots but the centroid "
                                                                                   "packing needs " + std::to_string(kSlotsUsed) +
                                     " (ring dimension >= " + std::to_string(2 * kSlotsUsed) + ")");
        }

        s.centroidMatrix = ReadMatrixFile(centroidsFile);
        if (s.centroidMatrix.empty()) {
            throw std::runtime_error("Centroids file " + centroidsFile + " is empty");
        }
        s.centroidDim = static_cast<int>(s.centroidMatrix.front().size());
        if (s.centroidDim <= 0 || s.centroidDim > kPaddedDim) {
            throw std::runtime_error("Centroid dimension must be > 0 and <= " + std::to_string(kPaddedDim));
        }
        s.centroidNorms.reserve(s.centroidMatrix.size());
        for (const auto &c: s.centroidMatrix) {
            if (static_cast<int>(c.size()) != s.centroidDim) {
                throw std::runtime_error("Centroid dimension mismatch in centroids file");
            }
            s.centroidNorms.push_back(std::inner_product(c.begin(), c.end(), c.begin(), 0.0));
        }
        std::cout << "Loaded " << s.centroidMatrix.size() << " centroids of dim " << s.centroidDim << std::endl;
        return s;
    }

    // Local simulation only: server-side distance computation and client-side
    // decryption happen in one process. Each batch is decrypted as soon as it is
    // computed, so only plaintext distances (8 bytes per centroid) are kept.
    void RunLocalCentroidSearch(const LocalCentroidSearcher &s, const std::string &inputVectorPath,
                                int topK, const std::string &outputJson) {
        const auto &cc = s.cc;
        const std::vector<double> query = ReadVectorFile(inputVectorPath);
        const int queryDim = static_cast<int>(query.size());
        if (queryDim != s.centroidDim) {
            throw std::runtime_error("Query " + inputVectorPath + " has dim " + std::to_string(queryDim) +
                                     ", centroids have dim " + std::to_string(s.centroidDim));
        }

        std::vector<double> packedQuery(kSlotsUsed, 0.0);
        for (int d = 0; d < queryDim; ++d) {
            for (int b = 0; b < kCentroidsPerCiphertext; ++b) {
                packedQuery[static_cast<size_t>(d * kCentroidsPerCiphertext + b)] = query[static_cast<size_t>(d)];
            }
        }
        const double normSquared = std::inner_product(query.begin(), query.end(), query.begin(), 0.0);
        std::vector<double> packedNorm(kSlotsUsed, 0.0);
        std::fill_n(packedNorm.begin(), kCentroidsPerCiphertext, normSquared);

        const auto encQuery = cc->Encrypt(s.pk, cc->MakeCKKSPackedPlaintext(packedQuery));
        const auto encNorm = cc->Encrypt(s.pk, cc->MakeCKKSPackedPlaintext(packedNorm));

        const int nCentroids = static_cast<int>(s.centroidMatrix.size());
        const int nBatches = (nCentroids + kCentroidsPerCiphertext - 1) / kCentroidsPerCiphertext;
        std::vector<double> distances(static_cast<size_t>(nCentroids), 0.0);
        std::atomic<bool> failed(false);
        std::string failureMessage;

        SetThreadCount(s.numThreads);
#ifdef _OPENMP
        // OpenFHE parallelises internally with OpenMP; nesting that inside this
        // loop oversubscribes the cores. Keep only the outer level active.
        const int prevLevels = omp_get_max_active_levels();
        omp_set_max_active_levels(1);
#endif
#pragma omp parallel for schedule(dynamic, 1) num_threads(s.numThreads)
        for (int batchIdx = 0; batchIdx < nBatches; ++batchIdx) {
            if (failed.load()) {
                continue;
            }
            try {
                const int start = batchIdx * kCentroidsPerCiphertext;
                const int inBatch = std::min(kCentroidsPerCiphertext, nCentroids - start);
                std::vector<double> centroidPacked(kSlotsUsed, 0.0);
                std::vector<double> centroidNormPacked(kSlotsUsed, 0.0);
                for (int b = 0; b < inBatch; ++b) {
                    const auto &c = s.centroidMatrix[static_cast<size_t>(start + b)];
                    centroidNormPacked[static_cast<size_t>(b)] = s.centroidNorms[static_cast<size_t>(start + b)];
                    for (int d = 0; d < s.centroidDim; ++d) {
                        centroidPacked[static_cast<size_t>(d * kCentroidsPerCiphertext + b)] = c[static_cast<size_t>(d)];
                    }
                }
                // Named Plaintexts: OpenFHE's EvalAdd/EvalMult(ciphertext, Plaintext&)
                // overloads take a non-const reference, so a temporary won't bind.
                Plaintext ptCentroid = cc->MakeCKKSPackedPlaintext(centroidPacked);
                Plaintext ptCentroidNorm = cc->MakeCKKSPackedPlaintext(centroidNormPacked);
                auto dot = cc->EvalMult(encQuery, ptCentroid);
                for (int step = 1; step < kPaddedDim; step <<= 1) {
                    dot = cc->EvalAdd(dot, cc->EvalAtIndex(dot, step * kCentroidsPerCiphertext));
                }
                const auto distance = cc->EvalSub(
                        cc->EvalAdd(encNorm, ptCentroidNorm),
                        cc->EvalAdd(dot, dot));

                Plaintext pt;
                cc->Decrypt(s.sk, distance, &pt);
                const auto values = pt->GetRealPackedValue();
                if (values.size() < static_cast<size_t>(inBatch)) {
                    throw std::runtime_error("Decrypted batch " + std::to_string(batchIdx) + " has only " +
                                             std::to_string(values.size()) + " slots");
                }
                for (int b = 0; b < inBatch; ++b) {
                    distances[static_cast<size_t>(start + b)] = values[static_cast<size_t>(b)];
                }
            } catch (const std::exception &ex) {
                if (!failed.exchange(true)) {
#pragma omp critical
                    { failureMessage = ex.what(); }
                }
            }
        }
#ifdef _OPENMP
        omp_set_max_active_levels(prevLevels);
#endif
        if (failed.load()) {
            throw std::runtime_error("Centroid distance computation failed: " + failureMessage);
        }

        const int keep = std::min(topK, nCentroids);
        std::vector<int> order(static_cast<size_t>(nCentroids));
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + keep, order.end(), [&](int a, int b) {
            return distances[static_cast<size_t>(a)] < distances[static_cast<size_t>(b)];
        });
        std::vector<double> topDistances;
        std::vector<int> topIndices;
        for (int i = 0; i < keep; ++i) {
            topIndices.push_back(order[static_cast<size_t>(i)]);
            topDistances.push_back(distances[static_cast<size_t>(order[static_cast<size_t>(i)])]);
        }
        WriteJsonTopK(outputJson, topDistances, topIndices);
        std::cout << "Scored " << nCentroids << " centroids in " << nBatches << " batches with "
                  << s.numThreads << " thread(s)" << std::endl;
    }

    // ----------------------------------------- Server-path centroid pipeline --

    void RunServerCentroidPipeline(const std::string &workDir, const std::string &inputVectorPath,
                                   const std::string &centroidsFile, const std::string &outputJson,
                                   int centroidTopK) {
        CliArgs base;
        base.kv.push_back({"--context-dir", workDir});
        base.kv.push_back({"--input-vector", inputVectorPath});
        base.kv.push_back({"--centroids-file", centroidsFile});

        CliArgs encArgs = base;
        encArgs.kv.push_back({"--output-dir", workDir + "/encrypted_query"});
        std::cout << "Run EncryptCentroid" << std::endl;
        RunEncryptCentroid(encArgs);

        CliArgs compArgs = base;
        compArgs.kv.push_back({"--encrypted-query", workDir + "/encrypted_query/encrypted_query_centroid_batched.bin"});
        compArgs.kv.push_back({"--encrypted-norm", workDir + "/encrypted_query/encrypted_norm_centroid_batched.bin"});
        compArgs.kv.push_back({"--output-dir", workDir + "/encrypted_distances"});
        std::cout << "Run ComputeCentroid" << std::endl;
        RunComputeCentroid(compArgs);

        CliArgs decArgs = base;
        decArgs.kv.push_back({"--encrypted-distances-dir", workDir + "/encrypted_distances"});
        decArgs.kv.push_back({"--output-json", outputJson});
        decArgs.kv.push_back({"--top-k", std::to_string(centroidTopK)});
        std::cout << "Run DecryptCentroid" << std::endl;
        RunDecryptCentroid(decArgs);
    }

    // --------------------------------------------------------- FAISS re-rank --

    std::vector<std::pair<uint64_t, float>> FaissTopK(const std::vector<std::vector<float>> &responses,
                                                      const std::vector<uint64_t> &responseIds,
                                                      const std::string &queryVectorPath, int topK) {
        std::vector<std::pair<uint64_t, float>> results;
        if (responses.empty()) {
            std::cerr << "  ! no documents retrieved for this query; skipping FAISS search, "
                         "emitting empty Top-K" << std::endl;
            return results;
        }
        const std::vector<float> queryVector = ReadFloatVectorFile(queryVectorPath);
        if (queryVector.size() != kEmbeddingDim) {
            throw std::runtime_error("Query vector has dim " + std::to_string(queryVector.size()) +
                                     ", expected " + std::to_string(kEmbeddingDim));
        }
        std::vector<float> candidates;
        candidates.reserve(responses.size() * kEmbeddingDim);
        for (const auto &v: responses) {
            candidates.insert(candidates.end(), v.begin(), v.end());
        }
        faiss::IndexFlatL2 index(static_cast<faiss::idx_t>(kEmbeddingDim));
        index.add(static_cast<faiss::idx_t>(responses.size()), candidates.data());

        const int k = std::min(topK, static_cast<int>(responses.size()));
        std::vector<float> dist(static_cast<size_t>(k));
        std::vector<faiss::idx_t> ids(static_cast<size_t>(k));
        index.search(1, queryVector.data(), k, dist.data(), ids.data());
        for (int i = 0; i < k; ++i) {
            if (ids[static_cast<size_t>(i)] >= 0) {  // FAISS returns -1 if not enough points
                results.emplace_back(responseIds[static_cast<size_t>(ids[static_cast<size_t>(i)])],
                                     dist[static_cast<size_t>(i)]);
            }
        }
        return results;
    }

    // ------------------------------------------------------------ Pipeline --

    int RunPipeline(int argc, char **argv) {
        const CliOptions opts = ParseCliOptions(argc, argv);

        const bool mac = GetBoolOpt(opts, "--mac", false);
        // Give every concurrently running process its own --work-dir.
        const std::string workDir = GetOpt(opts, "--work-dir", "test");
        fs::create_directories(workDir);
        const std::string outputJson = workDir + "/top_k_results.json";

        std::vector<std::string> queryPaths;
        if (opts.count("--query-list")) {
            queryPaths = ReadQueryList(GetOpt(opts, "--query-list", ""));
        } else {
            queryPaths.push_back(GetOpt(opts, "--input-vector", "../query_768d_from_k4096_centroid0.txt"));
        }

        const std::string centroidsFile = GetOpt(opts, "--centroids-file", "../centroids_65k_new.txt");
        const uint64_t dbSize = GetPositiveU64Opt(opts, "--db-size", 65000);  // 702873, 1120486
        const int topK = GetPositiveIntOpt(opts, "--top-k", 10);
        const int centroidTopK = GetPositiveIntOpt(opts, "--centroid-top-k", 10);
        const uint64_t failProbLog2 = GetPositiveU64Opt(opts, "--fail-prob-log2", 20);
        const unsigned hw = std::thread::hardware_concurrency();
        const int numThreads = GetPositiveIntOpt(opts, "--threads", hw > 0 ? static_cast<int>(hw) : 1);
        const bool printDocs = GetBoolOpt(opts, "--print-docs", false);
        const uint64_t batchSize = GetPositiveU64Opt(opts, "--batch-size", 64);
        const bool spreadClusters = GetBoolOpt(opts, "--spread-clusters", true);
        const uint64_t minBatches = GetOpt(opts, "--min-batches", "0") == "0"
                                    ? 0 : GetPositiveU64Opt(opts, "--min-batches", 0);
        if (batchSize < 2 || batchSize % 2 != 0) {
            throw std::invalid_argument("--batch-size must be an even number >= 2");
        }
        const uint64_t partitionNum = batchSize / kRealQueryPerPartition;
        if (partitionNum > dbSize) {
            throw std::invalid_argument("--batch-size " + std::to_string(batchSize) + " gives " +
                                        std::to_string(partitionNum) + " partitions, more than --db-size");
        }
        const uint64_t partitionSize = (dbSize + partitionNum - 1) / partitionNum;
        if ((partitionNum - 1) * partitionSize >= dbSize) {
            throw std::invalid_argument("--batch-size " + std::to_string(batchSize) + " leaves the last of " +
                                        std::to_string(partitionNum) + " partitions empty for --db-size " +
                                        std::to_string(dbSize) + "; pick a different batch size");
        }

        const std::string embeddingDbPath = GetOpt(opts, "--embedding-db", mac
                                                                           ? "/Users/antoniajanuszewicz/PycharmProjects/PIANO-RAG/faiss.json"
                                                                           : "/home/ajanusze/PIANO-RAG/faiss.json");
        // Optional raw float32 file (dbSize x 768). Avoids the JSON parse entirely.
        const std::string embeddingBinPath = GetOpt(opts, "--embedding-bin", "");
        const std::string centroidMappingPath = GetOpt(opts, "--centroid-mapping", mac
                                                                                   ? "/Users/antoniajanuszewicz/PycharmProjects/PIANO-RAG/prototype/data/lists.json"
                                                                                   : "../lists.json");
        const std::string textDbPath = GetOpt(opts, "--text-db", mac
                                                                 ? "/Users/antoniajanuszewicz/PycharmProjects/PIANO-RAG/uniform_index.txt"
                                                                 : "/home/ajanusze/PIANO-RAG/uniform_index_1024.txt");

        std::cout << "Config: db-size " << dbSize << ", queries " << queryPaths.size() << ", top-k " << topK
                  << ", centroid-top-k " << centroidTopK << ", threads " << numThreads
                  << ", work-dir " << workDir << (mac ? ", local centroid path" : ", server centroid path")
                  << ", PIR 1 batched: batch size " << batchSize << ", " << partitionNum << " partitions of "
                  << partitionSize << ", spread-clusters " << (spreadClusters ? "on" : "off")
                  << ", min-batches " << minBatches << std::endl;

        // ------------------------------------------------ One-time setup --
        const auto oneTimeStart = Clock::now();

        std::cout << "Loading embedding database ... " << std::endl;
        PackedDB embeddingDb = embeddingBinPath.empty()
                               ? LoadEmbeddingRawDBFromJson(embeddingDbPath, dbSize)
                               : LoadEmbeddingRawDBFromBinary(embeddingBinPath, dbSize);

        std::cout << "Loading centroid mapping ... " << std::endl;
        const std::unordered_map<uint64_t, std::vector<uint64_t>> centroidToIndex =
                load_json_mapping(centroidMappingPath);

        std::cout << "Loading text database ..." << std::endl;
        PackedDB textDb = LoadTextRawDB(textDbPath, dbSize);

        if (embeddingDb.sourceEntries != textDb.sourceEntries) {
            std::cerr << "warning: embedding file has " << embeddingDb.sourceEntries << " entries but text file has "
                      << textDb.sourceEntries << "; ids are assumed to line up for the first --db-size entries"
                      << std::endl;
        }
        // Ids at or beyond this are not in either file; ids in [dbSize, sourceEntries)
        // exist in the files but were truncated away for this run.
        const uint64_t sourceEntries = std::min(embeddingDb.sourceEntries, textDb.sourceEntries);
        std::cout << "Database for this run: " << dbSize << " of " << sourceEntries << " entries" << std::endl;
        ReportMappingCoverage(centroidToIndex, dbSize, sourceEntries);

        // Lay the embedding DB out for batching (spread each IVF list over the partitions).
        const auto layoutStart = Clock::now();
        const DocLayout layout = BuildDocLayout(centroidToIndex, dbSize, partitionNum, partitionSize, spreadClusters);
        if (spreadClusters) {
            PermuteRows(embeddingDb.raw, layout.posOfDoc, kEmbeddingWords);
        }
        std::cout << "PIR 1 layout: " << (spreadClusters ? "lists spread over " : "original order, ")
                  << partitionNum << " partitions in " << SecondsSince(layoutStart) << " s" << std::endl;

        std::cout << "Server and Client One-time Setup" << std::endl;
        // Moved in. Note: SimpleBatchPianoPIR copies each partition out of the
        // DB it is given, so PIR 1 briefly needs two copies during construction.
        BatchedEmbeddingPIR embeddingPir("PIR 1 (batched embeddings)", dbSize, kEmbeddingEntryBytes, batchSize,
                                         std::move(embeddingDb.raw), failProbLog2);
        BudgetedPianoPIR textPir("PIR 2 (text)", dbSize, kTextEntryBytes, std::move(textDb.raw), failProbLog2);
        std::cout << "Server Setup time: " << embeddingPir.ServerSetupSeconds() + textPir.ServerSetupSeconds()
                  << std::endl;

        embeddingPir.Preprocess();
        textPir.Preprocess();
        std::cout << "Client Setup time: " << embeddingPir.InitialPrepSeconds() + textPir.InitialPrepSeconds()
                  << std::endl;
        std::cout << "Offline communication (initial preprocessing): "
                  << BudgetedPianoPIR::FormatBytes(embeddingPir.OfflineBytes() + textPir.OfflineBytes()) << std::endl;
        embeddingPir.PrintBudget();
        textPir.PrintBudget();

        SetThreadCount(numThreads);
        std::cout << "Run KeyGen" << std::endl;
        {
            CliArgs keyArgs;
            keyArgs.kv.push_back({"--context-dir", workDir});
            keyArgs.kv.push_back({"--input-vector", queryPaths.front()});
            keyArgs.kv.push_back({"--centroids-file", centroidsFile});
            RunKeygen(keyArgs, "keygen-centroid");
        }

        std::optional<LocalCentroidSearcher> localSearcher;
        if (mac) {
            localSearcher.emplace(MakeLocalCentroidSearcher(workDir, centroidsFile, numThreads));
        }

        const double oneTimeSec = SecondsSince(oneTimeStart);
        std::cout << "One-time Setup time " << oneTimeSec << std::endl;

        // ---------------------------------------------------- Per query --
        double sumCentroidSec = 0, sumQuerySec = 0, sumRefreshSec = 0;
        uint64_t queriesWithNoResults = 0;

        for (size_t q = 0; q < queryPaths.size(); ++q) {
            const std::string &inputVectorPath = queryPaths[q];
            std::cout << "\n=== Query " << (q + 1) << "/" << queryPaths.size() << ": " << inputVectorPath
                      << " ===" << std::endl;
            const auto pir1Before = embeddingPir.Snapshot();
            const auto pir2Before = textPir.Snapshot();
            const auto queryStart = Clock::now();

            // Step 1: encrypted centroid selection. Remove the previous result
            // first so a failed step can't silently reuse it.
            std::error_code ec;
            fs::remove(outputJson, ec);
            auto t = Clock::now();
            if (mac) {
                RunLocalCentroidSearch(*localSearcher, inputVectorPath, centroidTopK, outputJson);
            } else {
                RunServerCentroidPipeline(workDir, inputVectorPath, centroidsFile, outputJson, centroidTopK);
            }
            if (!fs::exists(outputJson)) {
                throw std::runtime_error("Centroid step did not produce " + outputJson);
            }
            const double centroidSec = SecondsSince(t);
            std::cout << "Centroid time " << centroidSec << std::endl;

            // Step 2: centroids -> candidate ids. Padding (-1 arrives as 2^64-1),
            // ids truncated away by --db-size, out-of-range and duplicate ids are
            // skipped; the server only sees whole batches (see --min-batches).
            std::cout << "Loading centroid indices ... " << std::endl;
            const std::vector<uint64_t> centroidIndices = load_centroid_indices(outputJson);
            std::vector<uint64_t> candidates;
            uint64_t paddingIds = 0, truncatedIds = 0, invalidIds = 0, duplicateIds = 0;
            {
                std::unordered_set<uint64_t> seen;
                for (uint64_t idx: CentroidToIndex(centroidIndices, centroidToIndex)) {
                    if (idx == kPaddingId) {
                        ++paddingIds;
                    } else if (idx >= sourceEntries) {
                        ++invalidIds;
                    } else if (idx >= dbSize) {
                        ++truncatedIds;  // exists in the files, excluded by --db-size
                    } else if (!seen.insert(idx).second) {
                        ++duplicateIds;
                    } else {
                        candidates.push_back(idx);
                    }
                }
            }
            const uint64_t skippedIds = paddingIds + truncatedIds + invalidIds + duplicateIds;
            if (skippedIds > 0) {
                std::cout << "Skipping " << skippedIds << " ids (" << paddingIds << " padding, " << truncatedIds
                          << " beyond --db-size, " << invalidIds << " beyond the database files, " << duplicateIds
                          << " duplicate)" << std::endl;
            }
            std::cout << "Queries to batch : ";
            for (uint64_t idx: candidates) std::cout << idx << " ";
            std::cout << std::endl;

            // Step 3: PIR 1 in batches, every candidate retrieved (refreshing if needed).
            t = Clock::now();
            std::cout << "Processing queries ... " << candidates.size() << std::endl;
            const BatchRetrieval got = RetrieveBatched(embeddingPir, layout, candidates, minBatches);
            std::cout << "PIR 1 batching: " << candidates.size() << " candidates, max " << got.maxPerPartition
                      << " in one partition, " << got.realBatches << " batch(es) + " << got.dummyBatches
                      << " dummy, " << got.retried << " retried, " << got.dropped << " dropped" << std::endl;
            embeddingPir.PrintQueryCosts("PIR 1", pir1Before);
            std::cout << "PIR 1 time " << SecondsSince(t) << std::endl;

            // Step 4: local re-rank.
            std::cout << "Computing global distances " << std::endl;
            const auto results = FaissTopK(got.vectors, got.docIds, inputVectorPath, topK);
            if (results.empty()) {
                ++queriesWithNoResults;
            }
            std::cout << "Top-K: ";
            for (const auto &r: results) {
                std::cout << "[ " << r.first << " " << r.second << "] ";
            }
            std::cout << std::endl;

            // Step 5: PIR 2 for the text of every Top-K hit.
            t = Clock::now();
            std::cout << "Processing queries ... " << results.size() << std::endl;
            std::vector<std::string> retrievedDocs;
            retrievedDocs.reserve(results.size());
            for (const auto &r: results) {
                retrievedDocs.push_back(UnpackTextEntry(textPir.Query(r.first, /*realQuery=*/true)));
                if (printDocs) {
                    std::cout << "[" << r.first << "] " << retrievedDocs.back() << std::endl;
                }
            }
            textPir.PrintQueryCosts("PIR 2", pir2Before);
            std::cout << "PIR 2 time " << SecondsSince(t) << std::endl;

            const double querySec = SecondsSince(queryStart);
            const double refreshSec = (embeddingPir.RefreshSeconds() - pir1Before.refreshSeconds) +
                                      (textPir.RefreshSeconds() - pir2Before.refreshSeconds);
            std::cout << "Query time " << querySec << " (includes centroid step " << centroidSec
                      << " s and Piano refresh " << refreshSec << " s)" << std::endl;
            std::cout << "Total time " << querySec << std::endl;

            sumCentroidSec += centroidSec;
            sumQuerySec += querySec;
            sumRefreshSec += refreshSec;
        }

        // ------------------------------------------------------- Summary --
        const double n = static_cast<double>(queryPaths.size());
        std::cout << "\n=== Summary ===" << std::endl;
        std::cout << "Queries run: " << queryPaths.size() << " (" << queriesWithNoResults
                  << " with empty Top-K)" << std::endl;
        std::cout << "One-time setup: " << oneTimeSec << " s" << std::endl;
        std::cout << "Avg query time: " << sumQuerySec / n << " s (avg centroid step " << sumCentroidSec / n
                  << " s)" << std::endl;
        embeddingPir.PrintSummary();
        textPir.PrintSummary();
        std::cout << "Total Piano refresh time during queries: " << sumRefreshSec << " s" << std::endl;
        std::cout << "Total PIR communication: online "
                  << BudgetedPianoPIR::FormatBytes(embeddingPir.OnlineBytes() + textPir.OnlineBytes())
                  << ", offline "
                  << BudgetedPianoPIR::FormatBytes(embeddingPir.OfflineBytes() + textPir.OfflineBytes())
                  << std::endl;

        std::cout << "Successful RAG completion!" << std::endl;
        return 0;
    }

}  // namespace

int main(int argc, char **argv) {
    try {
        return RunPipeline(argc, argv);
    } catch (const std::exception &e) {
        std::cerr << "fatal: " << e.what() << std::endl;
        return 1;
    }
}