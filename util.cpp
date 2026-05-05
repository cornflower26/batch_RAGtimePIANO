/*
 * piano_prf.cpp
 *
 * PRF implementation for PIANO PIR — C++ port of Pacmann Go prf.go.
 *
 * Uses OpenSSL EVP AES-128-ECB for all AES operations.
 * No hand-rolled intrinsics or architecture-specific code.
 * Works identically on x86_64, Apple Silicon (AArch64), and any other
 * platform OpenSSL supports.
 *
 * The Go source uses AES-128 in Matyas-Meyer-Oseas (MMO) mode:
 *
 *   GetLongKey(key *PrfKey128) []uint32
 *     Returns the AES-128 key schedule as 44 uint32s.
 *     We store the raw 16-byte key and an initialised EVP_CIPHER_CTX
 *     inside the vector's backing store — see KeyState below.
 *
 *   PRFEvalWithLongKeyAndTag(longKey []uint32, tag uint64, x uint64) uint64
 *     input  = (tag << 35) + x, packed into 8 bytes little-endian
 *              (upper 8 bytes of the 16-byte AES block are zero)
 *     output = AES_K(input) XOR input   (MMO compression function)
 *     return first 8 bytes of output as uint64 little-endian
 *
 * Tag constraint: tag < 2^29  (so (tag<<35)+x fits in uint64 for x < 2^32)
 *
 * Build:  -lssl -lcrypto   (links OpenSSL; available on macOS and Linux)
 *   macOS:  brew install openssl
 *           g++ -I$(brew --prefix openssl)/include \
 *               -L$(brew --prefix openssl)/lib -lssl -lcrypto ...
 *   Linux:  apt install libssl-dev
 *           g++ ... -lssl -lcrypto
 */


#include <openssl/evp.h>

#include <cassert>
#include <chrono>
#include <cstring>
#include <random>
#include <stdexcept>
#include <map>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include "json.hpp"
// ============================================================================
// Internal: reusable EVP context wrapper
//
// Go's GetLongKey() returns []uint32 (the raw AES key schedule).
// We can't use EVP_CIPHER_CTX inside a std::vector<uint32_t> directly,
// so we store the EVP context pointer in a small heap allocation and stash
// its address in the first two words of the uint32 vector (as a uintptr_t).
// The vector always has exactly 44 uint32s: the first 4 words (16 bytes)
// hold the uintptr_t of the EVP_CIPHER_CTX*, and the remaining 40 words
// hold the raw 16-byte AES key (used by PRFEvalWithLongKeyAndTag to
// perform the MMO XOR check-free).
//
// Layout of the returned std::vector<uint32_t> (44 words = 176 bytes):
//   words[0..1]  : uintptr_t — pointer to EVP_CIPHER_CTX (on heap)
//   words[2..5]  : raw 16-byte AES key (for MMO: XOR back in after encrypt)
//   words[6..43] : unused (zero-padded to 44 total, matching Go's slice length)
//
// PRFEvalWithLongKeyAndTag extracts the context pointer and raw key from the
// first 6 words and issues a single EVP_EncryptUpdate call.
// ============================================================================

static_assert(sizeof(uintptr_t) <= 8, "uintptr_t must fit in 2 uint32s");

struct PrfKey128 {
    uint64_t lo;
    uint64_t hi;
};
using PrfKey = PrfKey128;

static uint64_t primaryNumParam(double /*Q*/, double ChunkSize, uint64_t target) {
    double k = std::ceil(std::log(2.0) * static_cast<double>(target));
    return static_cast<uint64_t>(k) * static_cast<uint64_t>(ChunkSize);
}


// Pack a pointer into words[0..1]
static void packPtr(std::vector<uint32_t>& v, EVP_CIPHER_CTX* ctx) {
    uintptr_t p = reinterpret_cast<uintptr_t>(ctx);
    v[0] = static_cast<uint32_t>(p);
    v[1] = static_cast<uint32_t>(p >> 32);
}

// Extract the context pointer from the vector
static EVP_CIPHER_CTX* unpackPtr(const std::vector<uint32_t>& v) {
    uintptr_t p = static_cast<uintptr_t>(v[0])
                  | (static_cast<uintptr_t>(v[1]) << 32);
    return reinterpret_cast<EVP_CIPHER_CTX*>(p);
}

// ============================================================================
// Little-endian helpers (matching Go's encoding/binary.LittleEndian)
// ============================================================================

static inline void leStoreU64(uint8_t* p, uint64_t v) {
    p[0]=(uint8_t)(v);     p[1]=(uint8_t)(v>>8);
    p[2]=(uint8_t)(v>>16); p[3]=(uint8_t)(v>>24);
    p[4]=(uint8_t)(v>>32); p[5]=(uint8_t)(v>>40);
    p[6]=(uint8_t)(v>>48); p[7]=(uint8_t)(v>>56);
}

static inline uint64_t leLoadU64(const uint8_t* p) {
    return  (uint64_t)p[0]        | ((uint64_t)p[1]<<8)
            | ((uint64_t)p[2]<<16)  | ((uint64_t)p[3]<<24)
            | ((uint64_t)p[4]<<32)  | ((uint64_t)p[5]<<40)
            | ((uint64_t)p[6]<<48)  | ((uint64_t)p[7]<<56);
}

// ============================================================================
// Public API
// ============================================================================

static PrfKey RandKey() {
    auto seed = static_cast<uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::mt19937_64 rng(seed);
    PrfKey key;
    uint8_t* k = reinterpret_cast<uint8_t*>(&key);
    leStoreU64(k,     rng());
    leStoreU64(k + 8, rng());
    return key;
}

// GetLongKey: initialise an OpenSSL AES-128-ECB context for the given key.
// Returns a 44-word vector whose first 2 words hold the EVP_CIPHER_CTX*
// and words 2-5 hold the raw 16-byte key (for MMO XOR).
// Caller must eventually call GetLongKeyFree() or let the PianoPIRClient
// destructor handle it.  (In practice the client recreates longKey_ on
// each Initialization(), so the old ctx is freed at that point.)
static std::vector<uint32_t> GetLongKey(const PrfKey128& key) {
    // Initialise EVP context
    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    if (!ctx) throw std::runtime_error("EVP_CIPHER_CTX_new failed");

    const uint8_t* rawKey = reinterpret_cast<const uint8_t*>(&key);

    if (EVP_EncryptInit_ex(ctx, EVP_aes_128_ecb(), nullptr, rawKey, nullptr) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        throw std::runtime_error("EVP_EncryptInit_ex failed");
    }
    EVP_CIPHER_CTX_set_padding(ctx, 0);  // no PKCS#7 padding — raw block

    // Build the 44-word vector
    std::vector<uint32_t> longKey(44, 0);

    // words[0..1]: EVP_CIPHER_CTX* (packed as two uint32s)
    packPtr(longKey, ctx);

    // words[2..5]: raw 16-byte key (needed for MMO: output XOR input)
    std::memcpy(longKey.data() + 2, rawKey, 16);

    return longKey;
}

// Free the EVP context stored inside a longKey vector.
// Call this if you need explicit cleanup; not needed if the vector goes
// out of scope and you don't care about the leaked OpenSSL context.
static void GetLongKeyFree(std::vector<uint32_t>& longKey) {
    if (longKey.size() >= 2) {
        EVP_CIPHER_CTX* ctx = unpackPtr(longKey);
        if (ctx) {
            EVP_CIPHER_CTX_free(ctx);
            longKey[0] = longKey[1] = 0;
        }
    }
}

// PRFEvalWithLongKeyAndTag: direct port of Go's function.
//
//   input  = (tag << 35) + x, stored LE in bytes 0-7; bytes 8-15 = 0
//   output = AES_K(input) XOR input   (Matyas-Meyer-Oseas)
//   return = output[0..7] as uint64 LE
//
// Tag constraint: tag < 2^29.
static uint64_t PRFEvalWithLongKeyAndTag(const std::vector<uint32_t>& longKey,
                                  uint64_t tag,
                                  uint64_t x) {
    assert(longKey.size() == 44);

    uint8_t src[16] = {};
    uint8_t dst[16] = {};
    leStoreU64(src, (tag << 35) + x);  // upper 8 bytes stay zero

    EVP_CIPHER_CTX* ctx = unpackPtr(longKey);

    int outlen = 16;
    EVP_EncryptUpdate(ctx, dst, &outlen, src, 16);

    // MMO: XOR plaintext back in
    for (int i = 0; i < 16; ++i) dst[i] ^= src[i];

    return leLoadU64(dst);
}

// EntryXor: XOR entrySize uint64 words from b into a.
// entrySize must be a multiple of 4 (Go constraint).
static void EntryXor(uint64_t* __restrict__ a,
              const uint64_t* __restrict__ b,
              uint64_t entrySize) {
    assert(entrySize % 4 == 0 && "entrySize must be a multiple of 4");
    for (uint64_t i = 0; i < entrySize; ++i)
        a[i] ^= b[i];
}

static std::vector<uint64_t> CentroidToIndex(std::vector<uint64_t> centroids, std::unordered_map<uint64_t,std::vector<uint64_t>> mapping){
    std::vector<uint64_t> indices;
    for (uint64_t i = 0; i < centroids.size();i++){
        indices.insert(indices.end(),mapping[centroids[i]].begin(),mapping[centroids[i]].end());
    }
    return indices;
}

static std::vector<uint64_t> load_centroid_indices(const std::string& filename) {
    std::ifstream file(filename);
    nlohmann::json j;
    file >> j;

    const auto& indices = j.at("centroid_indices");

    if (!indices.is_array()) {
        throw std::runtime_error("centroid_indices must be an array");
    }

    std::vector<uint64_t> result;
    result.reserve(indices.size());

    for (const auto& val : indices) {
        result.push_back(val.get<uint64_t>());
    }

    return result;
}

static std::unordered_map<uint64_t, std::vector<uint64_t>>
load_json_mapping(const std::string& filename) {
    std::ifstream file(filename);
    nlohmann::json j;
    file >> j;

    std::unordered_map<uint64_t, std::vector<uint64_t>> dictionary;

    for (auto& [key, value] : j.items()) {
        int int_key = std::stoi(key);
        dictionary[int_key] = value.get<std::vector<uint64_t>>();
    }

    return dictionary;
}

static std::vector<std::vector<uint64_t>>
load_json_distances(const std::string& filename) {
    std::ifstream file(filename);
    nlohmann::json j;
    file >> j;

    int num_vectors = j.at("num_vectors").get<int>();
    int dimension   = j.at("dimension").get<int>();

    const auto& vectors_json = j.at("vectors");

    std::vector<std::vector<uint64_t>> vectors;
    vectors.reserve(num_vectors);

    for (const auto& item : vectors_json) {
        std::vector<double> vec = item.at("vector").get<std::vector<double>>();

        if (vec.size() != static_cast<size_t>(dimension)) {
            throw std::runtime_error("Vector dimension does not match expected dimension");
        }
        std::vector<uint64_t> int_vec;
        for (uint64_t i = 0; i < vec.size(); i++)
            int_vec.push_back(std::bit_cast<uint64_t>(vec[i]));
        vectors.push_back(std::move(int_vec));
    }

    return vectors;
}

static std::vector<std::string> load_text_database(
        const std::string& filename,
        char delimiter = static_cast<char>(31)
) {
    std::ifstream file(filename);

    if (!file) {
        throw std::runtime_error("Could not open file");
    }

    std::vector<std::string> records;
    std::string record;

    while (std::getline(file, record, delimiter)) {
        if (!record.empty()) {
            records.push_back(record);
        }
    }

    return records;
}



