
//Servers
//home/ajanusze/PIANO-RAG/uniform_index.txt


//"/Users/antoniajanuszewicz/PycharmProjects/PIANO-RAG/faiss.json"
//"/Users/antoniajanuszewicz/PycharmProjects/PIANO-RAG/prototype/data/lists.json"
//"/Users/antoniajanuszewicz/Downloads/ground_truth.json"

//"/Users/antoniajanuszewicz/PycharmProjects/PIANO-RAG/decrypted_results/top_k_results.json"

//read in the server database, batch for centroid retrieval, regular for document retrieval?
// read in the client information, for centroids then make a function to read in lists and convert to centroids
//given centroids document and then read in the indices (ground_truth) document
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
#include <faiss/IndexFlat.h>
#include "openfhe.h"
#include "openfhe_batched_workload.cpp"

using namespace lbcrypto;

namespace {

    void centroids() {
        const uint32_t polyModulusDegree = 16384;
        const std::string coeffCsv = "60,40,40,60";
        const std::string securityLevel = "";
        const int paddedDim = 1024;
        const std::string inputVectorPath = "../query_768d_from_k4096_centroid0.txt";

        int lanes = 8;
        std::vector<int32_t> rotationIndices = RotationIndicesForLayout(lanes, paddedDim);
        const auto coeffSizes = ParseUintCsv(coeffCsv);

        CCParams<CryptoContextCKKSRNS> parameters;
        parameters.SetMultiplicativeDepth(coeffSizes.size() > 2 ? coeffSizes.size() - 2 : 2);
        parameters.SetScalingModSize(coeffSizes.size() > 1 ? coeffSizes[1] : 40);
        parameters.SetBatchSize(polyModulusDegree / 2);
        parameters.SetRingDim(polyModulusDegree);
        if (securityLevel == "none" || securityLevel == "notset" ||
            securityLevel == "HEStd_NotSet") {
            parameters.SetSecurityLevel(HEStd_NotSet);
        }

        CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
        cc->Enable(PKE);
        cc->Enable(KEYSWITCH);
        cc->Enable(LEVELEDSHE);
        cc->Enable(ADVANCEDSHE);

        KeyPair<DCRTPoly> keyPair = cc->KeyGen();
        if (!keyPair.good()) {
            throw std::runtime_error("Key generation failed");
        }
        cc->EvalMultKeyGen(keyPair.secretKey);
        cc->EvalSumKeyGen(keyPair.secretKey);
        if (!rotationIndices.empty()) {
            cc->EvalRotateKeyGen(keyPair.secretKey, rotationIndices);
        }

        const std::string outputDir = "test";
        std::filesystem::create_directories(outputDir);
        PublicKey <DCRTPoly> pk = keyPair.publicKey;
        const int centroidsPerCiphertext = 8;

        const std::vector<double> query = ReadVectorFile(inputVectorPath);
        const int queryDim = static_cast<int>(query.size());
        if (queryDim > paddedDim) {
            throw std::runtime_error("Query dimension exceeds --padded-dim");
        }

        const int slotsUsed = paddedDim * centroidsPerCiphertext;
        std::vector<double> packedQuery(static_cast<size_t>(slotsUsed), 0.0);
        for (int d = 0; d < queryDim; ++d) {
            for (int b = 0; b < centroidsPerCiphertext; ++b) {
                packedQuery[static_cast<size_t>(d * centroidsPerCiphertext + b)] =
                        query[static_cast<size_t>(d)];
            }
        }

        const double normSquared = std::inner_product(query.begin(), query.end(), query.begin(), 0.0);
        std::vector<double> packedNorm(static_cast<size_t>(slotsUsed), 0.0);
        std::fill_n(packedNorm.begin(), centroidsPerCiphertext, normSquared);

        auto encQuery = cc->Encrypt(pk, cc->MakeCKKSPackedPlaintext(packedQuery));
        auto encNorm = cc->Encrypt(pk, cc->MakeCKKSPackedPlaintext(packedNorm));

        std::ofstream out(outputDir + "/centroid_batch_query_metadata.json");
        out << "{\n";
        out << "  \"format_version\": \"openfhe_centroid_batch_query_v1\",\n";
        out << "  \"backend\": \"openfhe_cpp\",\n";
        out << "  \"query_dim\": " << queryDim << ",\n";
        out << "  \"padded_dim\": " << paddedDim << ",\n";
        out << "  \"centroids_per_ciphertext\": " << centroidsPerCiphertext << ",\n";
        out << "  \"slots_per_ciphertext\": " << slotsUsed << "\n";
        out << "}\n";

        std::cout << "Encrypted replicated query for centroid batching to " << outputDir << "\n";

        const auto centroids = ReadMatrixFile("../centroids.txt");
        const int nCentroids = static_cast<int>(centroids.size());
        const int centroidDim = static_cast<int>(centroids.front().size());
        if (centroidDim <= 0 || centroidDim > paddedDim) {
            throw std::runtime_error("Centroid dimension must be > 0 and <= --padded-dim");
        }
        for (const auto &centroid: centroids) {
            if (static_cast<int>(centroid.size()) != centroidDim) {
                throw std::runtime_error("Centroid dimension mismatch in centroids file");
            }
        }
        int numThreads = 20;
        int batchSize = 1;

        SetThreadCount(numThreads);
        const int nBatches = (nCentroids + centroidsPerCiphertext - 1) / centroidsPerCiphertext;
        std::cout << "Computing centroid-batched distances for " << nCentroids
                  << " centroids with " << MaxThreadCount() << " thread(s).\n";

        const auto wallStart = std::chrono::high_resolution_clock::now();
        std::atomic<bool> failed(false);
        std::string failureMessage;
        std::vector<Ciphertext<DCRTPoly>> distances;

#pragma omp parallel for schedule(dynamic, batchSize)
        for (int batchIdx = 0; batchIdx < nBatches; ++batchIdx) {
            if (failed.load()) {
                continue;
            }
            try {
                const int centroidStart = batchIdx * centroidsPerCiphertext;
                const int centroidsInBatch =
                        std::min(centroidsPerCiphertext, nCentroids - centroidStart);
                std::vector<double> centroidPacked(static_cast<size_t>(slotsUsed), 0.0);
                std::vector<double> centroidNormPacked(static_cast<size_t>(slotsUsed), 0.0);
                for (int b = 0; b < centroidsInBatch; ++b) {
                    const auto &centroid = centroids[static_cast<size_t>(centroidStart + b)];
                    centroidNormPacked[static_cast<size_t>(b)] =
                            std::inner_product(centroid.begin(), centroid.end(), centroid.begin(), 0.0);
                    for (int d = 0; d < centroidDim; ++d) {
                        centroidPacked[static_cast<size_t>(d * centroidsPerCiphertext + b)] =
                                centroid[static_cast<size_t>(d)];
                    }
                }
                Ciphertext<DCRTPoly> dot = cc->EvalMult(encQuery, cc->MakeCKKSPackedPlaintext(centroidPacked));
                for (int step = 1; step < paddedDim; step <<= 1) {
                    dot = cc->EvalAdd(dot, cc->EvalAtIndex(dot, step * centroidsPerCiphertext));
                }
                Ciphertext<DCRTPoly> distance = cc->EvalSub(
                        cc->EvalAdd(encNorm, cc->MakeCKKSPackedPlaintext(centroidNormPacked)),
                        cc->EvalAdd(dot, dot));
                distances.push_back(distance);
            } catch (const std::exception &ex) {
                failed.store(true);
#pragma omp critical
                { failureMessage = ex.what(); }
            }
        }
        if (failed.load()) {
            throw std::runtime_error(failureMessage);
        }

        std::ofstream nout(outputDir + "/distances_metadata.json");
        nout << "{\n";
        nout << "  \"backend\": \"openfhe_cpp\",\n";
        nout << "  \"format_version\": \"openfhe_centroid_batch_distances_v1\",\n";
        nout << "  \"n_centroids\": " << nCentroids << ",\n";
        nout << "  \"n_distances\": " << nCentroids << ",\n";
        nout << "  \"centroid_dim\": " << centroidDim << ",\n";
        nout << "  \"padded_dim\": " << paddedDim << ",\n";
        nout << "  \"centroids_per_ciphertext\": " << centroidsPerCiphertext << ",\n";
        nout << "  \"n_batches\": " << nBatches << "\n";
        nout << "}\n";

        const double elapsedSec =
                std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - wallStart).count();
        std::cout << "Computed " << nCentroids << " centroid distances in "
                  << elapsedSec << " s.\n";

        PrivateKey <DCRTPoly> sk = keyPair.secretKey;

        std::vector<std::pair<double, int>> scored;
        scored.reserve(nCentroids);
        for (int batchIdx = 0; batchIdx < nBatches; ++batchIdx) {
            Ciphertext<DCRTPoly> ct = distances[batchIdx];
            Plaintext pt;
            cc->Decrypt(sk, ct, &pt);
            const auto values = pt->GetRealPackedValue();
            for (int b = 0; b < centroidsPerCiphertext; ++b) {
                const int centroidIdx = batchIdx * centroidsPerCiphertext + b;
                if (centroidIdx >= nCentroids || static_cast<size_t>(b) >= values.size()) {
                    break;
                }
                scored.emplace_back(values[static_cast<size_t>(b)], centroidIdx);
            }
        }
        std::sort(scored.begin(), scored.end(), [](const auto &a, const auto &b) {
            return a.first < b.first;
        });
        const int topK = 100;
        const int keep = std::min(topK, static_cast<int>(scored.size()));
        std::vector<double> topDistances;
        std::vector<int> topIndices;
        for (int i = 0; i < keep; ++i) {
            topDistances.push_back(scored[static_cast<size_t>(i)].first);
            topIndices.push_back(scored[static_cast<size_t>(i)].second);
        }
        WriteJsonTopK("test/top_k_results.json", topDistances, topIndices);
        std::cout << "Decrypted " << scored.size() << " packed centroid distances.\n";
    }
} //namespace

int main(){
    //run-centroid --context-dir DIR --input-vector query.txt --centroids-file centroids.txt --work-dir DIR [--output-json FILE]
    //RunCentroidEndToEnd {{"--context-dir, "openfhe_core"},{"--input-vector","query.txt"},{"--centroids-file","centroids.txt"}}
    //RunKeygen (context directory), RunEncryptCentroid (output dir)
    //RunComputeCentroid (encrypted query, encrypted norm, output dir)
    //RunDecryptCentroid (distance dir, output json file)
    bool mac = false;
    if (mac){
        centroids();
    }
    else {
        CliArgs args;
        args.kv.push_back({"--context-dir", "test"});
        args.kv.push_back({"--input-vector", "../query_768d_from_k4096_centroid0.txt"});
        args.kv.push_back({"--centroids-file", "../centroids.txt"});
        const std::string workDir = "test";
        const std::string outputJson = "test/top_k_results.json";


        std::cout << "Run KeyGen" << std::endl;
        RunKeygen(args, "keygen-centroid");
        CliArgs encArgs = args;
        encArgs.kv.push_back({"--output-dir", workDir + "/encrypted_query"});

        std::cout << "Run EncryptCentroid" << std::endl;
        RunEncryptCentroid(encArgs);
        CliArgs compArgs = args;
        compArgs.kv.push_back({"--encrypted-query", workDir + "/encrypted_query/encrypted_query_centroid_batched.bin"});
        compArgs.kv.push_back({"--encrypted-norm", workDir + "/encrypted_query/encrypted_norm_centroid_batched.bin"});
        compArgs.kv.push_back({"--output-dir", workDir + "/encrypted_distances"});

        std::cout << "Run ComputeCentroid" << std::endl;
        RunComputeCentroid(compArgs);
        CliArgs decArgs = args;
        decArgs.kv.push_back({"--encrypted-distances-dir", workDir + "/encrypted_distances"});
        decArgs.kv.push_back({"--output-json", outputJson});

        std::cout << "Run DecryptCentroid" << std::endl;
        RunDecryptCentroid(decArgs);
    }

    constexpr uint64_t embedding_DBSize       = 65000;
    constexpr uint64_t embedding_DBEntrySize  = 768;
    constexpr uint64_t embedding_DBEntryBytes = embedding_DBEntrySize * 8;
    constexpr uint64_t embedding_BatchSize    = 32;
    constexpr uint64_t embedding_FailProb     = 20;

    std::cout << "Loading embedding database ... " << std::endl;
    std::vector<uint64_t> embedding_rawDB(embedding_DBEntrySize * embedding_DBSize);
    std::vector<std::vector<uint64_t>> embedding_DB;
    if (mac) embedding_DB = load_json_distances("/Users/antoniajanuszewicz/PycharmProjects/PIANO-RAG/faiss.json");
    else embedding_DB = load_json_distances("/home/ajanusze/PIANO-RAG/faiss.json");


    //std::cout << std::endl;
    for (uint64_t i = 0; i < embedding_DB.size(); ++i)
        for (uint64_t j = 0; j < embedding_DB[i].size(); ++j) {
            embedding_rawDB[i * embedding_DBEntrySize + j] = embedding_DB[i][j];
            //std::cout << " " << int(embedding_DB[i][j]);
        }
    //std::cout << std::endl;


    SimpleBatchPianoPIR embedding_pir(embedding_DBSize, embedding_DBEntryBytes, embedding_BatchSize,embedding_rawDB, embedding_FailProb);

    embedding_pir.Preprocessing();


    std::cout << "Loading centroid indices ... " << std::endl;
    std::vector<uint64_t> centroidIndices =
    //        load_centroid_indices("/Users/antoniajanuszewicz/PycharmProjects/PIANO-RAG/prototype/rag_operations/ground_truth.json");
    //        load_centroid_indices("/home/ajanusze/PIANO-RAG/decrypted_results/top_k_results.json");
            load_centroid_indices("test/top_k_results.json");
    std::cout << "Loading centroid mapping ... " << std::endl;
    std::unordered_map<uint64_t, std::vector<uint64_t>> centroidToIndex;
    if (mac) centroidToIndex = load_json_mapping("/Users/antoniajanuszewicz/PycharmProjects/PIANO-RAG/prototype/data/65000_lists.json");
    else centroidToIndex = load_json_mapping("/home/ajanusze/PIANO-RAG/prototype/data/65000_lists.json");

    std::vector<uint64_t> unbatched_query = CentroidToIndex(centroidIndices,centroidToIndex);
    while (unbatched_query.size()%embedding_BatchSize != 0) unbatched_query.push_back(0);
    std::cout << "Queries to batch : ";
    for (int i = 0; i < unbatched_query.size(); i++)
        std::cout << unbatched_query[i] << " ";

    std::cout << "Computing queries ... " << unbatched_query.size() << std::endl;
    std::vector<std::vector<float>> responses;
    std::vector<uint64_t> index_responses;
    for (uint64_t i = 0; i < unbatched_query.size(); i+=embedding_BatchSize) {
        std::unordered_set<uint64_t> querySet;
        std::vector<uint64_t> batchQuery;
        std::cout << std::endl << " Query " << i << ": ";
        for (uint64_t j = 0; j < embedding_BatchSize;j++) {
            if (i+j < unbatched_query.size())
                batchQuery.push_back(unbatched_query[i + j]);
            else batchQuery.push_back(0);
            std::cout << " " << batchQuery[j];
        }

        auto response = embedding_pir.Query(batchQuery);
        //std::cout << "Number of responses: " << response.size() << std::endl;

        std::cout << std::endl;
        for (int j = 0; j < response.size(); j++) {
            if (response[j].size() > 0) {
                std::cout << " " << std::setprecision(std::numeric_limits<double>::max_digits10)
                          << std::bit_cast<double>(response[j][0]);
                if (response[j][0] == 0){
                    unbatched_query.push_back(unbatched_query[i+j]);
                }
                else{
                    std::vector<float> res;
                    for (int k = 0; k < response[j].size();k++)
                        res.push_back((float)std::bit_cast<double>(response[j][k]));
                    responses.push_back(res);
                    index_responses.push_back(batchQuery[j]);
                }
            }
            //for (int k = 0; k < response[j].size(); k++)
            //    std::cout << " " << std::setprecision(std::numeric_limits<double>::max_digits10) << std::bit_cast<double>(response[j][k]);
        }
        std::cout << std::endl;
    }

    int top_k = 10;
    uint64_t dim = 728;
    std::vector<float> candidate_vectors;
    candidate_vectors.reserve(responses.size() * responses[0].size());
    for (const auto& vec : responses) {
        candidate_vectors.insert(candidate_vectors.end(), vec.begin(), vec.end());
    }
    faiss::IndexFlatL2 findex(dim);
    findex.add(responses.size(), candidate_vectors.data());
    std::vector<float> distances(top_k);
    std::vector<faiss::idx_t> indices(top_k);

    std::vector<float> query_vector = ReadFloatVectorFile("../query_768d_from_k4096_centroid0.txt");
    findex.search(
            1,                      // number of queries
            query_vector.data(),    // query vector
            top_k,
            distances.data(),
            indices.data()
    );
    std::vector<std::pair<uint64_t, float>> results;
    std::vector<uint64_t> queries;
    results.reserve(top_k);
    for (int i = 0; i < top_k; ++i) {
        if (indices[i] >= 0) { // FAISS may return -1 if not enough points
            results.emplace_back(
                    index_responses[indices[i]],
                    distances[i]
            );
            queries.push_back(index_responses[indices[i]]);
        }
    }

    std::cout << "Top-K: ";
    for (int i = 0; i < results.size(); i++){
        std::cout << "[ " << results[i].first << " " << results[i].second << "] ";
    }
    std::cout << std::endl;



    std::cout << "Loading text database ..." << std::endl;
    constexpr uint64_t text_DBSize       = 65000;
    constexpr uint64_t text_DBEntrySize  = 64;
    constexpr uint64_t text_DBEntryBytes = text_DBEntrySize * 8;
    constexpr uint64_t text_BatchSize    = 32;
    constexpr uint64_t text_FailProb     = 20;

    std::vector<uint64_t> text_rawDB(text_DBEntrySize * text_DBSize);
    std::vector<std::string> text_DB;
    if (mac) text_DB =  load_text_database("/Users/antoniajanuszewicz/PycharmProjects/PIANO-RAG/uniform_index.txt");
    else text_DB = load_text_database("/home/ajanusze/PIANO-RAG/uniform_index.txt");

    //std::cout << "Text DB: ";
    //for (uint64_t i = 0; i <text_DB.size(); i++)
    //    std::cout << text_DB[i];
    std::cout << std::endl;
    for (uint64_t i = 0; i < text_DB.size(); ++i)
        for (uint64_t j = 0; j < text_DB[i].size(); ++j)
            text_rawDB[i * text_DBEntrySize + j] = int(text_DB[i][j]);

    PianoPIR pir(text_DBSize, text_DBEntryBytes, text_rawDB, text_FailProb);

    pir.Preprocessing();

    const uint64_t maxQueryNum = pir.client_MaxQueryNum();

    std::cout << "Loading indices ... " << std::endl;
    //std::vector<uint64_t> queries =
            //load_centroid_indices("/Users/antoniajanuszewicz/PycharmProjects/PIANO-RAG/prototype/ground_truth/ground_truth.json");
            //load_centroid_indices("/home/ajanusze/PIANO-RAG/prototype/ground_truth/ground_truth.json");
    uint64_t retrieve_number = (maxQueryNum > queries.size()) ? queries.size() : maxQueryNum;

    std::cout << "Processing queries ... " << retrieve_number << std::endl;
    for (uint64_t i = 0; i < retrieve_number; ++i) {
        uint64_t idx = queries[i];

        std::vector<uint64_t> query;
        query = pir.Query(idx, /*realQuery=*/true);
        std::cout << "Size of return: " << query.size() << " for query " << idx << std::endl;
        for (uint64_t j = 0; j < query.size(); ++j)
            std::cout << char(query[j]);
        std::cout << std::endl;
    }

    std::cout << "Successful RAG completion!" << std::endl;
}