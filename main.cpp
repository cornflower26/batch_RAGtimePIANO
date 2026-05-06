
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
        const int paddedDim = 1024;
        const std::string inputVectorPath = "../query_768d_from_k4096_centroid0.txt";


        const std::string outputDir = "test";
        std::filesystem::create_directories(outputDir);
        PublicKey <DCRTPoly> pk = LoadPublicKey("test/public_key.bin");;
        CryptoContext<DCRTPoly> cc = pk->GetCryptoContext();
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
        std::vector<Ciphertext<DCRTPoly>> distances(nBatches);

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
                distances[batchIdx] = distance;
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

        PrivateKey <DCRTPoly> sk = LoadSecretKey("test/secret_key.bin");

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
    CliArgs args;
    args.kv.push_back({"--context-dir", "test"});
    args.kv.push_back({"--input-vector", "../query_768d_from_k4096_centroid0.txt"});
    args.kv.push_back({"--centroids-file", "../centroids_1m.txt"});
    const std::string workDir = "test";
    const std::string outputJson = "test/top_k_results.json";
    bool mac = false;
    // CHANGE ME !!!
    constexpr uint64_t DBsize = 702873;
            //702873, 1120486

    constexpr uint64_t embedding_DBSize       = DBsize;
    constexpr uint64_t embedding_DBEntrySize  = 768;
    constexpr uint64_t embedding_DBEntryBytes = embedding_DBEntrySize * 8;
    constexpr uint64_t embedding_BatchSize    = 32;
    constexpr uint64_t embedding_FailProb     = 20;

    constexpr uint64_t text_DBSize       = DBsize;
    constexpr uint64_t text_DBEntrySize  = 1024;
    constexpr uint64_t text_DBEntryBytes = text_DBEntrySize * 8;
    constexpr uint64_t text_BatchSize    = 32;
    constexpr uint64_t text_FailProb     = 20;

    std::cout << "Loading embedding database ... " << std::endl;
    std::vector<uint64_t> embedding_rawDB(embedding_DBEntrySize * embedding_DBSize);
    std::vector<std::vector<uint64_t>> embedding_DB;
    // CHANGE ME !!!
    if (mac) embedding_DB = load_json_distances("/Users/antoniajanuszewicz/PycharmProjects/PIANO-RAG/modified_faiss_10000.json");
    else embedding_DB = load_json_distances("/home/ajanusze/PIANO-RAG/modified_faiss_1000000.json");
    std::unordered_map<uint64_t, std::vector<uint64_t>> centroidToIndex;
    std::cout << "Loading centroid mapping ... " << std::endl;
    // CHANGE ME !!!
    if (mac) centroidToIndex = load_json_mapping("/Users/antoniajanuszewicz/PycharmProjects/PIANO-RAG/prototype/data/10000_lists.json");
    else centroidToIndex = load_json_mapping("/home/ajanusze/PIANO-RAG/prototype/data/1000000_lists.json");
    std::cout << "Loading text database ..." << std::endl;
    std::vector<std::string> text_DB;
    // CHANGE ME !!
    if (mac) text_DB =  load_text_database("/Users/antoniajanuszewicz/PycharmProjects/PIANO-RAG/uniform_index.txt");
    else text_DB = load_text_database("/home/ajanusze/PIANO-RAG/uniform_index_1000000_1024.txt");
    std::cout << "Embedding database to raw database size " << embedding_DB.size() << std::endl;
    for (uint64_t i = 0; i < DBsize; ++i)
        for (uint64_t j = 0; j < embedding_DB[i].size(); ++j) {
            embedding_rawDB[i * embedding_DBEntrySize + j] = embedding_DB[i][j];
        }
    std::cout << "Text database to raw database size " << text_DB.size() << std::endl;
    std::vector<uint64_t> text_rawDB(text_DBEntrySize * text_DBSize);
    for (uint64_t i = 0; i < DBsize; ++i)
        for (uint64_t j = 0; j < text_DB[i].size(); ++j)
            text_rawDB[i * text_DBEntrySize + j] = int(text_DB[i][j]);


    std::cout << "Server and Client One-time Setup" << std::endl;
    auto pre_setup_time_start = std::chrono::steady_clock::now();
    PianoPIR embedding_pir(embedding_DBSize, embedding_DBEntryBytes,embedding_rawDB, embedding_FailProb);
    embedding_pir.Preprocessing();
    PianoPIR pir(text_DBSize, text_DBEntryBytes, text_rawDB, text_FailProb);
    pir.Preprocessing();
    const uint64_t maxQueryNum = pir.client_MaxQueryNum();

    std::cout << "Run KeyGen" << std::endl;
    RunKeygen(args, "keygen-centroid");
    CliArgs encArgs = args;
    encArgs.kv.push_back({"--output-dir", workDir + "/encrypted_query"});
    auto elapsed_pre_setup_time =
            std::chrono::duration<double>(std::chrono::steady_clock::now()-pre_setup_time_start).count();
    std::cout << "One-time Setup time " << elapsed_pre_setup_time << std::endl;


    std::cout << "Setup time" << std::endl;
    auto setup_time_start = std::chrono::steady_clock::now();
    if (mac){
        centroids();
    }
    else {
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
        decArgs.kv.push_back({"--top-k","100"});

        std::cout << "Run DecryptCentroid" << std::endl;
        RunDecryptCentroid(decArgs);
    }
    auto elapsed_setup_time =
            std::chrono::duration<double>(std::chrono::steady_clock::now()-setup_time_start).count();
    std::cout << "Setup time " << elapsed_setup_time << std::endl;


    std::cout << "Query time start " << std::endl;
    auto query_time_start = std::chrono::steady_clock::now();
    std::cout << "Loading centroid indices ... " << std::endl;
    std::vector<uint64_t> centroidIndices =
    //        load_centroid_indices("/Users/antoniajanuszewicz/PycharmProjects/PIANO-RAG/prototype/rag_operations/ground_truth.json");
    //        load_centroid_indices("/home/ajanusze/PIANO-RAG/decrypted_results/top_k_results.json");
            load_centroid_indices("test/top_k_results.json");
    std::vector<uint64_t> unbatched_query = CentroidToIndex(centroidIndices,centroidToIndex);
    //while (unbatched_query.size()%embedding_BatchSize != 0) unbatched_query.push_back(0);
    std::cout << "Queries to batch : ";
    for (int i = 0; i < unbatched_query.size(); i++)
        std::cout << unbatched_query[i] << " ";
    std::cout << std::endl;

    std::cout << "Computing queries ... " << unbatched_query.size() << std::endl;
    std::vector<std::vector<float>> responses;
    std::vector<uint64_t> index_responses;
    uint64_t ret_num = (embedding_pir.client_MaxQueryNum() > unbatched_query.size())
            ? unbatched_query.size() : embedding_pir.client_MaxQueryNum();
    std::cout << "Processing queries ... " << ret_num << std::endl;
    for (uint64_t i = 0; i < ret_num; ++i) {
        uint64_t idx = unbatched_query[i];
        std::vector<uint64_t> query;
        query = embedding_pir.Query(idx, /*realQuery=*/true);
        std::vector<float> res;
        for (int k = 0; k < query.size();k++)
            res.push_back((float)std::bit_cast<double>(query[k]));
        responses.push_back(res);
        index_responses.push_back(idx);
        //std::cout << "Size of return: " << query.size()*sizeof(query[0]) << " for query " << idx << std::endl;

    }
    std::cout << "PIR 1 Upload " << embedding_pir.UploadCostPerQuery()*unbatched_query.size() << std::endl;
    std::cout << "PIR 1 Download " << embedding_pir.DownloadCostPerQuery()*unbatched_query.size() << std::endl;
    /**
    for (uint64_t i = 0; i < unbatched_query.size(); i+=embedding_BatchSize) {
        std::unordered_set<uint64_t> querySet;
        std::vector<uint64_t> batchQuery;
        std::cout << " Query " << i/embedding_BatchSize << std::endl;
        for (uint64_t j = 0; j < embedding_BatchSize;j++) {
            if (i+j < unbatched_query.size())
                batchQuery.push_back(unbatched_query[i + j]);
            else batchQuery.push_back(0);
            //std::cout << " " << batchQuery[j];
        }

        auto response = embedding_pir.Query(batchQuery);
        phase1_download += response.size()*response[0].size()*sizeof(response[0][0]);

        //std::cout << std::endl;
        for (int j = 0; j < response.size(); j++) {
            if (response[j].size() > 0) {
                //std::cout << " " << std::setprecision(std::numeric_limits<double>::max_digits10)
                //          << std::bit_cast<double>(response[j][0]);
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
        //std::cout << std::endl;
    }**/

    std::cout << "Computing global distances "<< std::endl;
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
        //std::cout << "Size of return: " << query.size()*sizeof(query[0]) << " for query " << idx << std::endl;

        /**
        for (uint64_t j = 0; j < query.size(); ++j)
            std::cout << char(query[j]);
        std::cout << std::endl;
         **/
    }
    std::cout << "PIR 2 Upload " << pir.UploadCostPerQuery()*queries.size() << std::endl;
    std::cout << "PIR 2 Download " << pir.DownloadCostPerQuery()*queries.size() << std::endl;
    auto elapsed_query_time =
            std::chrono::duration<double>(std::chrono::steady_clock::now()-query_time_start).count();
    std::cout << "Query time " << elapsed_query_time << std::endl;

    std::cout << "Total time " << elapsed_setup_time+elapsed_query_time << std::endl;


    std::cout << "Successful RAG completion!" << std::endl;
}