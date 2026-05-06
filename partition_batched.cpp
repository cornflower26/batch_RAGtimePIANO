
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


int main(){
    CliArgs args;
    args.kv.push_back({"--context-dir", "test"});
    args.kv.push_back({"--input-vector", "../query_768d_from_k4096_centroid0.txt"});
    args.kv.push_back({"--centroids-file", "../centroids_1k.txt"});
    const std::string workDir = "test";
    const std::string outputJson = "test/top_k_results.json";
    bool mac = false;
    constexpr uint64_t DBsize = 1000;
    //702873, 1120486

    uint64_t embedding_DBSize       = DBsize;
    constexpr uint64_t embedding_DBEntrySize  = 768;
    constexpr uint64_t embedding_DBEntryBytes = embedding_DBEntrySize * 8;
    uint64_t embedding_BatchSize    = 64;
    constexpr uint64_t embedding_FailProb     = 20;

    constexpr uint64_t text_DBSize       = DBsize;
    constexpr uint64_t text_DBEntrySize  = 1024;
    constexpr uint64_t text_DBEntryBytes = text_DBEntrySize * 8;
    constexpr uint64_t text_BatchSize    = 32;
    constexpr uint64_t text_FailProb     = 20;

    std::cout << "Loading embedding database ... " << std::endl;
    std::vector<std::vector<uint64_t>> embedding_DB;
    if (mac) embedding_DB = load_json_distances("/Users/antoniajanuszewicz/PycharmProjects/PIANO-RAG/modified_faiss_1000.json");
    else embedding_DB = load_json_distances("/home/ajanusze/PIANO-RAG/modified_faiss_1000.json");
    std::unordered_map<uint64_t, std::vector<uint64_t>> centroidToIndex;
    std::cout << "Loading centroid mapping ... " << std::endl;
    if (mac) centroidToIndex = load_json_mapping("/Users/antoniajanuszewicz/PycharmProjects/PIANO-RAG/prototype/data/1000_lists.json");
    else centroidToIndex = load_json_mapping("/home/ajanusze/PIANO-RAG/prototype/data/1000_lists.json");

    embedding_BatchSize = 256;
            //chooseBatchSize(centroidToIndex);
    std::cout << " Optimized batch size " << embedding_BatchSize << std::endl;

    uint64_t paddedPartitionSize = 0;
    std::unordered_map<uint64_t, uint64_t> embedding_index;
    embedding_DB = reorderDBForBatchPIR( embedding_DB, centroidToIndex, embedding_DBEntrySize,
                                         embedding_BatchSize, embedding_index);
    std::cout << std::endl;
    embedding_BatchSize = centroidToIndex.size() * kRealQueryPerPartition;
    embedding_DBSize = centroidToIndex.size() * paddedPartitionSize;
    std::cout << "Loading text database ..." << std::endl;
    std::vector<std::string> text_DB;
    if (mac) text_DB =  load_text_database("/Users/antoniajanuszewicz/PycharmProjects/PIANO-RAG/uniform_index.txt");
    else text_DB = load_text_database("/home/ajanusze/PIANO-RAG/uniform_index.txt");

    std::vector<uint64_t> embedding_rawDB(embedding_DB.size()*embedding_DBEntrySize);
    for (uint64_t i = 0; i < embedding_DB.size(); ++i)
        for (uint64_t j = 0; j < embedding_DB[i].size(); ++j) {
            embedding_rawDB[i * embedding_DBEntrySize + j] = embedding_DB[i][j];
        }

    std::vector<uint64_t> text_rawDB(text_DBEntrySize * text_DBSize);
    for (uint64_t i = 0; i < DBsize; ++i)
        for (uint64_t j = 0; j < text_DB[i].size(); ++j)
            text_rawDB[i * text_DBEntrySize + j] = int(text_DB[i][j]);


    std::cout << "Server and Client One-time Setup" << std::endl;
    auto pre_setup_time_start = std::chrono::steady_clock::now();
    SimpleBatchPianoPIR embedding_pir(embedding_DB.size(), embedding_DBEntryBytes, embedding_BatchSize,embedding_rawDB, embedding_FailProb);
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
        //centroids();
        std::cout << "Batching not available on mac currently" << std::endl;
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
        decArgs.kv.push_back({"--top-k","4"});

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
    for (int i = 0; i < unbatched_query.size(); i++) {
        std::cout << "(" << unbatched_query[i] << " , " <<  embedding_index[unbatched_query[i]] << ") ";
        unbatched_query[i] = embedding_index.at(unbatched_query[i]);
    }
    std::cout << std::endl;

    std::cout << "Computing queries ... " << unbatched_query.size() << std::endl;
    std::vector<std::vector<float>> responses;
    std::vector<uint64_t> index_responses;
    //uint64_t ret_num = (maxQueryNum > unbatched_query.size()) ? unbatched_query.size() : pir.client_MaxQueryNum();
    //std::cout << "Processing queries ... " << ret_num << std::endl;
    std::vector<std::vector<uint64_t>> batched_queries = makeOptimizedBatchesContiguous(
            unbatched_query,embedding_DBSize,
            paddedPartitionSize,2);
    for (uint64_t i = 0; i < batched_queries.size(); i++) {
        auto response = embedding_pir.Query(batched_queries[i]);

        //std::cout << std::endl;
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
                    index_responses.push_back(batched_queries[i][j]);
                }
            }
            //for (int k = 0; k < response[j].size(); k++)
            //    std::cout << " " << std::setprecision(std::numeric_limits<double>::max_digits10) << std::bit_cast<double>(response[j][k]);
        }
        //std::cout << std::endl;
    }
    std::cout << "PIR 1 Upload " << embedding_pir.UploadCostPerBatchOnline()*unbatched_query.size() << std::endl;
    std::cout << "PIR 1 Download " << embedding_pir.DownloadCostPerBatchOnline()*unbatched_query.size() << std::endl;

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