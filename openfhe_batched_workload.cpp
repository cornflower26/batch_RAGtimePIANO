#if __has_include(<openfhe.h>)
#include <openfhe.h>
#elif __has_include(<pke/openfhe.h>)
#include <pke/openfhe.h>
#elif __has_include(<openfhe/pke/openfhe.h>)
#include <openfhe/pke/openfhe.h>
#else
#error "OpenFHE header not found. Tried <openfhe.h>, <pke/openfhe.h>, <openfhe/pke/openfhe.h>."
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace lbcrypto;

namespace {

struct CliArgs {
  std::vector<std::string> positional;
  std::vector<std::pair<std::string, std::string>> kv;
};

struct ContextBundlePaths {
  std::string contextFile;
  std::string publicKeyFile;
  std::string secretKeyFile;
  std::string evalMultKeyFile;
  std::string evalSumKeyFile;
  std::string evalAutomorphismKeyFile;
};

bool gEvalKeysAvailableInProcess = false;

CliArgs ParseArgs(int argc, char** argv) {
  CliArgs out;
  for (int i = 1; i < argc; ++i) {
    std::string token(argv[i]);
    if (token.rfind("--", 0) == 0) {
      if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
        out.kv.push_back({token, std::string(argv[i + 1])});
        ++i;
      } else {
        out.kv.push_back({token, ""});
      }
    } else {
      out.positional.push_back(token);
    }
  }
  return out;
}

std::string GetArg(const CliArgs& args, const std::string& name,
                   const std::string& defaultValue = "") {
  for (const auto& kv : args.kv) {
    if (kv.first == name) {
      return kv.second;
    }
  }
  return defaultValue;
}

bool HasArg(const CliArgs& args, const std::string& name) {
  for (const auto& kv : args.kv) {
    if (kv.first == name) {
      return true;
    }
  }
  return false;
}

template <typename T>
void CheckSerialize(bool ok, const std::string& path, const std::string& op) {
  (void)sizeof(T);
  if (!ok) {
    throw std::runtime_error(op + " failed for path: " + path);
  }
}

ContextBundlePaths ResolveContextPaths(const std::string& contextDir) {
  ContextBundlePaths paths;
  paths.contextFile = contextDir + "/context.bin";
  paths.publicKeyFile = contextDir + "/public_key.bin";
  paths.secretKeyFile = contextDir + "/secret_key.bin";
  paths.evalMultKeyFile = contextDir + "/eval_mult_keys.bin";
  paths.evalSumKeyFile = contextDir + "/eval_sum_keys.bin";
  paths.evalAutomorphismKeyFile = contextDir + "/eval_automorphism_keys.bin";
  return paths;
}

std::string FormatIndex(int value) {
  std::ostringstream oss;
  oss << std::setw(4) << std::setfill('0') << value;
  return oss.str();
}

std::vector<uint32_t> ParseUintCsv(const std::string& csv) {
  std::vector<uint32_t> out;
  std::stringstream ss(csv);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (!token.empty()) {
      out.push_back(static_cast<uint32_t>(std::stoul(token)));
    }
  }
  return out;
}

std::vector<int32_t> ParseIntCsv(const std::string& csv) {
  std::vector<int32_t> out;
  std::stringstream ss(csv);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (!token.empty()) {
      out.push_back(static_cast<int32_t>(std::stoi(token)));
    }
  }
  return out;
}

std::vector<int32_t> RotationIndicesForLayout(int lanes, int paddedDim) {
  if (lanes <= 0 || paddedDim <= 0) {
    throw std::runtime_error("lanes and padded_dim must be positive");
  }
  std::vector<int32_t> out;
  for (int step = 1; step < paddedDim; step <<= 1) {
    out.push_back(static_cast<int32_t>(step * lanes));
  }
  return out;
}

std::vector<double> ReadVectorFile(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open vector file: " + path);
  }
  std::vector<double> out;
  double value = 0.0;
  while (in >> value) {
    out.push_back(value);
  }
  if (out.empty()) {
    throw std::runtime_error("Vector file is empty: " + path);
  }
  return out;
}

std::vector<float> ReadFloatVectorFile(const std::string& path) {
        std::ifstream in(path);
        if (!in) {
            throw std::runtime_error("Failed to open vector file: " + path);
        }
        std::vector<float> out;
        double value = 0.0;
        while (in >> value) {
            out.push_back(value);
        }
        if (out.empty()) {
            throw std::runtime_error("Vector file is empty: " + path);
        }
        return out;
}

std::vector<std::vector<double>> ReadMatrixFile(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open matrix file: " + path);
  }
  std::vector<std::vector<double>> rows;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    std::stringstream ss(line);
    std::vector<double> row;
    double value = 0.0;
    while (ss >> value) {
      row.push_back(value);
    }
    if (!row.empty()) {
      rows.push_back(row);
    }
  }
  if (rows.empty()) {
    throw std::runtime_error("Matrix file has no rows: " + path);
  }
  return rows;
}

int ParseRequiredInt(const std::string& jsonPath, const std::string& key) {
  std::ifstream in(jsonPath);
  if (!in) {
    throw std::runtime_error("Failed to open metadata file: " + jsonPath);
  }
  std::stringstream buffer;
  buffer << in.rdbuf();
  const std::string text = buffer.str();
  const std::regex rx("\"" + key + "\"\\s*:\\s*(\\d+)");
  std::smatch match;
  if (!std::regex_search(text, match, rx)) {
    throw std::runtime_error("Key not found in metadata: " + key);
  }
  return std::stoi(match[1].str());
}

void SaveContextAndKeys(const ContextBundlePaths& paths,
                        const CryptoContext<DCRTPoly>& cc,
                        const PublicKey<DCRTPoly>& pk,
                        const PrivateKey<DCRTPoly>& sk) {
  CheckSerialize<int>(Serial::SerializeToFile(paths.contextFile, cc, SerType::BINARY),
                      paths.contextFile, "Serialize crypto context");
  CheckSerialize<int>(Serial::SerializeToFile(paths.publicKeyFile, pk, SerType::BINARY),
                      paths.publicKeyFile, "Serialize public key");
  CheckSerialize<int>(Serial::SerializeToFile(paths.secretKeyFile, sk, SerType::BINARY),
                      paths.secretKeyFile, "Serialize secret key");

  std::ofstream evalMultOut(paths.evalMultKeyFile, std::ios::binary);
  if (!evalMultOut) {
    throw std::runtime_error("Failed to open eval mult key file for writing: " +
                             paths.evalMultKeyFile);
  }
  CheckSerialize<int>(
      CryptoContextImpl<DCRTPoly>::SerializeEvalMultKey(
          evalMultOut, SerType::BINARY, sk->GetKeyTag()),
      paths.evalMultKeyFile, "Serialize eval mult keys");

  std::ofstream evalSumOut(paths.evalSumKeyFile, std::ios::binary);
  if (!evalSumOut) {
    throw std::runtime_error("Failed to open eval sum key file for writing: " +
                             paths.evalSumKeyFile);
  }
  CheckSerialize<int>(
      CryptoContextImpl<DCRTPoly>::SerializeEvalSumKey(
          evalSumOut, SerType::BINARY, sk->GetKeyTag()),
      paths.evalSumKeyFile, "Serialize eval sum keys");
}

CryptoContext<DCRTPoly> LoadContext(const std::string& contextFile) {
  CryptoContext<DCRTPoly> cc;
  CheckSerialize<int>(Serial::DeserializeFromFile(contextFile, cc, SerType::BINARY),
                      contextFile, "Deserialize crypto context");
  return cc;
}

PublicKey<DCRTPoly> LoadPublicKey(const std::string& publicKeyFile) {
  PublicKey<DCRTPoly> pk;
  CheckSerialize<int>(Serial::DeserializeFromFile(publicKeyFile, pk, SerType::BINARY),
                      publicKeyFile, "Deserialize public key");
  return pk;
}

PrivateKey<DCRTPoly> LoadSecretKey(const std::string& secretKeyFile) {
  PrivateKey<DCRTPoly> sk;
  CheckSerialize<int>(Serial::DeserializeFromFile(secretKeyFile, sk, SerType::BINARY),
                      secretKeyFile, "Deserialize secret key");
  return sk;
}

void SaveEvalAutomorphismKeys(const std::string& path,
                              const CryptoContext<DCRTPoly>& cc) {
  std::ofstream evalAutoOut(path, std::ios::binary);
  if (!evalAutoOut) {
    throw std::runtime_error("Failed to open eval automorphism key file for writing: " + path);
  }
  CheckSerialize<int>(
      CryptoContextImpl<DCRTPoly>::SerializeEvalAutomorphismKey(
          evalAutoOut, SerType::BINARY, cc),
      path, "Serialize eval automorphism keys");
}

void LoadEvalKeys(const ContextBundlePaths& paths) {
  if (gEvalKeysAvailableInProcess) {
    return;
  }

  std::ifstream evalMultIn(paths.evalMultKeyFile, std::ios::binary);
  if (!evalMultIn) {
    throw std::runtime_error("Failed to open eval mult key file for reading: " +
                             paths.evalMultKeyFile);
  }
  CheckSerialize<int>(
      CryptoContextImpl<DCRTPoly>::DeserializeEvalMultKey(evalMultIn, SerType::BINARY),
      paths.evalMultKeyFile, "Deserialize eval mult keys");

  std::ifstream evalSumIn(paths.evalSumKeyFile, std::ios::binary);
  if (!evalSumIn) {
    throw std::runtime_error("Failed to open eval sum key file for reading: " +
                             paths.evalSumKeyFile);
  }
  CheckSerialize<int>(
      CryptoContextImpl<DCRTPoly>::DeserializeEvalSumKey(evalSumIn, SerType::BINARY),
      paths.evalSumKeyFile, "Deserialize eval sum keys");

  if (std::filesystem::exists(paths.evalAutomorphismKeyFile) &&
      std::filesystem::file_size(paths.evalAutomorphismKeyFile) > 0) {
    std::ifstream evalAutoIn(paths.evalAutomorphismKeyFile, std::ios::binary);
    if (!evalAutoIn) {
      throw std::runtime_error("Failed to open eval automorphism key file for reading: " +
                               paths.evalAutomorphismKeyFile);
    }
    CheckSerialize<int>(
        CryptoContextImpl<DCRTPoly>::DeserializeEvalAutomorphismKey(
            evalAutoIn, SerType::BINARY),
        paths.evalAutomorphismKeyFile, "Deserialize eval automorphism keys");
  }
  gEvalKeysAvailableInProcess = true;
}

void SaveCiphertext(const std::string& path, const Ciphertext<DCRTPoly>& ct) {
  CheckSerialize<int>(Serial::SerializeToFile(path, ct, SerType::BINARY),
                      path, "Serialize ciphertext");
}

Ciphertext<DCRTPoly> LoadCiphertext(const std::string& path) {
  Ciphertext<DCRTPoly> ct;
  CheckSerialize<int>(Serial::DeserializeFromFile(path, ct, SerType::BINARY),
                      path, "Deserialize ciphertext");
  return ct;
}

void SetThreadCount(int numThreads) {
#ifdef _OPENMP
  if (numThreads > 0) {
    omp_set_num_threads(numThreads);
  }
#else
  (void)numThreads;
#endif
}

int MaxThreadCount() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

void WriteJsonTopK(const std::string& path,
                   const std::vector<double>& distances,
                   const std::vector<int>& indices) {
  if (distances.size() != indices.size()) {
    throw std::runtime_error("distances and indices size mismatch");
  }
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Failed to write file: " + path);
  }
  out << "{\n";
  out << "  \"top_k\": " << distances.size() << ",\n";
  out << "  \"distances\": [";
  for (size_t i = 0; i < distances.size(); ++i) {
    out << distances[i] << (i + 1 < distances.size() ? ", " : "");
  }
  out << "],\n";
  out << "  \"centroid_indices\": [";
  for (size_t i = 0; i < indices.size(); ++i) {
    out << indices[i] << (i + 1 < indices.size() ? ", " : "");
  }
  out << "]\n";
  out << "}\n";
}

void WriteMultiQueryTopK(
    const std::string& path,
    const std::vector<std::vector<std::pair<double, int>>>& perQueryTopK) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Failed to write output JSON: " + path);
  }
  out << "{\n";
  out << "  \"n_queries\": " << perQueryTopK.size() << ",\n";
  out << "  \"results\": [\n";
  for (size_t q = 0; q < perQueryTopK.size(); ++q) {
    out << "    {\n";
    out << "      \"query_index\": " << q << ",\n";
    out << "      \"distances\": [";
    for (size_t i = 0; i < perQueryTopK[q].size(); ++i) {
      out << perQueryTopK[q][i].first << (i + 1 < perQueryTopK[q].size() ? ", " : "");
    }
    out << "],\n";
    out << "      \"centroid_indices\": [";
    for (size_t i = 0; i < perQueryTopK[q].size(); ++i) {
      out << perQueryTopK[q][i].second << (i + 1 < perQueryTopK[q].size() ? ", " : "");
    }
    out << "]\n";
    out << "    }" << (q + 1 < perQueryTopK.size() ? "," : "") << "\n";
  }
  out << "  ]\n";
  out << "}\n";
}

int RunKeygen(const CliArgs& args, const std::string& command) {
  const std::string contextDir = GetArg(args, "--context-dir");
  if (contextDir.empty()) {
    throw std::runtime_error("--context-dir is required");
  }

  const uint32_t polyModulusDegree =
      static_cast<uint32_t>(std::stoul(GetArg(args, "--poly-modulus-degree", "16384")));
  const std::string coeffCsv = GetArg(args, "--coeff-mod-bit-sizes", "60,40,40,60");
  const std::string securityLevel = GetArg(args, "--security-level", "");
  const int paddedDim = std::stoi(GetArg(args, "--padded-dim", "1024"));

  std::vector<int32_t> rotationIndices = ParseIntCsv(GetArg(args, "--rotation-indices", ""));
  if (rotationIndices.empty()) {
    int lanes = 0;
    if (command == "keygen-centroid") {
      lanes = std::stoi(GetArg(args, "--centroids-per-ciphertext", "8"));
    } else if (command == "keygen-query-centroid") {
      lanes = std::stoi(GetArg(args, "--queries-per-batch", "2")) *
              std::stoi(GetArg(args, "--centroids-per-batch", "4"));
    } else if (HasArg(args, "--lanes")) {
      lanes = std::stoi(GetArg(args, "--lanes"));
    }
    if (lanes > 0) {
      rotationIndices = RotationIndicesForLayout(lanes, paddedDim);
    }
  }

  const auto coeffSizes = ParseUintCsv(coeffCsv);
  if (coeffSizes.empty()) {
    throw std::runtime_error("No coeff modulus sizes provided");
  }

  std::filesystem::create_directories(contextDir);
  const ContextBundlePaths paths = ResolveContextPaths(contextDir);
  if (std::filesystem::exists(paths.contextFile) &&
      std::filesystem::exists(paths.publicKeyFile) &&
      std::filesystem::exists(paths.secretKeyFile) &&
      std::filesystem::exists(paths.evalMultKeyFile) &&
      std::filesystem::exists(paths.evalSumKeyFile)) {
    if (!rotationIndices.empty() &&
        (!std::filesystem::exists(paths.evalAutomorphismKeyFile) ||
         std::filesystem::file_size(paths.evalAutomorphismKeyFile) == 0)) {
      PrivateKey<DCRTPoly> sk = LoadSecretKey(paths.secretKeyFile);
      CryptoContext<DCRTPoly> cc = sk->GetCryptoContext();
      cc->EvalRotateKeyGen(sk, rotationIndices);
      SaveEvalAutomorphismKeys(paths.evalAutomorphismKeyFile, cc);
      std::cout << "Generated OpenFHE rotation keys in " << contextDir << "\n";
      return 0;
    }
    std::cout << "Context and keys already exist, skipping key generation.\n";
    return 0;
  }

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

  SaveContextAndKeys(paths, cc, keyPair.publicKey, keyPair.secretKey);
  if (!rotationIndices.empty()) {
    SaveEvalAutomorphismKeys(paths.evalAutomorphismKeyFile, cc);
  }
  gEvalKeysAvailableInProcess = true;
  std::cout << "Generated OpenFHE context and keys in " << contextDir << "\n";
  return 0;
}

int RunEncryptCentroid(const CliArgs& args) {
  const std::string contextDir = GetArg(args, "--context-dir");
  const std::string inputVectorPath = GetArg(args, "--input-vector");
  const std::string outputDir = GetArg(args, "--output-dir");
  const int centroidsPerCiphertext =
      std::stoi(GetArg(args, "--centroids-per-ciphertext", "8"));
  const int paddedDim = std::stoi(GetArg(args, "--padded-dim", "1024"));
  if (contextDir.empty() || inputVectorPath.empty() || outputDir.empty()) {
    throw std::runtime_error("--context-dir, --input-vector, and --output-dir are required");
  }
  if (centroidsPerCiphertext <= 0 || paddedDim <= 0) {
    throw std::runtime_error("--centroids-per-ciphertext and --padded-dim must be positive");
  }

  std::filesystem::create_directories(outputDir);
  const ContextBundlePaths paths = ResolveContextPaths(contextDir);
  PublicKey<DCRTPoly> pk = LoadPublicKey(paths.publicKeyFile);
  CryptoContext<DCRTPoly> cc = pk->GetCryptoContext();

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

  int upload = 0;
  SaveCiphertext(outputDir + "/encrypted_query_centroid_batched.bin",
                 cc->Encrypt(pk, cc->MakeCKKSPackedPlaintext(packedQuery)));
  upload += sizeof(cc->Encrypt(pk, cc->MakeCKKSPackedPlaintext(packedQuery)));
  SaveCiphertext(outputDir + "/encrypted_norm_centroid_batched.bin",
                 cc->Encrypt(pk, cc->MakeCKKSPackedPlaintext(packedNorm)));
  upload += sizeof(cc->Encrypt(pk, cc->MakeCKKSPackedPlaintext(packedNorm)));
  std::cout << " FHE Distance upload " << upload << std::endl;

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
  return 0;
}

int RunComputeCentroid(const CliArgs& args) {
  const std::string contextDir = GetArg(args, "--context-dir");
  const std::string centroidsFile = GetArg(args, "--centroids-file");
  const std::string encryptedQueryPath = GetArg(args, "--encrypted-query");
  const std::string encryptedNormPath = GetArg(args, "--encrypted-norm");
  const std::string outputDir = GetArg(args, "--output-dir");
  const int centroidsPerCiphertext =
      std::stoi(GetArg(args, "--centroids-per-ciphertext", "8"));
  const int paddedDim = std::stoi(GetArg(args, "--padded-dim", "1024"));
  const int batchSize = std::max(1, std::stoi(GetArg(args, "--batch-size", "1")));
  const int numThreads = std::stoi(GetArg(args, "--num-threads", "0"));
  if (contextDir.empty() || centroidsFile.empty() || encryptedQueryPath.empty() ||
      encryptedNormPath.empty() || outputDir.empty()) {
    throw std::runtime_error(
        "--context-dir, --centroids-file, --encrypted-query, --encrypted-norm, and --output-dir are required");
  }

  std::filesystem::create_directories(outputDir);
  const ContextBundlePaths paths = ResolveContextPaths(contextDir);
  LoadEvalKeys(paths);
  Ciphertext<DCRTPoly> encQuery = LoadCiphertext(encryptedQueryPath);
  Ciphertext<DCRTPoly> encNorm = LoadCiphertext(encryptedNormPath);
  CryptoContext<DCRTPoly> cc = encQuery->GetCryptoContext();

  const auto centroids = ReadMatrixFile(centroidsFile);
  const int nCentroids = static_cast<int>(centroids.size());
  const int centroidDim = static_cast<int>(centroids.front().size());
  if (centroidDim <= 0 || centroidDim > paddedDim) {
    throw std::runtime_error("Centroid dimension must be > 0 and <= --padded-dim");
  }
  for (const auto& centroid : centroids) {
    if (static_cast<int>(centroid.size()) != centroidDim) {
      throw std::runtime_error("Centroid dimension mismatch in centroids file");
    }
  }
  SetThreadCount(numThreads);

  const int slotsUsed = paddedDim * centroidsPerCiphertext;
  const int nBatches = (nCentroids + centroidsPerCiphertext - 1) / centroidsPerCiphertext;
  std::cout << "Computing centroid-batched distances for " << nCentroids
            << " centroids with " << MaxThreadCount() << " thread(s).\n";

  const auto wallStart = std::chrono::high_resolution_clock::now();
  std::atomic<bool> failed(false);
  std::string failureMessage;

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
        const auto& centroid = centroids[static_cast<size_t>(centroidStart + b)];
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
      SaveCiphertext(outputDir + "/encrypted_distance_batch_" + FormatIndex(batchIdx) + ".bin",
                     distance);
    } catch (const std::exception& ex) {
      failed.store(true);
#pragma omp critical
      { failureMessage = ex.what(); }
    }
  }
  if (failed.load()) {
    throw std::runtime_error(failureMessage);
  }

  std::ofstream out(outputDir + "/distances_metadata.json");
  out << "{\n";
  out << "  \"backend\": \"openfhe_cpp\",\n";
  out << "  \"format_version\": \"openfhe_centroid_batch_distances_v1\",\n";
  out << "  \"n_centroids\": " << nCentroids << ",\n";
  out << "  \"n_distances\": " << nCentroids << ",\n";
  out << "  \"centroid_dim\": " << centroidDim << ",\n";
  out << "  \"padded_dim\": " << paddedDim << ",\n";
  out << "  \"centroids_per_ciphertext\": " << centroidsPerCiphertext << ",\n";
  out << "  \"n_batches\": " << nBatches << "\n";
  out << "}\n";

  const double elapsedSec =
      std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - wallStart).count();
  std::cout << "Computed " << nCentroids << " centroid distances in "
            << elapsedSec << " s.\n";
  return 0;
}

int RunDecryptCentroid(const CliArgs& args) {
  const std::string contextDir = GetArg(args, "--context-dir");
  const std::string encryptedDistancesDir = GetArg(args, "--encrypted-distances-dir");
  const std::string outputJson = GetArg(args, "--output-json");
  const int topK = std::stoi(GetArg(args, "--top-k", "100"));
  if (contextDir.empty() || encryptedDistancesDir.empty() || outputJson.empty()) {
    throw std::runtime_error("--context-dir, --encrypted-distances-dir, --output-json are required");
  }

  const ContextBundlePaths paths = ResolveContextPaths(contextDir);
  PrivateKey<DCRTPoly> sk = LoadSecretKey(paths.secretKeyFile);
  CryptoContext<DCRTPoly> cc = sk->GetCryptoContext();
  LoadEvalKeys(paths);

  const std::string metadataPath = encryptedDistancesDir + "/distances_metadata.json";
  const int nCentroids = ParseRequiredInt(metadataPath, "n_centroids");
  const int centroidsPerCiphertext = ParseRequiredInt(metadataPath, "centroids_per_ciphertext");
  const int nBatches = ParseRequiredInt(metadataPath, "n_batches");

  std::vector<std::pair<double, int>> scored;
  int distance_sizes = 0;
  scored.reserve(nCentroids);
  for (int batchIdx = 0; batchIdx < nBatches; ++batchIdx) {
    const std::string inFile =
        encryptedDistancesDir + "/encrypted_distance_batch_" + FormatIndex(batchIdx) + ".bin";
    if (!std::filesystem::exists(inFile)) {
      continue;
    }
    Ciphertext<DCRTPoly> ct = LoadCiphertext(inFile);
    distance_sizes += sizeof(ct);
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
  std::cout << "FHE Distance download " << distance_sizes << std::endl;
  std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
    return a.first < b.first;
  });
  const int keep = std::min(topK, static_cast<int>(scored.size()));
  std::vector<double> topDistances;
  std::vector<int> topIndices;
  for (int i = 0; i < keep; ++i) {
    topDistances.push_back(scored[static_cast<size_t>(i)].first);
    topIndices.push_back(scored[static_cast<size_t>(i)].second);
  }
  WriteJsonTopK(outputJson, topDistances, topIndices);
  std::cout << "Decrypted " << scored.size() << " packed centroid distances.\n";
  return 0;
}

int RunEncryptQueryCentroid(const CliArgs& args) {
  const std::string contextDir = GetArg(args, "--context-dir");
  const std::string inputMatrixPath = GetArg(args, "--input-matrix");
  const std::string outputDir = GetArg(args, "--output-dir");
  const int queriesPerBatch = std::stoi(GetArg(args, "--queries-per-batch", "2"));
  const int centroidsPerBatch = std::stoi(GetArg(args, "--centroids-per-batch", "4"));
  const int paddedDim = std::stoi(GetArg(args, "--padded-dim", "1024"));
  if (contextDir.empty() || inputMatrixPath.empty() || outputDir.empty()) {
    throw std::runtime_error("--context-dir, --input-matrix, and --output-dir are required");
  }

  std::filesystem::create_directories(outputDir);
  const ContextBundlePaths paths = ResolveContextPaths(contextDir);
  PublicKey<DCRTPoly> pk = LoadPublicKey(paths.publicKeyFile);
  CryptoContext<DCRTPoly> cc = pk->GetCryptoContext();

  const auto queries = ReadMatrixFile(inputMatrixPath);
  const int nQueries = static_cast<int>(queries.size());
  const int queryDim = static_cast<int>(queries.front().size());
  if (queryDim <= 0 || queryDim > paddedDim) {
    throw std::runtime_error("Query dimension must be > 0 and <= --padded-dim");
  }
  for (const auto& query : queries) {
    if (static_cast<int>(query.size()) != queryDim) {
      throw std::runtime_error("All query vectors must have identical dimensions");
    }
  }

  const int lanes = queriesPerBatch * centroidsPerBatch;
  const int slotsUsed = paddedDim * lanes;
  const int nQueryBatches = (nQueries + queriesPerBatch - 1) / queriesPerBatch;
  for (int qb = 0; qb < nQueryBatches; ++qb) {
    const int queryStart = qb * queriesPerBatch;
    const int queriesInBatch = std::min(queriesPerBatch, nQueries - queryStart);
    std::vector<double> packedQuery(static_cast<size_t>(slotsUsed), 0.0);
    std::vector<double> packedNorm(static_cast<size_t>(slotsUsed), 0.0);
    for (int q = 0; q < queriesInBatch; ++q) {
      const auto& query = queries[static_cast<size_t>(queryStart + q)];
      const double normSquared = std::inner_product(query.begin(), query.end(), query.begin(), 0.0);
      for (int c = 0; c < centroidsPerBatch; ++c) {
        packedNorm[static_cast<size_t>(q * centroidsPerBatch + c)] = normSquared;
      }
      for (int d = 0; d < queryDim; ++d) {
        for (int c = 0; c < centroidsPerBatch; ++c) {
          packedQuery[static_cast<size_t>(d * lanes + q * centroidsPerBatch + c)] =
              query[static_cast<size_t>(d)];
        }
      }
    }
    SaveCiphertext(outputDir + "/encrypted_query_qbatch_" + FormatIndex(qb) + ".bin",
                   cc->Encrypt(pk, cc->MakeCKKSPackedPlaintext(packedQuery)));
    SaveCiphertext(outputDir + "/encrypted_norm_qbatch_" + FormatIndex(qb) + ".bin",
                   cc->Encrypt(pk, cc->MakeCKKSPackedPlaintext(packedNorm)));
  }

  std::ofstream out(outputDir + "/query_centroid_batch_metadata.json");
  out << "{\n";
  out << "  \"format_version\": \"openfhe_query_centroid_batch_queries_v1\",\n";
  out << "  \"backend\": \"openfhe_cpp\",\n";
  out << "  \"n_queries\": " << nQueries << ",\n";
  out << "  \"n_query_batches\": " << nQueryBatches << ",\n";
  out << "  \"queries_per_batch\": " << queriesPerBatch << ",\n";
  out << "  \"centroids_per_batch\": " << centroidsPerBatch << ",\n";
  out << "  \"query_dim\": " << queryDim << ",\n";
  out << "  \"padded_dim\": " << paddedDim << ",\n";
  out << "  \"slots_used\": " << slotsUsed << "\n";
  out << "}\n";

  std::cout << "Encrypted " << nQueries << " queries into "
            << nQueryBatches << " query batches.\n";
  return 0;
}

int RunComputeQueryCentroid(const CliArgs& args) {
  const std::string contextDir = GetArg(args, "--context-dir");
  const std::string centroidsFile = GetArg(args, "--centroids-file");
  const std::string encryptedQueriesDir = GetArg(args, "--encrypted-queries-dir");
  const std::string outputDir = GetArg(args, "--output-dir");
  const int batchSize = std::max(1, std::stoi(GetArg(args, "--batch-size", "1")));
  const int numThreads = std::stoi(GetArg(args, "--num-threads", "0"));
  if (contextDir.empty() || centroidsFile.empty() || encryptedQueriesDir.empty() ||
      outputDir.empty()) {
    throw std::runtime_error(
        "--context-dir, --centroids-file, --encrypted-queries-dir, and --output-dir are required");
  }

  std::filesystem::create_directories(outputDir);
  const std::string queryMetadataPath =
      encryptedQueriesDir + "/query_centroid_batch_metadata.json";
  const int nQueries = ParseRequiredInt(queryMetadataPath, "n_queries");
  const int nQueryBatches = ParseRequiredInt(queryMetadataPath, "n_query_batches");
  const int queriesPerBatch = ParseRequiredInt(queryMetadataPath, "queries_per_batch");
  const int centroidsPerBatch = ParseRequiredInt(queryMetadataPath, "centroids_per_batch");
  const int queryDim = ParseRequiredInt(queryMetadataPath, "query_dim");
  const int paddedDim = ParseRequiredInt(queryMetadataPath, "padded_dim");
  const int lanes = queriesPerBatch * centroidsPerBatch;
  const int slotsUsed = paddedDim * lanes;

  const ContextBundlePaths paths = ResolveContextPaths(contextDir);
  LoadEvalKeys(paths);
  const auto centroids = ReadMatrixFile(centroidsFile);
  const int nCentroids = static_cast<int>(centroids.size());
  const int centroidDim = static_cast<int>(centroids.front().size());
  if (centroidDim != queryDim || centroidDim > paddedDim) {
    throw std::runtime_error("Centroid dimension must match query_dim and be <= padded_dim");
  }
  for (const auto& centroid : centroids) {
    if (static_cast<int>(centroid.size()) != centroidDim) {
      throw std::runtime_error("Centroid dimension mismatch in centroids file");
    }
  }
  SetThreadCount(numThreads);

  const int nCentroidBatches = (nCentroids + centroidsPerBatch - 1) / centroidsPerBatch;
  const auto wallStart = std::chrono::high_resolution_clock::now();
  for (int qb = 0; qb < nQueryBatches; ++qb) {
    Ciphertext<DCRTPoly> encQuery = LoadCiphertext(
        encryptedQueriesDir + "/encrypted_query_qbatch_" + FormatIndex(qb) + ".bin");
    Ciphertext<DCRTPoly> encNorm = LoadCiphertext(
        encryptedQueriesDir + "/encrypted_norm_qbatch_" + FormatIndex(qb) + ".bin");
    CryptoContext<DCRTPoly> cc = encQuery->GetCryptoContext();
    std::atomic<bool> failed(false);
    std::string failureMessage;

#pragma omp parallel for schedule(dynamic, batchSize)
    for (int cb = 0; cb < nCentroidBatches; ++cb) {
      if (failed.load()) {
        continue;
      }
      try {
        const int centroidStart = cb * centroidsPerBatch;
        const int centroidsInBatch = std::min(centroidsPerBatch, nCentroids - centroidStart);
        std::vector<double> centroidPacked(static_cast<size_t>(slotsUsed), 0.0);
        std::vector<double> centroidNormPacked(static_cast<size_t>(slotsUsed), 0.0);
        for (int c = 0; c < centroidsInBatch; ++c) {
          const auto& centroid = centroids[static_cast<size_t>(centroidStart + c)];
          const double centroidNorm =
              std::inner_product(centroid.begin(), centroid.end(), centroid.begin(), 0.0);
          for (int q = 0; q < queriesPerBatch; ++q) {
            centroidNormPacked[static_cast<size_t>(q * centroidsPerBatch + c)] = centroidNorm;
          }
          for (int d = 0; d < centroidDim; ++d) {
            for (int q = 0; q < queriesPerBatch; ++q) {
              centroidPacked[static_cast<size_t>(d * lanes + q * centroidsPerBatch + c)] =
                  centroid[static_cast<size_t>(d)];
            }
          }
        }
        Ciphertext<DCRTPoly> dot =
            cc->EvalMult(encQuery, cc->MakeCKKSPackedPlaintext(centroidPacked));
        for (int step = 1; step < paddedDim; step <<= 1) {
          dot = cc->EvalAdd(dot, cc->EvalAtIndex(dot, step * lanes));
        }
        Ciphertext<DCRTPoly> distance = cc->EvalSub(
            cc->EvalAdd(encNorm, cc->MakeCKKSPackedPlaintext(centroidNormPacked)),
            cc->EvalAdd(dot, dot));
        SaveCiphertext(outputDir + "/encrypted_distance_qbatch_" + FormatIndex(qb) +
                           "_cbatch_" + FormatIndex(cb) + ".bin",
                       distance);
      } catch (const std::exception& ex) {
        failed.store(true);
#pragma omp critical
        { failureMessage = ex.what(); }
      }
    }
    if (failed.load()) {
      throw std::runtime_error(failureMessage);
    }
  }

  std::ofstream out(outputDir + "/distances_metadata.json");
  out << "{\n";
  out << "  \"backend\": \"openfhe_cpp\",\n";
  out << "  \"format_version\": \"openfhe_query_centroid_batch_distances_v1\",\n";
  out << "  \"n_queries\": " << nQueries << ",\n";
  out << "  \"n_centroids\": " << nCentroids << ",\n";
  out << "  \"query_dim\": " << queryDim << ",\n";
  out << "  \"padded_dim\": " << paddedDim << ",\n";
  out << "  \"queries_per_batch\": " << queriesPerBatch << ",\n";
  out << "  \"centroids_per_batch\": " << centroidsPerBatch << ",\n";
  out << "  \"n_query_batches\": " << nQueryBatches << ",\n";
  out << "  \"n_centroid_batches\": " << nCentroidBatches << "\n";
  out << "}\n";

  const double elapsedSec =
      std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - wallStart).count();
  std::cout << "Computed " << (static_cast<long long>(nQueries) * nCentroids)
            << " query-centroid distances in " << elapsedSec << " s.\n";
  return 0;
}

int RunDecryptQueryCentroid(const CliArgs& args) {
  const std::string contextDir = GetArg(args, "--context-dir");
  const std::string encryptedDistancesDir = GetArg(args, "--encrypted-distances-dir");
  const std::string outputJson = GetArg(args, "--output-json");
  const int topK = std::stoi(GetArg(args, "--top-k", "100"));
  if (contextDir.empty() || encryptedDistancesDir.empty() || outputJson.empty()) {
    throw std::runtime_error("--context-dir, --encrypted-distances-dir, --output-json are required");
  }

  const ContextBundlePaths paths = ResolveContextPaths(contextDir);
  PrivateKey<DCRTPoly> sk = LoadSecretKey(paths.secretKeyFile);
  CryptoContext<DCRTPoly> cc = sk->GetCryptoContext();
  LoadEvalKeys(paths);

  const std::string metadataPath = encryptedDistancesDir + "/distances_metadata.json";
  const int nQueries = ParseRequiredInt(metadataPath, "n_queries");
  const int nCentroids = ParseRequiredInt(metadataPath, "n_centroids");
  const int queriesPerBatch = ParseRequiredInt(metadataPath, "queries_per_batch");
  const int centroidsPerBatch = ParseRequiredInt(metadataPath, "centroids_per_batch");
  const int nQueryBatches = ParseRequiredInt(metadataPath, "n_query_batches");
  const int nCentroidBatches = ParseRequiredInt(metadataPath, "n_centroid_batches");

  std::vector<std::vector<std::pair<double, int>>> scored(static_cast<size_t>(nQueries));
  for (auto& row : scored) {
    row.reserve(static_cast<size_t>(nCentroids));
  }
  for (int qb = 0; qb < nQueryBatches; ++qb) {
    const int queryStart = qb * queriesPerBatch;
    const int queriesInBatch = std::min(queriesPerBatch, nQueries - queryStart);
    for (int cb = 0; cb < nCentroidBatches; ++cb) {
      const std::string inFile = encryptedDistancesDir + "/encrypted_distance_qbatch_" +
                                 FormatIndex(qb) + "_cbatch_" + FormatIndex(cb) + ".bin";
      if (!std::filesystem::exists(inFile)) {
        continue;
      }
      Ciphertext<DCRTPoly> ct = LoadCiphertext(inFile);
      Plaintext pt;
      cc->Decrypt(sk, ct, &pt);
      const auto values = pt->GetRealPackedValue();
      const int centroidStart = cb * centroidsPerBatch;
      const int centroidsInBatch = std::min(centroidsPerBatch, nCentroids - centroidStart);
      for (int q = 0; q < queriesInBatch; ++q) {
        for (int c = 0; c < centroidsInBatch; ++c) {
          const int slot = q * centroidsPerBatch + c;
          if (static_cast<size_t>(slot) < values.size()) {
            scored[static_cast<size_t>(queryStart + q)].emplace_back(
                values[static_cast<size_t>(slot)], centroidStart + c);
          }
        }
      }
    }
  }

  std::vector<std::vector<std::pair<double, int>>> topResults(static_cast<size_t>(nQueries));
  for (int q = 0; q < nQueries; ++q) {
    auto& row = scored[static_cast<size_t>(q)];
    std::sort(row.begin(), row.end(), [](const auto& a, const auto& b) {
      return a.first < b.first;
    });
    const int keep = std::min(topK, static_cast<int>(row.size()));
    topResults[static_cast<size_t>(q)].assign(row.begin(), row.begin() + keep);
  }
  WriteMultiQueryTopK(outputJson, topResults);
  std::cout << "Decrypted packed distances and wrote top " << topK
            << " for " << nQueries << " queries.\n";
  return 0;
}

int RunCentroidEndToEnd(const CliArgs& args) {
  const std::string workDir = GetArg(args, "--work-dir");
  const std::string outputJson = GetArg(args, "--output-json", workDir + "/top_k_results.json");
  if (workDir.empty()) {
    throw std::runtime_error("--work-dir is required");
  }
  RunKeygen(args, "keygen-centroid");
  CliArgs encArgs = args;
  encArgs.kv.push_back({"--output-dir", workDir + "/encrypted_query"});
  RunEncryptCentroid(encArgs);
  CliArgs compArgs = args;
  compArgs.kv.push_back({"--encrypted-query", workDir + "/encrypted_query/encrypted_query_centroid_batched.bin"});
  compArgs.kv.push_back({"--encrypted-norm", workDir + "/encrypted_query/encrypted_norm_centroid_batched.bin"});
  compArgs.kv.push_back({"--output-dir", workDir + "/encrypted_distances"});
  RunComputeCentroid(compArgs);
  CliArgs decArgs = args;
  decArgs.kv.push_back({"--encrypted-distances-dir", workDir + "/encrypted_distances"});
  decArgs.kv.push_back({"--output-json", outputJson});
  return RunDecryptCentroid(decArgs);
}

int RunQueryCentroidEndToEnd(const CliArgs& args) {
  const std::string workDir = GetArg(args, "--work-dir");
  const std::string outputJson = GetArg(args, "--output-json", workDir + "/top_k_results.json");
  if (workDir.empty()) {
    throw std::runtime_error("--work-dir is required");
  }
  RunKeygen(args, "keygen-query-centroid");
  CliArgs encArgs = args;
  encArgs.kv.push_back({"--output-dir", workDir + "/encrypted_queries"});
  RunEncryptQueryCentroid(encArgs);
  CliArgs compArgs = args;
  compArgs.kv.push_back({"--encrypted-queries-dir", workDir + "/encrypted_queries"});
  compArgs.kv.push_back({"--output-dir", workDir + "/encrypted_distances"});
  RunComputeQueryCentroid(compArgs);
  CliArgs decArgs = args;
  decArgs.kv.push_back({"--encrypted-distances-dir", workDir + "/encrypted_distances"});
  decArgs.kv.push_back({"--output-json", outputJson});
  return RunDecryptQueryCentroid(decArgs);
}

void PrintUsage() {
  std::cerr
      << "Usage: openfhe_batched_workload <command> [args]\n\n"
      << "Centroid-batched single-query commands:\n"
      << "  keygen-centroid --context-dir DIR [--poly-modulus-degree 16384] [--padded-dim 1024] [--centroids-per-ciphertext 8]\n"
      << "  encrypt-centroid --context-dir DIR --input-vector query.txt --output-dir DIR [--padded-dim 1024]\n"
      << "  compute-centroid --context-dir DIR --centroids-file centroids.txt --encrypted-query FILE --encrypted-norm FILE --output-dir DIR\n"
      << "  decrypt-centroid --context-dir DIR --encrypted-distances-dir DIR --output-json FILE [--top-k 100]\n"
      << "  run-centroid --context-dir DIR --input-vector query.txt --centroids-file centroids.txt --work-dir DIR [--output-json FILE]\n\n"
      << "Query+centroid batched multi-query commands:\n"
      << "  keygen-query-centroid --context-dir DIR [--queries-per-batch 2] [--centroids-per-batch 4] [--padded-dim 1024]\n"
      << "  encrypt-query-centroid --context-dir DIR --input-matrix queries.txt --output-dir DIR\n"
      << "  compute-query-centroid --context-dir DIR --centroids-file centroids.txt --encrypted-queries-dir DIR --output-dir DIR\n"
      << "  decrypt-query-centroid --context-dir DIR --encrypted-distances-dir DIR --output-json FILE [--top-k 100]\n"
      << "  run-query-centroid --context-dir DIR --input-matrix queries.txt --centroids-file centroids.txt --work-dir DIR [--output-json FILE]\n\n"
      << "Input vectors/matrices are whitespace-delimited text. One matrix row is one vector.\n";
}

}  // namespace

/**
int main(int argc, char** argv) {
  try {
    const CliArgs args = ParseArgs(argc, argv);
    if (args.positional.empty() || args.positional.front() == "help" ||
        args.positional.front() == "--help") {
      PrintUsage();
      return args.positional.empty() ? 1 : 0;
    }
    const std::string command = args.positional.front();
    if (command == "keygen" || command == "keygen-centroid" ||
        command == "keygen-query-centroid") {
      return RunKeygen(args, command);
    }
    if (command == "encrypt-centroid") {
      return RunEncryptCentroid(args);
    }
    if (command == "compute-centroid") {
      return RunComputeCentroid(args);
    }
    if (command == "decrypt-centroid") {
      return RunDecryptCentroid(args);
    }
    if (command == "run-centroid") {
      return RunCentroidEndToEnd(args);
    }
    if (command == "encrypt-query-centroid") {
      return RunEncryptQueryCentroid(args);
    }
    if (command == "compute-query-centroid") {
      return RunComputeQueryCentroid(args);
    }
    if (command == "decrypt-query-centroid") {
      return RunDecryptQueryCentroid(args);
    }
    if (command == "run-query-centroid") {
      return RunQueryCentroidEndToEnd(args);
    }
    throw std::runtime_error("Unknown command: " + command);
  } catch (const std::exception& ex) {
    std::cerr << "openfhe_batched_workload error: " << ex.what() << "\n";
    return 1;
  }
}**/
