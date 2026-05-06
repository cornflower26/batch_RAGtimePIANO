cmake_minimum_required(VERSION 3.16)
if(APPLE)
    #set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "" FORCE)
    set(CMAKE_OSX_ARCHITECTURES "x86_64" CACHE STRING "" FORCE)
endif()
project(batch_pir LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
add_compile_options(-O2 -Wall -Wextra)

# ---- OpenSSL ----------------------------------------------------------------
if(APPLE)
    #set(OPENSSL_ROOT_DIR /opt/homebrew/opt/openssl@3)
    #set(OpenMP_ROOT /opt/homebrew/opt/libomp)

    set(OPENSSL_ROOT_DIR /usr/local/Cellar/openssl@3/3.6.2.reinstall/)
    set(OpenMP_ROOT_DIR /usr/local/Cellar/libomp/21.1.8/)

    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/Cellar/libomp/21.1.8/include")
    #set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/usr/local/Cellar/libomp/21.1.8/lib/libomp.dylib")

    #link_directories("/opt/homebrew/opt/libomp/lib")
    link_directories("/usr/local/Cellar/openssl@3/3.6.2.reinstall/lib")
    link_directories("/usr/local/Cellar/libomp/21.1.8/lib")
    link_directories("/usr/local/Cellar/faiss/1.14.1/lib")

    set(CMAKE_CXX_VISIBILITY_PRESET default)
endif()
find_package(OpenSSL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(faiss CONFIG REQUIRED)
message(STATUS "OpenSSL ${OPENSSL_VERSION} at ${OPENSSL_INCLUDE_DIR}")
message(STATUS "FAISS ${faiss_VERSION} at ${faiss_INCLUDE_DIR}")

# ---- Library ----------------------------------------------------------------
add_library(pianopir STATIC
        openfhe_batched_workload.cpp
        piano_pir.cpp
        util.cpp
        batch_piano_pir.cpp
)
target_link_libraries(pianopir PUBLIC faiss)

if(OpenMP_CXX_FOUND)
        target_link_libraries(pianopir PUBLIC OpenMP::OpenMP_CXX)
    if(APPLE)
        target_link_directories(pianopir PUBLIC /usr/local/Cellar/libomp/21.1.8/lib)
    endif()
endif()
target_include_directories(pianopir PUBLIC /usr/local/include)

target_include_directories(pianopir PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OPENSSL_INCLUDE_DIR}
)
target_link_libraries(pianopir PUBLIC OpenSSL::SSL OpenSSL::Crypto)

# ---- Executable -------------------------------------------------------------
add_executable(batch_pir regular_test.cpp)
add_executable(main main.cpp)
add_executable(regular_batch_main regular_batched.cpp)
add_executable(partition_batch_main partition_batched.cpp)
target_link_libraries(main PRIVATE pianopir)
target_link_libraries(regular_batch_main PRIVATE pianopir)
target_link_libraries(batch_pir PRIVATE pianopir)
target_link_libraries(partition_batch_main PRIVATE pianopir)

add_library(openfhecorelib SHARED IMPORTED)
add_library(openfhepkelib SHARED IMPORTED)

set_target_properties(openfhepkelib PROPERTIES IMPORTED_LOCATION "/usr/local/lib/libOPENFHEpke.so")
set_target_properties(openfhecorelib PROPERTIES IMPORTED_LOCATION "/usr/local/lib/libOPENFHEcore.so")

include_directories("/usr/local/include/openfhe")
include_directories("/usr/local/include/openfhe/core")
include_directories("/usr/local/include/openfhe/pke")
include_directories("/usr/local/include/openfhe/binfhe")

link_directories("/usr/local/lib")
link_directories("/usr/include")

#target_link_libraries(SLAP omp)
target_link_libraries(pianopir PRIVATE OPENFHEcore)
target_link_libraries(pianopir PRIVATE OPENFHEpke)
target_link_libraries(pianopir PRIVATE OPENFHEbinfhe)

set(CMAKE_PREFIX_PATH "/usr/local/lib/OpenFHE")
find_package("OpenFHE" REQUIRED)