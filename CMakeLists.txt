cmake_minimum_required(VERSION 3.20)
project(qwfnfer LANGUAGES C CXX)

# qwfnfer -- a purpose-built inference engine for Qwen3.8-Flash-Next (GGUF arch "qwen4exp").
#
# We own the model graph and the memory hierarchy; ggml supplies the quantized
# math kernels. The entire performance thesis of this engine lives in
# qwfn_io / qwfn_expert_cache / qwfn_ple: fetch expert slices at their natural
# 627KB-921KB granularity over io_uring instead of 4KiB demand paging.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Where llama.cpp/ggml lives. Override with -DLLAMA_CPP_ROOT=...
# On Windows the default is %USERPROFILE%\.unsloth\llama.cpp (LLAMA_CPP_ROOT
# still overrides it); the build dir defaults to <root>/build/bin, where
# llama.cpp's CMake puts ggml-base.dll / ggml.dll / llama.dll.
if(WIN32)
  if(DEFINED ENV{USERPROFILE} AND "$ENV{HOME}" STREQUAL "")
    set(_QWFN_HOME_DEFAULT "$ENV{USERPROFILE}")
  else()
    set(_QWFN_HOME_DEFAULT "$ENV{HOME}")
  endif()
  set(LLAMA_CPP_ROOT "${_QWFN_HOME_DEFAULT}/.unsloth/llama.cpp" CACHE PATH "llama.cpp source root")
else()
  set(LLAMA_CPP_ROOT "$ENV{HOME}/.unsloth/llama.cpp" CACHE PATH "llama.cpp source root")
endif()
set(LLAMA_CPP_BUILD "${LLAMA_CPP_ROOT}/build/bin"   CACHE PATH "llama.cpp built libraries")
# QWFN_PORTABLE builds the release bundle (scripts/package.sh): baseline x86-64-v3 code
# instead of this machine's -march=native, a glibc 2.34 floor (cmake/glibc_compat.h keeps
# the pre-C23 strtol/sscanf symbols, -fno-math-errno the pre-2.43 sqrtf), and the ggml/CUDA
# libraries found next to the binaries ($ORIGIN). Every target the bundle ships takes
# QWFN_ARCH_FLAGS, the -O2 ones included: without it they are built for whatever -march the
# build machine's compiler happens to default to, which is nobody's guarantee of x86-64-v3.
#
# -static-libstdc++ -static-libgcc used to be here as well, to keep the bundle off the user's
# C++ runtime. They also copy the build machine's runtime into the binaries, and the linker
# ORs the ISA property of every object it links into the output: on a distribution that
# compiles libstdc++.a and libgcc.a for AVX-512 that leaves the binaries marked "x86-64-v4
# needed" whatever -march the engine itself was built with, after which glibc's loader
# refuses to start them on any CPU without AVX-512. It bought little in return -- libllama
# and the ggml libraries link libstdc++ dynamically, so the bundle needs the user's copy
# either way -- and the C++ runtime is one thing every distribution has.
option(QWFN_PORTABLE "Relocatable release build" OFF)
if(WIN32)
  # MSVC has no -march; /arch:AVX2 is the portable floor (matches x86-64-v3:
  # AVX, AVX2, BMI1/2, FMA, LZCNT, MOVBE). Non-portable builds take no arch
  # flag and get whatever the host compiler defaults to.
  if(QWFN_PORTABLE)
    if(MSVC)
      set(QWFN_ARCH_FLAGS /arch:AVX2)
    else()
      set(QWFN_ARCH_FLAGS -march=x86-64-v3 -mtune=generic)
    endif()
  else()
    set(QWFN_ARCH_FLAGS "")
  endif()
  # No glibc compat header, no RPATH on Windows: DLLs are found next to the
  # binaries (package.ps1) or via PATH.
elseif(QWFN_PORTABLE)
  set(QWFN_ARCH_FLAGS -march=x86-64-v3 -mtune=generic)
  add_compile_options(-include "${CMAKE_SOURCE_DIR}/cmake/glibc_compat.h" -fno-math-errno)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
  set(CMAKE_INSTALL_RPATH "$ORIGIN")
else()
  set(QWFN_ARCH_FLAGS -march=native)
endif()

if(NOT EXISTS "${LLAMA_CPP_ROOT}/ggml/include/ggml.h")
  message(FATAL_ERROR "ggml headers not found under ${LLAMA_CPP_ROOT}. Pass -DLLAMA_CPP_ROOT=<path>.")
endif()

if(WIN32)
  # llama.cpp builds shared libs as ggml-base.dll + ggml-base.lib (import lib)
  # on Windows. IMPORTED_LOCATION takes the DLL (needed at runtime),
  # IMPORTED_IMPLIB the .lib (needed at link time). Distinct variable names
  # per target: find_* caches by variable name, so reusing one name across
  # targets would freeze the first result in the cache.
  find_file(GGML_BASE_DLL NAMES ggml-base.dll PATHS "${LLAMA_CPP_BUILD}" NO_DEFAULT_PATH)
  find_library(GGML_BASE_LIB NAMES ggml-base PATHS "${LLAMA_CPP_BUILD}" NO_DEFAULT_PATH)
  find_file(GGML_DLL NAMES ggml.dll PATHS "${LLAMA_CPP_BUILD}" NO_DEFAULT_PATH)
  find_library(GGML_LIB NAMES ggml PATHS "${LLAMA_CPP_BUILD}" NO_DEFAULT_PATH)
  find_file(LLAMA_DLL NAMES llama.dll PATHS "${LLAMA_CPP_BUILD}" NO_DEFAULT_PATH)
  find_library(LLAMA_LIB NAMES llama PATHS "${LLAMA_CPP_BUILD}" NO_DEFAULT_PATH)
  if(NOT GGML_BASE_DLL AND NOT GGML_BASE_LIB)
    message(FATAL_ERROR "ggml-base not found under ${LLAMA_CPP_BUILD} (looked for ggml-base.dll/.lib). Build llama.cpp first: cmake -S <llama.cpp> -B <llama.cpp>/build -DGGML_CUDA=ON, then build.")
  endif()
  if(NOT GGML_DLL AND NOT GGML_LIB)
    message(FATAL_ERROR "ggml not found under ${LLAMA_CPP_BUILD} (looked for ggml.dll/.lib). Build llama.cpp first.")
  endif()
  if(NOT LLAMA_DLL AND NOT LLAMA_LIB)
    message(FATAL_ERROR "llama not found under ${LLAMA_CPP_BUILD} (looked for llama.dll/.lib). Build llama.cpp first.")
  endif()
  add_library(ggml_base SHARED IMPORTED)
  if(GGML_BASE_LIB)
    set_target_properties(ggml_base PROPERTIES IMPORTED_IMPLIB "${GGML_BASE_LIB}")
    if(GGML_BASE_DLL)
      set_target_properties(ggml_base PROPERTIES IMPORTED_LOCATION "${GGML_BASE_DLL}")
    else()
      set_target_properties(ggml_base PROPERTIES IMPORTED_LOCATION "${GGML_BASE_LIB}")
    endif()
  else()
    set_target_properties(ggml_base PROPERTIES IMPORTED_LOCATION "${GGML_BASE_DLL}")
  endif()
  add_library(ggml_all SHARED IMPORTED)
  if(GGML_LIB)
    set_target_properties(ggml_all PROPERTIES IMPORTED_IMPLIB "${GGML_LIB}")
    if(GGML_DLL)
      set_target_properties(ggml_all PROPERTIES IMPORTED_LOCATION "${GGML_DLL}")
    else()
      set_target_properties(ggml_all PROPERTIES IMPORTED_LOCATION "${GGML_LIB}")
    endif()
  else()
    set_target_properties(ggml_all PROPERTIES IMPORTED_LOCATION "${GGML_DLL}")
  endif()
  # Two things need libllama: qwfn-refdump drives llama.cpp's own graph to
  # produce the reference intermediates our graph is bisected against, and
  # qwfn-chat borrows its tokenizer (opened vocab_only, so no weights load).
  add_library(llama_ref SHARED IMPORTED)
  if(LLAMA_LIB)
    set_target_properties(llama_ref PROPERTIES IMPORTED_IMPLIB "${LLAMA_LIB}")
    if(LLAMA_DLL)
      set_target_properties(llama_ref PROPERTIES IMPORTED_LOCATION "${LLAMA_DLL}")
    else()
      set_target_properties(llama_ref PROPERTIES IMPORTED_LOCATION "${LLAMA_LIB}")
    endif()
  else()
    set_target_properties(llama_ref PROPERTIES IMPORTED_LOCATION "${LLAMA_DLL}")
  endif()
else()
  add_library(ggml_base SHARED IMPORTED)
  set_target_properties(ggml_base PROPERTIES IMPORTED_LOCATION "${LLAMA_CPP_BUILD}/libggml-base.so.0")
  add_library(ggml_all SHARED IMPORTED)
  set_target_properties(ggml_all PROPERTIES IMPORTED_LOCATION "${LLAMA_CPP_BUILD}/libggml.so.0")

  # Two things need libllama: qwfn-refdump drives llama.cpp's own graph to produce
  # the reference intermediates our graph is bisected against, and qwfn-chat
  # borrows its tokenizer (opened vocab_only, so no weights are loaded).
  add_library(llama_ref SHARED IMPORTED)
  set_target_properties(llama_ref PROPERTIES IMPORTED_LOCATION "${LLAMA_CPP_BUILD}/libllama.so.0")
endif()

find_package(Threads REQUIRED)

# liburing is the Linux fast path; on Windows the threads backend (ReadFile +
# OVERLAPPED) is the only one and no extra library is needed.
if(WIN32)
  set(URING_INCLUDE_DIR "")
  set(URING_LIBRARY "")
else()
  # liburing is mandatory: it is the whole point of the engine.
  find_path(URING_INCLUDE_DIR liburing.h)
  find_library(URING_LIBRARY NAMES uring liburing)
  if(NOT URING_INCLUDE_DIR OR NOT URING_LIBRARY)
    message(FATAL_ERROR "liburing not found. Install liburing (Arch: pacman -S liburing).")
  endif()
  message(STATUS "liburing: ${URING_LIBRARY}")
endif()

add_library(qwfn_core STATIC
  src/qwfn_model.cpp
  src/qwfn_io.cpp
  src/qwfn_expert_cache.cpp
  src/qwfn_ple.cpp
  src/qwfn_weights.cpp
  src/qwfn_graph.cpp
  src/qwfn_state.cpp
  src/qwfn_engine.cpp
  src/qwfn_prefill.cpp
  src/qwfn_vision.cpp
)
target_include_directories(qwfn_core PUBLIC
  "${LLAMA_CPP_ROOT}/vendor/stb"
  src
  "${LLAMA_CPP_ROOT}/ggml/include"
  "${LLAMA_CPP_ROOT}/ggml/src"       # ggml-impl.h: the graph struct, for the per-node profiler only
  "${URING_INCLUDE_DIR}"
)
if(WIN32)
  target_link_libraries(qwfn_core PUBLIC ggml_base ggml_all Threads::Threads ws2_32)
  if(MSVC)
    target_compile_options(qwfn_core PRIVATE /O2 /W2 ${QWFN_ARCH_FLAGS})
    target_compile_definitions(qwfn_core PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
  else()
    target_compile_options(qwfn_core PRIVATE -O3 ${QWFN_ARCH_FLAGS} -Wall -Wextra -Wno-unused-parameter)
  endif()
else()
  target_link_libraries(qwfn_core PUBLIC ggml_base ggml_all Threads::Threads "${URING_LIBRARY}")
  target_compile_options(qwfn_core PRIVATE -O3 ${QWFN_ARCH_FLAGS} -Wall -Wextra -Wno-unused-parameter)
endif()

# Per-target optimization flags: MSVC uses /O2, GCC/Clang -O2/-O3.
if(MSVC)
  set(QWFN_OPT_O3 /O2)
  set(QWFN_OPT_O2 /O2)
  set(QWFN_WARN "")
else()
  set(QWFN_OPT_O3 -O3)
  set(QWFN_OPT_O2 -O2)
  set(QWFN_WARN -Wall -Wextra -Wno-unused-parameter)
endif()

add_executable(qwfn-iobench tools/qwfn_iobench.cpp)
target_link_libraries(qwfn-iobench PRIVATE qwfn_core)
target_compile_options(qwfn-iobench PRIVATE ${QWFN_OPT_O3} ${QWFN_ARCH_FLAGS} ${QWFN_WARN})

add_executable(qwfn-inspect tools/qwfn_inspect.cpp)
target_link_libraries(qwfn-inspect PRIVATE qwfn_core)
target_compile_options(qwfn-inspect PRIVATE ${QWFN_OPT_O3} ${QWFN_ARCH_FLAGS} ${QWFN_WARN})

add_executable(qwfn-test-hc tools/qwfn_test_hc.cpp)
target_link_libraries(qwfn-test-hc PRIVATE qwfn_core)
target_compile_options(qwfn-test-hc PRIVATE ${QWFN_OPT_O3} ${QWFN_ARCH_FLAGS} ${QWFN_WARN})

add_executable(qwfn-logits tools/qwfn_logits.cpp)
target_link_libraries(qwfn-logits PRIVATE qwfn_core)
target_compile_options(qwfn-logits PRIVATE ${QWFN_OPT_O3} ${QWFN_ARCH_FLAGS} ${QWFN_WARN})

add_executable(qwfn-refdump tools/qwfn_refdump.cpp)
target_include_directories(qwfn-refdump PRIVATE "${LLAMA_CPP_ROOT}/include" "${LLAMA_CPP_ROOT}/ggml/include")
target_link_libraries(qwfn-refdump PRIVATE llama_ref ggml_base ggml_all)
target_compile_options(qwfn-refdump PRIVATE ${QWFN_OPT_O2})

add_executable(qwfn-run tools/qwfn_run.cpp)
target_link_libraries(qwfn-run PRIVATE qwfn_core)
target_compile_options(qwfn-run PRIVATE ${QWFN_OPT_O3} ${QWFN_ARCH_FLAGS} ${QWFN_WARN})

# qwfn_vocab.cpp is deliberately NOT in qwfn_core: only this binary should
# carry a libllama dependency.
add_executable(qwfn-vtest tools/qwfn_vtest.cpp)
target_link_libraries(qwfn-vtest PRIVATE qwfn_core)
target_compile_options(qwfn-vtest PRIVATE ${QWFN_OPT_O3} ${QWFN_ARCH_FLAGS} ${QWFN_WARN})

# cpp-httplib 0.54 is split header/source, so its .cpp joins the target.
add_executable(qwfn-server tools/qwfn_server.cpp src/qwfn_vocab.cpp
               "${LLAMA_CPP_ROOT}/vendor/cpp-httplib/httplib.cpp")
target_include_directories(qwfn-server PRIVATE
  "${LLAMA_CPP_ROOT}/include"
  "${LLAMA_CPP_ROOT}/vendor"
  "${LLAMA_CPP_ROOT}/vendor/cpp-httplib")
target_link_libraries(qwfn-server PRIVATE qwfn_core llama_ref)
if(WIN32)
  target_link_libraries(qwfn-server PRIVATE ws2_32 crypt32)
endif()
target_compile_options(qwfn-server PRIVATE ${QWFN_OPT_O2} ${QWFN_ARCH_FLAGS} ${QWFN_WARN})

add_executable(qwfn-chat tools/qwfn_chat.cpp src/qwfn_vocab.cpp)
target_include_directories(qwfn-chat PRIVATE "${LLAMA_CPP_ROOT}/include")
target_link_libraries(qwfn-chat PRIVATE qwfn_core llama_ref)
target_compile_options(qwfn-chat PRIVATE ${QWFN_OPT_O3} ${QWFN_ARCH_FLAGS} ${QWFN_WARN})

add_executable(qwfn-tok tools/qwfn_tok.cpp src/qwfn_vocab.cpp)
target_include_directories(qwfn-tok PRIVATE "${LLAMA_CPP_ROOT}/include")
target_link_libraries(qwfn-tok PRIVATE qwfn_core llama_ref)
target_compile_options(qwfn-tok PRIVATE ${QWFN_OPT_O2} ${QWFN_ARCH_FLAGS} ${QWFN_WARN})

add_executable(qwfn-gen tools/qwfn_gen.cpp)
target_link_libraries(qwfn-gen PRIVATE qwfn_core)
target_compile_options(qwfn-gen PRIVATE ${QWFN_OPT_O3} ${QWFN_ARCH_FLAGS} ${QWFN_WARN})

add_executable(qwfn-train-predictor tools/qwfn_train_predictor.cpp)
target_link_libraries(qwfn-train-predictor PRIVATE qwfn_core)
target_compile_options(qwfn-train-predictor PRIVATE ${QWFN_OPT_O3} ${QWFN_ARCH_FLAGS} ${QWFN_WARN})
target_compile_definitions(qwfn-train-predictor PRIVATE LLAMA_CPP_BUILD_DIR="${LLAMA_CPP_BUILD}")
