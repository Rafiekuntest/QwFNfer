# Building qwfnfer on Windows

Linux is the reference platform. This is the Windows port: same engine, same
console, same tiers — the NVMe layer that was `io_uring` is a thread pool
reading with positional `ReadFile` + `OVERLAPPED` (handles opened with
`FILE_FLAG_NO_BUFFERING` when direct I/O is on), `mmap` is `MapViewOfFile`,
`/proc/meminfo` is `GlobalMemoryStatusEx`, and `liburing` is not needed.

## Requirements

- 64-bit Windows 10/11, NVIDIA GPU, driver 580 or newer (CUDA 13 runtime)
- Visual Studio 2022 (MSVC + Windows SDK) **or** MinGW-w64; CMake 3.20+; Ninja
- CUDA toolkit 13 (`nvcc` on PATH)
- Python 3.10+ (console + launcher)
- A built [llama.cpp](https://github.com/unslothai/llama.cpp) tree at
  `%USERPROFILE%\.unsloth\llama.cpp`, branch `b10798-mix-659e406` — the mix
  the forward pass is validated against:

```powershell
git clone https://github.com/unslothai/llama.cpp $env:USERPROFILE\.unsloth\llama.cpp
cd $env:USERPROFILE\.unsloth\llama.cpp; git checkout b10798-mix-659e406
cmake -S . -B build -DGGML_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=75;80;86;89;90;120" `
  -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF `
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF `
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build build --config Release
```

## Build the engine

```powershell
cmake -S . -B build -DGGML_CUDA=ON  # -DLLAMA_CPP_ROOT=<path> if the tree is elsewhere
cmake --build build --config Release
```

Binaries land in `build\Release\` (Ninja: `build\`): `qwfn-server.exe`,
`qwfn-tok.exe`, plus the dev tools (`qwfn-chat.exe`, `qwfn-iobench.exe`, …).

## Run it

```powershell
hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "UD-Q4_K_XL/*" "mmproj-F16.gguf"
scripts\qwfnfer.bat
```

It opens `http://127.0.0.1:8090`. Pick a tier and press **Auto-tune & start** —
the console measures the drive (buffered random 2 MiB reads on Windows;
`FILE_FLAG_NO_BUFFERING` in the engine itself), sweeps the CPU thread count,
sizes the RAM tier with `GlobalMemoryStatusEx`, and verifies on a short chat
plus a 16K–32K-token document. Endpoint: `http://127.0.0.1:8080/v1`.

`scripts\qwfnfer.bat --no-browser` runs the console in the terminal;
`--port 8091` picks another console port.

## Install / bundle

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install.ps1        # from a release zip
powershell -ExecutionPolicy Bypass -File scripts\package.ps1        # build dist\qwfnfer-windows-x86_64-cuda.zip
```

`package.ps1` writes the zip that `install.ps1` consumes (`$env:QWFN_ZIP`
for an offline path). Uninstall: delete `%LOCALAPPDATA%\qwfnfer` and
`%USERPROFILE%\.local\bin\qwfnfer.bat`.

## Notes and limits

- `--io-uring` is accepted and silently mapped to the threads backend.
- `FILE_FLAG_NO_BUFFERING` needs sector-aligned offset/length/buffer; the
  engine's `dio_align()`/`dio_alloc()` layout already guarantees it, and falls
  back to buffered reads per file when the volume refuses it.
- The console stores state in `%LOCALAPPDATA%\qwfn-console` (`config.json`,
  `server.log`); the Hugging Face cache resolves via `HF_HUB_CACHE` /
  `HF_HOME` / `%USERPROFILE%\.cache\huggingface\hub`.
- One engine at a time (same as Linux); `taskkill /F /IM qwfn-server.exe`
  stops an orphan.
- The MX230-class GPUs (2 GB) cannot hold the ~5 GB dense core: the engine
  will refuse to size a VRAM tier there. A 16 GB GPU is the reference target.
