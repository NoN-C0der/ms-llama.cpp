# Same size as unsloth. Half the quality loss.

**MS (Mad Shift)** quants have **1.61-2.31x lower quality loss** than unsloth's files of about the same size: **1.61-1.63x**
at the low tiers, **1.90-2.31x** from MS3.7 up. Quality loss = KL divergence from the original BF16 model (Qwen3.8-27B).
**ms-llama.cpp** is the llama.cpp fork that runs them.

| MS file | Size | KL divergence | unsloth UD3 file | its size | its KL divergence | MS is |
|---|---|---|---|---|---|---|
| MS5 | 19.41 GB | 0.00153 | UD-Q5_K_M | 19.42 GB | 0.00354 | **2.31x** closer |
| MS4.3 | 17.30 GB | 0.00298 | UD-Q4_K_XL | 17.21 GB | 0.00646 | **2.16x** closer |
| MS4 | 16.11 GB | 0.00442 | UD-Q4_K_M | 16.11 GB | 0.00899 | **2.04x** closer |
| MS3.7 | 15.00 GB | 0.00623 | UD-Q4_K_S | 15.01 GB | 0.01182 | **1.90x** closer |
| MS3.5 | 13.89 GB | 0.01000 | UD-IQ4_XS | 13.90 GB | 0.01614 | **1.61x** closer |
| MS3 | 12.79 GB | 0.01570 | UD-Q3_K_XL | 12.80 GB | 0.02554 | **1.63x** closer |
| MS2.7 | 10.58 GB | 0.03406 | UD-IQ3_XXS | 10.58 GB | 0.05502 | **1.62x** closer |

Compared against Unsloth Dynamic 3 (UD3): unsloth's files at revision 4ca72078 (August 2026). Mean KL divergence over
English wiki, web, code and math text the files were not made from; lower is better. unsloth's sizes are without the MTP
layer built into their files. Every MS file is no larger than its unsloth file except MS4.3, which is 0.09 GB larger.
unsloth updated four of these files on 2026-11-12; against the new versions, MS3 is 1.60x and MS4 1.84x closer. Method
and per-text numbers: [model card](https://huggingface.co/devvexus/Qwen3.8-27B-MS).

## Run it now

Windows with an NVIDIA GPU: paste this one line into PowerShell.

```powershell
$d="$env:LOCALAPPDATA\ms-llama"; if(!(Test-Path "$d\llama-server.exe")){$z=if($env:PROCESSOR_ARCHITECTURE -eq "ARM64"){"arm64-cuda"}else{"x64-cuda12"}; ni -ItemType Directory -Force $d|Out-Null; curl.exe -L -o "$d\ms.zip" "https://github.com/devvexus/ms-llama.cpp/releases/latest/download/ms-llama-win-$z.zip"; tar -xf "$d\ms.zip" -C $d}; & "$d\llama-server.exe" -hf devvexus/Qwen3.8-27B-MS:MS4.3 --spec-type draft-mtp --port 8080
```

What it does:

- Installs the prebuilt engine (x64 or ARM64, picked for you; CUDA and C++ runtimes included) into
  `%LOCALAPPDATA%\ms-llama`. It never touches an existing llama.cpp. The next run skips this step.
- Downloads Qwen3.8-27B-MS4.3 and the MTP file from Hugging Face (into the Hugging Face cache, once).
- Serves a chat page and an API on **http://localhost:8080**. Stop it with Ctrl+C; paste the same line to start it again.

Pick the file for your GPU by changing only the tag after the colon:

| GPU memory | Tag |
|---|---|
| 32 GB or more | `MS5` |
| 24 GB | `MS4.3` |
| 16 GB | `MS2.7` |
| 12 GB | `MS2.7` with a short context (the server shortens it to fit by itself, or add `-c 4096`) |

The other tiers (MS3, MS3.5, MS3.7, MS4) go by file name instead of a tag, e.g. replace
`-hf devvexus/Qwen3.8-27B-MS:MS4.3` with `-hf devvexus/Qwen3.8-27B-MS --hf-file MS4/Qwen3.8-27B-MS4.gguf`.

**Linux and macOS:** build from source like llama.cpp, then run the same server:

```bash
git clone https://github.com/devvexus/ms-llama.cpp
cd ms-llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release   # CPU only (and macOS): drop -DGGML_CUDA=ON
cmake --build build --config Release -j
./build/bin/llama-server -hf devvexus/Qwen3.8-27B-MS:MS4.3 --spec-type draft-mtp --port 8080
```

## Use it in your apps

With the server running:

- **Claude Code:** set `ANTHROPIC_BASE_URL=http://localhost:8080` and `ANTHROPIC_AUTH_TOKEN` to any value, then start
  `claude`. The server speaks Anthropic's Messages API (`/v1/messages`).
  PowerShell: `$env:ANTHROPIC_BASE_URL="http://localhost:8080"; $env:ANTHROPIC_AUTH_TOKEN="local"; claude`
- **OpenAI-compatible apps:** base URL `http://localhost:8080/v1`, any API key.
- **Apps that run llama.cpp themselves** (they ask for a llama.cpp folder or a `llama-server` path): point them at
  `%LOCALAPPDATA%\ms-llama`.
- **Ollama:** Ollama can't load MS files itself (nor can LM Studio or stock llama.cpp). If your app can use an
  OpenAI-compatible connection instead of Ollama, give it `http://localhost:8080/v1`.

## Models

- [devvexus/Qwen3.8-27B-MS](https://huggingface.co/devvexus/Qwen3.8-27B-MS): 7 tiers, MS2.7 (10.6 GB) to MS5 (19.4 GB),
  plus one MTP file, `mtp-Qwen3.8-27B-MS.gguf` (0.30 GB, in the top folder).
- [devvexus/Qwen3.5-2B-MS](https://huggingface.co/devvexus/Qwen3.5-2B-MS).

These repositories are private until launch.

## MTP: one small file, twice the speed

One 0.30 GB MTP file works with every tier: it holds only the MTP block and borrows the loaded tier's token table and
output head. The one-line command turns it on (`--spec-type draft-mtp`). The model checks every drafted token.

RTX Spark (GB10), 20 chat prompts x 256 tokens, greedy, tokens per second:

| Model | without MTP | with MTP | speedup | extra GPU memory for MTP |
|---|---|---|---|---|
| **MS4.3** + our 0.30 GB MTP file | 14.4 | **31.6** | **2.20x** | ~0.9 GB |
| **MS2.7** + our 0.30 GB MTP file | 20.8 | **43.7** | **2.10x** | ~0.9 GB |
| unsloth UD-Q4_K_XL + their 1.37 GB MTP file | 14.7 | 32.1 | 2.18x | ~1.7 GB |
| unsloth UD-IQ3_XXS + their 1.37 GB MTP file | 21.4 | 38.9 | 1.82x | ~1.7 GB |

With MTP on, MS4.3 runs at 98% of the speed of unsloth's file of the same size with its MTP file, and MS2.7 at 112%.

## Speed

- **Generation** is at parity with unsloth's files of the same size (without MTP in the benchmark above: MS4.3 14.4 vs
  14.7 tokens/s, MS2.7 20.8 vs 21.4).
- **Prompt processing** on the RTX Spark (GB10), relative to unsloth's file of the same size: MS2.7 1.04x, MS3 1.00x,
  MS4 0.94x, MS4.3 0.92x, a kernel limit on that machine.

## What is MS

MS (Mad Shift) is a family of GGUF weight types (`MS_C1` ... `MS_C6`, 1 to 6 bits per weight) with its own CUDA and CPU
(x86, ARM) kernels in this fork. The files are made with QTIP's bitshift trellis + our own calibration-aware,
layer-by-layer conversion. The trellis code builds on **QTIP** (Tseng, Sun, Hou and De Sa, NeurIPS 2024,
[arXiv:2406.11235](https://arxiv.org/abs/2406.11235)). To our knowledge, MS (Mad Shift) is the first GGUF format built on
QTIP's bitshift trellis code. This repository runs MS files; format details are in
[docs/ms.md](docs/ms.md).

## Hardware

- **NVIDIA GPUs (CUDA):** supported. The Windows x64 build needs an RTX 20 series or newer card; the ARM64 build is for
  Windows on ARM with an NVIDIA GPU (RTX Spark, DGX Spark).
- **CPU (x86, ARM):** works, slower.
- **Apple Metal** and other GPU backends: not yet.

## Credit and citation

MS (Mad Shift) was created by **Matthew Priester** ([GitHub](https://github.com/devvexus),
[ORCID 0009-0000-0032-5292](https://orcid.org/0009-0000-0032-5292)). Please credit *MS (Mad Shift) by Matthew Priester*.

- How to cite: [CITATION.cff](CITATION.cff) (GitHub's "Cite this repository"). Please cite QTIP too.
- [NOTICE](NOTICE): the credit notice; keep it when you copy or adapt the MS code.
- [LICENSE](LICENSE): MIT, like llama.cpp: "Copyright (c) 2023-2026 The ggml authors" and "Copyright (c) 2026 Matthew
  Priester (MS (Mad Shift) additions)".

This project is not affiliated with or endorsed by ggml-org/llama.cpp, Qwen/Alibaba, unsloth or NVIDIA. Other product
names (Claude Code, Ollama, LM Studio and others) are trademarks of their owners, named only to describe compatibility.

We plan to propose MS for upstream llama.cpp.

---

*The original [llama.cpp](https://github.com/ggml-org/llama.cpp) README follows. Its install commands install stock
llama.cpp, which cannot load MS files; build this repository instead (see Run it now).*

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Asanmai%20OR%20author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Athelittlefireman%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3AmasterFoad%20OR%20author%3Aallozaur%20OR%20author%3AAndreas-Krebbel%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Achaxu01%20OR%20author%3Alhez%20OR%20author%3AAlekseiNikiforovIBM%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming%20OR%20author%3Aynankani)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
