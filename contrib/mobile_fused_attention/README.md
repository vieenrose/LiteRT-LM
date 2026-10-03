# contrib/mobile_fused_attention: Gemma-4 mobile under 3 GB on a phone CPU

Google's Gemma-4 **mobile** graphs (`litert-community/gemma-4-E2B-it-litert-lm`, `-E4B-`) need far more
memory on CPU than their weights:

| Reno7 (Dimensity 900), CPU, 4k context | E2B | E4B |
|---|---|---|
| LiteRT-LM 0.17.1, peak RSS | 2.28 GB | 4.58 GB |
| of which anonymous memory | 0.97 GB | 1.91 GB |

**Cause.** On CPU, each `odml.runtime_bmm` composite runs through its decomposition. It dequantizes a
slice of the int8 KV cache, builds a score tensor as wide as the whole cache (FILL +
DYNAMIC_UPDATE_SLICE), and then the main graph broadcasts the mask, selects and softmaxes over the full
width. XNNPACK gives every partition its own workspace for this. The workspaces are allocated by
`xnn_reshape_runtime` at the first prefill, one per partition (about 18 for E2B and 20 for E4B), each of
**ctx² × 4 bytes × KV heads** (traced with an `LD_PRELOAD` malloc hook). For E4B that is 128 MB each at a
4k context, and 512 MB each at 8k. The KV cache itself is int8 and small (~0.1 GB at 4k), so KV
quantization (TurboQuant) would not help here. The GPU path (ML Drift) has native
`runtime_batched_matmul` kernels; the CPU path is the fallback.

**Fix.** Three parts:
1. **Graph rewrite** (`python/rewrite_mobile_attention.py`). Each attention block,
   `runtime_bmm(q,K) → SELECT_V2(mask) → SOFTMAX → runtime_bmm(P,V)`, becomes one custom op,
   `voxsum.i8_attention(q, K, V, mask, param, meta_f, meta_i)`. The KV cache, `odml.cache_update`, the
   mask computation and every other op are untouched. Its small constant inputs are stored 64-byte
   aligned after the flatbuffer, because the custom-op dispatcher rejects unaligned host buffers. E2B:
   35 blocks fused in `decode`, 14 in `prefill_128`. E4B: 42 and 23.
2. **Kernel** (`cpp/i8_attn.cc`). It reads the int8 cache on the live columns only (mask true and
   column < `param[2]`), converted to float once per call. Q·Kᵀ and P·Vᵀ run as 4×4 NEON dot-product
   tiles, the softmax normalizer is accumulated in double, and nothing is allocated at the size of the
   cache squared. It uses 8 threads for prefill and 2 for decode. `KMP_BLOCKTIME=0` keeps its OpenMP
   threads from spinning against XNNPACK's pool: prefill went from 80 to 130 tok/s on the Reno7 with it.
3. **Engine** (`cpp/mfa_engine.cc`). A standalone CPU driver on the stock `libLiteRt.so` (the
   `com.google.ai.edge.litert:litert:2.1.6` AAR that VoxSumDroid already ships). It reproduces
   LiteRT-LM's CPU protocol (`llm_litert_compiled_model_executor.cc`):
   - **Context length.** The magic cache length 32003 is replaced at load (`MagicNumberConfigs`). The
     model must be loaded from a *private writable* mapping, because the replacement writes into
     constant buffers and a read-only file mapping crashes in `ReplaceMagicNumbersIfAny`.
   - **Embeddings.** The embedder and per-layer-embedder models run per token.
   - **Prefill.** `prefill_128` gets a causal bool mask and `param_tensor = [start, end, end]`.
   - **KV cache.** The 30 (E2B) or 48 (E4B) int8 KV buffers are shared between signatures and updated
     in place.
   - **Decoding** is greedy, and the XNNPACK weight cache is a file.

## Results

Same prompt (a 2,881-token reading prompt), `maxNumTokens` 4096, warm runs (the weight cache exists):

| Reno7, CPU, 4k | LiteRT-LM 0.17.1 | **mfa_engine + fused graph** |
|---|---|---|
| E2B peak RSS | 2.28 GB | **1.26 GB** (anonymous 0.25 GB) |
| E2B prefill / decode | 118 / ~9.8 tok/s | **130 / 7.6–9.3 tok/s** |
| E4B peak RSS | 4.58 GB | **2.97 GB** (anonymous 0.40 GB) |
| E4B prefill / decode | 41 / ~4 tok/s | **43 / 4.1 tok/s** |

**Output.** Greedy tokens are identical to the unfused graph run by the same engine on x86 (E2B and
E4B, the first 30+ tokens compared). The x86 host shows the same picture. E2B: anonymous memory
0.94 → 0.24 GB, prefill 1,199 → 1,556 tok/s. E4B: anonymous memory 1.90 → 0.36 GB, prefill
603 → 623 tok/s, decode 14.9 → 16.0 tok/s.

**First run.** The first run builds the XNNPACK weight cache and peaks higher (E4B: 4.3 GB). Build
the cache once per library build and CPU class and ship it, as in `vieenrose/LiteRT` turboquant-tq3.

## Build

```bash
python python/rewrite_mobile_attention.py <unpack>/Section10_TFLiteModel_tf_lite_prefill_decode.tflite fused.tflite
mkdir build && cd build
cmake .. -DLITERT_INCLUDE=<dir with litert/c/*.h> -DLITERT_LIB=<libLiteRt.so>          # host
cmake .. -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-26 -DMFA_MARCH=armv8.2-a+dotprod -DLITERT_INCLUDE=... \
  -DLITERT_LIB=<arm64 libLiteRt.so> -DOpenMP_CXX_FLAGS=-fopenmp -DOpenMP_CXX_LIB_NAMES=omp \
  -DOpenMP_omp_LIBRARY=$NDK/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/17/lib/linux/aarch64/libomp.a
make
./mfa_engine --dir <unpack> --main fused.tflite --fused --ctx 4096 --threads 8 \
  --weight-cache model.wcache --ids prompt_ids.txt --max-new 400
```

`<unpack>` is `litert-lm unpack model.litertlm`. Prompt ids include `<bos>` (2). Generation stops at
`<turn|>` (106) or `<eos>` (1).

## Limits

- CPU only. The GPU path keeps Google's graph, so use the unfused file there.
- Greedy decoding, and the tokenizer is outside the engine (the app supplies token ids).
- The engine is a driver, not a library yet. The VoxSumDroid integration should follow
  `mosslite/moss_lite_engine.cc`: a resident engine and a JNI surface.
