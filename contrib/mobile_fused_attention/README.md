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
   cache squared. The dot products are summed in float over blocks of 64 and the blocks in double: with
   a plain float sum over 4k columns, E4B's greedy output left the unfused graph's at token 44. It uses 8 threads for prefill and 2 for decode. `KMP_BLOCKTIME=0` keeps its OpenMP
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
   - **Decoding** is greedy or sampled (temperature, top-k, top-p, seed), and the XNNPACK weight cache
     is a file.
   - **Server mode** (`--serve`): a line protocol on stdin/stdout keeps the engine resident and reuses
     the longest common prefix of the KV cache between requests. `python/serve_openai.py` wraps it as
     an OpenAI-compatible `/v1/chat/completions` server (chat template from the `.litertlm` metadata,
     HF tokenizer, stop strings, llama-server-style `timings`), one engine process per slot.

## Results

Same prompt (a 2,881-token reading prompt), `maxNumTokens` 4096, warm runs (the weight cache exists):

| Reno7, CPU, 4k | LiteRT-LM 0.17.1 | **mfa_engine + fused graph** |
|---|---|---|
| E2B peak RSS | 2.28 GB | **1.08 GB** (1.26 GB before the table release) |
| E2B prefill / decode | 118 / ~9.8 tok/s | **130 / 7.6–9.3 tok/s** |
| E4B peak RSS | 4.58 GB | **2.65 GB** (anonymous 0.40 GB; 2.97 GB before the table release) |
| E4B prefill / decode | 41 / ~4 tok/s | **43 / 4.1 tok/s** |

**Output.** Greedy tokens are identical to the unfused graph run by the same engine on x86 (E2B and
E4B, the first 30+ tokens compared). The x86 host shows the same picture. E2B: anonymous memory
0.94 → 0.24 GB, prefill 1,199 → 1,556 tok/s. E4B: anonymous memory 1.90 → 0.36 GB, prefill
603 → 623 tok/s, decode 14.9 → 16.0 tok/s.

**Context length.** Reno7, CPU, forked engine, peak RSS / prefill / decode: E2B 4k 1.08 GB / 118 / 8.6 tok/s; E2B 8k 1.16 GB / 90 / 6.4; E4B 4k 2.65 GB / 43 / 4.1; E4B 8k 2.82 GB / 38 / 3.3. On 13 held-out meetings the 8k context improves only decision recall (E4B 73 → 80 %), so 4k is recommended.

**First run.** When the weight cache file does not exist, the engine first builds it in a
compile-only pass. XNNPACK writes each packed step to the file and maps it back, while the original
weights it has read stay resident; the two together peaked at 4.5 GB for E4B (anonymous memory only
0.58 GB). During the pass a thread drops the cache file's pages from the process (`MADV_DONTNEED` on
its shared mapping; the data stays in the file). The pass then unmaps everything and the engine loads
warm. Reno7, cold start: E4B peak RSS 4.54 → **3.01 GB** (the pass itself peaks at 2.66 GB), E2B
1.48 GB. The cache is byte-identical to one built the old way, and so are the output tokens, so
nothing needs to be shipped pre-built.

**Table release.** The embedder and per-layer-embedder tables are lookup tables whose touched pages stay resident (0.24 GB after one 3k-token prompt for E4B, growing with the vocabulary seen). Every 8 tokens the engine `MADV_DONTNEED`s their mappings, only those with `Private_Dirty = 0` in `smaps`; lookups read the page cache again. Reno7, E4B: `VmHWM` 2,821 → 2,586 MB, `dumpsys` RSS peak 2.65 GB, identical tokens, same speed. `MFA_KEEP_TABLES=1` disables it.

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
- The tokenizer is outside the engine (the app supplies token ids; `serve_openai.py` uses the HF one).
- The engine is a driver, not a library yet. The VoxSumDroid integration should follow
  `mosslite/moss_lite_engine.cc`: a resident engine and a JNI surface.
