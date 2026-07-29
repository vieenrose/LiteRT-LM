# contrib/turboquant — TurboQuant TQ3 KV cache engines for Gemma 4 E2B

Engine-side integration of **TurboQuant** (Zandieh et al., ICLR 2026) 3-bit KV
caching for Gemma 4 E2B (16k context) on LiteRT.

**Canonical home of the port** — including the fused
`voxsum.tq3_attention` custom-op kernel, the flatbuffer lowering
(`rewrite_tq3.py`), asset prep, committed codebooks/rotations, and the full
README with acceptance numbers — is the `turboquant-tq3` branch of
[vieenrose/LiteRT](https://github.com/vieenrose/LiteRT/tree/turboquant-tq3/litert/samples/llm/turboquant).
This directory carries the engine drivers so LiteRT-LM readers can find them,
mirrored from that branch.

## Why standalone CMake, not Bazel

LiteRT-LM is currently **not buildable from public sources** (see
google-ai-edge/LiteRT issue #3002 and the CMake defects #2932 / #2945), so this
contribution deliberately does not touch the Bazel tree. Both engines build
standalone with CMake against a **stock prebuilt `libLiteRt.so`** plus the
LiteRT C headers; the fused custom op is registered at runtime via
`LiteRtAddCustomOpKernelOption` — no runtime rebuild.

```bash
mkdir -p build && cd build
cmake .. -DLITERT_INCLUDE=/path/to/litert/c/headers \
         -DLITERT_LIB=/path/to/libLiteRt.so
make   # -> engine2 (staging/fused auto-detect), engine (staging-only reference)
```

## Contents

- `cpp/engine2.cc` — driver: auto-detects staging vs fused mode from signature
  inputs (`kv_cache_*` vs `packed_*`), zero-copy packed-tensor binding,
  custom-op registration, quantize-on-write TQ3 scatter, teacher-forced /
  free-running loops, `--attn-threads`, `--dump-logits`.
- `cpp/engine.cc` — earlier staging-only engine (packed TQ3 side-cache as
  source of truth, fp32 staging as its dequantized image); kept as A/B
  reference.
- `cpp/tq3.{c,h}` — torch-matched TQ3 quantize/dequant (per-vector L2 norm,
  seed-42 QR rotation, 3-bit Lloyd-Max codebooks).
- `cpp/tq3_attn.{cc,h}` — fused attention kernel (double accumulation,
  mask-derived live rows, generation-keyed dequant memo, OpenMP ≤8 threads).
- `CMakeLists.txt` — standalone build of both engines.

Model, assets (codebooks/rotations/prompts), lowering script, and acceptance
results: see the LiteRT branch above.

## Headline numbers (9950X3D, XNNPACK 32 threads)

- Teacher-forced top-1 vs fp32-KV baseline: **0.9455 (en) / 0.9688 (zh)**.
- KV-attributable memory at 16k: ~1152 MiB (fp32 harness) → **65–93 MiB**.
- Peak RSS 6.13 → **5.12 GB**; prefill **410 tok/s**, decode **14.3 tok/s**
  (~2.5× the unfused staging engine).
