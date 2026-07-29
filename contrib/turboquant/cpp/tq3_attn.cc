// Phase 3 fused TQ3 attention kernel. See tq3_attn.h for the contract.
#include "tq3_attn.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

#include <chrono>
#include <map>
#include <mutex>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"

namespace {
constexpr int kCacheLen = 16384;
constexpr int kHeads = 8;
constexpr float kMaskedBelow = -50.0f;  // mask values are {0, -100}

double now_s() {
  using namespace std::chrono;
  return duration_cast<duration<double>>(steady_clock::now().time_since_epoch())
      .count();
}
}  // namespace

struct tq3_attn_core {
  const tq3_ctx *tq256 = nullptr, *tq512 = nullptr;
  int threads = 0;
  uint64_t generation = 0;
  struct Memo {
    uint64_t gen = ~0ull;
    int lo = 0, hi = 0, d = 0;
    std::vector<float> k, v;  // (hi-lo) x d each
  };
  std::map<const void *, Memo> memo;  // keyed by packed_k base pointer
  std::mutex mu;
  double t_dequant = 0, t_total = 0;
};

struct Tq3AttnOp {
  tq3_attn_core *core;
  int transposed;
};

extern "C" {

tq3_attn_core *tq3_attn_create(const tq3_ctx *tq256, const tq3_ctx *tq512,
                               int threads) {
  auto *c = new tq3_attn_core;
  c->tq256 = tq256;
  c->tq512 = tq512;
  c->threads = threads;
  return c;
}
void tq3_attn_destroy(tq3_attn_core *c) { delete c; }
void tq3_attn_bump_generation(tq3_attn_core *c) { ++c->generation; }
size_t tq3_attn_memo_bytes(const tq3_attn_core *c) {
  size_t n = 0;
  for (auto &kv : c->memo)
    n += (kv.second.k.capacity() + kv.second.v.capacity()) * sizeof(float);
  return n;
}
double tq3_attn_dequant_seconds(const tq3_attn_core *c) { return c->t_dequant; }
double tq3_attn_total_seconds(const tq3_attn_core *c) { return c->t_total; }

}  // extern "C"

namespace {

LiteRtStatus AttnInit(void *, const void *, size_t) { return kLiteRtStatusOk; }
LiteRtStatus AttnDestroy(void *) { return kLiteRtStatusOk; }

LiteRtStatus AttnGetOutputLayouts(void *, size_t num_inputs,
                                  const LiteRtLayout *in, size_t num_outputs,
                                  LiteRtLayout *out) {
  if (num_inputs != 6 || num_outputs != 1) return kLiteRtStatusErrorInvalidArgument;
  memset(&out[0], 0, sizeof(out[0]));
  out[0].rank = in[0].rank;  // ctx has q's shape (1,1,8T,d)
  for (unsigned i = 0; i < in[0].rank; ++i)
    out[0].dimensions[i] = in[0].dimensions[i];
  out[0].has_strides = false;
  return kLiteRtStatusOk;
}

LiteRtStatus AttnRun(void *user_data, size_t num_inputs,
                     const LiteRtTensorBuffer *inputs, size_t num_outputs,
                     LiteRtTensorBuffer *outputs) {
  if (num_inputs != 6 || num_outputs != 1) return kLiteRtStatusErrorInvalidArgument;
  auto *op = static_cast<Tq3AttnOp *>(user_data);
  tq3_attn_core *core = op->core;
  const double tw0 = now_s();

  size_t nb[6], ob;
  const void *p[6];
  void *po;
  for (int i = 0; i < 6; ++i) {
    if (LiteRtGetTensorBufferSize(inputs[i], &nb[i]) != kLiteRtStatusOk ||
        LiteRtLockTensorBuffer(inputs[i], const_cast<void **>(&p[i]),
                               kLiteRtTensorBufferLockModeRead) != kLiteRtStatusOk)
      return kLiteRtStatusErrorRuntimeFailure;
  }
  if (LiteRtGetTensorBufferSize(outputs[0], &ob) != kLiteRtStatusOk ||
      LiteRtLockTensorBuffer(outputs[0], &po, kLiteRtTensorBufferLockModeWrite) !=
          kLiteRtStatusOk)
    return kLiteRtStatusErrorRuntimeFailure;

  const float *q = (const float *)p[0];
  const float *k_new = (const float *)p[1];
  const float *v_new = (const float *)p[2];
  const float *mask = (const float *)p[3];
  const uint8_t *pk = (const uint8_t *)p[4];
  const uint8_t *pv = (const uint8_t *)p[5];
  float *ctx_out = (float *)po;

  const size_t bb = nb[4] / kCacheLen;             // 100 (d=256) or 196 (d=512)
  const int d = bb == 100 ? 256 : 512;
  const tq3_ctx *tq = d == 256 ? core->tq256 : core->tq512;
  const size_t mask_f = nb[3] / 4;
  // mask is (T, C+T): T*(C+T) floats -> solve for T
  int T = -1;
  for (int cand : {1, 128})
    if ((size_t)cand * (kCacheLen + cand) == mask_f) T = cand;
  if (T < 0 || nb[0] != (size_t)kHeads * T * d * 4 || ob != nb[0])
    return kLiteRtStatusErrorInvalidArgument;
  const int C = kCacheLen;
  const int W = C + T;  // mask row width

  // live cache range = union over tokens of unmasked cache columns
  int lo = C, hi = 0;
  for (int t = 0; t < T; ++t) {
    const float *mr = mask + (size_t)t * W;
    int j = 0;
    while (j < C && mr[j] < kMaskedBelow) ++j;
    if (j < C) {
      if (j < lo) lo = j;
      int e = C;
      while (e > j && mr[e - 1] < kMaskedBelow) --e;
      if (e > hi) hi = e;
    }
  }
  if (lo > hi) lo = hi = 0;
  const int live = hi - lo;

  // memo: dequantize live K/V rows once per generation per packed pair
  tq3_attn_core::Memo *mm;
  {
    std::lock_guard<std::mutex> g(core->mu);
    mm = &core->memo[pk];
  }
  const int nt = core->threads > 0 ? core->threads : omp_get_max_threads();
  if (mm->gen != core->generation || mm->lo != lo || mm->hi != hi || mm->d != d) {
    const double td0 = now_s();
    mm->k.resize((size_t)live * d);
    mm->v.resize((size_t)live * d);
#pragma omp parallel for schedule(static) num_threads(nt)
    for (int j = 0; j < live; ++j) {
      tq3_dequantize(tq, pk + (size_t)(lo + j) * bb, mm->k.data() + (size_t)j * d);
      tq3_dequantize(tq, pv + (size_t)(lo + j) * bb, mm->v.data() + (size_t)j * d);
    }
    mm->gen = core->generation;
    mm->lo = lo;
    mm->hi = hi;
    mm->d = d;
    core->t_dequant += now_s() - td0;
  }
  const float *K = mm->k.data();
  const float *V = mm->v.data();

  // All accumulation in DOUBLE: the fp32 inputs are exact, so the op output is
  // within ~1e-7 of exact fp64 attention. This matters because the surrounding
  // graph's dynamic-activation-quant FCs amplify any per-op epsilon by ~3
  // orders of magnitude (a 2e-5 fp32-accumulation wobble became ~1e-2 relative
  // on the next kv slice); double accumulation keeps the injected noise at the
  // rounding floor of the fp32 output itself.
  const int R = kHeads * T;
#pragma omp parallel num_threads(nt)
  {
    std::vector<double> srow(live + T);
    std::vector<double> acc(d);
#pragma omp for schedule(static)
    for (int r = 0; r < R; ++r) {
      const int t = r % T;
      const float *qv = q + (size_t)r * d;
      const float *mr = mask + (size_t)t * W;
      double mx = -INFINITY;
      // scores vs live cache rows
      for (int j = 0; j < live; ++j) {
        const float mv = mr[lo + j];
        if (mv < kMaskedBelow) {
          srow[j] = -INFINITY;
          continue;
        }
        const float *kr = K + (size_t)j * d;
        double s = 0.0;
        for (int x = 0; x < d; ++x) s += (double)qv[x] * kr[x];
        s += mv;
        srow[j] = s;
        if (s > mx) mx = s;
      }
      // scores vs new tokens
      for (int u = 0; u < T; ++u) {
        const float mv = mr[C + u];
        if (mv < kMaskedBelow) {
          srow[live + u] = -INFINITY;
          continue;
        }
        double s = 0.0;
        if (op->transposed)
          for (int x = 0; x < d; ++x) s += (double)qv[x] * k_new[(size_t)x * T + u];
        else
          for (int x = 0; x < d; ++x) s += (double)qv[x] * k_new[(size_t)u * d + x];
        s += mv;
        srow[live + u] = s;
        if (s > mx) mx = s;
      }
      // softmax (masked entries excluded)
      double sum = 0.0;
      for (int j = 0; j < live + T; ++j) {
        if (srow[j] == -INFINITY) {
          srow[j] = 0.0;
        } else {
          srow[j] = exp(srow[j] - mx);
          sum += srow[j];
        }
      }
      const double inv = 1.0 / sum;
      // context = P * V
      float *o = ctx_out + (size_t)r * d;
      memset(acc.data(), 0, (size_t)d * sizeof(double));
      for (int j = 0; j < live; ++j) {
        const double pj = srow[j] * inv;
        if (pj == 0.0) continue;
        const float *vr = V + (size_t)j * d;
        for (int x = 0; x < d; ++x) acc[x] += pj * vr[x];
      }
      for (int u = 0; u < T; ++u) {
        const double pu = srow[live + u] * inv;
        if (pu == 0.0) continue;
        if (op->transposed)
          for (int x = 0; x < d; ++x) acc[x] += pu * v_new[(size_t)u * d + x];
        else
          for (int x = 0; x < d; ++x) acc[x] += pu * v_new[(size_t)x * T + u];
      }
      for (int x = 0; x < d; ++x) o[x] = (float)acc[x];
    }
  }

  // debug: dump the first invocation's IO once for offline replication
  static bool dumped = false;
  if (!dumped) {
    if (const char* dir = getenv("TQ3_DUMP_OP")) {
      dumped = true;
      const char* nm[7] = {"q", "k_new", "v_new", "mask", "packed_k", "packed_v", "out"};
      for (int i = 0; i < 7; ++i) {
        char pth[512];
        snprintf(pth, sizeof(pth), "%s/%s.bin", dir, nm[i]);
        FILE* f = fopen(pth, "wb");
        if (i < 6) fwrite(p[i], 1, nb[i], f);
        else fwrite(po, 1, ob, f);
        fclose(f);
      }
      fprintf(stderr, "[tq3_attn] dumped first op (d=%d T=%d transposed=%d live=%d lo=%d) to %s\n",
              d, T, op->transposed, live, lo, getenv("TQ3_DUMP_OP"));
    }
  }

  for (int i = 0; i < 6; ++i) LiteRtUnlockTensorBuffer(inputs[i]);
  LiteRtUnlockTensorBuffer(outputs[0]);
  core->t_total += now_s() - tw0;
  return kLiteRtStatusOk;
}

}  // namespace

extern "C" void tq3_attn_kernel(tq3_attn_core *core, int transposed,
                                LiteRtCustomOpKernel *kernel, void **user_data) {
  static Tq3AttnOp ops[2];
  ops[transposed ? 1 : 0] = {core, transposed};
  kernel->Init = AttnInit;
  kernel->GetOutputLayouts = AttnGetOutputLayouts;
  kernel->Run = AttnRun;
  kernel->Destroy = AttnDestroy;
  *user_data = &ops[transposed ? 1 : 0];
}
