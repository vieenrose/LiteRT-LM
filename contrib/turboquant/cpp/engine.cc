// Phase 2b: standalone x86 C++ engine for Gemma 4 E2B LiteRT summarization with
// a packed TQ3 KV side-cache (TurboQuant, seed-42 rotation, Lloyd-Max codebooks).
//
// Data flow per step (same semantics as phase2a_harness.py):
//   embedder_quantized.tflite : token_ids -> embeddings
//   PLE                       : mmap'd bf16 row gather from model.safetensors, x16
//   auxiliary.tflite          : rope (input_pos) + masks (tokens, time_step, valid)
//   model_quantized.tflite    : prefill_128 / decode -> kv_slice_* (+ logits)
//   engine                    : TQ3 quantize slice rows -> packed side-cache,
//                               dequantize -> fp32 staging (the model's own
//                               kv_cache_* input buffers; scatter at absolute pos)
//
// The fp32 staging is the interpreter's OWN input tensor memory (LiteRt managed
// host buffers, aliased between prefill and decode) — there is no second fp32
// copy of the KV cache anywhere in the process. The packed side-cache is the
// source of truth; staging rows are its dequantized image.
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <map>
#include <string>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"

#include "tq3.h"

#define DIE(...) do { fprintf(stderr, "FATAL: " __VA_ARGS__); fprintf(stderr, "\n"); exit(1); } while (0)
#define ENSURE(expr) do { LiteRtStatus s_ = (expr); if (s_ != kLiteRtStatusOk) \
    DIE("%s:%d %s -> %d", __FILE__, __LINE__, #expr, (int)s_); } while (0)

namespace {

constexpr int kCacheLen = 16384;
constexpr int kNumLayers = 15;
constexpr int kPrefill = 128;
constexpr int kWindow = 512;
bool is_global_layer(int l) { return l == 4 || l == 9 || l == 14; }
int layer_dim(int l) { return is_global_layer(l) ? 512 : 256; }

double now_s() {
  using namespace std::chrono;
  return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

void rss_mb(long* rss, long* hwm) {
  *rss = *hwm = -1;
  FILE* f = fopen("/proc/self/status", "r");
  if (!f) return;
  char line[256];
  while (fgets(line, sizeof(line), f)) {
    if (!strncmp(line, "VmRSS:", 6)) *rss = atol(line + 7) / 1024;
    if (!strncmp(line, "VmHWM:", 6)) *hwm = atol(line + 7) / 1024;
  }
  fclose(f);
}

LiteRtOpaqueOptions make_cpu_options(int num_threads, const std::string& weight_cache) {
  char toml[1024];
  int off = 0;
  if (num_threads > 0)
    off += snprintf(toml + off, sizeof(toml) - off, "num_threads = %d\n", num_threads);
  if (!weight_cache.empty())
    off += snprintf(toml + off, sizeof(toml) - off,
                    "weight_cache_file_path = \"%s\"\n", weight_cache.c_str());
  if (off <= 0) return nullptr;
  char* payload = strdup(toml);
  LiteRtOpaqueOptions oo = nullptr;
  if (LiteRtCreateOpaqueOptions("xnnpack", payload, [](void* p) { free(p); }, &oo)
      != kLiteRtStatusOk) { free(payload); return nullptr; }
  return oo;
}

// ---------------- minimal JSON helpers (machine-generated inputs only) -------
std::string slurp(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) DIE("cannot open %s", path.c_str());
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::string s(n, 0);
  if (fread(&s[0], 1, n, f) != (size_t)n) DIE("short read %s", path.c_str());
  fclose(f);
  return s;
}
std::vector<int32_t> json_int_array(const std::string& j, const std::string& key) {
  std::vector<int32_t> out;
  size_t p = j.find("\"" + key + "\"");
  if (p == std::string::npos) return out;
  p = j.find('[', p);
  if (p == std::string::npos) return out;
  size_t e = j.find(']', p);
  const char* c = j.c_str() + p + 1;
  const char* end = j.c_str() + e;
  while (c < end) {
    char* nx = nullptr;
    long v = strtol(c, &nx, 10);
    if (nx == c) { ++c; continue; }
    out.push_back((int32_t)v);
    c = nx;
  }
  return out;
}
long json_long(const std::string& j, const std::string& key) {
  size_t p = j.find("\"" + key + "\"");
  if (p == std::string::npos) DIE("json key %s missing", key.c_str());
  p = j.find(':', p);
  return strtol(j.c_str() + p + 1, nullptr, 10);
}
std::string json_str(const std::string& j, const std::string& key) {
  size_t p = j.find("\"" + key + "\"");
  if (p == std::string::npos) DIE("json key %s missing", key.c_str());
  p = j.find('"', j.find(':', p));
  size_t e = j.find('"', p + 1);
  return j.substr(p + 1, e - p - 1);
}

// ---------------- LiteRT component (moss_lite_engine pattern) ----------------
struct SigIO {
  LiteRtParamIndex index = 0;
  std::vector<LiteRtTensorBuffer> in, out;
  std::vector<std::string> in_names, out_names;
  int in_idx(const char* needle) const {
    for (size_t i = 0; i < in_names.size(); ++i)
      if (in_names[i].find(needle) != std::string::npos) return (int)i;
    return -1;
  }
  int out_idx(const char* needle) const {
    for (size_t i = 0; i < out_names.size(); ++i)
      if (out_names[i].find(needle) != std::string::npos) return (int)i;
    return -1;
  }
};

// Buffers aliased across signatures by input-tensor name (kv_cache_* only):
// prefill_128 and decode then share ONE fp32 staging per KV tensor.
struct Component {
  LiteRtEnvironment env_ = nullptr;
  LiteRtModel model_ = nullptr;
  LiteRtCompiledModel cm_ = nullptr;
  std::map<std::string, SigIO> sigs_;
  std::vector<LiteRtTensorBuffer> owned_;
  std::map<std::string, LiteRtTensorBuffer> alias_;   // shared kv_cache_* inputs
  std::map<std::string, size_t> alias_bytes_;

  Component(LiteRtEnvironment env, const std::string& path, int threads,
            const std::string& weight_cache, bool alias_kv)
      : env_(env) {
    ENSURE(LiteRtCreateModelFromFile(env, path.c_str(), &model_));
    LiteRtOptions opts;
    ENSURE(LiteRtCreateOptions(&opts));
    ENSURE(LiteRtSetOptionsHardwareAccelerators(opts, kLiteRtHwAcceleratorCpu));
    if (LiteRtOpaqueOptions oo = make_cpu_options(threads, weight_cache))
      ENSURE(LiteRtAddOpaqueOptions(opts, oo));
    ENSURE(LiteRtCreateCompiledModel(env, model_, opts, &cm_));
    LiteRtParamIndex nsigs = 0;
    ENSURE(LiteRtGetNumModelSignatures(model_, &nsigs));
    for (LiteRtParamIndex si = 0; si < nsigs; ++si) {
      LiteRtSignature sig;
      ENSURE(LiteRtGetModelSignature(model_, si, &sig));
      const char* key = nullptr;
      ENSURE(LiteRtGetSignatureKey(sig, &key));
      SigIO io;
      io.index = si;
      LiteRtParamIndex nin = 0, nout = 0;
      ENSURE(LiteRtGetNumSignatureInputs(sig, &nin));
      ENSURE(LiteRtGetNumSignatureOutputs(sig, &nout));
      for (LiteRtParamIndex i = 0; i < nin; ++i) {
        const char* nm = nullptr;
        ENSURE(LiteRtGetSignatureInputName(sig, i, &nm));
        io.in_names.push_back(nm);
        bool alias = alias_kv && !strncmp(nm, "kv_cache_", 9);
        LiteRtTensorBuffer b = nullptr;
        if (alias) {
          auto it = alias_.find(nm);
          if (it != alias_.end()) b = it->second;
        }
        if (!b) {
          b = make_buffer(sig, si, i, true);
          if (alias) {
            alias_[nm] = b;
            alias_bytes_[nm] = buf_bytes(b);
            zero_buf(b);
          }
        }
        io.in.push_back(b);
      }
      for (LiteRtParamIndex i = 0; i < nout; ++i) {
        const char* nm = nullptr;
        ENSURE(LiteRtGetSignatureOutputName(sig, i, &nm));
        io.out_names.push_back(nm);
        io.out.push_back(make_buffer(sig, si, i, false));
      }
      sigs_[key] = io;
    }
  }
  ~Component() {
    for (auto b : owned_) LiteRtDestroyTensorBuffer(b);
    if (cm_) LiteRtDestroyCompiledModel(cm_);
    if (model_) LiteRtDestroyModel(model_);
  }

  LiteRtTensorBuffer make_buffer(LiteRtSignature sig, LiteRtParamIndex si,
                                 LiteRtParamIndex ti, bool is_input) {
    LiteRtTensor tensor;
    ENSURE(is_input ? LiteRtGetSignatureInputTensorByIndex(sig, ti, &tensor)
                    : LiteRtGetSignatureOutputTensorByIndex(sig, ti, &tensor));
    LiteRtRankedTensorType tt;
    ENSURE(LiteRtGetRankedTensorType(tensor, &tt));
    LiteRtTensorBufferRequirements reqs;
    ENSURE(is_input ? LiteRtGetCompiledModelInputBufferRequirements(cm_, si, ti, &reqs)
                    : LiteRtGetCompiledModelOutputBufferRequirements(cm_, si, ti, &reqs));
    size_t bytes = 0;
    ENSURE(LiteRtGetTensorBufferRequirementsBufferSize(reqs, &bytes));
    LiteRtTensorBuffer buf;
    ENSURE(LiteRtCreateManagedTensorBuffer(env_, kLiteRtTensorBufferTypeHostMemory,
                                           &tt, bytes, &buf));
    owned_.push_back(buf);
    return buf;
  }

  SigIO& sig(const std::string& name) {
    auto it = sigs_.find(name);
    if (it == sigs_.end()) DIE("signature %s not found", name.c_str());
    return it->second;
  }
  void run(SigIO& io) {
    ENSURE(LiteRtRunCompiledModel(cm_, io.index, io.in.size(), io.in.data(),
                                  io.out.size(), io.out.data()));
  }
  static size_t buf_bytes(LiteRtTensorBuffer b) {
    size_t n = 0; LiteRtGetTensorBufferSize(b, &n); return n;
  }
  static void* lock_w(LiteRtTensorBuffer b) {
    void* p = nullptr;
    ENSURE(LiteRtLockTensorBuffer(b, &p, kLiteRtTensorBufferLockModeWrite));
    return p;
  }
  static const void* lock_r(LiteRtTensorBuffer b) {
    void* p = nullptr;
    ENSURE(LiteRtLockTensorBuffer(b, &p, kLiteRtTensorBufferLockModeRead));
    return p;
  }
  static void unlock(LiteRtTensorBuffer b) { LiteRtUnlockTensorBuffer(b); }
  static void write_buf(LiteRtTensorBuffer b, const void* src, size_t bytes) {
    void* p = lock_w(b); memcpy(p, src, bytes); unlock(b);
  }
  static void read_buf(LiteRtTensorBuffer b, void* dst, size_t bytes) {
    const void* p = lock_r(b); memcpy(dst, p, bytes); unlock(b);
  }
  static void zero_buf(LiteRtTensorBuffer b) {
    size_t n = buf_bytes(b); void* p = lock_w(b); memset(p, 0, n); unlock(b);
  }
  static void copy_buf(LiteRtTensorBuffer src, LiteRtTensorBuffer dst, size_t bytes) {
    const void* s = lock_r(src); void* d = lock_w(dst);
    memcpy(d, s, bytes); unlock(dst); unlock(src);
  }
};

// ---------------- PLE (mmap'd bf16 row gather) -------------------------------
struct Ple {
  const uint16_t* base = nullptr;  // bf16 rows
  size_t map_len = 0; void* map_addr = nullptr;
  long rows = 0, cols = 0; float scale = 16.0f;
  void init(const std::string& meta_path) {
    std::string j = slurp(meta_path);
    std::string path = json_str(j, "path");
    long off = json_long(j, "offset");
    rows = json_long(j, "rows"); cols = json_long(j, "cols");
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) DIE("open %s", path.c_str());
    struct stat st; fstat(fd, &st);
    map_len = st.st_size;
    map_addr = mmap(nullptr, map_len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map_addr == MAP_FAILED) DIE("mmap %s", path.c_str());
    close(fd);
    base = (const uint16_t*)((const uint8_t*)map_addr + off);
  }
  // dst: n_tok * cols floats (cols = 35*256)
  void gather(const int32_t* toks, int n_tok, float* dst) const {
    for (int t = 0; t < n_tok; ++t) {
      const uint16_t* row = base + (size_t)toks[t] * cols;
      float* d = dst + (size_t)t * cols;
      for (long c = 0; c < cols; ++c) {
        uint32_t bits = (uint32_t)row[c] << 16;
        float v; memcpy(&v, &bits, 4);
        d[c] = v * scale;
      }
    }
  }
};

// ---------------- engine -----------------------------------------------------
struct Engine {
  LiteRtEnvironment env = nullptr;
  Component *model = nullptr, *aux = nullptr, *emb = nullptr;
  Ple ple;
  tq3_ctx tq256, tq512;
  bool use_tq = true;
  // packed side-cache: [layer][2(k,v)] -> kCacheLen * block_bytes
  std::vector<uint8_t> packed[kNumLayers][2];
  double t_model = 0, t_quant = 0, t_dequant = 0, t_glue = 0;

  void init(const std::string& final_dir, const std::string& assets, int threads,
            const std::string& weight_cache, bool tq_mode) {
    use_tq = tq_mode;
    ENSURE(LiteRtCreateEnvironment(0, nullptr, &env));
    model = new Component(env, final_dir + "/model_quantized.tflite", threads,
                          weight_cache, /*alias_kv=*/true);
    aux = new Component(env, final_dir + "/auxiliary.tflite", threads, "", false);
    emb = new Component(env, final_dir + "/embedder_quantized.tflite", threads, "", false);
    ple.init(assets + "/ple.json");
    if (tq3_init(&tq256, 256, (assets + "/rot_d256.bin").c_str(),
                 (assets + "/cb_d256_b3.bin").c_str()))
      DIE("tq3_init d=256");
    if (tq3_init(&tq512, 512, (assets + "/rot_d512.bin").c_str(),
                 (assets + "/cb_d512_b3.bin").c_str()))
      DIE("tq3_init d=512");
    if (use_tq)
      for (int l = 0; l < kNumLayers; ++l) {
        size_t bb = (is_global_layer(l) ? tq512 : tq256).block_bytes;
        packed[l][0].assign((size_t)kCacheLen * bb, 0);
        packed[l][1].assign((size_t)kCacheLen * bb, 0);
      }
  }

  size_t packed_bytes() const {
    size_t n = 0;
    for (int l = 0; l < kNumLayers; ++l) n += packed[l][0].size() + packed[l][1].size();
    return n;
  }

  // Scatter slice outputs of `io` (T tokens at absolute pos0) into staging,
  // via the TQ3 round-trip when enabled.
  void scatter(SigIO& io, int pos0, int T) {
    float vec[512], deq[512], scr[512];
    for (size_t oi = 0; oi < io.out_names.size(); ++oi) {
      const std::string& nm = io.out_names[oi];
      if (nm.rfind("kv_slice_", 0) != 0) continue;
      const char role = nm[9];              // 'k' or 'v'
      const int layer = atoi(nm.c_str() + 11);
      const int d = layer_dim(layer);
      tq3_ctx* q = is_global_layer(layer) ? &tq512 : &tq256;
      const float* s = (const float*)Component::lock_r(io.out[oi]);
      std::string cache_name = std::string("kv_cache_") + role + "_" + std::to_string(layer);
      LiteRtTensorBuffer cb = model->alias_.at(cache_name);
      float* dst = (float*)Component::lock_w(cb);
      for (int t = 0; t < T; ++t) {
        const int pos = pos0 + t;
        // gather source vector
        if (role == 'k') {
          memcpy(vec, s + (size_t)t * d, d * sizeof(float));
        } else {
          for (int j = 0; j < d; ++j) vec[j] = s[(size_t)j * T + t];   // (d,T) col t
        }
        const float* w = vec;
        if (use_tq) {
          uint8_t* blk = packed[layer][role == 'v'].data() + (size_t)pos * q->block_bytes;
          double t0 = now_s();
          tq3_quantize(q, vec, blk, scr);
          t_quant += now_s() - t0;
          t0 = now_s();
          tq3_dequantize(q, blk, deq);
          t_dequant += now_s() - t0;
          w = deq;
        }
        if (role == 'k') {
          memcpy(dst + (size_t)pos * d, w, d * sizeof(float));
        } else {
          for (int j = 0; j < d; ++j) dst[(size_t)j * kCacheLen + pos] = w[j];
        }
      }
      Component::unlock(cb);
      Component::unlock(io.out[oi]);
    }
  }

  // Re-dequantize row range [lo,hi) of one layer from packed into staging and
  // report max |staging - packed_dequant| (0 expected: staging IS the image).
  float verify_packed(int layer, int lo, int hi) {
    if (!use_tq) return -1.f;
    const int d = layer_dim(layer);
    tq3_ctx* q = is_global_layer(layer) ? &tq512 : &tq256;
    float deq[512], m = 0.f;
    for (int role = 0; role < 2; ++role) {
      std::string nm = std::string("kv_cache_") + (role ? "v" : "k") + "_" + std::to_string(layer);
      const float* stg = (const float*)Component::lock_r(model->alias_.at(nm));
      for (int pos = lo; pos < hi; ++pos) {
        tq3_dequantize(q, packed[layer][role].data() + (size_t)pos * q->block_bytes, deq);
        for (int j = 0; j < d; ++j) {
          float sv = role ? stg[(size_t)j * kCacheLen + pos] : stg[(size_t)pos * d + j];
          float diff = fabsf(sv - deq[j]);
          if (diff > m) m = diff;
        }
      }
      Component::unlock(model->alias_.at(nm));
    }
    return m;
  }

  void run_aux_masks(SigIO& msig, const int32_t* toks, int T, int time_step,
                     SigIO& target) {
    int i_t = msig.in_idx("input_tokens"), i_s = msig.in_idx("time_step"),
        i_v = msig.in_idx("valid_mask");
    if (i_t < 0 || i_s < 0 || i_v < 0) DIE("mask sig inputs");
    Component::write_buf(msig.in[i_t], toks, (size_t)T * 4);
    int32_t ts = time_step;
    Component::write_buf(msig.in[i_s], &ts, 4);
    std::vector<uint8_t> valid(Component::buf_bytes(msig.in[i_v]), 1);
    Component::write_buf(msig.in[i_v], valid.data(), valid.size());
    aux->run(msig);
    for (const char* nm : {"mask_global", "mask_local"}) {
      int o = msig.out_idx(nm), mi = target.in_idx(nm);
      if (o < 0 || mi < 0) DIE("mask io %s", nm);
      Component::copy_buf(msig.out[o], target.in[mi],
                          Component::buf_bytes(target.in[mi]));
    }
  }

  void run_aux_rope(SigIO& rsig, int pos0, int T, SigIO& target) {
    int i_p = rsig.in_idx("input_pos");
    if (i_p < 0) DIE("rope input_pos");
    std::vector<int32_t> pos(T);
    for (int i = 0; i < T; ++i) pos[i] = pos0 + i;
    Component::write_buf(rsig.in[i_p], pos.data(), (size_t)T * 4);
    aux->run(rsig);
    for (const char* nm : {"pos_emb_cos", "pos_emb_sin", "pos_emb_local_cos",
                           "pos_emb_local_sin"}) {
      int o = rsig.out_idx(nm), mi = target.in_idx(nm);
      if (o < 0 || mi < 0) DIE("rope io %s", nm);
      Component::copy_buf(rsig.out[o], target.in[mi],
                          Component::buf_bytes(target.in[mi]));
    }
  }

  void fill_common(SigIO& esig, SigIO& target, const int32_t* toks, int T) {
    int i_t = esig.in_idx("token_ids");
    if (i_t < 0) DIE("embedder token_ids");
    Component::write_buf(esig.in[i_t], toks, (size_t)T * 4);
    emb->run(esig);
    int o = esig.out_idx("embeddings"), mi = target.in_idx("embeddings");
    if (o < 0 || mi < 0) DIE("embeddings io");
    Component::copy_buf(esig.out[o], target.in[mi],
                        Component::buf_bytes(target.in[mi]));
    int pl = target.in_idx("per_layer_embeddings");
    if (pl < 0) DIE("ple input");
    float* p = (float*)Component::lock_w(target.in[pl]);
    ple.gather(toks, T, p);
    Component::unlock(target.in[pl]);
  }

  void prefill(const int32_t* toks, int pos0, bool do_scatter = true) {
    double g0 = now_s();
    SigIO& pf = model->sig("prefill_128");
    fill_common(emb->sig("prefill_embedder_128"), pf, toks, kPrefill);
    run_aux_rope(aux->sig("prefill_rope_128"), pos0, kPrefill, pf);
    run_aux_masks(aux->sig("prefill_mask_128"), toks, kPrefill, pos0, pf);
    t_glue += now_s() - g0;
    double t0 = now_s();
    model->run(pf);
    t_model += now_s() - t0;
    if (do_scatter) scatter(pf, pos0, kPrefill);
  }

  // returns argmax token; logits_out optionally receives full row
  int decode(int32_t token, int pos, bool do_scatter = true,
             float* logits_out = nullptr) {
    double g0 = now_s();
    SigIO& dc = model->sig("decode");
    fill_common(emb->sig("decode_embedder"), dc, &token, 1);
    run_aux_rope(aux->sig("decode_rope"), pos, 1, dc);
    run_aux_masks(aux->sig("decode_mask"), &token, 1, pos, dc);
    t_glue += now_s() - g0;
    double t0 = now_s();
    model->run(dc);
    t_model += now_s() - t0;
    if (do_scatter) scatter(dc, pos, 1);
    int lo = dc.out_idx("logits");
    const float* lg = (const float*)Component::lock_r(dc.out[lo]);
    size_t vocab = Component::buf_bytes(dc.out[lo]) / 4;
    int best = 0;
    for (size_t i = 1; i < vocab; ++i)
      if (lg[i] > lg[best]) best = (int)i;
    if (logits_out) memcpy(logits_out, lg, vocab * 4);
    Component::unlock(dc.out[lo]);
    return best;
  }

  // zero staging rows outside [pos-kWindow+1, pos] for all sliding layers
  void zero_outside_window(int pos) {
    int lo = pos - kWindow + 1;
    if (lo <= 0) return;
    for (int l = 0; l < kNumLayers; ++l) {
      if (is_global_layer(l)) continue;
      const int d = layer_dim(l);
      for (int role = 0; role < 2; ++role) {
        std::string nm = std::string("kv_cache_") + (role ? "v" : "k") + "_" + std::to_string(l);
        float* stg = (float*)Component::lock_w(model->alias_.at(nm));
        if (role == 0) {
          memset(stg, 0, (size_t)lo * d * sizeof(float));
        } else {
          for (int j = 0; j < d; ++j)
            memset(stg + (size_t)j * kCacheLen, 0, (size_t)lo * sizeof(float));
        }
        Component::unlock(model->alias_.at(nm));
      }
    }
  }
};

}  // namespace

int main(int argc, char** argv) {
  std::string final_dir, assets, prompt_file, out_file = "engine_out.json",
              weight_cache;
  int threads = 32, steps = 64, max_new = 256;
  bool tq_mode = true, teacher_force = false, free_run = false, window_check = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--final") final_dir = next();
    else if (a == "--assets") assets = next();
    else if (a == "--prompt-file") prompt_file = next();
    else if (a == "--out") out_file = next();
    else if (a == "--threads") threads = atoi(next().c_str());
    else if (a == "--steps") steps = atoi(next().c_str());
    else if (a == "--max-new") max_new = atoi(next().c_str());
    else if (a == "--mode") tq_mode = next() == "tq";
    else if (a == "--teacher") teacher_force = true;
    else if (a == "--free") free_run = true;
    else if (a == "--window-check") window_check = true;
    else if (a == "--weight-cache") weight_cache = next();
    else DIE("unknown arg %s", a.c_str());
  }
  if (final_dir.empty() || assets.empty() || prompt_file.empty())
    DIE("usage: engine --final DIR --assets DIR --prompt-file F [--mode tq|baseline] "
        "[--teacher] [--free] [--window-check] [--steps N] [--threads N] [--out F]");

  std::string pj = slurp(prompt_file);
  std::vector<int32_t> ids = json_int_array(pj, "prompt_ids");
  std::vector<int32_t> teacher = json_int_array(pj, "teacher");
  std::vector<int32_t> eos = json_int_array(pj, "eos");
  fprintf(stderr, "prompt: %zu tokens, teacher: %zu\n", ids.size(), teacher.size());

  long rss, hwm;
  Engine eng;
  double t0 = now_s();
  eng.init(final_dir, assets, threads, weight_cache, tq_mode);
  rss_mb(&rss, &hwm);
  fprintf(stderr, "loaded in %.1fs rss=%ld MB hwm=%ld MB packed_side_cache=%.1f MB\n",
          now_s() - t0, rss, hwm, eng.packed_bytes() / 1048576.0);
  long rss_load = rss;

  // ---- prompt ingestion: prefill_128 chunks + catch-up decode ----
  const int n = (int)ids.size();
  const int m = ((n - 1) / kPrefill) * kPrefill;
  double tp0 = now_s();
  for (int c = 0; c < m; c += kPrefill) {
    eng.prefill(ids.data() + c, c);
    fprintf(stderr, "prefill @%d\n", c);
  }
  double prefill_s = now_s() - tp0;
  long rss_prefill; rss_mb(&rss_prefill, &hwm);
  double tc0 = now_s();
  int cur = -1;
  for (int i = m; i < n; ++i) cur = eng.decode(ids[i], i);
  double catchup_s = now_s() - tc0;
  int catchup_toks = n - m;
  fprintf(stderr, "ingested %d tokens: prefill %.1fs (%.1f tok/s), catch-up %d in %.1fs "
          "(%.2f tok/s), rss=%ld MB\n", n, prefill_s, m / (prefill_s + 1e-9),
          catchup_toks, catchup_s, catchup_toks / (catchup_s + 1e-9), rss_prefill);

  if (eng.use_tq) {
    float v0 = eng.verify_packed(0, 0, std::min(n, kCacheLen));
    float v4 = eng.verify_packed(4, 0, std::min(n, kCacheLen));
    fprintf(stderr, "packed-vs-staging max|diff|: layer0=%g layer4=%g\n", v0, v4);
  }

  // ---- window check mode ----
  if (window_check) {
    if (n <= kWindow) DIE("window check needs a prompt longer than %d tokens", kWindow);
    size_t vocab = 262144;
    std::vector<float> la(vocab), lb(vocab);
    eng.decode(cur, n, /*do_scatter=*/false, la.data());
    eng.zero_outside_window(n);
    eng.decode(cur, n, /*do_scatter=*/false, lb.data());
    double mx = 0; size_t ndiff = 0; int am_a = 0, am_b = 0;
    for (size_t i = 0; i < vocab; ++i) {
      if (la[i] != lb[i]) ++ndiff;
      double d = fabs((double)la[i] - lb[i]);
      if (d > mx) mx = d;
      if (la[i] > la[am_a]) am_a = (int)i;
      if (lb[i] > lb[am_b]) am_b = (int)i;
    }
    printf("{\"window_check\": {\"pos\": %d, \"bit_identical\": %s, \"n_diff\": %zu, "
           "\"max_abs_diff\": %g, \"argmax_equal\": %s}}\n",
           n, ndiff == 0 ? "true" : "false", ndiff, mx,
           am_a == am_b ? "true" : "false");
    return 0;
  }

  // ---- generation ----
  std::vector<int> gen;
  double tg0 = now_s();
  int pos = n;
  for (int s = 0; s < (teacher_force ? steps : max_new); ++s) {
    gen.push_back(cur);
    int feed = cur;
    if (teacher_force && s < (int)teacher.size()) feed = teacher[s];
    bool stop = false;
    for (int e : eos) if (feed == e) stop = true;
    if (stop) break;
    cur = eng.decode(feed, pos);
    ++pos;
  }
  double gen_s = now_s() - tg0;
  rss_mb(&rss, &hwm);

  // metrics vs teacher
  int agree = 0, compared = 0, diverge = -1;
  for (size_t s = 0; s < gen.size() && s < teacher.size(); ++s) {
    ++compared;
    if (gen[s] == teacher[s]) ++agree;
    else if (diverge < 0) diverge = (int)s;
  }

  FILE* f = fopen(out_file.c_str(), "w");
  fprintf(f, "{\"mode\": \"%s\", \"n_prompt\": %d, \"gen\": [", tq_mode ? "tq" : "baseline", n);
  for (size_t i = 0; i < gen.size(); ++i) fprintf(f, "%s%d", i ? "," : "", gen[i]);
  fprintf(f, "], \"top1\": %.4f, \"diverge_step\": %d, \"steps_compared\": %d,\n",
          compared ? (double)agree / compared : -1, diverge, compared);
  fprintf(f, " \"prefill_s\": %.3f, \"prefill_tok_s\": %.2f, \"catchup_tok_s\": %.3f,\n",
          prefill_s, m / (prefill_s + 1e-9), catchup_toks / (catchup_s + 1e-9));
  fprintf(f, " \"decode_s\": %.3f, \"decode_tok_s\": %.3f, \"n_gen\": %zu,\n",
          gen_s, (gen.size() - 1) / (gen_s + 1e-9), gen.size());
  fprintf(f, " \"t_model\": %.2f, \"t_quant\": %.3f, \"t_dequant\": %.3f, \"t_glue\": %.2f,\n",
          eng.t_model, eng.t_quant, eng.t_dequant, eng.t_glue);
  fprintf(f, " \"rss_load_mb\": %ld, \"rss_prefill_mb\": %ld, \"rss_end_mb\": %ld, "
          "\"rss_hwm_mb\": %ld, \"packed_mb\": %.1f, \"staging_mb\": %.1f}\n",
          rss_load, rss_prefill, rss, hwm, eng.packed_bytes() / 1048576.0,
          [&]{ size_t t = 0; for (auto& kv : eng.model->alias_bytes_) t += kv.second;
               return t / 1048576.0; }());
  fclose(f);
  fprintf(stderr, "done: %zu gen, top1=%.4f (vs %d teacher), decode %.2f tok/s, "
          "rss_hwm=%ld MB -> %s\n", gen.size(),
          compared ? (double)agree / compared : -1.0, compared,
          (gen.size() - 1) / (gen_s + 1e-9), hwm, out_file.c_str());
  return 0;
}
