// Phase 3 engine v2: Phase 2b engine + support for the FUSED model
// (model_tq3.tflite, voxsum.tq3_attention custom ops consuming the packed TQ3
// side-cache directly — no fp32 KV staging exists at all).
//
// One binary, two modes, auto-detected from the model's signature inputs:
//   kv_cache_* inputs  -> staging mode (exact Phase 2b math, A/B reference)
//   packed_*   inputs  -> fused mode (packed caches bound zero-copy, custom
//                         op kernels registered on the stock libLiteRt.so)
// --dump-logits FILE appends the fp32 logits row of every generation-loop
// decode, for bitwise A/B between the two modes.
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
#include <cerrno>
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

#include "litert/c/litert_custom_op_kernel.h"

#include "tq3.h"
#include "tq3_attn.h"

#define DIE(...) do { fprintf(stderr, "FATAL: " __VA_ARGS__); fprintf(stderr, "\n"); exit(1); } while (0)
#define ENSURE(expr) do { LiteRtStatus s_ = (expr); if (s_ != kLiteRtStatusOk) \
    DIE("%s:%d %s -> %d", __FILE__, __LINE__, #expr, (int)s_); } while (0)

namespace {

int kCacheLen = 16384;   // runtime: --cache-len (4k device export uses 4096)
constexpr int kMaxLayers = 64;
int kNumLayers = 15;     // --layers  (E2B 15 ext pairs, Gemma-3-1B 26)
constexpr int kPrefill = 128;
int kWindow = 512;       // --window
int kGlobalEvery = 5;    // --global-every (E2B 5 -> 4,9,14; G3-1B 6 -> 5,11,17,23)
int kGlobalDim = 512;    // --global-dim (G3-1B head_dim is 256 everywhere)
bool is_global_layer(int l) { return (l + 1) % kGlobalEvery == 0; }
int layer_dim(int l) { return is_global_layer(l) ? kGlobalDim : 256; }

// Windowed side-cache: a sliding-window layer can never attend outside its last
// kWindow positions, so its ring buffer only needs kWindow rows (absolute
// column c lives at c % kWindow).  Global layers keep the full cache length --
// windowing those WOULD lose information.  --window-kv 0 disables (full length
// everywhere, the original layout).
bool kWindowKv = false;
int layer_rows(int l) {
  return (kWindowKv && !is_global_layer(l) && kWindow < kCacheLen) ? kWindow
                                                                  : kCacheLen;
}

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
  std::map<std::string, LiteRtTensorBuffer> alias_;   // shared kv_cache_*/packed_* inputs
  std::map<std::string, size_t> alias_bytes_;
  // name -> caller memory bound zero-copy (fused mode packed side-cache)
  const std::map<std::string, std::pair<void*, size_t>>* ext_ = nullptr;

  Component(LiteRtEnvironment env, const std::string& path, int threads,
            const std::string& weight_cache, bool alias_kv,
            const std::map<std::string, std::pair<void*, size_t>>* ext = nullptr,
            tq3_attn_core* attn = nullptr)
      : env_(env), ext_(ext) {
    ENSURE(LiteRtCreateModelFromFile(env, path.c_str(), &model_));
    LiteRtOptions opts;
    ENSURE(LiteRtCreateOptions(&opts));
    ENSURE(LiteRtSetOptionsHardwareAccelerators(opts, kLiteRtHwAcceleratorCpu));
    if (LiteRtOpaqueOptions oo = make_cpu_options(threads, weight_cache))
      ENSURE(LiteRtAddOpaqueOptions(opts, oo));
    if (attn) {
      // Register both fused-attention custom codes on the STOCK runtime.
      LiteRtCustomOpKernel k;
      void* ud = nullptr;
      tq3_attn_kernel(attn, 0, &k, &ud);
      ENSURE(LiteRtAddCustomOpKernelOption(opts, "voxsum.tq3_attention", 1, &k, ud));
      tq3_attn_kernel(attn, 1, &k, &ud);
      ENSURE(LiteRtAddCustomOpKernelOption(opts, "voxsum.tq3_attention_t", 1, &k, ud));
    }
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
        bool alias = alias_kv && (!strncmp(nm, "kv_cache_", 9) ||
                                  !strncmp(nm, "packed_", 7));
        LiteRtTensorBuffer b = nullptr;
        if (alias) {
          auto it = alias_.find(nm);
          if (it != alias_.end()) b = it->second;
        }
        if (!b && ext_) {
          auto it = ext_->find(nm);
          if (it != ext_->end()) {
            // zero-copy: wrap the engine's packed side-cache memory
            LiteRtTensor tensor;
            ENSURE(LiteRtGetSignatureInputTensorByIndex(sig, i, &tensor));
            LiteRtRankedTensorType tt;
            ENSURE(LiteRtGetRankedTensorType(tensor, &tt));
            ENSURE(LiteRtCreateTensorBufferFromHostMemory(
                &tt, it->second.first, it->second.second, nullptr, &b));
            owned_.push_back(b);
            alias_[nm] = b;
            alias_bytes_[nm] = it->second.second;
          }
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

// ---------------- PLE --------------------------------------------------------
// Two sources:
//  * ple.json  : mmap model.safetensors at the recorded offset (bf16, desktop)
//  * --ple-table FILE : standalone table (make_ple_table.py), 32-byte header
//    "PLETBL01" + u32 dtype(0=fp32,1=fp16,2=bf16,3=int8,4=int4) + u32 rows +
//    u32 cols + f32 scale + 8 pad. dtype>=3: cols f32 per-column quant scales
//    follow the header, then rows*cols int8 values (dtype 3) or rows*cols/2
//    bytes of little-nibble-first int4 (dtype 4, signed [-8,7]).
//    Dequant: value * colscale[c] * scale. bf16 is bit-identical to
//    the safetensors path; fp16 loses only sub-normal-range values.
static inline float half_to_float(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1f;
  uint32_t man = h & 0x3ff;
  uint32_t bits;
  if (exp == 0) {
    if (man == 0) { bits = sign; }
    else {                       // subnormal half -> normalized float
      int sh = 0; while (!(man & 0x400)) { man <<= 1; ++sh; }
      man &= 0x3ff;
      bits = sign | ((127 - 15 - sh + 1) << 23) | (man << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7f800000u | (man << 13);
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float f; memcpy(&f, &bits, 4); return f;
}


// ---- page-cache footprint control ------------------------------------------
// Apply |advice| to every mapping whose backing path contains |needle|.
// LiteRT/XNNPACK own the model and weight-cache mmaps, so /proc/self/maps is
// the only handle we have on them.
static size_t madvise_mapped_file(const char* needle, int advice) {
  FILE* f = fopen("/proc/self/maps", "r");
  if (!f) return 0;
  char line[512]; size_t total = 0;
  while (fgets(line, sizeof line, f)) {
    if (!strstr(line, needle)) continue;
    unsigned long lo = 0, hi = 0;
    if (sscanf(line, "%lx-%lx", &lo, &hi) != 2 || hi <= lo) continue;
    if (madvise((void*)lo, hi - lo, advice) == 0) total += hi - lo;
  }
  fclose(f);
  return total;
}

// Periodically drop the weight pages already streamed through.  The weight set
// is read in full every token and dwarfs the reclaimable RAM of a small device,
// so those pages have no reuse value -- but they keep counting toward RSS,
// which is what Android's lowmemorykiller ranks victims by.  Clean file pages:
// dropping is always safe and costs only a refault we would have taken anyway.
static void trim_file_cache() {
  madvise_mapped_file(".tflite", MADV_DONTNEED);
  madvise_mapped_file("wcache", MADV_DONTNEED);
}

// One-shot: after the graph is built and (with a warm XNNPACK weight cache)
// the repacked weights are being served from wcache, the pages faulted in from
// the .tflite during construction are dead weight that still counts toward RSS
// -- and RSS is what Android's lowmemorykiller ranks victims by.  Measured on a
// Boox Tab Mini C (Gemma 4 E2B): this returns 2196 MB.  Clean MAP_PRIVATE file
// pages, so dropping is always safe; anything still needed simply refaults.
// TQ3_DROP_MODEL_CACHE=0 disables.
static void drop_model_page_cache() {
  const char* e = getenv("TQ3_DROP_MODEL_CACHE");
  if (e && (e[0] == '0' || e[0] == 'n' || e[0] == 'N')) return;
  size_t n = madvise_mapped_file(".tflite", MADV_DONTNEED);
  fprintf(stderr, "dropped %.0f MB of .tflite page cache from RSS\n",
          n / 1048576.0);
}

// Whole-process resident split from smaps_rollup, in kB.
static void rollup_kb(long* rss, long* anon) {
  *rss = *anon = 0;
  FILE* f = fopen("/proc/self/smaps_rollup", "r");
  if (!f) return;
  char line[256]; long v;
  while (fgets(line, sizeof line, f)) {
    if (sscanf(line, "Rss: %ld", &v) == 1) *rss = v;
    else if (sscanf(line, "Anonymous: %ld", &v) == 1) *anon = v;
  }
  fclose(f);
}

static int g_trim_every = -1;
static int g_since_trim = 0;
static long g_peak_file_kb = 0, g_peak_rss_kb = 0;
static void maybe_trim() {
  if (g_trim_every < 0) {
    const char* e = getenv("TQ3_TRIM_EVERY");
    g_trim_every = e ? atoi(e) : 0;
  }
  long rss, anon; rollup_kb(&rss, &anon);
  if (rss > g_peak_rss_kb) g_peak_rss_kb = rss;
  if (rss - anon > g_peak_file_kb) g_peak_file_kb = rss - anon;
  if (g_trim_every > 0 && ++g_since_trim >= g_trim_every) {
    g_since_trim = 0;
    trim_file_cache();
  }
}

struct Ple {
  const uint8_t* base = nullptr;
  size_t map_len = 0; void* map_addr = nullptr;
  long rows = 0, cols = 0; float scale = 16.0f;
  int dtype = 2;  // 0 fp32, 1 fp16, 2 bf16, 3 int8+colscale, 4 int4+colscale
  std::vector<float> colscale;  // per-column scale * global scale (dtype>=3)
  // I/O strategy for the row gather.  A gather touches one ~8.75 KB row per
  // token at a uniformly random offset in a multi-GiB table.  Under mmap the
  // kernel's default readahead pulls 128 KB+ per fault and every page stays
  // resident, so the mapping's RSS grows toward the whole table for no reuse
  // benefit -- fatal on memory-tight devices, where it makes the process the
  // fattest lowmemorykiller target.  pread() copies just the row into a
  // reusable buffer and never grows RSS.  TQ3_PLE_IO_PREAD=0 restores mmap.
  bool use_pread = true;
  int fd = -1; size_t row_bytes = 0; off_t data_off = 0;
  mutable std::vector<uint8_t> rowbuf;
  ~Ple() { if (map_addr) munmap(map_addr, map_len); if (fd >= 0) close(fd); }
  void map_file(const std::string& path, long off) {
    fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) DIE("open %s", path.c_str());
    struct stat st; fstat(fd, &st);
    map_len = st.st_size;
    data_off = (off_t)off;
    const char* e = getenv("TQ3_PLE_IO_PREAD");
    use_pread = !(e && (e[0] == '0' || e[0] == 'n' || e[0] == 'N'));
    if (use_pread) return;  // no mapping at all in pread mode
    map_addr = mmap(nullptr, map_len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map_addr == MAP_FAILED) DIE("mmap %s", path.c_str());
    // Random row gather: suppress readahead, which otherwise inflates this
    // mapping's resident set by ~15x for no benefit.
    madvise(map_addr, map_len, MADV_RANDOM);
    base = (const uint8_t*)map_addr + off;
  }
  // Raw bytes of row |tok|.  Under mmap this points into the mapping; under
  // pread into the reusable buffer, valid until the next call.  Identical
  // bytes either way.
  const uint8_t* row_ptr(int32_t tok) const {
    size_t r = (size_t)(tok < 0 ? 0 : (tok >= rows ? rows - 1 : tok));
    if (!use_pread) return base + r * row_bytes;
    off_t off = data_off + (off_t)r * (off_t)row_bytes;
    size_t got = 0;
    while (got < row_bytes) {
      ssize_t n = pread(fd, rowbuf.data() + got, row_bytes - got,
                        off + (off_t)got);
      if (n > 0) got += (size_t)n;
      else if (n < 0 && errno == EINTR) continue;
      else { memset(rowbuf.data() + got, 0, row_bytes - got); break; }
    }
    return rowbuf.data();
  }
  void set_row_bytes() {
    row_bytes = dtype == 0 ? (size_t)cols * 4
              : dtype <= 2 ? (size_t)cols * 2
              : dtype == 3 ? (size_t)cols : (size_t)cols / 2;
    rowbuf.resize(row_bytes);
  }
  void init(const std::string& meta_path) {
    std::string j = slurp(meta_path);
    long off = json_long(j, "offset");
    rows = json_long(j, "rows"); cols = json_long(j, "cols");
    dtype = 2;
    map_file(json_str(j, "path"), off);
    set_row_bytes();
  }
  void init_table(const std::string& path) {
    struct { char magic[8]; uint32_t dtype, rows, cols; float scale;
             char pad[8]; } hdr;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f || fread(&hdr, sizeof hdr, 1, f) != 1) DIE("read %s", path.c_str());
    fclose(f);
    if (memcmp(hdr.magic, "PLETBL01", 8)) DIE("%s: bad magic", path.c_str());
    dtype = (int)hdr.dtype; rows = hdr.rows; cols = hdr.cols; scale = hdr.scale;
    size_t extra = dtype >= 3 ? (size_t)cols * 4 : 0;
    size_t data_bytes = dtype == 0 ? (size_t)rows * cols * 4
                      : dtype <= 2 ? (size_t)rows * cols * 2
                      : dtype == 3 ? (size_t)rows * cols
                      : (size_t)rows * (cols / 2);
    map_file(path, (long)(sizeof hdr + extra));
    set_row_bytes();
    if (map_len < sizeof hdr + extra + data_bytes)
      DIE("%s: truncated", path.c_str());
    if (dtype >= 3) {
      std::vector<float> cs(cols);
      if (pread(fd, cs.data(), (size_t)cols * 4, sizeof hdr) !=
          (ssize_t)((size_t)cols * 4))
        DIE("%s: colscale read", path.c_str());
      colscale.resize(cols);
      for (long c = 0; c < cols; ++c) colscale[c] = cs[c] * scale;
    }
    static const char* dn[] = {"fp32", "fp16", "bf16", "int8", "int4"};
    fprintf(stderr, "PLE table %s: dtype=%s rows=%ld cols=%ld scale=%g\n",
            path.c_str(), dn[dtype], rows, cols, scale);
  }
  // dst: n_tok * cols floats (cols = 35*256)
  void gather(const int32_t* toks, int n_tok, float* dst) const {
    for (int t = 0; t < n_tok; ++t) {
      float* d = dst + (size_t)t * cols;
      const uint8_t* raw = row_ptr(toks[t]);
      if (dtype == 0) {
        const float* row = (const float*)raw;
        for (long c = 0; c < cols; ++c) d[c] = row[c] * scale;
      } else if (dtype == 3) {
        const int8_t* row = (const int8_t*)raw;
        for (long c = 0; c < cols; ++c) d[c] = (float)row[c] * colscale[c];
      } else if (dtype == 4) {
        const uint8_t* row = raw;
        for (long c = 0; c < cols; c += 2) {
          uint8_t b = row[c >> 1];
          int lo = (int)(int8_t)(uint8_t)(b << 4) >> 4;
          int hi = (int)(int8_t)b >> 4;
          d[c] = (float)lo * colscale[c];
          d[c + 1] = (float)hi * colscale[c + 1];
        }
      } else {
        const uint16_t* row = (const uint16_t*)raw;
        if (dtype == 1) {
          for (long c = 0; c < cols; ++c) d[c] = half_to_float(row[c]) * scale;
        } else {
          for (long c = 0; c < cols; ++c) {
            uint32_t bits = (uint32_t)row[c] << 16;
            float v; memcpy(&v, &bits, 4);
            d[c] = v * scale;
          }
        }
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
  tq3_attn_core* attn = nullptr;
  bool use_tq = true;
  bool fused = false;   // model consumes packed_* inputs via the custom op
  // packed side-cache: [layer][2(k,v)] -> kCacheLen * block_bytes
  std::vector<uint8_t> packed[kMaxLayers][2];
  std::map<std::string, std::pair<void*, size_t>> ext_inputs;
  double t_model = 0, t_quant = 0, t_dequant = 0, t_glue = 0;

  int attn_threads = 8;   // OMP threads for the fused kernel; more
                        // collides with the XNNPACK pool (oversubscription)
  int global_memo = 0;    // 0 full fp32, 1 fp16, 2 stream (see tq3_attn.h)
  int kv_bits = 3;       // --kv-bits (3 = TQ3, 4 = TQ4)
  bool has_ple = true;
  std::string ple_table;  // optional standalone PLE table (--ple-table)
  void init(const std::string& model_path, const std::string& final_dir,
            const std::string& assets, int threads,
            const std::string& weight_cache, bool tq_mode) {
    use_tq = tq_mode;
    ENSURE(LiteRtCreateEnvironment(0, nullptr, &env));
    if (kv_bits == 16) {              // exact fp16: no rotation, no codebook
      if (tq3_init(&tq256, 256, 16, nullptr, nullptr) ||
          tq3_init(&tq512, 512, 16, nullptr, nullptr))
        DIE("tq3_init fp16");
      fprintf(stderr, "KV codec: EXACT fp16, block %zu B at d=256\n",
              tq256.block_bytes);
    } else {
      char cb[512];
      snprintf(cb, sizeof cb, "%s/cb_d256_b%d.bin", assets.c_str(), kv_bits);
      if (tq3_init(&tq256, 256, kv_bits, (assets + "/rot_d256.bin").c_str(), cb))
        DIE("tq3_init d=256 bits=%d (%s)", kv_bits, cb);
      snprintf(cb, sizeof cb, "%s/cb_d512_b%d.bin", assets.c_str(), kv_bits);
      if (tq3_init(&tq512, 512, kv_bits, (assets + "/rot_d512.bin").c_str(), cb))
        DIE("tq3_init d=512 bits=%d (%s)", kv_bits, cb);
      fprintf(stderr, "KV codec: %d-bit TurboQuant, block %zu B at d=256\n",
              kv_bits, tq256.block_bytes);
    }
    if (use_tq) {
      size_t kv_total = 0;
      for (int l = 0; l < kNumLayers; ++l) {
        size_t bb = (layer_dim(l) == 512 ? tq512 : tq256).block_bytes;
        const size_t rows = (size_t)layer_rows(l);
        // +64: LiteRT host-memory buffers require 64-byte alignment
        packed[l][0].assign(rows * bb + 64, 0);
        packed[l][1].assign(rows * bb + 64, 0);
        kv_total += 2 * rows * bb;
        for (int role = 0; role < 2; ++role)
          ext_inputs[std::string("packed_") + (role ? "v" : "k") + "_" +
                     std::to_string(l)] = {pdata(l, role), rows * bb};
      }
      fprintf(stderr,
              "KV side-cache: %.1f MB (%d layers, window-kv %s, %d sliding at "
              "%d rows / %d global at %d rows)\n",
              kv_total / 1048576.0, kNumLayers, kWindowKv ? "on" : "off",
              kNumLayers - kNumLayers / kGlobalEvery, layer_rows(0),
              kNumLayers / kGlobalEvery, kCacheLen);
    }
    attn = tq3_attn_create(&tq256, &tq512, attn_threads, global_memo);
    model = new Component(env, model_path, threads, weight_cache,
                          /*alias_kv=*/true, &ext_inputs, attn);
    aux = new Component(env, final_dir + "/auxiliary.tflite", threads, "", false);
    emb = new Component(env, final_dir + "/embedder_quantized.tflite", threads, "", false);
    has_ple = model->sig("decode").in_idx("per_layer_embeddings") >= 0;
    if (has_ple) {
      if (!ple_table.empty()) ple.init_table(ple_table);
      else ple.init(assets + "/ple.json");
    } else {
      fprintf(stderr, "no per_layer_embeddings input: PLE disabled (dense model)"
              "\n");
    }
    fused = model->sig("decode").in_idx("packed_k_0") >= 0;
    if (fused && !use_tq) DIE("fused model has no baseline mode");
    fprintf(stderr, "model %s: %s mode\n", model_path.c_str(),
            fused ? "FUSED (packed custom-op attention)" : "staging (phase 2b)");
  }

  uint8_t* pdata(int l, int role) {
    uintptr_t a = (uintptr_t)packed[l][role].data();
    return (uint8_t*)((a + 63) & ~(uintptr_t)63);
  }
  size_t packed_bytes() const {
    size_t n = 0;
    for (int l = 0; l < kNumLayers; ++l)
      n += packed[l][0].size() + packed[l][1].size() - 128;
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
      tq3_ctx* q = layer_dim(layer) == 512 ? &tq512 : &tq256;
      const float* s = (const float*)Component::lock_r(io.out[oi]);
      LiteRtTensorBuffer cb = nullptr;
      float* dst = nullptr;
      if (!fused) {
        std::string cache_name = std::string("kv_cache_") + role + "_" + std::to_string(layer);
        cb = model->alias_.at(cache_name);
        dst = (float*)Component::lock_w(cb);
      }
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
          const int rws = layer_rows(layer);
          uint8_t* blk = pdata(layer, role == 'v') +
                         (size_t)(pos % rws) * q->block_bytes;
          double t0 = now_s();
          tq3_quantize(q, vec, blk, scr);   // quantize-on-write; packed = truth
          t_quant += now_s() - t0;
          if (!fused) {
            t0 = now_s();
            tq3_dequantize(q, blk, deq);
            t_dequant += now_s() - t0;
            w = deq;
          }
        }
        if (fused) continue;                 // no fp32 staging exists at all
        if (role == 'k') {
          memcpy(dst + (size_t)pos * d, w, d * sizeof(float));
        } else {
          for (int j = 0; j < d; ++j) dst[(size_t)j * kCacheLen + pos] = w[j];
        }
      }
      if (!fused) Component::unlock(cb);
      Component::unlock(io.out[oi]);
    }
  }

  void run_model(SigIO& io) {
    if (fused) tq3_attn_bump_generation(attn);  // invalidate the kernel memo
    double t0 = now_s();
    model->run(io);
    t_model += now_s() - t0;
  }

  // Re-dequantize row range [lo,hi) of one layer from packed into staging and
  // report max |staging - packed_dequant| (0 expected: staging IS the image).
  float verify_packed(int layer, int lo, int hi) {
    if (!use_tq || fused) return -1.f;
    const int d = layer_dim(layer);
    tq3_ctx* q = layer_dim(layer) == 512 ? &tq512 : &tq256;
    float deq[512], m = 0.f;
    for (int role = 0; role < 2; ++role) {
      std::string nm = std::string("kv_cache_") + (role ? "v" : "k") + "_" + std::to_string(layer);
      const float* stg = (const float*)Component::lock_r(model->alias_.at(nm));
      for (int pos = lo; pos < hi; ++pos) {
        tq3_dequantize(q, pdata(layer, role) +
                              (size_t)(pos % layer_rows(layer)) * q->block_bytes,
                       deq);
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
    if (!has_ple) return;
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
    run_model(pf);
    if (const char* dir = getenv("DUMP_SLICES")) {   // debug: first chunk only
      if (pos0 == 0)
        for (size_t oi = 0; oi < pf.out_names.size(); ++oi) {
          std::string p = std::string(dir) + "/" + pf.out_names[oi] + ".bin";
          FILE* f = fopen(p.c_str(), "wb");
          size_t nb = Component::buf_bytes(pf.out[oi]);
          const void* src = Component::lock_r(pf.out[oi]);
          fwrite(src, 1, nb, f);
          Component::unlock(pf.out[oi]);
          fclose(f);
        }
    }
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
    run_model(dc);
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
              weight_cache, model_path, dump_logits, ple_table;
  int threads = 32, steps = 64, max_new = 256, attn_threads = 8;
  int kv_bits = 3;
  bool window_kv = false;
  int global_memo = 0;
  bool tq_mode = true, teacher_force = false, free_run = false, window_check = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--model") model_path = next();
    else if (a == "--dump-logits") dump_logits = next();
    else if (a == "--final") final_dir = next();
    else if (a == "--assets") assets = next();
    else if (a == "--prompt-file") prompt_file = next();
    else if (a == "--out") out_file = next();
    else if (a == "--threads") threads = atoi(next().c_str());
    else if (a == "--attn-threads") attn_threads = atoi(next().c_str());
    else if (a == "--global-memo") {
      std::string v = next();
      global_memo = v == "fp16" ? 1 : v == "stream" ? 2 : 0;
      if (v != "full" && v != "fp16" && v != "stream")
        DIE("--global-memo full|fp16|stream");
    }
    else if (a == "--steps") steps = atoi(next().c_str());
    else if (a == "--max-new") max_new = atoi(next().c_str());
    else if (a == "--mode") tq_mode = next() == "tq";
    else if (a == "--teacher") teacher_force = true;
    else if (a == "--free") free_run = true;
    else if (a == "--window-check") window_check = true;
    else if (a == "--weight-cache") weight_cache = next();
    else if (a == "--ple-table") ple_table = next();
    else if (a == "--cache-len") kCacheLen = atoi(next().c_str());
    else if (a == "--kv-bits") kv_bits = atoi(next().c_str());
    else if (a == "--window-kv") window_kv = atoi(next().c_str()) != 0;
    else if (a == "--layers") kNumLayers = atoi(next().c_str());
    else if (a == "--window") kWindow = atoi(next().c_str());
    else if (a == "--global-every") kGlobalEvery = atoi(next().c_str());
    else if (a == "--global-dim") kGlobalDim = atoi(next().c_str());
    else DIE("unknown arg %s", a.c_str());
  }
  if (final_dir.empty() || assets.empty() || prompt_file.empty())
    DIE("usage: engine2 --final DIR --assets DIR --prompt-file F [--model F] "
        "[--mode tq|baseline] [--teacher] [--free] [--window-check] [--steps N] "
        "[--threads N] [--out F] [--dump-logits F]");
  if (model_path.empty()) model_path = final_dir + "/model_quantized.tflite";

  std::string pj = slurp(prompt_file);
  std::vector<int32_t> ids = json_int_array(pj, "prompt_ids");
  std::vector<int32_t> teacher = json_int_array(pj, "teacher");
  std::vector<int32_t> eos = json_int_array(pj, "eos");
  fprintf(stderr, "prompt: %zu tokens, teacher: %zu\n", ids.size(), teacher.size());

  long rss, hwm;
  Engine eng;
  double t0 = now_s();
  eng.attn_threads = attn_threads;
  eng.global_memo = global_memo;
  eng.ple_table = ple_table;
  eng.kv_bits = kv_bits;
  kWindowKv = window_kv;
  eng.init(model_path, final_dir, assets, threads, weight_cache, tq_mode);
  drop_model_page_cache();
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
    maybe_trim();
    fprintf(stderr, "prefill @%d\n", c);
  }
  double prefill_s = now_s() - tp0;
  long rss_prefill; rss_mb(&rss_prefill, &hwm);
  // when dumping logits, catch-up decodes are dumped too (steps align between
  // modes since the fed tokens are the prompt itself)
  FILE* dumpc = nullptr;
  std::vector<float> cl;
  if (!dump_logits.empty()) {
    dumpc = fopen((dump_logits + ".catchup").c_str(), "wb");
    cl.resize(262144);
  }
  double tc0 = now_s();
  int cur = -1;
  for (int i = m; i < n; ++i) {
    cur = eng.decode(ids[i], i, true, dumpc ? cl.data() : nullptr);
    maybe_trim();
    if (dumpc) fwrite(cl.data(), 4, cl.size(), dumpc);
  }
  if (dumpc) fclose(dumpc);
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
    if (eng.fused) DIE("--window-check is a staging-mode probe");
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
  FILE* dumpf = nullptr;
  std::vector<float> logits_buf;
  if (!dump_logits.empty()) {
    dumpf = fopen(dump_logits.c_str(), "wb");
    if (!dumpf) DIE("cannot open %s", dump_logits.c_str());
    logits_buf.resize(262144);
  }
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
    cur = eng.decode(feed, pos, true, dumpf ? logits_buf.data() : nullptr);
    maybe_trim();
    if (dumpf) fwrite(logits_buf.data(), 4, logits_buf.size(), dumpf);
    ++pos;
  }
  if (dumpf) fclose(dumpf);
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
  fprintf(f, " \"peak_rss_kb\": %ld, \"peak_file_kb\": %ld, \"trim_every\": %d,\n",
          g_peak_rss_kb, g_peak_file_kb, g_trim_every);
  fprintf(f, " \"decode_s\": %.3f, \"decode_tok_s\": %.3f, \"n_gen\": %zu,\n",
          gen_s, (gen.size() - 1) / (gen_s + 1e-9), gen.size());
  fprintf(f, " \"t_model\": %.2f, \"t_quant\": %.3f, \"t_dequant\": %.3f, \"t_glue\": %.2f,\n",
          eng.t_model, eng.t_quant, eng.t_dequant, eng.t_glue);
  fprintf(f, " \"fused\": %s, \"t_attn\": %.3f, \"t_attn_dequant\": %.3f, "
          "\"memo_mb\": %.1f,\n", eng.fused ? "true" : "false",
          tq3_attn_total_seconds(eng.attn), tq3_attn_dequant_seconds(eng.attn),
          tq3_attn_memo_bytes(eng.attn) / 1048576.0);
  fprintf(f, " \"rss_load_mb\": %ld, \"rss_prefill_mb\": %ld, \"rss_end_mb\": %ld, "
          "\"rss_hwm_mb\": %ld, \"packed_mb\": %.1f, \"staging_mb\": %.1f}\n",
          rss_load, rss_prefill, rss, hwm, eng.packed_bytes() / 1048576.0,
          [&]{ size_t t = 0; for (auto& kv : eng.model->alias_bytes_)
                 if (!strncmp(kv.first.c_str(), "kv_cache_", 9)) t += kv.second;
               return t / 1048576.0; }());
  fclose(f);
  fprintf(stderr, "done: %zu gen, top1=%.4f (vs %d teacher), decode %.2f tok/s, "
          "rss_hwm=%ld MB -> %s\n", gen.size(),
          compared ? (double)agree / compared : -1.0, compared,
          (gen.size() - 1) / (gen_s + 1e-9), hwm, out_file.c_str());
  return 0;
}
