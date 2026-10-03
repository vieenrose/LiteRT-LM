// mfa_engine: a standalone CPU engine for Google's Gemma-4 mobile graphs (E2B, E4B) on the
// stock libLiteRt.so, with the attention fused into voxsum.i8_attention.
//
// It reproduces LiteRT-LM's CPU protocol (runtime/executor/llm_litert_compiled_model_executor.cc):
//   - the cache length is the magic number 32003 in the graph, replaced at load by --ctx
//     (environment option MagicNumberConfigs);
//   - token -> embedder.tflite -> embeddings, token -> per_layer_embedder.tflite -> PLE;
//   - prefill_128: positions start.., causal bool mask (row t sees columns <= start+t),
//     param_tensor = [start, start+n, start+n]; padding rows have position 0 and no mask;
//   - the KV caches (30 int8 tensors) are inputs and outputs of both signatures: one buffer each,
//     updated in place;
//   - the last prompt token is fed by the first decode step; greedy decoding on `logits`.
//
// Usage: mfa_engine --dir <litert-lm unpack dir> [--main fused.tflite] [--fused] --ctx 4096
//                   --threads 8 [--weight-cache f] --ids prompt_ids.txt --max-new 400
//                   [--stop 106,1] [--out gen_ids.txt]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#if !defined(__ANDROID__)
#include <execinfo.h>
#endif
#include <signal.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "i8_attn.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_custom_op_kernel.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"

#define DIE(...) do { fprintf(stderr, "FATAL: " __VA_ARGS__); fprintf(stderr, "\n"); exit(1); } while (0)
#define ENSURE(x) do { LiteRtStatus s_ = (x); if (s_ != kLiteRtStatusOk) DIE("%s -> %d (%s:%d)", #x, (int)s_, __FILE__, __LINE__); } while (0)

static double now_s() { timeval tv; gettimeofday(&tv, nullptr); return tv.tv_sec + tv.tv_usec * 1e-6; }

static long status_kb(const char* key) {
  FILE* f = fopen("/proc/self/status", "r");
  char line[256]; long v = -1;
  while (f && fgets(line, sizeof line, f))
    if (!strncmp(line, key, strlen(key))) v = atol(line + strlen(key));
  if (f) fclose(f);
  return v;
}
static long g_peak_anon = 0;
static void sample_mem() { g_peak_anon = std::max(g_peak_anon, status_kb("RssAnon:")); }

static LiteRtOpaqueOptions cpu_options(int threads, const std::string& cache) {
  char toml[1024]; int off = 0;
  if (threads > 0) off += snprintf(toml + off, sizeof toml - off, "num_threads = %d\n", threads);
  if (!cache.empty()) off += snprintf(toml + off, sizeof toml - off, "weight_cache_file_path = \"%s\"\n", cache.c_str());
  if (off <= 0) return nullptr;
  char* payload = strdup(toml);
  LiteRtOpaqueOptions oo = nullptr;
  if (LiteRtCreateOpaqueOptions("xnnpack", payload, [](void* p) { free(p); }, &oo) != kLiteRtStatusOk) {
    free(payload);
    return nullptr;
  }
  return oo;
}

struct Sig {
  LiteRtParamIndex index = 0;
  std::vector<std::string> in_names, out_names;
  std::vector<LiteRtTensorBuffer> in, out;
  int in_idx(const std::string& n) const {
    for (size_t i = 0; i < in_names.size(); ++i) if (in_names[i] == n) return (int)i;
    return -1;
  }
  int out_idx(const std::string& n) const {
    for (size_t i = 0; i < out_names.size(); ++i) if (out_names[i] == n) return (int)i;
    return -1;
  }
};

struct Model {
  LiteRtEnvironment env = nullptr;
  LiteRtModel model = nullptr;
  LiteRtCompiledModel cm = nullptr;
  std::map<std::string, Sig> sigs;
  std::map<std::string, LiteRtTensorBuffer>* shared = nullptr;   // KV caches, by name

  Model(LiteRtEnvironment e, const std::string& path, int threads, const std::string& cache,
        bool fused, std::map<std::string, LiteRtTensorBuffer>* kv, const std::vector<std::string>& only,
        int attn_threads = 0)
      : env(e), shared(kv) {
    fprintf(stderr, "[load] %s\n", path.c_str());
    // Private writable mapping: the runtime rewrites the magic cache length inside a few
    // constant buffers at load (a read-only file mapping crashes it). Untouched pages stay
    // file-backed; only the rewritten pages become anonymous memory.
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) DIE("cannot open %s", path.c_str());
    struct stat st; fstat(fd, &st);
    void* p = mmap(nullptr, st.st_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) DIE("mmap %s", path.c_str());
    ENSURE(LiteRtCreateModelFromBuffer(env, p, st.st_size, &model));
    LiteRtOptions opts;
    ENSURE(LiteRtCreateOptions(&opts));
    ENSURE(LiteRtSetOptionsHardwareAccelerators(opts, kLiteRtHwAcceleratorCpu));
    if (LiteRtOpaqueOptions oo = cpu_options(threads, cache)) ENSURE(LiteRtAddOpaqueOptions(opts, oo));
    if (fused) {
      LiteRtCustomOpKernel k; void* ud = nullptr;
      i8_attn_kernel(attn_threads > 0 ? attn_threads : threads, &k, &ud);
      ENSURE(LiteRtAddCustomOpKernelOption(opts, "voxsum.i8_attention", 1, &k, ud));
    }
    ENSURE(LiteRtCreateCompiledModel(env, model, opts, &cm));
    fprintf(stderr, "[compiled] %s\n", path.c_str());
    LiteRtParamIndex n = 0;
    ENSURE(LiteRtGetNumModelSignatures(model, &n));
    for (LiteRtParamIndex si = 0; si < n; ++si) {
      LiteRtSignature sig; ENSURE(LiteRtGetModelSignature(model, si, &sig));
      const char* key = nullptr; ENSURE(LiteRtGetSignatureKey(sig, &key));
      if (!only.empty() && std::find(only.begin(), only.end(), std::string(key)) == only.end()) continue;
      Sig s; s.index = si;
      LiteRtParamIndex nin = 0, nout = 0;
      ENSURE(LiteRtGetNumSignatureInputs(sig, &nin));
      ENSURE(LiteRtGetNumSignatureOutputs(sig, &nout));
      for (LiteRtParamIndex i = 0; i < nin; ++i) {
        const char* nm = nullptr; ENSURE(LiteRtGetSignatureInputName(sig, i, &nm));
        s.in_names.push_back(nm);
        s.in.push_back(buffer(sig, si, i, true, nm));
      }
      for (LiteRtParamIndex i = 0; i < nout; ++i) {
        const char* nm = nullptr; ENSURE(LiteRtGetSignatureOutputName(sig, i, &nm));
        s.out_names.push_back(nm);
        s.out.push_back(buffer(sig, si, i, false, nm));
      }
      sigs[key] = s;
    }
  }

  LiteRtTensorBuffer buffer(LiteRtSignature sig, LiteRtParamIndex si, LiteRtParamIndex ti, bool in,
                            const char* name) {
    const bool is_kv = shared && !strncmp(name, "kv_cache_", 9);
    if (is_kv) {
      auto it = shared->find(name);
      if (it != shared->end()) return it->second;
    }
    LiteRtTensor t;
    ENSURE(in ? LiteRtGetSignatureInputTensorByIndex(sig, ti, &t) : LiteRtGetSignatureOutputTensorByIndex(sig, ti, &t));
    LiteRtRankedTensorType tt; ENSURE(LiteRtGetRankedTensorType(t, &tt));
    LiteRtTensorBufferRequirements req;
    ENSURE(in ? LiteRtGetCompiledModelInputBufferRequirements(cm, si, ti, &req)
              : LiteRtGetCompiledModelOutputBufferRequirements(cm, si, ti, &req));
    size_t bytes = 0; ENSURE(LiteRtGetTensorBufferRequirementsBufferSize(req, &bytes));
    LiteRtTensorBuffer b;
    ENSURE(LiteRtCreateManagedTensorBuffer(env, kLiteRtTensorBufferTypeHostMemory, &tt, bytes, &b));
    void* p; ENSURE(LiteRtLockTensorBuffer(b, &p, kLiteRtTensorBufferLockModeWrite));
    memset(p, 0, bytes); LiteRtUnlockTensorBuffer(b);
    if (is_kv) (*shared)[name] = b;
    return b;
  }
  Sig& sig(const std::string& k) {
    auto it = sigs.find(k); if (it == sigs.end()) DIE("no signature %s", k.c_str()); return it->second;
  }
  void run(Sig& s) {
    ENSURE(LiteRtRunCompiledModel(cm, s.index, s.in.size(), s.in.data(), s.out.size(), s.out.data()));
  }
};

static void* lockw(LiteRtTensorBuffer b) { void* p; ENSURE(LiteRtLockTensorBuffer(b, &p, kLiteRtTensorBufferLockModeWrite)); return p; }
static const void* lockr(LiteRtTensorBuffer b) { void* p; ENSURE(LiteRtLockTensorBuffer(b, &p, kLiteRtTensorBufferLockModeRead)); return p; }
static void unlock(LiteRtTensorBuffer b) { LiteRtUnlockTensorBuffer(b); }
static size_t bytes_of(LiteRtTensorBuffer b) { size_t n = 0; LiteRtGetTensorBufferSize(b, &n); return n; }

static void on_segv(int) {
#if !defined(__ANDROID__)
  void* bt[48]; int n = backtrace(bt, 48);
  backtrace_symbols_fd(bt, n, 2);
#endif
  _exit(139);
}

int main(int argc, char** argv) {
  signal(SIGSEGV, on_segv);
  // OpenMP threads of the attention op must not spin between ops: XNNPACK runs its own pool on
  // the same cores (measured on a Reno7: prefill 80 -> 130 tok/s). Must be set before the
  // OpenMP runtime starts.
  setenv("KMP_BLOCKTIME", "0", 0);
  setenv("OMP_WAIT_POLICY", "PASSIVE", 0);
  std::string dir, main_path, cache, ids_path, out_path;
  int ctx = 4096, threads = 8, max_new = 400, attn_threads = 0;
  bool fused = false;
  std::vector<int> stop = {106, 1};
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { if (i + 1 >= argc) DIE("missing value for %s", a.c_str()); return std::string(argv[++i]); };
    if (a == "--dir") dir = next();
    else if (a == "--main") main_path = next();
    else if (a == "--weight-cache") cache = next();
    else if (a == "--ids") ids_path = next();
    else if (a == "--out") out_path = next();
    else if (a == "--ctx") ctx = atoi(next().c_str());
    else if (a == "--threads") threads = atoi(next().c_str());
    else if (a == "--attn-threads") attn_threads = atoi(next().c_str());
    else if (a == "--max-new") max_new = atoi(next().c_str());
    else if (a == "--fused") fused = true;
    else if (a == "--stop") { stop.clear(); std::string s = next(); for (char* t = strtok(&s[0], ","); t; t = strtok(nullptr, ",")) stop.push_back(atoi(t)); }
    else DIE("unknown argument %s", a.c_str());
  }
  if (dir.empty() || ids_path.empty()) DIE("--dir and --ids are required");
  if (main_path.empty()) main_path = dir + "/Section10_TFLiteModel_tf_lite_prefill_decode.tflite";
  std::vector<int> ids;
  { FILE* f = fopen(ids_path.c_str(), "r"); if (!f) DIE("cannot open %s", ids_path.c_str());
    int v; while (fscanf(f, "%d", &v) == 1) ids.push_back(v); fclose(f); }
  if (ids.size() < 2) DIE("need at least 2 prompt tokens");

  // the cache length: 32003 in the graph, replaced at load
  std::vector<char> mn(sizeof(LiteRtMagicNumberConfigs) + sizeof(LiteRtMagicNumberConfig));
  auto* cfg = reinterpret_cast<LiteRtMagicNumberConfigs*>(mn.data());
  cfg->num_configs = 1;
  cfg->configs[0].magic_number = 32003;
  cfg->configs[0].target_number = ctx;
  cfg->configs[0].signature_prefix = "";   // all signatures (nullptr crashes the 2.x runtime)
  LiteRtEnvOption eo; eo.tag = kLiteRtEnvOptionTagMagicNumberConfigs;
  eo.value.type = kLiteRtAnyTypeVoidPtr; eo.value.ptr_value = cfg;
  LiteRtEnvironment env; ENSURE(LiteRtCreateEnvironment(getenv("MFA_NO_MAGIC") ? 0 : 1, &eo, &env));

  double t0 = now_s();
  std::map<std::string, LiteRtTensorBuffer> kv;
  Model emb(env, dir + "/Section2_TFLiteModel_tf_lite_embedder.tflite", threads, "", false, nullptr, {});
  Model ple(env, dir + "/Section3_TFLiteModel_tf_lite_per_layer_embedder.tflite", threads, "", false, nullptr, {});
  Model lm(env, main_path, threads, cache, fused, &kv, {"prefill_128", "decode"}, attn_threads);
  sample_mem();
  fprintf(stderr, "load %.1fs, %zu kv buffers, VmHWM %ld MB RssAnon %ld MB\n", now_s() - t0, kv.size(),
          status_kb("VmHWM:") / 1024, status_kb("RssAnon:") / 1024);

  Sig& es = emb.sig("embedder"); Sig& ps = ple.sig("per_layer_embedder");
  const size_t hid = bytes_of(es.out[0]) / 4, ple_n = bytes_of(ps.out[0]) / 4;
  double t_embed = 0, t_main = 0;
  auto embed = [&](int tok, float* e, float* pl) {
    const double t = now_s();
    *(int32_t*)lockw(es.in[0]) = tok; unlock(es.in[0]); emb.run(es);
    memcpy(e, lockr(es.out[0]), hid * 4); unlock(es.out[0]);
    *(int32_t*)lockw(ps.in[0]) = tok; unlock(ps.in[0]); ple.run(ps);
    memcpy(pl, lockr(ps.out[0]), ple_n * 4); unlock(ps.out[0]);
    t_embed += now_s() - t;
  };

  Sig& pf = lm.sig("prefill_128"); Sig& dc = lm.sig("decode");
  auto bind_kv_outputs = [&](Sig& s) {   // in place: the outputs of the caches are the inputs
    for (size_t i = 0; i < s.out_names.size(); ++i) {
      auto it = kv.find(s.out_names[i]);
      if (it != kv.end()) s.out[i] = it->second;
    }
  };
  bind_kv_outputs(pf); bind_kv_outputs(dc);
  const int T = 128;
  const int iE = pf.in_idx("embeddings"), iP = pf.in_idx("per_layer_embeddings"), iPos = pf.in_idx("input_pos"),
            iM = pf.in_idx("mask"), iPar = pf.in_idx("param_tensor");
  if (iE < 0 || iP < 0 || iPos < 0 || iM < 0 || iPar < 0) DIE("prefill signature inputs not found");
  const size_t C = bytes_of(pf.in[iM]) / T;
  fprintf(stderr, "cache length %zu (requested %d)\n", C, ctx);

  // prefill all prompt tokens but the last
  const int n_prompt = (int)ids.size() - 1;
  std::vector<float> e(hid), pl(ple_n);
  double tp = now_s();
  for (int start = 0; start < n_prompt; start += T) {
    const int n = std::min(T, n_prompt - start);
    float* E = (float*)lockw(pf.in[iE]); float* P = (float*)lockw(pf.in[iP]);
    memset(E, 0, bytes_of(pf.in[iE])); memset(P, 0, bytes_of(pf.in[iP]));
    for (int t = 0; t < n; ++t) {
      embed(ids[start + t], e.data(), pl.data());
      memcpy(E + (size_t)t * hid, e.data(), hid * 4); memcpy(P + (size_t)t * ple_n, pl.data(), ple_n * 4);
    }
    unlock(pf.in[iE]); unlock(pf.in[iP]);
    int32_t* pos = (int32_t*)lockw(pf.in[iPos]); memset(pos, 0, T * 4);
    for (int t = 0; t < n; ++t) pos[t] = start + t;
    unlock(pf.in[iPos]);
    uint8_t* M = (uint8_t*)lockw(pf.in[iM]); memset(M, 0, (size_t)T * C);
    for (int t = 0; t < n; ++t) memset(M + (size_t)t * C, 1, std::min<size_t>(start + t + 1, C));
    unlock(pf.in[iM]);
    int32_t* par = (int32_t*)lockw(pf.in[iPar]); memset(par, 0, bytes_of(pf.in[iPar]));
    par[0] = start; par[1] = start + n; par[2] = start + n; unlock(pf.in[iPar]);
    { const double t = now_s(); lm.run(pf); t_main += now_s() - t; }
    sample_mem();
  }
  const double prefill_s = now_s() - tp;

  const int dE = dc.in_idx("embeddings"), dP = dc.in_idx("per_layer_embeddings"), dPos = dc.in_idx("input_pos"),
            dM = dc.in_idx("mask"), dPar = dc.in_idx("param_tensor"), dL = dc.out_idx("logits");
  if (dE < 0 || dP < 0 || dPos < 0 || dM < 0 || dPar < 0 || dL < 0) DIE("decode signature names not found");
  const size_t vocab = bytes_of(dc.out[dL]) / 4;
  std::vector<int> gen;
  int tok = ids.back(), pos = n_prompt;
  double td = now_s();
  for (int step = 0; step < max_new && pos < (int)C; ++step, ++pos) {
    embed(tok, e.data(), pl.data());
    memcpy(lockw(dc.in[dE]), e.data(), hid * 4); unlock(dc.in[dE]);
    memcpy(lockw(dc.in[dP]), pl.data(), ple_n * 4); unlock(dc.in[dP]);
    *(int32_t*)lockw(dc.in[dPos]) = pos; unlock(dc.in[dPos]);
    uint8_t* M = (uint8_t*)lockw(dc.in[dM]); memset(M, 0, C); memset(M, 1, pos + 1); unlock(dc.in[dM]);
    int32_t* par = (int32_t*)lockw(dc.in[dPar]); memset(par, 0, bytes_of(dc.in[dPar]));
    par[0] = pos; par[1] = pos + 1; par[2] = pos + 1; unlock(dc.in[dPar]);
    { const double t = now_s(); lm.run(dc); t_main += now_s() - t; }
    const float* L = (const float*)lockr(dc.out[dL]);
    tok = (int)(std::max_element(L, L + vocab) - L);
    unlock(dc.out[dL]);
    sample_mem();
    gen.push_back(tok);
    if (std::find(stop.begin(), stop.end(), tok) != stop.end()) break;
  }
  const double decode_s = now_s() - td;
  fprintf(stderr, "prefill %d tokens in %.2fs (%.1f tok/s), decode %zu tokens in %.2fs (%.2f tok/s)\n",
          n_prompt, prefill_s, n_prompt / prefill_s, gen.size(), decode_s, gen.size() / decode_s);
  fprintf(stderr, "time in embedder+PLE %.2fs, in the main graph %.2fs\n", t_embed, t_main);
  fprintf(stderr, "peak VmHWM %ld MB, peak RssAnon %ld MB\n", status_kb("VmHWM:") / 1024, g_peak_anon / 1024);
  FILE* fo = out_path.empty() ? stdout : fopen(out_path.c_str(), "w");
  for (int g : gen) fprintf(fo, "%d ", g);
  fprintf(fo, "\n");
  if (fo != stdout) fclose(fo);
  return 0;
}
