#!/usr/bin/env python3
"""OpenAI-compatible chat server over tools/litert_fused mfa_engine (--serve), for evaluation harnesses.

Each worker is one mfa_engine process (its own KV cache). A request with "id_slot" goes to the
worker slot % N, so a reading session keeps extending the prompt already in that worker's cache;
the engine reuses the longest cached prefix. The chat template is the one stored in the
.litertlm metadata (what litert-lm serve applies); tokens come from the Gemma-4 HF tokenizer.
Stop strings are checked here on the decoded text; the engine is told to cancel.

  python3 serve_openai.py --engine build/mfa_engine --dir /dev/shm/e4bfin.dir --main /dev/shm/e4b_fused.tflite --fused \
      --weight-cache /dev/shm/mfa/e4b_fused.wcache --ctx 4096 --workers 2 --threads 8 --port 9379
"""
import argparse, json, re, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from transformers import AutoTokenizer

ENGINE = "mfa_engine"


def litertlm_template(unpack_dir):
    pb = open(f"{unpack_dir}/LlmMetadataProto.pbtext", encoding="utf-8").read()
    raw = re.search(r'jinja_prompt_template: "((?:[^"\\]|\\.)*)"', pb, re.S).group(1)
    return raw.encode("utf-8").decode("unicode_escape").encode("latin-1").decode("utf-8")


class Worker:
    def __init__(self, i, a):
        cmd = [a.engine, "--serve", "--dir", a.dir, "--ctx", str(a.ctx), "--threads", str(a.threads)]
        if a.main:
            cmd += ["--main", a.main]
        if a.fused:
            cmd += ["--fused"]
        if a.weight_cache:
            cmd += ["--weight-cache", a.weight_cache]
        self.log = open(f"{a.log_prefix}.{i}.log", "w")
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                  text=True, bufsize=1)
        line = self.p.stdout.readline()
        if not line.startswith("READY"):
            sys.exit(f"worker {i} failed to start: {line!r}")
        self.ctx = int(line.split()[1])
        self.lock = threading.Lock()

    def generate(self, ids, max_new, temp, top_k, top_p, seed, on_token):
        """Stream tokens to on_token(id) -> bool (True: cancel). Returns the D line fields."""
        self.p.stdin.write(f"R {max_new} {temp} {top_k} {top_p} {seed} {len(ids)} {' '.join(map(str, ids))}\n")
        self.p.stdin.flush()
        cancelled = False
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine exited")
            if line.startswith("T "):
                if not cancelled and on_token(int(line.split()[1])):
                    cancelled = True
                    self.p.stdin.write("C\n")
                    self.p.stdin.flush()
            elif line.startswith("D "):
                _, reason, prefilled, reused, ps, ds = line.split()
                return reason, int(prefilled), int(reused), float(ps), float(ds)
            elif line.startswith("E "):
                raise ValueError(line[2:].strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=ENGINE)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--main")
    ap.add_argument("--fused", action="store_true")
    ap.add_argument("--weight-cache")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--tokenizer", default="google/gemma-4-E4B-it")
    ap.add_argument("--port", type=int, default=9379)
    ap.add_argument("--log-prefix", default="logs/mfa_server")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    template = litertlm_template(a.dir)
    # the engine stops on these ids (model metadata); never part of the text
    stop_ids = {1, 50, 106}
    # the cache of a worker is reused only if each worker is created once; a pending
    # request waits for its worker
    workers = [Worker(i, a) for i in range(a.workers)]
    rr = [0]
    rr_lock = threading.Lock()
    print(f"{a.workers} workers ready, cache {workers[0].ctx}", flush=True)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                return self.reply(200, {"object": "list", "data": [{"id": "mfa", "object": "model"}]})
            self.reply(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.rstrip("/").endswith("/chat/completions"):
                return self.reply(404, {"error": "not found"})
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            kw = req.get("chat_template_kwargs") or {}
            kw.setdefault("enable_thinking", False)
            text = tok.apply_chat_template(req["messages"], tokenize=False, add_generation_prompt=True,
                                           chat_template=template, **kw)
            ids = tok(text, add_special_tokens=False).input_ids
            slot = req.get("id_slot")
            if slot is None or slot < 0:
                with rr_lock:
                    slot = rr[0]
                    rr[0] += 1
            w = workers[slot % len(workers)]
            stops = req.get("stop") or []
            stops = [stops] if isinstance(stops, str) else stops
            max_new = int(req.get("max_tokens") or 1024)
            max_new = max(1, min(max_new, w.ctx - len(ids)))
            temp = float(req.get("temperature", 0.2))
            gen, state = [], {"cut": None}

            def on_token(t):
                if t in stop_ids:
                    return False
                gen.append(t)
                if stops:
                    s = tok.decode(gen, skip_special_tokens=False)
                    hits = [s.find(x) for x in stops if x in s]
                    if hits:
                        state["cut"] = min(hits)
                        return True
                return False

            t0 = time.time()
            with w.lock:
                try:
                    reason, prefilled, reused, ps, ds = w.generate(
                        ids, max_new, temp, a.top_k, a.top_p, int(req.get("seed", time.time_ns() % 2**31)), on_token)
                except ValueError as e:
                    return self.reply(400, {"error": {"message": str(e), "type": "exceed_context_size_error",
                                                      "n_prompt_tokens": len(ids), "n_ctx": w.ctx}})
            out = tok.decode(gen, skip_special_tokens=False)
            finish = "length"
            if state["cut"] is not None:
                out, finish = out[:state["cut"]], "stop"
            elif reason == "stop":
                finish = "stop"
            self.reply(200, {
                "id": f"mfa-{time.time_ns()}", "object": "chat.completion", "model": req.get("model", "mfa"),
                "choices": [{"index": 0, "finish_reason": finish,
                             "message": {"role": "assistant", "content": out}}],
                "usage": {"prompt_tokens": len(ids), "completion_tokens": len(gen),
                          "total_tokens": len(ids) + len(gen)},
                # llama-server style: what was actually computed (the cached prefix is reused)
                "timings": {"prompt_n": prefilled, "cache_n": reused, "predicted_n": len(gen),
                            "prompt_ms": ps * 1e3, "predicted_ms": ds * 1e3, "wall_ms": (time.time() - t0) * 1e3}})

    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
