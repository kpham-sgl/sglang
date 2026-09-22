"""Turn-2 with decode radix hit vs cold decode cache: greedy parity and accept length."""
import json, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

LB = f"http://127.0.0.1:{os.environ.get('LB_PORT', '8000')}"
DECODE = f"http://127.0.0.1:{os.environ.get('DECODE_PORT', '30200')}"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
OUT = sys.argv[2] if len(sys.argv) > 2 else "/scratch/repro-out/hit-vs-cold.json"
GSM8K = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
CACHE = "/scratch/repro-out/gsm8k_test.jsonl"

def post(url, body, timeout=600):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    return json.loads(data) if data.strip().startswith(b"{") else data.decode()

def gen(text, n):
    out = post(f"{LB}/generate", {"text": text, "sampling_params": {"temperature": 0.0, "max_new_tokens": n}})
    m = out["meta_info"]
    return out["text"], {"prompt": m["prompt_tokens"], "cached": m["cached_tokens"],
                         "completion": m["completion_tokens"], "verify_ct": m.get("spec_verify_ct")}

if not os.path.exists(CACHE):
    open(CACHE, "wb").write(urllib.request.urlopen(GSM8K, timeout=60).read())
qs = [json.loads(l)["question"] for l in open(CACHE)][200:200 + N]
p1 = [f"Problem {i}: {q}\nSolve it step by step, showing all work.\nSolution:" for i, q in enumerate(qs)]
with ThreadPoolExecutor(4) as ex:
    t1 = list(ex.map(lambda p: gen(p, 384), p1))
print("turn1 done", flush=True)
p2 = [p + o[0] + "\n\nNow verify the result by solving the problem a second, different way.\nVerification:" for p, o in zip(p1, t1)]
with ThreadPoolExecutor(4) as ex:
    hit = list(ex.map(lambda p: gen(p, 256), p2))
print("turn2(hit) done", flush=True)
if os.environ.get("SKIP_COLD") != "1":
    print("flush decode:", post(f"{DECODE}/flush_cache", {}), flush=True)
    with ThreadPoolExecutor(4) as ex:
        cold = list(ex.map(lambda p: gen(p, 256), p2))
else:
    cold = hit
def acc(rs): return sum(r[1]["completion"] for r in rs) / max(1, sum(r[1]["verify_ct"] or 0 for r in rs))
rows = [{"conv": i, "t1": a[1], "hit": h[1], "cold": c[1], "text_equal": h[0] == c[0]}
        for i, (a, h, c) in enumerate(zip(t1, hit, cold))]
summary = {"n": N, "parity": sum(r["text_equal"] for r in rows), "accept_t1": acc(t1),
           "accept_hit": acc(hit), "accept_cold": acc(cold),
           "mean_cached_hit": sum(r["hit"]["cached"] for r in rows) / N,
           "mean_cached_cold": sum(r["cold"]["cached"] for r in rows) / N}
print("SUMMARY", json.dumps(summary))
json.dump({"rows": rows, "summary": summary}, open(OUT, "w"), indent=1)
