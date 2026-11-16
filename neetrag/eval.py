"""Run the evaluation set and print a score table.  python eval.py"""
import json, time
from collections import defaultdict
from pathlib import Path

import neetrag as n

ROWS = {r["id"]: r for r in n.load_rows()}

def check(case):
    k, e = case["kind"], case["expect"]
    if k == "predict":
        try:
            opts = n.predict(**case["args"])
        except ValueError:
            return e == "error", "ValueError"
        if e == "empty":
            return not opts, f"{len(opts)} options"
        return e == "valid" and bool(opts) and all(ROWS[o["id"]]["closing_rank"] >= case["args"]["rank"] for o in opts), f"{len(opts)} options"
    r = n.ask(case["q"])
    s = r["status"]
    if k == "answer":
        ok = s == "answer" and e["contains"].lower() in r["text"].replace(",", "").lower() and e["cite"] in r["cites"]
    elif k == "refuse":
        ok = s in ("refused", "blocked")
    else:  # advice: must give the no-guarantee reply, not an answer
        ok = s == "advice" and "can't give guarantees" in r["text"]
    return ok, f"{s}: {r['why'] or r['text'][:60]}"

if __name__ == "__main__":
    cases = json.loads((Path(__file__).parent / "eval_set.json").read_text())
    tally, fails, times = defaultdict(lambda: [0, 0]), [], []
    for c in cases:
        t = time.time()
        ok, info = check(c)
        times.append(time.time() - t)
        tally[c["kind"]][0] += ok
        tally[c["kind"]][1] += 1
        if not ok:
            fails.append((c["id"], c.get("q") or c["args"], info))
    print(f"\nmodel={n.CHAT_MODEL}  embed={n.EMBED_MODEL}  min_cos={n.MIN_COS}")
    print(f"{'kind':<10}{'pass':>6}{'total':>7}")
    for k, (p, t) in tally.items():
        print(f"{k:<10}{p:>6}{t:>7}")
    P, T = sum(v[0] for v in tally.values()), sum(v[1] for v in tally.values())
    print(f"{'ALL':<10}{P:>6}{T:>7}   avg {sum(times)/len(times):.1f}s/case")
    for f in fails:
        print("FAIL", *f)
