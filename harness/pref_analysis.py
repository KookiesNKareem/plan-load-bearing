import json


def load(path):
    return json.load(open(path))


for name, bind_path, uns_path in (
        ("QWEN", "data/qwen_v2_full.json", "data/unseeded_qwen_v2.json"),):
    bind = {r["task_id"]: r for r in load(bind_path)["rows"] if r["status"] == "ok"}
    uns = {r["task_id"]: r for r in load(uns_path)["rows"]}
    bins = {}
    pts = []
    for tid, u in uns.items():
        if tid not in bind:
            continue
        us = [x for x in u["realized"] if x in ("A", "B")]
        if len(us) < 2:
            continue
        p = sum(1 for x in us if x == "A") / len(us)
        b = bind[tid]["realized_all"]
        ovA = [1 if v == "B" else 0 for k, v in b.items() if k.startswith("A") and v in ("A", "B")]
        ovB = [1 if v == "A" else 0 for k, v in b.items() if k.startswith("B") and v in ("A", "B")]
        pts.append((tid, p, ovA, ovB))
        key = "pure_A" if p == 1 else ("pure_B" if p == 0 else "mixed")
        d = bins.setdefault(key, {"n": 0, "ovA": [], "ovB": []})
        d["n"] += 1
        d["ovA"] += ovA
        d["ovB"] += ovB
    print(f"== {name}: {len(pts)} tasks with >=2 scored unseeded rollouts")
    for key in ("pure_A", "mixed", "pure_B"):
        if key not in bins:
            continue
        d = bins[key]
        ra = round(sum(d["ovA"]) / len(d["ovA"]), 3) if d["ovA"] else None
        rb = round(sum(d["ovB"]) / len(d["ovB"]), 3) if d["ovB"] else None
        print(f"  {key:7s} n={d['n']:2d} | override-under-A-seed: {ra} ({len(d['ovA'])} rolls)"
              f" | override-under-B-seed: {rb} ({len(d['ovB'])} rolls)")
    if name == "QWEN":
        theater = ["0", "14", "45", "52", "75", "77", "82", "85"]
        out = []
        for tid in theater:
            if tid not in uns:
                continue
            us = [x for x in uns[tid]["realized"] if x in ("A", "B")]
            out.append((tid, round(sum(1 for x in us if x == "A") / len(us), 2) if us else "unscored"))
        print("  theater tasks unseeded P(A):", out)
