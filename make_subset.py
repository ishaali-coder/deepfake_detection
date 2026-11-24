"""Make a smaller, still-balanced subset of frames_balanced.csv by keeping only a few
spread-out frames per video.

Each FAKE video keeps k frames; each REAL video keeps k x (number of fake methods)
frames, so real and fake stay balanced (there are 4x more fake videos than real).
After that, each split is trimmed so real = fake and every fake method is equal.

Usage:  python make_subset.py --data data --k 3
Writes data/frames_small.csv (same columns and relative image paths as the input).
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--data", default="data")
p.add_argument("--k", type=int, default=3, help="frames kept per fake video (train)")
p.add_argument("--k_eval", type=int, default=None, help="frames kept per fake video in val/test (default: same as --k)")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
k_eval = a.k_eval or a.k

data = Path(a.data)
rows = list(csv.DictReader(open(data / "frames_balanced.csv")))
fieldnames = list(rows[0].keys())

by_video = defaultdict(list)
for r in rows:
    by_video[(r["split"], r["method"], r["video"])].append(r)


def spread(frames, n):
    """n frames evenly spaced through the video's kept frames (all of them if fewer than n)."""
    frames = sorted(frames, key=lambda r: int(r["frame"]))
    if len(frames) <= n:
        return frames
    if n == 1:
        return [frames[len(frames) // 2]]
    idx = sorted({round(i * (len(frames) - 1) / (n - 1)) for i in range(n)})
    return [frames[i] for i in idx]


rng = random.Random(a.seed)
out = []
print(f"{'split':>6} {'real':>7} {'fake':>7}  {'per fake method':>15}  frames/fake video  frames/real video")
for split in ["train", "val", "test"]:
    k = a.k if split == "train" else k_eval
    methods = sorted({m for (s, m, _) in by_video if s == split and m != "original"})
    k_real = k * len(methods)

    pool = {m: [] for m in methods}
    real_pool = []
    for (s, m, v), frames in by_video.items():
        if s != split:
            continue
        if m == "original":
            real_pool += spread(frames, k_real)
        else:
            pool[m] += spread(frames, k)

    # videos that lost frames to filtering can leave the counts slightly uneven: trim to equal
    per_method = min(min(len(v) for v in pool.values()), len(real_pool) // len(methods))
    for m in methods:
        out += rng.sample(pool[m], per_method)
    out += rng.sample(real_pool, per_method * len(methods))
    total = per_method * len(methods)
    print(f"{split:>6} {total:>7} {total:>7}  {per_method:>15}  {k:>17}  {k_real:>17}")

with open(data / "frames_small.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(out)
print(f"\nWrote {data / 'frames_small.csv'} ({len(out)} frames, seed {a.seed}). "
      f"Zip it with: python make_zip.py --csv frames_small.csv")
