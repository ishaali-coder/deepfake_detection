#!/usr/bin/env python
"""
Clean and prepare FaceForensics++ (c23) for training.

Step 1: extract
    - builds a label table from the official split files (pairs of original IDs)
    - samples evenly spaced frames from every video (more from real than fake)
    - detects the face, crops it with a margin, resizes it
    - drops frames with no face / small face / blurry face (and optionally
      multiple faces), and logs every drop
    - saves crops + one meta.json per video (so an interrupted run can resume)

Step 2: balance
    - within each split, makes real and fake frame counts equal, and the four
      fake methods equal inside the fake class
    - writes frames_balanced.csv (the file your Dataset class should read)

Usage:
    python prepare_data.py extract --root ff --splits splits --out data --limit_pairs 3
    python prepare_data.py extract --root ff --splits splits --out data
    python prepare_data.py balance --out data

Requires: pip install opencv-python numpy tqdm
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

FAKE_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
SPLITS = ["train", "val", "test"]
COMPRESSION = "c23"
DETECT_MAX_SIDE = 640  # detect on a downscaled copy for speed
DROP_REASONS = ["read_error", "no_face", "multi_face", "small_face", "blurry"]


# --------------------------------------------------------------------------
# Step 1a: label table from the official splits
# --------------------------------------------------------------------------
def norm_id(x):
    return f"{x:03d}" if isinstance(x, int) else str(x)


def load_pairs(splits_dir, limit_pairs=None):
    pairs = {}
    for s in SPLITS:
        path = Path(splits_dir) / f"{s}.json"
        data = json.load(open(path))
        out = []
        for entry in data:
            if len(entry) != 2:
                raise ValueError(f"{path}: expected [id_a, id_b] pairs, got {entry!r}")
            out.append((norm_id(entry[0]), norm_id(entry[1])))
        pairs[s] = out[:limit_pairs] if limit_pairs else out
    return pairs


def build_table(root, pairs):
    root = Path(root)
    rows = []
    for split, plist in pairs.items():
        for a, b in plist:
            for v in (a, b):
                rows.append(dict(
                    video=v, label=0, method="original", split=split,
                    path=str(root / "original_sequences" / "youtube" / COMPRESSION / "videos" / f"{v}.mp4")))
            for m in FAKE_METHODS:
                for v in (f"{a}_{b}", f"{b}_{a}"):
                    rows.append(dict(
                        video=v, label=1, method=m, split=split,
                        path=str(root / "manipulated_sequences" / m / COMPRESSION / "videos" / f"{v}.mp4")))
    return rows


def check_no_leakage(rows):
    """Every original ID must live in exactly one split, and every fake must be in
    the same split as both of its source IDs."""
    id_split = {}
    for r in rows:
        if r["label"] == 0:
            if r["video"] in id_split and id_split[r["video"]] != r["split"]:
                raise AssertionError(f"original {r['video']} appears in two splits")
            id_split[r["video"]] = r["split"]
    for r in rows:
        if r["label"] == 1:
            a, b = r["video"].split("_")
            if id_split[a] != r["split"] or id_split[b] != r["split"]:
                raise AssertionError(f"fake {r['video']} is in a different split than its sources")


# --------------------------------------------------------------------------
# Step 1b: per-video processing (runs in worker processes)
# --------------------------------------------------------------------------
_DET = None
_DET_KIND = None


def init_worker(yunet_model):
    global _DET, _DET_KIND
    cv2.setNumThreads(1)
    if yunet_model:
        _DET = cv2.FaceDetectorYN.create(yunet_model, "", (320, 320), 0.8, 0.3, 5000)
        _DET_KIND = "yunet"
    else:
        _DET = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if _DET.empty():
            raise RuntimeError("Could not load the Haar cascade from your OpenCV install. "
                               "Run: pip install opencv-python  (not the -headless build), "
                               "or pass --yunet_model path/to/face_detection_yunet_2023mar.onnx")
        _DET_KIND = "haar"


def detect_faces(frame):
    """Returns a list of (x, y, w, h) boxes in original-frame coordinates."""
    h, w = frame.shape[:2]
    ds = min(1.0, DETECT_MAX_SIDE / max(h, w))
    small = cv2.resize(frame, (int(w * ds), int(h * ds))) if ds < 1.0 else frame
    if _DET_KIND == "yunet":
        _DET.setInputSize((small.shape[1], small.shape[0]))
        _, faces = _DET.detect(small)
        boxes = [] if faces is None else [tuple(f[:4]) for f in faces]
    else:
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        boxes = _DET.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        boxes = [tuple(b) for b in boxes]
    return [tuple(float(v) / ds for v in b) for b in boxes]


def face_crop(frame, box, scale, size):
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    side = max(8, int(round(max(w, h) * scale)))
    padded = cv2.copyMakeBorder(frame, side, side, side, side, cv2.BORDER_CONSTANT, value=0)
    x0 = max(0, int(round(cx - side / 2)) + side)
    y0 = max(0, int(round(cy - side / 2)) + side)
    crop = padded[y0:y0 + side, x0:x0 + side]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def process_video(task):
    out_root = Path(task["out"])
    rel_dir = Path("crops") / task["split"] / task["method"] / task["video"]
    out_dir = out_root / rel_dir
    meta_path = out_dir / "meta.json"
    if meta_path.exists():  # resume support
        return json.load(open(meta_path))
    out_dir.mkdir(parents=True, exist_ok=True)

    drops = Counter()
    kept = []
    cap = cv2.VideoCapture(task["path"])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    n_want = task["n_frames"]

    if total <= 0:
        sampled = n_want
        drops["read_error"] += n_want
    else:
        idx = sorted(set(np.linspace(0, total - 1, n_want).astype(int).tolist()))
        sampled = len(idx)
        want, last, i = set(idx), idx[-1], 0
        while i <= last:
            if not cap.grab():
                break
            if i in want:
                ok, frame = cap.retrieve()
                if not ok:
                    drops["read_error"] += 1
                else:
                    boxes = detect_faces(frame)
                    if not boxes:
                        drops["no_face"] += 1
                    elif task["drop_multi"] and len(boxes) > 1:
                        drops["multi_face"] += 1
                    else:
                        box = max(boxes, key=lambda b: b[2] * b[3])
                        face_px = int(min(box[2], box[3]))
                        if face_px < task["min_face"]:
                            drops["small_face"] += 1
                        else:
                            crop = face_crop(frame, box, task["scale"], task["size"])
                            blur = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                            if blur < task["blur_thresh"]:
                                drops["blurry"] += 1
                            else:
                                fname = f"{task['video']}_f{i:05d}.{task['ext']}"
                                params = [cv2.IMWRITE_JPEG_QUALITY, 95] if task["ext"] == "jpg" else []
                                cv2.imwrite(str(out_dir / fname), crop, params)
                                kept.append(dict(path=(rel_dir / fname).as_posix(), frame=i,
                                                 n_faces=len(boxes), face_px=face_px, blur=round(blur, 1)))
            i += 1
        drops["read_error"] += sum(1 for j in want if j >= i)
    cap.release()

    meta = dict(video=task["video"], split=task["split"], method=task["method"], label=task["label"],
                sampled=sampled, kept=kept, drops=dict(drops))
    json.dump(meta, open(meta_path, "w"))
    return meta


# --------------------------------------------------------------------------
# Step 1: extract
# --------------------------------------------------------------------------
def cmd_extract(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = build_table(a.root, load_pairs(a.splits, a.limit_pairs))
    check_no_leakage(rows)
    print("Leakage check passed: every original ID is in one split and every fake "
          "shares a split with both of its sources.")

    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "label", "method", "split", "path"])
        w.writeheader()
        w.writerows(rows)

    tasks, missing = [], Counter()
    for r in rows:
        if not Path(r["path"]).exists():
            missing[(r["split"], r["method"])] += 1
            continue
        tasks.append(dict(r, out=str(out), ext=a.ext, size=a.size, scale=a.scale,
                          min_face=a.min_face, blur_thresh=a.blur_thresh, drop_multi=a.drop_multi,
                          n_frames=a.real_frames if r["label"] == 0 else a.fake_frames))
    print(f"{len(rows)} videos in label table, {len(tasks)} found on disk, "
          f"{sum(missing.values())} missing")
    if missing:
        print("Missing by (split, method):", dict(missing))

    det = "YuNet" if a.yunet_model else "Haar cascade"
    print(f"Face detector: {det}. Processing with {a.workers} workers...")
    metas = []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=init_worker,
                             initargs=(a.yunet_model,)) as pool:
        for m in tqdm(pool.map(process_video, tasks, chunksize=4), total=len(tasks)):
            metas.append(m)

    # frames.csv (every kept crop) and a summary of what was dropped
    stats = defaultdict(lambda: dict(videos=0, sampled=0, kept=0, **{r: 0 for r in DROP_REASONS}))
    with open(out / "frames.csv", "w", newline="") as f:
        cols = ["path", "video", "label", "method", "split", "frame", "n_faces", "face_px", "blur"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for m in metas:
            s = stats[(m["split"], m["method"])]
            s["videos"] += 1
            s["sampled"] += m["sampled"]
            s["kept"] += len(m["kept"])
            for reason, n in m["drops"].items():
                s[reason] += n
            for k in m["kept"]:
                w.writerow(dict(k, video=m["video"], label=m["label"], method=m["method"], split=m["split"]))

    cols = ["split", "method", "videos", "sampled", "kept"] + DROP_REASONS
    with open(out / "summary_extract.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for (split, method), s in sorted(stats.items()):
            w.writerow([split, method] + [s[c] for c in cols[2:]])
    print("\n" + "  ".join(f"{c:>12}" for c in cols))
    for (split, method), s in sorted(stats.items()):
        print("  ".join(f"{v:>12}" for v in [split, method] + [s[c] for c in cols[2:]]))
    print(f"\nWrote {out / 'frames.csv'} and {out / 'summary_extract.csv'}. "
          f"Look at the numbers above before balancing: a method or class that lost "
          f"far more frames than the others means a bias to investigate.")


# --------------------------------------------------------------------------
# Step 2: balance
# --------------------------------------------------------------------------
def cmd_balance(a):
    out = Path(a.out)
    rng = random.Random(a.seed)
    rows = list(csv.DictReader(open(out / "frames.csv")))
    by = defaultdict(list)
    for r in rows:
        by[(r["split"], r["method"])].append(r)

    selected = []
    print(f"{'split':>6} {'real_before':>12} {'fake_before':>12} {'per_method':>11} {'real_after':>11} {'fake_after':>11}")
    for split in SPLITS:
        real = by.get((split, "original"), [])
        fake = {m: by.get((split, m), []) for m in FAKE_METHODS if by.get((split, m))}
        if not real or not fake:
            print(f"{split:>6}: skipped (real={len(real)}, fake methods={len(fake)})")
            continue
        n_methods = len(fake)
        per_method = min(min(len(v) for v in fake.values()), len(real) // n_methods)
        picks = []
        for m in fake:
            picks += rng.sample(fake[m], per_method)
        picks += rng.sample(real, per_method * n_methods)
        selected += picks
        print(f"{split:>6} {len(real):>12} {sum(len(v) for v in fake.values()):>12} "
              f"{per_method:>11} {per_method * n_methods:>11} {per_method * n_methods:>11}")

    with open(out / "frames_balanced.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(selected)
    print(f"\nWrote {out / 'frames_balanced.csv'} ({len(selected)} frames, seed {a.seed}). "
          f"Read this file in your Dataset class, not frames.csv.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="sample frames, crop faces, filter")
    e.add_argument("--root", required=True, help="folder containing original_sequences/ and manipulated_sequences/")
    e.add_argument("--splits", required=True, help="folder with train.json, val.json, test.json")
    e.add_argument("--out", default="data")
    e.add_argument("--real_frames", type=int, default=40, help="frames sampled per real video")
    e.add_argument("--fake_frames", type=int, default=10, help="frames sampled per fake video")
    e.add_argument("--size", type=int, default=224, help="output crop size in pixels")
    e.add_argument("--scale", type=float, default=1.3, help="crop side = scale x face box size")
    e.add_argument("--min_face", type=int, default=64, help="drop faces smaller than this many pixels")
    e.add_argument("--blur_thresh", type=float, default=20.0, help="drop crops with Laplacian variance below this")
    e.add_argument("--drop_multi", action="store_true", help="drop frames with more than one detected face")
    e.add_argument("--ext", choices=["png", "jpg"], default="png")
    e.add_argument("--workers", type=int, default=4)
    e.add_argument("--limit_pairs", type=int, default=None, help="only use the first N pairs per split (for testing)")
    e.add_argument("--yunet_model", default=None, help="path to YuNet .onnx file (default: Haar cascade)")
    e.set_defaults(func=cmd_extract)

    b = sub.add_parser("balance", help="equalize real/fake and per-method counts in each split")
    b.add_argument("--out", default="data")
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=cmd_balance)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
