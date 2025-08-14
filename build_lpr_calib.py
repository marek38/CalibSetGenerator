#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
import sys
import csv
import glob
import argparse
import random
import numpy as np
from datetime import datetime
from typing import Tuple, Optional, List

# --- cieľ pre LPRNet calib_set ---
TARGET_W, TARGET_H = 300, 75  # width x height (RGB)

# heuristiky pre auto-detekciu
ASPECT_MIN, ASPECT_MAX = 2.0, 6.0       # typický pomer ŠPZ ~4:1
MIN_REL_AREA, MAX_REL_AREA = 0.01, 0.6  # plocha boxu vs. plocha obrázka

def imread_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img

def resize_plate_bgr_to_rgb(crop_bgr: np.ndarray) -> np.ndarray:
    # LPRNet má normalizáciu on‑chip → stačí len BGR->RGB a resize
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

def guess_plate_bbox(img_bgr: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    """
    Jednoduchá heuristika:
      - Canny hrany + dilatácia
      - kontúry filtrované podľa pomeru strán a relatívnej plochy
      - skórovanie podľa blízkosti k 4:1, veľkosti a kontrastu
    Vráti (x1, y1, x2, y2) alebo None.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)

    v = float(np.median(gray))
    lo = max(0, int(0.66 * v))
    hi = min(255, int(1.33 * v))
    edges = cv2.Canny(gray, lo, hi)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, k, iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1e9
    img_area = float(w * h)

    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area < img_area * MIN_REL_AREA or area > img_area * MAX_REL_AREA:
            continue
        aspect = bw / float(bh)
        if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
            continue

        # “rektangularita”
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        rect_like = 1.0 if len(approx) in (4, 5) else 0.0

        # lokálny kontrast
        xs1, ys1 = max(0, x - 4), max(0, y - 4)
        xs2, ys2 = min(w, x + bw + 4), min(h, y + bh + 4)
        roi = gray[ys1:ys2, xs1:xs2]
        if roi.size == 0:
            continue
        contrast = float(np.std(roi))

        # skóre – preferuj blízko 4:1 a rozumnú relatívnu veľkosť (~8 % plochy)
        aspect_score = -abs(aspect - (TARGET_W / TARGET_H))
        size_score = -abs((area / img_area) - 0.08)
        score = 2.0 * aspect_score + 1.5 * size_score + 0.003 * contrast + 0.5 * rect_like

        if score > best_score:
            best_score = score
            best = (x, y, x + bw, y + bh)

    return best

# ---------- Interaktívny výber myšou ----------
class BoxSelector:
    def __init__(self, window_name: str, img: np.ndarray, init_box=None):
        self.win = window_name
        self.img = img.copy()
        self.preview = img.copy()
        self.drag = False
        self.x0 = self.y0 = self.x1 = self.y1 = 0
        if init_box:
            self.x0, self.y0, self.x1, self.y1 = init_box
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, min(1280, img.shape[1]), min(720, img.shape[0]))
        cv2.setMouseCallback(self.win, self.on_mouse)
        self.draw()

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag = True
            self.x0, self.y0 = x, y
            self.x1, self.y1 = x, y
        elif event == cv2.EVENT_MOUSEMOVE and self.drag:
            self.x1, self.y1 = x, y
            self.draw()
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag = False
            self.x1, self.y1 = x, y
            self.draw()

    def current_box(self):
        x1, x2 = sorted([self.x0, self.x1])
        y1, y2 = sorted([self.y0, self.y1])
        return x1, y1, x2, y2

    def draw(self):
        self.preview = self.img.copy()
        x1, y1, x2, y2 = self.current_box()
        cv2.rectangle(self.preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow(self.win, self.preview)

def save_crop(img_bgr: np.ndarray, box: Tuple[int,int,int,int], out_path: str):
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h, y2))
    if x2 - x1 < 5 or y2 - y1 < 5:
        raise RuntimeError("Box too small")
    crop = img_bgr[y1:y2, x1:x2]
    resized_rgb = resize_plate_bgr_to_rgb(crop)
    # uložíme ako JPG (imwrite chce BGR)
    save_bgr = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, save_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

def collect_images(src_dir: str, exts: Tuple[str,...]=(".jpg",".jpeg",".png",".bmp",".webp")) -> List[str]:
    files: List[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(src_dir, f"**/*{ext}"), recursive=True))
    return sorted(files)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def process_auto(files: List[str], out_dir: str, csv_path: str, limit: Optional[int] = None):
    ensure_dir(out_dir)
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["out_file","src_file","x1","y1","x2","y2","mode","ts"])
        count = 0
        for src in files:
            if limit and count >= limit: break
            try:
                img = imread_rgb(src)
                box = guess_plate_bbox(img)
                if not box:
                    print(f"[SKIP] {os.path.basename(src)} : no candidate")
                    continue
                out_name = f"plate_{count:06d}.jpg"
                save_crop(img, box, os.path.join(out_dir, out_name))
                wr.writerow([out_name, src, *box, "auto", datetime.utcnow().isoformat()])
                count += 1
                print(f"[OK] auto -> {out_name}")
            except Exception as e:
                print(f"[WARN] {src}: {e}")
    print(f"Saved {count} images to {out_dir}\nLog: {csv_path}")

def process_fixed(files: List[str], out_dir: str, csv_path: str, y1:int, y2:int, x1:int, x2:int, limit: Optional[int] = None):
    ensure_dir(out_dir)
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["out_file","src_file","x1","y1","x2","y2","mode","ts"])
        count = 0
        box = (x1, y1, x2, y2)
        for src in files:
            if limit and count >= limit: break
            try:
                img = imread_rgb(src)
                out_name = f"plate_{count:06d}.jpg"
                save_crop(img, box, os.path.join(out_dir, out_name))
                wr.writerow([out_name, src, *box, "fixed", datetime.utcnow().isoformat()])
                count += 1
                print(f"[OK] fixed -> {out_name}")
            except Exception as e:
                print(f"[WARN] {src}: {e}")
    print(f"Saved {count} images to {out_dir}\nLog: {csv_path}")

def process_gui(files: List[str], out_dir: str, csv_path: str, limit: Optional[int] = None):
    ensure_dir(out_dir)
    count = 0
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["out_file","src_file","x1","y1","x2","y2","mode","ts"])
        for src in files:
            if limit and count >= limit: break
            img = imread_rgb(src)
            guess = guess_plate_bbox(img)

            title = "PlateCrop – [ENTER]=Accept  [SPACE]=Draw  [N]=Skip  [Q]=Quit"
            disp = img.copy()
            if guess:
                x1,y1,x2,y2 = guess
                cv2.rectangle(disp, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.imshow(title, disp)
            key = cv2.waitKey(0) & 0xFF

            if key in (ord('q'), 27):
                break
            if key in (ord('n'), ord('s')):
                continue

            if key in (13, 10) and guess:
                box = guess
                mode = "auto"
            else:
                sel = BoxSelector(title, img, init_box=guess)
                while True:
                    k2 = cv2.waitKey(20) & 0xFF
                    if k2 in (13, 10):  # Enter
                        break
                    if k2 in (27, ord('q')):
                        box = None
                        break
                cv2.destroyWindow(title)
                if not sel: 
                    continue
                box = sel.current_box()
                mode = "manual"

            if not box:
                continue
            try:
                out_name = f"plate_{count:06d}.jpg"
                save_crop(img, box, os.path.join(out_dir, out_name))
                wr.writerow([out_name, src, *box, mode, datetime.utcnow().isoformat()])
                count += 1
                print(f"[OK] {mode} -> {out_name}")
            except Exception as e:
                print(f"[WARN] {src}: {e}")

        cv2.destroyAllWindows()
    print(f"Saved {count} images to {out_dir}\nLog: {csv_path}")

def main():
    ap = argparse.ArgumentParser(
        description="Build LPR calibration set (RGB 300x75) from full-scene images."
    )
    ap.add_argument("--src", required=True, help="Zdrojový priečinok s obrázkami.")
    ap.add_argument("--dst", required=True, help="Cieľový priečinok pre 300x75 JPG.")
    ap.add_argument("--mode", choices=["auto","gui","fixed"], default="auto",
                    help="Spôsob výberu boxu ŠPZ.")
    ap.add_argument("--limit", type=int, default=None, help="Max. počet výstupov.")
    ap.add_argument("--shuffle", action="store_true", help="Náhodné poradie vstupov.")
    ap.add_argument("--exts", default=".jpg,.jpeg,.png,.bmp,.webp",
                    help="Prípony súborov, oddelené čiarkou.")
    ap.add_argument("--csv", default=None, help="Cesta k CSV logu (default: <dst>/calib_log.csv).")

    # pre --mode fixed
    ap.add_argument("--y1", type=int, help="Top Y súradnica pre fixed režim.")
    ap.add_argument("--y2", type=int, help="Bottom Y súradnica pre fixed režim.")
    ap.add_argument("--x1", type=int, help="Left X súradnica pre fixed režim.")
    ap.add_argument("--x2", type=int, help="Right X súradnica pre fixed režim.")

    args = ap.parse_args()

    exts = tuple(x.strip() for x in args.exts.split(",") if x.strip())
    files = collect_images(args.src, exts=exts)
    if not files:
        print("Nenájdené žiadne obrázky.")
        sys.exit(1)
    if args.shuffle:
        random.shuffle(files)

    ensure_dir(args.dst)
    csv_path = args.csv or os.path.join(args.dst, "calib_log.csv")

    if args.mode == "fixed":
        required = [args.y1, args.y2, args.x1, args.x2]
        if any(v is None for v in required):
            print("Pre --mode fixed musíš zadať --y1 --y2 --x1 --x2")
            sys.exit(2)
        process_fixed(files, args.dst, csv_path, args.y1, args.y2, args.x1, args.x2, args.limit)

    elif args.mode == "gui":
        process_gui(files, args.dst, csv_path, args.limit)

    else:  # auto
        process_auto(files, args.dst, csv_path, args.limit)

if __name__ == "__main__":
    main()
