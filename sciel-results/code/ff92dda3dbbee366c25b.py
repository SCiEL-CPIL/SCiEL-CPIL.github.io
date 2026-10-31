"""Fresh design: combinatorial search over compact triangular-lattice subsets of
11 unit hexagons, enclosing each in the minimal regular hexagon via support
functions. Strict wall-clock deadline guard."""
import time
import math
import itertools

# EVOLVE-BLOCK-START

import numpy as np

N = 11
AP = math.sqrt(3.0) / 2.0          # apothem of a unit regular hexagon
S3 = math.sqrt(3.0)                # lattice spacing (hexagon tiling distance)

# 6 unit direction vectors of a regular hexagon (apothem directions), angle 0 -> flat-top
DIRS = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

def _support(side):
    # side: (2,) array of (x,y). returns (6,) support values.
    return side @ DIRS.T

def _enclosing_side(centers):
    # centers: (k,2). minimal side length of a regular hexagon (any center, rot 0)
    # containing all, apothem AP. side = max over directions (max_proj + AP).
    sup = _support(centers)
    return float(np.max(sup) + AP)

def _enclosing_center(centers, side):
    # center of the enclosing hexagon for side length 'side' (rot 0).
    sup = _support(centers)
    # For each direction i, constraint: c_i - c_{i+3} <= side - 2*AP  (since sup_i - sup_{i+3} = c_i - c_{i+3})
    # and c_i - c_{i+3} >= -(side - 2*AP)
    # We want c minimizing max_i (sup_i - c_i + AP) = side.
    # c_i - c_{i+3} is fixed by centers. We need c_i - c_{i+3} = side - 2*AP for the tight pair.
    # Simplest: choose c so that the max is minimized. With rot 0 and our construction the
    # tight direction is the global max; set c so that all constraints are satisfied.
    # c_i - c_{i+3} must equal side - 2*AP (the tight value). Solve least squares over the 3 pairs.
    A = np.zeros((3, 2))
    b = np.zeros(3)
    for i in range(3):
        A[i] = DIRS[i] - DIRS[i + 3]   # = 2*DIRS[i]
        b[i] = side - 2 * AP
    # least squares
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return (float(c[0]), float(c[1]))

def _build():
    best = None
    best_side = float('inf')
    start = time.time()
    deadline = start + 140.0   # safety margin under 150s kill

    # ---- candidate center sets ----
    candidates = []

    # 1) explicit compact cluster shapes (coordinates on triangular lattice, spacing S3)
    def hx(i, j):
        return (S3 * (i + 0.5 * j), S3 * 0.75 * j)

    # ring-based compact sets
    # ring0: (0,0)
    # ring1: 6 neighbors
    # ring2: 12 neighbors
    def ring1():
        pts = []
        for (di, dj) in [(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]:
            pts.append(hx(di, dj))
        return pts
    def ring2():
        pts = []
        # 12 cells at hex distance 2
        for (di, dj) in [(2,0),(-2,0),(0,2),(0,-2),(2,-2),(-2,2),(1,1),(-1,-1),(1,-2),(-1,2),(2,-1),(-2,1)]:
            pts.append(hx(di, dj))
        return pts

    r0 = [hx(0, 0)]
    r1 = ring1()
    r2 = ring2()

    # shape A: full honeycomb 1 + 6 + 4 (11 cells) - take ring1 (6) + 4 of ring2
    # shape B: 1 + 6 + 4 different
    # shape C: two rows of 5 + 1
    def rows5():
        pts = []
        for k in range(5):
            pts.append(hx(k, 0))
        for k in range(5):
            pts.append(hx(k, 1))
        pts.append(hx(2, 2))
        return pts
    def rows5b():
        pts = []
        for k in range(5):
            pts.append(hx(k, 0))
        for k in range(5):
            pts.append(hx(k, 1))
        pts.append(hx(0, -1))
        return pts

    candidates.append(r0 + r1 + r2[:4])
    candidates.append(r0 + r1 + r2[4:8])
    candidates.append(r0 + r1 + r2[8:12])
    candidates.append(r0 + r1 + [r2[0], r2[5], r2[10], r2[1]])
    candidates.append(r0 + r1 + [r2[0], r2[3], r2[6], r2[9]])
    candidates.append(rows5())
    candidates.append(rows5b())
    # 3x3 block minus 2 + 1
    def block33():
        pts = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                pts.append(hx(i, j))
        return pts
    candidates.append(block33() + [hx(2, 0), hx(0, 2)])
    candidates.append(block33() + [hx(2, 0), hx(2, 1)])
    candidates.append(block33() + [hx(0, 2), hx(1, 2)])

    # 2) combinatorial: pick 10 from ring1(6)+ring2(12) plus center
    pool = r1 + r2
    # Precompute support of each pool cell
    pool_arr = np.array(pool)
    pool_sup = _support(pool_arr)
    center_sup = _support(np.array(r0))

    # We want to choose 10 cells from pool (18 choose 10 is large ~ 43M) - too big.
    # Instead, do a greedy + local search: start from a compact 10, then try swaps to reduce enclosing side.
    def greedy10():
        # start: 6 ring1 + 4 ring2 (closest to center)
        idx = list(range(6)) + list(range(6, 10))
        return idx

    def enclosing_from_idx(idx):
        c = np.vstack([r0, [pool[i] for i in idx]])
        return _enclosing_side(c)

    # local search: try replacing each chosen with an unchosen to reduce side
    def local_search(idx):
        idx = set(idx)
        chosen = list(idx)
        cur = enclosing_from_idx(chosen)
        improved = True
        iters = 0
        while improved and iters < 400:
            improved = False
            iters += 1
            unchosen = [i for i in range(len(pool)) if i not in idx]
            # try each swap
            best_swap = None
            best_val = cur
            for out_i in chosen:
                for in_i in unchosen:
                    if in_i == out_i:
                        continue
                    trial = list(idx)
                    trial.remove(out_i)
                    trial.append(in_i)
                    val = enclosing_from_idx(trial)
                    if val < best_val - 1e-9:
                        best_val = val
                        best_swap = (out_i, in_i)
            if best_swap is not None:
                out_i, in_i = best_swap
                idx.remove(out_i)
                idx.add(in_i)
                cur = best_val
                improved = True
        return cur, list(idx)

    # run local search from a few seeds
    seeds = [
        greedy10(),
        list(range(6)) + [6, 9, 12, 15],
        list(range(6)) + [7, 10, 13, 16],
        [0,1,2,3,4,5,6,7,8,9],
        [0,1,2,3,4,5,11,12,13,14],
        [2,3,4,5,6,7,8,9,10,11],
    ]
    for s in seeds:
        val, idx = local_search(s)
        centers = np.vstack([r0, [pool[i] for i in idx]])
        candidates.append(centers.tolist())
        if time.time() > deadline - 5:
            break

    # ---- evaluate all candidates ----
    for cand in candidates:
        if time.time() > deadline - 2:
            break
        centers = np.array(cand)
        side = _enclosing_side(centers)
        if side < best_side:
            best_side = side
            cx, cy = _enclosing_center(centers, side)
            best = (centers, (cx, cy), side)

    if best is None:
        # fallback: simple row layout (valid, large)
        centers = np.array([[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)])[:, :2]
        side = _enclosing_side(centers)
        cx, cy = _enclosing_center(centers, side)
        best = (centers, (cx, cy), side)

    centers, (cx, cy), side = best
    inner = [[float(centers[i][0]), float(centers[i][1]), 0.0] for i in range(len(centers))]
    return inner, (cx, cy), float(side), 0.0

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    try:
        inner, center, side, angle = _build()
    except Exception:
        # robust fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    if time.time() > deadline:
        # still return best found (it was computed before deadline guard inside _build)
        pass
    return inner, center, side, angle

# EVOLVE-BLOCK-END
