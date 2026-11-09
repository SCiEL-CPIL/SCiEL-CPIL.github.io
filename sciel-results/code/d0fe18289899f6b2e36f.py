import numpy as np
import time
import math
from itertools import product, combinations
from math import gcd

def _normalize_line(A, B, C):
    """Normalize a line (A, B, C) by dividing by gcd and making first nonzero positive."""
    A, B, C = int(A), int(B), int(C)
    if A == 0 and B == 0:
        return None
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 1:
        A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidences(points, lines):
    """Count exact integer incidences between points and lines."""
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    total = 0
    for i in range(len(lns)):
        A, B, C = lns[i]
        val = A * pts[:, 0] + B * pts[:, 1] + C
        total += int(np.sum(val == 0))
    return total

def _is_valid(points, lines):
    """Validate points and lines per the task contract."""
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    if pts.shape != (54, 2) or lns.shape != (27, 3):
        return False
    if not np.all(np.isfinite(pts.astype(np.float64))):
        return False
    if not np.all(np.isfinite(lns.astype(np.float64))):
        return False
    seen_pts = set()
    for p in pts:
        t = (int(p[0]), int(p[1]))
        if t in seen_pts:
            return False
        seen_pts.add(t)
    seen_lines = set()
    for l in lns:
        A, B, C = int(l[0]), int(l[1]), int(l[2])
        if A == 0 and B == 0:
            return False
        norm = _normalize_line(A, B, C)
        if norm is None:
            return False
        if norm in seen_lines:
            return False
        seen_lines.add(norm)
    return True

def _build_grid(a, b):
    """Build a a x b grid and pick top 27 lines."""
    xs = np.arange(a)
    ys = np.arange(b)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    if len(points) != 54:
        return None, None, 0

    candidates = []
    # Vertical
    for c in range(a):
        candidates.append((1, 0, -c))
    # Horizontal
    for r in range(b):
        candidates.append((0, 1, -r))
    # Diagonals
    for d in range(-(a-1), a):
        candidates.append((-1, 1, d))
    # Anti-diagonals
    for s in range(0, a + b - 1):
        candidates.append((1, 1, -s))
    # Steeper slopes
    for slope in [2, 3, -2, -3, 4, -4]:
        for c in range(-b*2, b*2):
            candidates.append((slope, -1, c))
            
    # Deduplicate candidates
    unique_candidates = []
    seen = set()
    for c in candidates:
        norm = _normalize_line(*c)
        if norm and norm not in seen:
            seen.add(norm)
            unique_candidates.append(norm)
            
    scores = []
    for idx, (A, B, C) in enumerate(unique_candidates):
        val = A * points[:, 0] + B * points[:, 1] + C
        cnt = int(np.sum(val == 0))
        scores.append((cnt, idx))
    scores.sort(reverse=True)
    
    top_indices = [idx for _, idx in scores[:27]]
    lines_arr = np.array([unique_candidates[i] for i in top_indices], dtype=np.int64)
    
    if _is_valid(points, lines_arr):
        inc = _count_incidences(points, lines_arr)
        return points, lines_arr, inc
    
    return None, None, 0

def szemeredi_trotter_incidences():
    deadline = time.time() + 580.0
    
    best_pts = None
    best_lns = None
    best_inc = -1
    
    # Try various grid dimensions
    for a in range(2, 28):
        if 54 % a != 0:
            continue
        b = 54 // a
        if time.time() > deadline:
            break
        pts, lns, inc = _build_grid(a, b)
        if pts is not None and inc > best_inc:
            best_inc = inc
            best_pts, best_lns = pts, lns
            
    if best_pts is None:
        # Fallback
        pts, lns = _build_grid(6, 9)
        return pts, lns
        
    return best_pts, best_lns
