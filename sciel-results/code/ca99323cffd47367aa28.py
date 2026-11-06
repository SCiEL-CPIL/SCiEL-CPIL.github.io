"""Rewritten program: grid construction + time-bounded local search."""
import time
import itertools
import numpy as np

_TIME_LIMIT = 85.0  # safety margin under the 90 s kill


def _area_matrix(pts):
    """Return the doubled (unsigned) area for every triple as a (n,n,n) array."""
    x = pts[:, 0].astype(np.int64)
    y = pts[:, 1].astype(np.int64)
    # 2*signed area = x1(y2-y3) + x2(y3-y1) + x3(y1-y2)
    d = x[:, None, None] * (y[None, :, None] - y[None, None, :]) \
      + x[None, :, None] * (y[None, None, :] - y[:, None, None]) \
      + x[None, None, :] * (y[:, None, None] - y[None, :, None])
    return np.abs(d)


def _multiplicity(arr):
    """Largest multiplicity of a single positive value (zero excluded)."""
    nz = arr[arr > 0]
    if nz.size == 0:
        return 0
    vals, counts = np.unique(nz, return_counts=True)
    return int(counts.max())


def _eval(pts):
    """Score = largest multiplicity of a positive doubled area."""
    arr = _area_matrix(pts)
    return _multiplicity(arr)


def _grid(w, h):
    return [(x, y) for x in range(w) for y in range(h)]


def construct():
    t0 = time.time()
    deadline = t0 + _TIME_LIMIT

    # 1) Strong baseline: rectangular grid (30 points).
    best_pts = np.array(_grid(6, 5), dtype=np.int64)
    best_score = _eval(best_pts)

    # 2) Try a few other grid shapes for robustness.
    for (w, h) in [(5, 6), (10, 3), (3, 10), (15, 2), (2, 15)]:
        if time.time() > deadline:
            break
        cand = np.array(_grid(w, h), dtype=np.int64)
        s = _eval(cand)
        if s > best_score:
            best_score = s
            best_pts = cand

    # 3) Time-bounded local search: perturb one point to an empty lattice cell.
    #    Only accept if the multiplicity strictly increases.
    while time.time() < deadline - 0.5:
        improved = False
        pts = best_pts.tolist()
        occupied = set(pts)
        # candidate cells in a bounded box around the current configuration
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        lo_x, hi_x = min(xs) - 2, max(xs) + 2
        lo_y, hi_y = min(ys) - 2, max(ys) + 2
        cells = [(cx, cy) for cx in range(lo_x, hi_x + 1)
                 for cy in range(lo_y, hi_y + 1) if (cx, cy) not in occupied]
        for i, (ox, oy) in enumerate(pts):
            if time.time() > deadline - 0.5:
                break
            for (nx, ny) in cells:
                if time.time() > deadline - 0.5:
                    break
                new_pts = pts[:i] + [(nx, ny)] + pts[i + 1:]
                arr = _area_matrix(np.array(new_pts, dtype=np.int64))
                s = _multiplicity(arr)
                if s > best_score:
                    best_score = s
                    best_pts = np.array(new_pts, dtype=np.int64)
                    improved = True
                    occupied = set(new_pts)
                    break
        if not improved:
            break

    return [tuple(p) for p in best_pts]
