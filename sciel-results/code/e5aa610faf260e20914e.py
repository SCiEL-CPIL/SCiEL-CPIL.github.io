# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd


def _line_from_two_points(p1, p2):
    """Return (A,B,C) int tuple for line through p1=(x1,y1), p2=(x2,y2)."""
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        g = 1
    A //= g
    B //= g
    C //= g
    if A < 0 or (A == 0 and (B < 0 or (B == 0 and C < 0))):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _incidence_count(points, lines):
    """Count incidences exactly in int64."""
    P = points.astype(np.int64)
    L = lines.astype(np.int64)
    # A*x + B*y + C
    vals = L[:, 0:1] * P[:, 0:1].T + L[:, 1:2] * P[:, 1:2].T + L[:, 2:3]
    return int(np.count_nonzero(vals == 0))


def _normalize_line(A, B, C):
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    if A < 0 or (A == 0 and (B < 0 or (B == 0 and C < 0))):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _is_valid(points, lines):
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    if not np.issubdtype(points.dtype, np.integer):
        return False
    if not np.issubdtype(lines.dtype, np.integer):
        return False
    # repeated points
    if len(np.unique(points, axis=0)) != 54:
        return False
    # lines: A=B=0 or duplicates
    seen = set()
    for row in lines:
        A, B, C = int(row[0]), int(row[1]), int(row[2])
        if A == 0 and B == 0:
            return False
        key = _normalize_line(A, B, C)
        if key in seen:
            return False
        seen.add(key)
    return True


def _grid_construction():
    """Classical Szemeredi-Trotter grid: 54 points, 27 lines, 81 incidences."""
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    rows = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            rows.append((a, -1, b))
    lines = np.array(rows, dtype=np.int64)
    return points, lines


def _star_construction():
    """
    Alternative: place points on a small grid and lines through pairs.
    Try a 6x9 grid of points (54) and pick 27 lines that maximize incidences.
    """
    # 6 x 9 grid
    xs = np.arange(6)
    ys = np.arange(9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    # Candidate lines: horizontal (9), vertical (6), diagonals slope +/-1, +/-2, etc.
    # We'll do a greedy selection over all lines through pairs of points.
    n = len(points)
    line_scores = {}
    # Precompute all lines through pairs
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_from_two_points(points[i], points[j])
            line_scores[key] = line_scores.get(key, 0) + 1
    # Sort by score descending, pick top 27 unique
    sorted_keys = sorted(line_scores.items(), key=lambda x: -x[1])
    lines = []
    seen = set()
    for key, _ in sorted_keys:
        if key in seen:
            continue
        lines.append(key)
        seen.add(key)
        if len(lines) == 27:
            break
    lines = np.array(lines, dtype=np.int64)
    return points, lines


def _local_search(points, lines, max_iters=5000, deadline=None):
    """
    Local search: try replacing a point or a line to increase incidences.
    """
    n_pts = len(points)
    n_lines = len(lines)
    best_inc = _incidence_count(points, lines)
    P = points.copy()
    L = lines.copy()
    rng = np.random.default_rng(42)
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Try moving a random point to a nearby integer location
        idx = rng.integers(0, n_pts)
        old_pt = P[idx].copy()
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                new_pt = old_pt + np.array([dx, dy])
                P[idx] = new_pt
                # Check no duplicate points
                if len(np.unique(P, axis=0)) < n_pts:
                    P[idx] = old_pt
                    continue
                inc = _incidence_count(P, L)
                if inc > best_inc:
                    best_inc = inc
                    improved = True
                    break
                P[idx] = old_pt
            if improved:
                break
        if improved:
            continue
        # Try replacing a random line with a line through two random points
        lidx = rng.integers(0, n_lines)
        old_line = L[lidx].copy()
        for _ in range(10):
            i = rng.integers(0, n_pts)
            j = rng.integers(0, n_pts)
            if i == j:
                continue
            new_line = _line_from_two_points(P[i], P[j])
            # Check no duplicate lines
            dup = False
            for k in range(n_lines):
                if k == lidx:
                    continue
                if _normalize_line(int(L[k, 0]), int(L[k, 1]), int(L[k, 2])) == new_line:
                    dup = True
                    break
            if dup:
                continue
            L[lidx] = new_line
            inc = _incidence_count(P, L)
            if inc > best_inc:
                best_inc = inc
                improved = True
                break
            L[lidx] = old_line
        if improved:
            continue
        # Try swapping two points
        if rng.random() < 0.3:
            i = rng.integers(0, n_pts)
            j = rng.integers(0, n_pts)
            if i == j:
                continue
            P[[i, j]] = P[[j, i]]
            if len(np.unique(P, axis=0)) < n_pts:
                P[[i, j]] = P[[j, i]]
                continue
            inc = _incidence_count(P, L)
            if inc > best_inc:
                best_inc = inc
                improved = True
            else:
                P[[i, j]] = P[[j, i]]
    return P, L, best_inc


def szemeredi_trotter_incidences():
    start = time.time()
    deadline = start + 600  # safety margin before 630s kill

    # Start with grid construction
    points, lines = _grid_construction()
    best_inc = _incidence_count(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()

    # Try star construction
    pts2, lns2 = _star_construction()
    inc2 = _incidence_count(pts2, lns2)
    if inc2 > best_inc:
        best_inc = inc2
        best_points = pts2.copy()
        best_lines = lns2.copy()

    # Local search from best starting point
    if time.time() < deadline - 60:
        P, L, inc = _local_search(best_points, best_lines, max_iters=20000, deadline=deadline - 30)
        if inc > best_inc:
            best_inc = inc
            best_points = P
            best_lines = L

    # Final local search round
    if time.time() < deadline - 30:
        P, L, inc = _local_search(best_points, best_lines, max_iters=10000, deadline=deadline - 15)
        if inc > best_inc:
            best_inc = inc
            best_points = P
            best_lines = L

    # Ensure validity
    if not _is_valid(best_points, best_lines):
        # Fallback to grid
        best_points, best_lines = _grid_construction()

    return best_points.astype(np.int64), best_lines.astype(np.int64)


# EVOLVE-BLOCK-END
