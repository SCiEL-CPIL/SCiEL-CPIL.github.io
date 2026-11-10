import numpy as np
import time
from math import gcd
from collections import Counter
import random


def _normalize_line(a, b, c):
    a, b, c = int(a), int(b), int(c)
    if a == 0 and b == 0:
        return None
    g = gcd(abs(a), gcd(abs(b), abs(c)))
    if g == 0:
        return None
    a, b, c = a // g, b // g, c // g
    for v in (a, b, c):
        if v != 0:
            if v < 0:
                a, b, c = -a, -b, -c
            break
    return (a, b, c)


def _is_valid(points, lines):
    try:
        p = np.asarray(points)
        l = np.asarray(lines)
    except Exception:
        return False
    if p.shape != (54, 2) or l.shape != (27, 3):
        return False
    for arr in (p, l):
        if not np.all(np.isfinite(arr)):
            return False
        if not np.all(np.abs(arr - np.round(arr)) < 1e-6):
            return False
    if len(set(map(tuple, np.round(p, 6).astype(np.int64).tolist()))) != 54:
        return False
    seen = set()
    for row in np.round(l, 6).astype(np.int64).tolist():
        canon = _normalize_line(row[0], row[1], row[2])
        if canon is None:
            return False
        if canon in seen:
            return False
        seen.add(canon)
    return True


def _incidence_count(points, lines):
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    vals = lns[:, 0, None] * pts[None, :, 0] + lns[:, 1, None] * pts[None, :, 1] + lns[:, 2, None]
    return int(np.sum(vals == 0))


def _line_from_points(p1, p2):
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return _normalize_line(A, B, C)


def _build_candidate_pool(pts, max_coeff=9):
    """Pool = all small-coefficient lines + lines through all point pairs."""
    pool = set()
    for A in range(-max_coeff, max_coeff + 1):
        for B in range(-max_coeff, max_coeff + 1):
            if A == 0 and B == 0:
                continue
            for C in range(-30, 31):
                c = _normalize_line(A, B, C)
                if c:
                    pool.add(c)
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            c = _line_from_points(pts[i], pts[j])
            if c:
                pool.add(c)
    return list(pool)


def _build_candidate_pool2(pts, max_coeff=12):
    pool = set()
    for A in range(-max_coeff, max_coeff + 1):
        for B in range(-max_coeff, max_coeff + 1):
            if A == 0 and B == 0:
                continue
            for C in range(-40, 41):
                c = _normalize_line(A, B, C)
                if c:
                    pool.add(c)
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            c = _line_from_points(pts[i], pts[j])
            if c:
                pool.add(c)
    return list(pool)


def _generate_grid_initialization():
    w, h = 6, 9
    pts = np.array([(x, y) for x in range(w) for y in range(h)], dtype=np.int64)

    lines = []
    for x in range(w):
        lines.append((1, 0, -x))
    for y in range(h):
        lines.append((0, 1, -y))
    for c in range(-(h - 1), w):
        lines.append((1, -1, -c))
    for c in range(0, w + h - 1):
        lines.append((1, 1, -c))

    # keep only distinct lines
    uniq = []
    seen = set()
    for l in lines:
        c = _normalize_line(l[0], l[1], l[2])
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)

    # score each candidate line by # of points it passes through
    scored = []
    for (A, B, C) in uniq:
        vals = A * pts[:, 0] + B * pts[:, 1] + C
        scored.append((int(np.sum(vals == 0)), (A, B, C)))
    scored.sort(key=lambda t: -t[0])

    selected = []
    seen2 = set()
    for cnt, l in scored:
        if l not in seen2:
            seen2.add(l)
            selected.append(l)
            if len(selected) == 27:
                break
    # pad if necessary (shouldn't happen)
    for l in uniq:
        if l not in seen2:
            seen2.add(l)
            selected.append(l)
            if len(selected) == 27:
                break
    return pts, np.array(selected, dtype=np.int64)


def _local_search(pts, lns, deadline, pool, seed=42, T_start=8.0, T_end=0.05):
    rng = random.Random(seed)
    pts = pts.copy()
    lns = lns.copy()
    n_pts, n_lines = 54, 27
    pool = np.asarray(pool, dtype=np.int64)

    inc = (lns[:, 0, None] * pts[None, :, 0] +
           lns[:, 1, None] * pts[None, :, 1] +
           lns[:, 2, None]) == 0

    current = int(inc.sum())
    best = current
    best_pts = pts.copy()
    best_lns = lns.copy()

    it = 0
    max_iters = 600000
    while time.time() < deadline - 20:
        it += 1
        progress = min(it / max_iters, 1.0)
        T = T_start * (T_end / T_start) ** progress

        if rng.random() < 0.55:
            # ---- move a point ----
            i = rng.randint(0, n_pts - 1)
            ox, oy = int(pts[i, 0]), int(pts[i, 1])
            moved = False
            for _ in range(12):
                dx = rng.randint(-3, 3)
                dy = rng.randint(-3, 3)
                if dx == 0 and dy == 0:
                    continue
                nx, ny = ox + dx, oy + dy
                # duplicate check (vectorized)
                if np.any((pts[:, 0] == nx) & (pts[:, 1] == ny)):
                    continue
                delta = int(np.sum(lns[:, 0] * nx + lns[:, 1] * ny + lns[:, 2] == 0)) - int(inc[:, i].sum())
                if delta >= 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                    pts[i] = (nx, ny)
                    inc[:, i] = (lns[:, 0] * nx + lns[:, 1] * ny + lns[:, 2] == 0)
                    current += delta
                    moved = True
                    break
            if not moved:
                continue
        else:
            # ---- replace a line ----
            j = rng.randint(0, n_lines - 1)
            old = tuple(lns[j])
            if rng.random() < 0.5:
                new = tuple(pool[rng.randint(0, len(pool) - 1)])
            else:
                i1 = rng.randint(0, n_pts - 1)
                i2 = rng.randint(0, n_pts - 1)
                if i1 == i2:
                    continue
                new = _line_from_points(pts[i1], pts[i2])
                if new is None:
                    continue
            if new == old:
                continue
            if new in [tuple(lns[k]) for k in range(n_lines) if k != j]:
                continue
            A, B, C = new
            delta = int(np.sum(A * pts[:, 0] + B * pts[:, 1] + C == 0)) - int(inc[j].sum())
            if delta >= 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                lns[j] = new
                inc[j] = (A * pts[:, 0] + B * pts[:, 1] + C == 0)
                current += delta

        if current > best:
            best = current
            best_pts = pts.copy()
            best_lns = lns.copy()

    return best_pts, best_lns, best


def _polish(pts, lns, deadline, pool):
    """Exhaustive local-optimum polish: single point moves + best line replacement."""
    pts = pts.copy()
    lns = lns.copy()
    n_pts, n_lines = 54, 27
    inc = (lns[:, 0, None] * pts[None, :, 0] +
           lns[:, 1, None] * pts[None, :, 1] +
           lns[:, 2, None]) == 0
    current = int(inc.sum())
    best = current
    best_pts = pts.copy()
    best_lns = lns.copy()

    # ---- point polish: try every small move for every point, greedily ----
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(n_pts):
            if time.time() > deadline - 10:
                break
            x0, y0 = int(pts[i, 0]), int(pts[i, 1])
            best_delta = 0
            best_pos = None
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x0 + dx, y0 + dy
                    if np.any((pts[:, 0] == nx) & (pts[:, 1] == ny)):
                        continue
                    delta = int(np.sum(lns[:, 0] * nx + lns[:, 1] * ny + lns[:, 2] == 0)) - int(inc[:, i].sum())
                    if delta > best_delta:
                        best_delta = delta
                        best_pos = (nx, ny)
            if best_pos is not None:
                pts[i] = best_pos
                inc[:, i] = (lns[:, 0] * best_pos[0] + lns[:, 1] * best_pos[1] + lns[:, 2] == 0)
                current += best_delta
                improved = True
                if current > best:
                    best = current
                    best_pts = pts.copy()
                    best_lns = lns.copy()

    # ---- line polish: replace each line with the best non-duplicate candidate ----
    # rebuild a pool that includes all pairwise lines through current points
    pair_lines = set()
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            c = _line_from_points(pts[i], pts[j])
            if c:
                pair_lines.add(c)
    full_pool = list(set(pool) | pair_lines)
    full_pool_arr = np.asarray(full_pool, dtype=np.int64)

    for j in range(n_lines):
        if time.time() > deadline - 5:
            break
        old = tuple(lns[j])
        others = [tuple(lns[k]) for k in range(n_lines) if k != j]
        # delta for every candidate line, vectorized over points
        cand_vals = (full_pool_arr[:, 0, None] * pts[None, :, 0] +
                     full_pool_arr[:, 1, None] * pts[None, :, 1] +
                     full_pool_arr[:, 2, None])
        cand_counts = (cand_vals == 0).sum(axis=1)  # (len(pool),)
        old_count = int(inc[j].sum())
        best_delta = 0
        best_idx = None
        for idx in range(len(full_pool)):
            if tuple(full_pool_arr[idx]) == old:
                continue
            if tuple(full_pool_arr[idx]) in others:
                continue
            delta = int(cand_counts[idx]) - old_count
            if delta > best_delta:
                best_delta = delta
                best_idx = idx
        if best_idx is not None:
            lns[j] = full_pool_arr[best_idx]
            inc[j] = (lns[j, 0] * pts[:, 0] + lns[j, 1] * pts[:, 1] + lns[j, 2] == 0)
            current += best_delta
            if current > best:
                best = current
                best_pts = pts.copy()
                best_lns = lns.copy()

    return best_pts, best_lns, best


def szemeredi_trotter_incidences():
    start = time.time()
    deadline = start + 560  # safety margin before 600s hard limit

    pts, lns = _generate_grid_initialization()
    pool = _build_candidate_pool(pts)

    # First SA pass
    pts, lns, score = _local_search(pts, lns, deadline, pool, seed=42)

    # Second SA pass with a richer pool, from the best found
    if time.time() < deadline - 60:
        pool2 = _build_candidate_pool2(pts)
        pts2, lns2, score2 = _local_search(pts, lns, deadline, pool2, seed=7, T_start=4.0, T_end=0.03)
        if score2 > score:
            pts, lns, score = pts2, lns2, score2

    # Final exhaustive polish
    if time.time() < deadline - 25:
        pts, lns, score = _polish(pts, lns, deadline, pool2 if time.time() < deadline else pool)

    pts = np.array(pts, dtype=np.int64)
    lns = np.array(lns, dtype=np.int64)

    if not _is_valid(pts, lns):
        xs = np.arange(6)
        ys = np.arange(9)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = []
        for x in range(6):
            lines.append((1, 0, -x))
        for y in range(9):
            lines.append((0, 1, -y))
        for k in range(12):
            lines.append((1, 1, -k))
        lns = np.array(lines, dtype=np.int64)

    return pts, lns
