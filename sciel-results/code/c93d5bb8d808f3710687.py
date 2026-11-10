import time
import random
import numpy as np
from collections import Counter

def _score(pts):
    n = len(pts)
    areas = []
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            for k in range(j + 1, n):
                xk, yk = pts[k]
                d = abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                if d > 0:
                    areas.append(d)
    if not areas:
        return 0
    return Counter(areas).most_common(1)[0][1]

def _init_counts(pts):
    counts = Counter()
    n = len(pts)
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            for k in range(j + 1, n):
                xk, yk = pts[k]
                d = abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                if d > 0:
                    counts[d] += 1
    return counts

def _top(counts):
    if not counts:
        return 0, 0
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))

def _try_move(pts, pos, new_pt, counts, top_val, top_cnt):
    # Apply move pts[pos] -> new_pt, return (new_top_val, new_top_cnt, applied)
    old_pt = pts[pos]
    if new_pt == old_pt:
        return top_val, top_cnt, False
    n = len(pts)
    for a in range(n):
        if a == pos:
            continue
        for b in range(a + 1, n):
            if b == pos:
                continue
            xa, ya = pts[a]
            xb, yb = pts[b]
            d_old = abs(old_pt[0] * (ya - yb) + xa * (yb - old_pt[1]) + xb * (old_pt[1] - ya))
            d_new = abs(new_pt[0] * (ya - yb) + xa * (yb - new_pt[1]) + xb * (new_pt[1] - ya))
            if d_old == d_new:
                continue
            if d_old > 0:
                c = counts[d_old]
                if c == 1:
                    del counts[d_old]
                else:
                    counts[d_old] = c - 1
            if d_new > 0:
                counts[d_new] = counts.get(d_new, 0) + 1
    # recompute top
    tv, tc = _top(counts)
    return tv, tc, True

def _revert_move(pts, pos, old_pt, counts, top_val, top_cnt, new_pt):
    # Undo a move that set pts[pos] to new_pt; restore old_pt
    # Reapply inverse: treat as move from new_pt back to old_pt
    return _try_move(pts, pos, old_pt, counts, top_val, top_cnt)

def _seed_grid(scale, ox, oy):
    pts = []
    for x in range(6):
        for y in range(5):
            pts.append((x * scale + ox, y * scale + oy))
    return pts

def _random_seed(rng):
    pts = set()
    while len(pts) < 30:
        pts.add((rng.randint(-30, 30), rng.randint(-30, 30)))
    return list(pts)

def _anneal(pts, rng, budget, start_temp):
    pts = list(pts)
    counts = _init_counts(pts)
    top_val, top_cnt = _top(counts)
    best_top = top_cnt
    best_pts = list(pts)
    temp = start_temp
    cooldown = 0.9999
    deadline = time.time() + budget
    while time.time() < deadline:
        i = rng.randrange(30)
        new_pt = (rng.randint(-30, 30), rng.randint(-30, 30))
        if new_pt == pts[i]:
            continue
        old_pt = pts[i]
        tv, tc, applied = _try_move(pts, i, new_pt, counts, top_val, top_cnt)
        if not applied:
            continue
        pts[i] = new_pt
        delta = tc - top_cnt
        if delta >= 0 or rng.random() < np.exp(delta / max(temp, 1e-6)):
            top_val, top_cnt = tv, tc
            if top_cnt > best_top:
                best_top = top_cnt
                best_pts = list(pts)
        else:
            # revert
            rv, rc, _ = _revert_move(pts, i, old_pt, counts, tv, tc, new_pt)
            pts[i] = old_pt
            top_val, top_cnt = rv, rc
        temp *= cooldown
    return best_pts, best_top

def construct():
    deadline = time.time() + 80
    rng = random.Random(98765)
    best_pts = None
    best_score = 0

    seeds = []
    for scale in [1, 2, 3]:
        for ox in range(-30, 31, 4):
            for oy in range(-30, 31, 4):
                pts = _seed_grid(scale, ox, oy)
                if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
                    seeds.append(pts)
    for _ in range(8):
        seeds.append(_random_seed(rng))

    scored = []
    for s in seeds:
        sc = _score(s)
        scored.append((sc, s))
    scored.sort(key=lambda t: -t[0])
    top_seeds = [s for sc, s in scored[:10]]

    budget_each = 5.5
    for s in top_seeds:
        if time.time() > deadline - 8:
            break
        pts, sc = _anneal(s, rng, budget_each, start_temp=2.0)
        if sc > best_score:
            best_score = sc
            best_pts = pts

    if best_pts is not None and time.time() < deadline - 4:
        pts, sc = _anneal(best_pts, rng, deadline - time.time() - 3, start_temp=0.4)
        if sc > best_score:
            best_score = sc
            best_pts = pts

    if best_pts is None:
        return _seed_grid(1, 0, 0)
    return best_pts
