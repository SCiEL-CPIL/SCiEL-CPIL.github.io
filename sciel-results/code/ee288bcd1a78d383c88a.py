import math
import time
import random

N = 11
EPS = 1e-9
START = time.time()
DEADLINE = 150.0

def _outer_vertices(cx, cy, s, rot):
    # Flat-top hexagon: vertices at angles 0,60,...
    pts = []
    for k in range(6):
        a = math.radians(rot) + math.radians(60 * k)
        pts.append((cx + s * math.cos(a), cy + s * math.sin(a)))
    return pts

def _inner_hex(cx, cy, ang_deg):
    # Flat-top unit hexagon (apothem = sqrt(3)/2, radius = 1)
    pts = []
    a0 = math.radians(ang_deg)
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + math.cos(a), cy + math.sin(a)))
    return pts

def _cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def _point_in_poly(pt, poly):
    n = len(poly)
    sign = 0
    for i in range(n):
        c = _cross(poly[i], poly[(i+1) % n], pt)
        if abs(c) < 1e-9:
            continue
        s = 1 if c > 0 else -1
        if sign == 0:
            sign = s
        elif sign != s:
            return False
    return True

def _poly_overlap(A, B):
    # SAT: two convex polygons are disjoint iff there is a separating axis
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            ax = poly[(i+1) % n][0] - poly[i][0]
            ay = poly[(i+1) % n][1] - poly[i][1]
            nx, ny = -ay, ax
            # project both
            minA = min(p[0]*nx + p[1]*ny for p in A)
            maxA = max(p[0]*nx + p[1]*ny for p in A)
            minB = min(p[0]*nx + p[1]*ny for p in B)
            maxB = max(p[0]*nx + p[1]*ny for p in B)
            if maxA + 1e-9 < minB or maxB + 1e-9 < minA:
                return False  # separated
    return True

def _in_outer(inner_pts, outer_pts):
    return all(_point_in_poly(p, outer_pts) for p in inner_pts)

def _valid(inner, outer_c, s, rot):
    outer_pts = _outer_vertices(outer_c[0], outer_c[1], s, rot)
    polys = [_inner_hex(x, y, a) for (x, y, a) in inner]
    for i in range(len(polys)):
        if not _in_outer(polys[i], outer_pts):
            return False
    for i in range(len(polys)):
        for j in range(i+1, len(polys)):
            if _poly_overlap(polys[i], polys[j]):
                return False
    return True

def _objective(inner, outer_c, s, rot):
    # returns penalty (0 if valid) = number of violations, else 1.0
    return 0.0 if _valid(inner, outer_c, s, rot) else 1.0

def _try_refine(inner, outer_c, s, rot, rng, iters, budget):
    """Local search: random perturbations + small step reduction of s."""
    best = (inner, outer_c, s, rot)
    best_pen = _objective(inner, outer_c, s, rot)
    cur_pen = best_pen
    cur = (inner, outer_c, s, rot)
    it = 0
    while it < iters:
        if time.time() > budget:
            break
        it += 1
        # candidate: perturb positions/angles
        cand = [[x + rng.uniform(-0.2, 0.2), y + rng.uniform(-0.2, 0.2), a + rng.uniform(-10, 10)] for (x, y, a) in cur[0]]
        cand_oc = (cur[1][0] + rng.uniform(-0.1, 0.1), cur[1][1] + rng.uniform(-0.1, 0.1))
        cand_s = cur[2] * (1.0 + rng.uniform(-0.02, 0.02))
        cand_rot = cur[3] + rng.uniform(-5, 5)
        pen = _objective(cand, cand_oc, cand_s, cand_rot)
        if pen < cur_pen:
            cur = (cand, cand_oc, cand_s, cand_rot)
            cur_pen = pen
            if pen < best_pen:
                best = cur
                best_pen = pen
        # if valid, try to shrink s
        if cur_pen == 0.0:
            try_s = cur[2] * 0.99
            if _objective(cur[0], cur[1], try_s, cur[3]) == 0.0:
                cur = (cur[0], cur[1], try_s, cur[3])
                best = cur
                best_pen = 0.0
    return best

def _make_seed(rng):
    # random scattered layout
    inner = [[rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(0, 60)] for _ in range(N)]
    return inner, (0.0, 0.0), 12.0, 0.0

def construct():
    best = None
    best_score = float('inf')
    budget = START + DEADLINE - 5.0  # safety margin
    rng = random.Random(12345)

    # 1) try a few heuristic seeds
    seeds = []
    # grid seed
    seeds.append(([[-6.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 6.0, 0.0],
                   [0.0, -6.0, 0.0], [3.5, 3.5, 0.0], [3.5, -3.5, 0.0],
                   [-3.5, 3.5, 0.0], [-3.5, -3.5, 0.0], [7.0, 4.0, 0.0],
                   [7.0, -4.0, 0.0], [-7.0, 4.0, 0.0]], (0.0, 0.0), 16.0, 0.0))
    # random seeds
    for _ in range(6):
        seeds.append(_make_seed(rng))

    for seed in seeds:
        if time.time() > budget:
            break
        inner, oc, s, rot = seed
        res = _try_refine(inner, oc, s, rot, rng, iters=400, budget=budget)
        inner, oc, s, rot = res
        if _valid(inner, oc, s, rot):
            if s < best_score:
                best_score = s
                best = (inner, oc, s, rot)

    # 2) if none valid, run more random restarts
    restarts = 0
    while time.time() < budget - 1.0 and best is None:
        restarts += 1
        inner, oc, s, rot = _make_seed(rng)
        res = _try_refine(inner, oc, s, rot, rng, iters=300, budget=budget)
        inner, oc, s, rot = res
        if _valid(inner, oc, s, rot):
            if s < best_score:
                best_score = s
                best = (inner, oc, s, rot)
        if restarts >= 40:
            break

    if best is None:
        # fallback: return seed-like valid-ish large box (may be invalid -> score 0)
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

    inner, oc, s, rot = best
    return inner, (oc[0], oc[1]), s, rot
