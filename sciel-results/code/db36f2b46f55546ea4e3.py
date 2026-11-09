# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    thetas = a + np.deg2rad(60.0 * np.arange(6))
    return np.stack([cx + s * np.cos(thetas), cy + s * np.sin(thetas)], axis=1)


def _hex_edges_and_normals(cx, cy, s, a_deg):
    v = _hex_vertices(cx, cy, s, a_deg)
    edges = np.diff(np.vstack([v, v[:1]]), axis=0)
    # outward normals
    normals = np.empty_like(edges)
    for i in range(6):
        e = edges[i]
        n = np.array([-e[1], e[0]])
        # orient outward: point from center to edge midpoint should align with normal
        mid = (v[i] + v[(i + 1) % 6]) / 2.0
        if np.dot(mid - np.array([cx, cy]), n) < 0:
            n = -n
        normals[i] = n
    return edges, normals


def _pair_min_gap(h1, h2):
    # h1,h2: (cx,cy,s,a). Return minimum signed separation over all 12 edge normals.
    # Positive gap means separated; negative means overlap.
    cx1, cy1, s1, a1 = h1
    cx2, cy2, s2, a2 = h2
    v1 = _hex_vertices(cx1, cy1, s1, a1)
    v2 = _hex_vertices(cx2, cy2, s2, a2)
    e1, n1 = _hex_edges_and_normals(cx1, cy1, s1, a1)
    e2, n2 = _hex_edges_and_normals(cx2, cy2, s2, a2)
    best = -np.inf
    for n in list(n1) + list(n2):
        n = n / (np.linalg.norm(n) + 1e-15)
        p1 = v1 @ n
        p2 = v2 @ n
        lo1, hi1 = p1.min(), p1.max()
        lo2, hi2 = p2.min(), p2.max()
        # separation: how far apart are the intervals
        sep = max(lo2 - hi1, lo1 - hi2)
        if sep > best:
            best = sep
    return best


def _all_gaps(inner):
    # inner: (11,3) (x,y,angle). Return min gap over all pairs (negative=overlap)
    n = len(inner)
    min_gap = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            g = _pair_min_gap((inner[i, 0], inner[i, 1], 1.0, inner[i, 2]),
                              (inner[j, 0], inner[j, 1], 1.0, inner[j, 2]))
            if g < min_gap:
                min_gap = g
    return min_gap


def _outer_side_for(inner, angle_deg=0.0):
    # Minimal side length of a regular hexagon (centered at origin, given angle)
    # that contains all inner unit hexagons (their vertices inside).
    # For a regular hexagon of side s centered at origin with angle a, the
    # support function in direction u is s * max over edges of projection.
    # A point p is inside iff for all 6 outward normals n_k: p . n_k <= s * h_k
    # where h_k is the support of the hexagon in direction n_k.
    # For a regular hexagon of side s, the distance from center to each edge
    # (inward normal direction) is s * sqrt(3)/2. The outward normals point
    # to edge midpoints. So condition: p . n_k <= s * sqrt(3)/2 for all k.
    a = np.deg2rad(angle_deg)
    thetas = a + np.deg2rad(60.0 * np.arange(6))
    # outward normal directions (perpendicular to edges, pointing out)
    # Edge k is between vertex k and k+1. Outward normal direction = angle of
    # midpoint of edge = a + 60*k + 30 deg? Let's compute: for a regular hexagon
    # with vertex 0 at angle a, the edge between vertex0 and vertex1 has outward
    # normal at angle a + 30 deg.
    normals = np.stack([np.cos(a + np.deg2rad(30.0) + np.deg2rad(60.0 * np.arange(6))),
                        np.sin(a + np.deg2rad(30.0) + np.deg2rad(60.0 * np.arange(6)))], axis=1)
    # For each inner hexagon vertex, project onto each normal; need s such that
    # max projection <= s * sqrt(3)/2
    max_proj = 0.0
    for i in range(len(inner)):
        v = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        projs = v @ normals.T
        m = projs.max()
        if m > max_proj:
            max_proj = m
    return max_proj / (np.sqrt(3.0) / 2.0)


def _objective(x):
    # x: 11*3 = 33 vars (x,y,angle_deg per hexagon)
    inner = x.reshape(11, 3)
    # penalty for overlaps
    min_gap = _all_gaps(inner)
    pen = max(0.0, -min_gap)
    side = _outer_side_for(inner)
    # objective: side + big penalty for overlap
    return side + 1000.0 * pen ** 2


def _seed_positions():
    # A good initial seed: honeycomb-like 11 hexagons
    # 3-4-3-1? Let's do a compact arrangement.
    # Use axial coordinates for hex grid, convert to cartesian.
    # Hex grid with pointy-top: positions
    # Let's place 11 hexagons in a "flower" + extras
    seed = []
    # center
    seed.append([0.0, 0.0, 0.0])
    # 6 around center at distance 2*sqrt(3)/2*? Unit hexagons touching:
    # center-to-center distance for touching unit hexagons = 2 * (sqrt(3)/2) = sqrt(3)
    # if aligned, adjacent centers are 2*apothem apart = sqrt(3)
    d = np.sqrt(3.0)
    for k in range(6):
        ang = np.deg2rad(60.0 * k)
        seed.append([d * np.cos(ang), d * np.sin(ang), 0.0])
    # 4 more: place further out
    # second ring partial
    for k in range(0, 6, 2):
        ang = np.deg2rad(60.0 * k)
        seed.append([2 * d * np.cos(ang), 2 * d * np.sin(ang), 0.0])
    # that's 1+6+3 = 10, need 1 more
    ang = np.deg2rad(30.0)
    seed.append([2 * d * np.cos(ang), 2 * d * np.sin(ang), 0.0])
    return np.array(seed[:11])


def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin before 600s

    best_inner = None
    best_side = np.inf

    def _eval_and_record(inner):
        nonlocal best_inner, best_side
        # check all finite
        if not np.all(np.isfinite(inner)):
            return
        # check overlaps strictly
        min_gap = _all_gaps(inner)
        if min_gap < -1e-6:
            return
        side = _outer_side_for(inner)
        if not np.isfinite(side):
            return
        if side < best_side:
            best_side = side
            best_inner = inner.copy()

    # Initial seed
    inner = _seed_positions()
    _eval_and_record(inner)

    # Try multiple starting angles for the outer hexagon to find best orientation
    # We optimize inner positions; outer angle we can also try a few.

    # L-BFGS-B optimization on inner positions
    try:
        res = minimize(_objective, inner.reshape(-1), method='L-BFGS-B',
                       options={'maxiter': 400, 'maxfun': 20000})
        if time.time() > deadline:
            pass
        else:
            inner_opt = res.x.reshape(11, 3)
            _eval_and_record(inner_opt)
    except Exception:
        pass

    # Local refinement: Nelder-Mead style small perturbations around best
    if best_inner is not None and time.time() < deadline:
        # gradient-free local search
        cur = best_inner.copy()
        cur_side = best_side
        step = 0.05
        for it in range(3000):
            if time.time() > deadline:
                break
            # random perturbation
            idx = np.random.randint(0, 11)
            dim = np.random.randint(0, 3)
            old = cur[idx, dim].copy()
            cur[idx, dim] = old + np.random.uniform(-step, step)
            if not np.all(np.isfinite(cur)):
                cur[idx, dim] = old
                continue
            min_gap = _all_gaps(cur)
            if min_gap < -1e-6:
                cur[idx, dim] = old
                continue
            side = _outer_side_for(cur)
            if side < cur_side - 1e-9:
                cur_side = side
                best_inner = cur.copy()
                best_side = side
            else:
                cur[idx, dim] = old
            # occasionally shrink step
            if it % 500 == 499:
                step *= 0.8

    # Final: try optimizing outer angle for the best inner set
    if best_inner is not None:
        # sweep outer angle finely
        best_angle = 0.0
        best_side_final = best_side
        for ang in np.linspace(0, 60, 601):
            s = _outer_side_for(best_inner, angle_deg=ang)
            if s < best_side_final:
                best_side_final = s
                best_angle = ang
        best_side = best_side_final
        outer_angle = best_angle
    else:
        outer_angle = 0.0
        best_side = 8.0
        best_inner = _seed_positions()

    # Ensure validity: if overlaps, fall back to a known-valid large packing
    # (the seed grid is valid with side 8)
    min_gap_final = _all_gaps(best_inner)
    if min_gap_final < -1e-6:
        # fallback to seed grid with big outer
        best_inner = np.array([
            [0, 0, 0],
            [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0],
            [-1.25, -2.17, 0], [1.25, -2.17, 0],
            [-3.75, 2.17, 0], [3.75, 2.17, 0],
            [-3.75, -2.17, 0], [3.75, -2.17, 0],
        ], dtype=float)
        best_side = 8.0
        outer_angle = 0.0

    inner_out = best_inner.astype(float)
    outer_out = np.array([0.0, 0.0, outer_angle], dtype=float)
    side_out = float(best_side)

    return inner_out, outer_out, side_out


# EVOLVE-BLOCK-END
