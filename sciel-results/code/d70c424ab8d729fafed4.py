import time
import math
import numpy as np

N = 12
START = time.time()
DEADLINE = 120.0
MARGIN = 8.0

def _time_left():
    return DEADLINE - MARGIN - (time.time() - START)

def _hex_vertices(cx, cy, s, ang_deg):
    a = math.radians(ang_deg)
    # flat-top regular hexagon vertices
    pts = np.empty((6, 2))
    for k in range(6):
        th = a + math.radians(60.0 * k)
        pts[k, 0] = cx + s * math.cos(th)
        pts[k, 1] = cy + s * math.sin(th)
    return pts

def _outer_halfwidths(side, ang_deg):
    # width/2 of a regular hexagon of side s along x and y for its orientation
    a = math.radians(ang_deg)
    # support of hexagon (flat-top base) along a unit direction
    # For a regular hexagon, support function h(u) = s * max_k cos(angle(u)-theta_k)
    # but for axis-aligned bounding we need x-extent and y-extent.
    # Compute exact extents from vertices.
    xs = np.empty(6)
    ys = np.empty(6)
    for k in range(6):
        th = a + math.radians(60.0 * k)
        xs[k] = side * math.cos(th)
        ys[k] = side * math.sin(th)
    return (xs.max() - xs.min()) / 2.0, (ys.max() - ys.min()) / 2.0

def _inner_halfwidths(ang_deg):
    # unit hexagon extents along x,y for orientation ang_deg
    a = math.radians(ang_deg)
    xs = np.empty(6)
    ys = np.empty(6)
    for k in range(6):
        th = a + math.radians(60.0 * k)
        xs[k] = math.cos(th)
        ys[k] = math.sin(th)
    return (xs.max() - xs.min()) / 2.0, (ys.max() - ys.min()) / 2.0

def _outer_contains_inner(cx, cy, side, oang_deg, px, py, pang_deg):
    # True if inner hexagon (center p, angle pang) is inside outer hexagon (center c, side, oang)
    # Equivalent: for every direction, support_outer(u) - support_inner(u) >= |p-c| .u
    # Use 12 critical directions (6 of outer + 6 of inner) for a safe test.
    a_o = math.radians(oang_deg)
    a_i = math.radians(pang_deg)
    dirs = []
    for k in range(6):
        dirs.append(a_o + math.radians(60.0 * k))
        dirs.append(a_i + math.radians(60.0 * k))
    dx = px - cx
    dy = py - cy
    for th in dirs:
        ux, uy = math.cos(th), math.sin(th)
        # support of outer hexagon along u: s * max_k cos(theta_k - u)
        ho = side * max(math.cos(a_o + math.radians(60.0 * k) - th) for k in range(6))
        hi = 1.0 * max(math.cos(a_i + math.radians(60.0 * k) - th) for k in range(6))
        # need ho - hi >= u . (p - c)
        if ho - hi + 1e-9 < ux * dx + uy * dy:
            return False
    return True

def _max_side_for_centers(centers, angles, oang_deg):
    # Given inner centers and angles, and outer orientation, find minimal outer side
    # such that all inner fit. Binary search on side.
    lo, hi = 1.0, 20.0
    # ensure hi is large enough
    for _ in range(30):
        if hi < 10.0:
            hi *= 2.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        ok = True
        for (px, py), pa in zip(centers, angles):
            if not _outer_contains_inner(0.0, 0.0, mid, oang_deg, px, py, pa):
                ok = False
                break
        if ok:
            hi = mid
        else:
            lo = mid
    return hi

def construct():
    best = None
    # Initial guess: centers on a circle, angles 0, outer angle 0
    centers = np.zeros((N, 2))
    angles = np.zeros(N)
    # place on a ring
    for i in range(N):
        th = 2.0 * math.pi * i / N
        centers[i, 0] = 3.0 * math.cos(th)
        centers[i, 1] = 3.0 * math.sin(th)
    oang = 0.0

    # Greedy local improvement: for each inner, nudge to reduce required side.
    # We'll do a simple coordinate-descent style with random restarts within budget.
    rng = np.random.default_rng(0)
    best_side = _max_side_for_centers(centers.tolist(), angles.tolist(), oang)
    best_state = (centers.copy(), angles.copy(), oang, best_side)

    while _time_left() > 2.0:
        # random restart with a different arrangement
        mode = rng.integers(0, 3)
        if mode == 0:
            # ring
            for i in range(N):
                th = 2.0 * math.pi * i / N + rng.uniform(-0.2, 0.2)
                r = rng.uniform(1.5, 5.0)
                centers[i, 0] = r * math.cos(th)
                centers[i, 1] = r * math.sin(th)
            angles[:] = 0.0
            oang = 0.0
        elif mode == 1:
            # grid-ish
            for i in range(N):
                centers[i, 0] = rng.uniform(-4, 4)
                centers[i, 1] = rng.uniform(-4, 4)
            angles[:] = rng.uniform(0, 60, size=N)
            oang = 0.0
        else:
            # perturb current best
            centers = best_state[0].copy() + rng.normal(0, 0.5, size=(N, 2))
            angles = best_state[1].copy() + rng.uniform(-10, 10, size=N)
            oang = best_state[2]

        # local search: adjust one inner at a time to reduce side
        cur_side = _max_side_for_centers(centers.tolist(), angles.tolist(), oang)
        for _it in range(40):
            if _time_left() < 2.0:
                break
            improved = False
            for i in range(N):
                # try small moves
                for dx in (-0.3, -0.1, 0.0, 0.1, 0.3):
                    for dy in (-0.3, -0.1, 0.0, 0.1, 0.3):
                        c2 = centers.copy()
                        c2[i, 0] += dx
                        c2[i, 1] += dy
                        s2 = _max_side_for_centers(c2.tolist(), angles.tolist(), oang)
                        if s2 < cur_side - 1e-9:
                            centers = c2
                            cur_side = s2
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break

        if cur_side < best_side - 1e-9:
            best_side = cur_side
            best_state = (centers.copy(), angles.copy(), oang, best_side)
            best = (centers, angles, oang, best_side)

    if best is None:
        best = best_state

    centers, angles, oang, side = best
    inner = [[float(centers[i, 0]), float(centers[i, 1]), float(angles[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(side), float(oang)
