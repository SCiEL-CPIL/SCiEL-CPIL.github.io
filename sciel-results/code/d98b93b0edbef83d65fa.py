import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3)
# Minimum center-to-center distance for two unit hexagons (any orientation)
MIN_DIST = SQRT3 - 1e-4


def construct():
    deadline = time.time() + 140.0

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
        return verts

    def compute_outer_side(centers, angles, ocx, ocy, otheta):
        """Smallest outer-hex side (centered at (ocx,ocy), rotated otheta) that
        contains every inner hexagon.  A regular hexagon of side S is the
        intersection of 6 half-planes  dot(p-c, n_k) <= S*SQRT3/2  where the
        outward normals n_k are spaced 60 deg apart."""
        max_proj = 0.0
        for k in range(6):
            na = otheta + math.pi / 6 + k * math.pi / 3
            nx, ny = math.cos(na), math.sin(na)
            for i in range(len(centers)):
                cx, cy = centers[i]
                ang = math.radians(angles[i])
                for vi in range(6):
                    va = ang + vi * math.pi / 3
                    vx = cx + math.cos(va)
                    vy = cy + math.sin(va)
                    proj = (vx - ocx) * nx + (vy - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
        return 2.0 * max_proj / SQRT3

    def check_disjoint(centers, angles, min_dist=MIN_DIST):
        n = len(centers)
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                if dx * dx + dy * dy < min_dist * min_dist:
                    return False
        return True

    def polish_outer(centers, angles, ocx, ocy, otheta):
        """High-precision coordinate-descent polish of the outer-hex parameters."""
        best = compute_outer_side(centers, angles, ocx, ocy, otheta)
        for scale in [0.5, 0.1, 0.02, 0.004, 0.0008]:
            improved = True
            while improved and time.time() < deadline - 12:
                improved = False
                for dx in (-scale, 0.0, scale):
                    for dy in (-scale, 0.0, scale):
                        for da_deg in (-scale * 20.0, 0.0, scale * 20.0):
                            s = compute_outer_side(centers, angles,
                                                   ocx + dx, ocy + dy,
                                                   (otheta + da_deg) % 60.0)
                            if s < best - 1e-12:
                                best = s
                                ocx += dx
                                ocy += dy
                                otheta = (otheta + da_deg) % 60.0
                                improved = True
        return ocx, ocy, otheta, best

    def slsqp_optimize(centers, angles, max_iters=40):
        """Constrained SLSQP: variables = [x_i, y_i, a_i ...] + [ocx, ocy, otheta].
        Objective = outer side; constraints = pairwise center distance >= MIN_DIST."""
        n = len(centers)
        x0 = []
        for i in range(n):
            x0 += [centers[i][0], centers[i][1], angles[i]]
        x0 += [0.0, 0.0, 0.0]

        def objective(x):
            if time.time() > deadline - 25:
                return 1e9
            c = [[x[3 * i], x[3 * i + 1]] for i in range(n)]
            a = [x[3 * i + 2] for i in range(n)]
            return compute_outer_side(c, a, x[3 * n], x[3 * n + 1], x[3 * n + 2])

        cons = []
        for i in range(n):
            for j in range(i + 1, n):
                def distc(x, i=i, j=j):
                    dx = x[3 * i] - x[3 * j]
                    dy = x[3 * i + 1] - x[3 * j + 1]
                    return dx * dx + dy * dy - MIN_DIST * MIN_DIST
                cons.append({'type': 'ineq', 'fun': distc})

        bounds = [(-6, 6), (-6, 6), (0, 60)] * n + [(-3, 3), (-3, 3), (0, 60)]
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons,
                           options={'maxiter': max_iters, 'ftol': 1e-10})
            if res.success:
                c = [[res.x[3 * i], res.x[3 * i + 1]] for i in range(n)]
                a = [res.x[3 * i + 2] for i in range(n)]
                return c, a, res.x[3 * n], res.x[3 * n + 1], res.x[3 * n + 2]
        except Exception:
            pass
        return None

    # ---- Build a good lattice-ring seed ----
    eps = 1e-3
    d = SQRT3 + eps
    u = np.array([d, 0.0])
    v = np.array([d / 2.0, d * SQRT3 / 2.0])
    basis = [u, v, v - u, -u, -v, u - v]

    ring1 = [u, v, v - u, -u, -v, u - v]
    centers = [[0.0, 0.0]] + [[p[0], p[1]] for p in ring1]

    ring2 = []
    for p1 in ring1:
        for b in basis:
            p2 = p1 + b
            if np.allclose(p2, np.array([0.0, 0.0]), atol=0.05):
                continue
            if any(np.allclose(p2, q, atol=0.05) for q in ring1):
                continue
            if not any(np.allclose(p2, q, atol=0.05) for q in ring2):
                ring2.append(p2)
    ring2.sort(key=lambda p: np.linalg.norm(p))
    centers += [[p[0], p[1]] for p in ring2[:5]]

    assert len(centers) == 12
    angles = [30.0] * 12

    # Seed result: polish outer hex only.
    allv = []
    for i in range(12):
        allv += get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
    cx = sum(p[0] for p in allv) / len(allv)
    cy = sum(p[1] for p in allv) / len(allv)
    best_side = float('inf')
    best_angle = 0.0
    for th in np.linspace(0, 60, 61):
        s = compute_outer_side(centers, angles, cx, cy, math.radians(th))
        if s < best_side:
            best_side = s
            best_angle = th
    best = (centers, angles, cx, cy, best_angle, best_side)

    # ---- Optimize several candidate seeds, keep the best ----
    seeds = []
    seeds.append((list(centers), list(angles)))
    # alternate ring-2 selections
    for k in range(1, 4):
        idxs = list(range(len(ring2)))
        random.seed(1000 + k)
        random.shuffle(idxs)
        sel = [ring2[i] for i in idxs[:5]]
        c2 = [[0.0, 0.0]] + [[p[0], p[1]] for p in ring1] + [[p[0], p[1]] for p in sel]
        if len(c2) == 12 and check_disjoint(c2, angles):
            seeds.append((c2, list(angles)))

    for si, (c0, a0) in enumerate(seeds):
        if time.time() > deadline - 35:
            break
        res = slsqp_optimize(c0, a0, max_iters=40)
        if res is None:
            continue
        c, a, ocx, ocy, oth = res
        if not check_disjoint(c, a):
            continue
        ocx, ocy, oth, s = polish_outer(c, a, ocx, ocy, oth)
        if s < best_side:
            best = (c, a, ocx, ocy, oth, s)
            best_side = s

    centers, angles, ocx, ocy, oth, best_side = best

    # Final safety polish
    ocx, ocy, oth, best_side = polish_outer(centers, angles, ocx, ocy, oth)

    inner = [[centers[i][0], centers[i][1], angles[i]] for i in range(12)]
    return inner, (ocx, ocy), best_side, oth
