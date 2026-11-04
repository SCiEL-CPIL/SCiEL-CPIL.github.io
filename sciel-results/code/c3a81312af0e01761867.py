"""Hexagon packing: 11 unit hexagons inside a regular hexagon.
Structural change: SAT-based fast overlap check + contact-graph-aware Nelder-Mead
with outer hexagon rotation as an optimization variable.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11
APOTHEM = math.sqrt(3.0) / 2.0  # inradius of unit hexagon

def _hex_vertices(cx, cy, angle_deg):
    """6 vertices of unit regular hexagon centered at (cx,cy) rotated by angle_deg."""
    a = math.radians(angle_deg)
    v = np.empty((6, 2))
    for i in range(6):
        theta = a + i * math.pi / 3.0
        v[i, 0] = cx + math.cos(theta)
        v[i, 1] = cy + math.sin(theta)
    return v

def _sat_overlap(poly1, poly2):
    """SAT test: return True if two convex polygons overlap."""
    for poly in (poly1, poly2):
        n = len(poly)
        for i in range(n):
            ax, ay = poly[i]
            bx, by = poly[(i + 1) % n]
            # normal to edge (a->b)
            dx, dy = bx - ax, by - ay
            nx, ny = -dy, dx
            # project all vertices of both polygons onto normal
            min1, max1 = float('inf'), float('-inf')
            for j in range(len(poly1)):
                p = poly1[j][0] * nx + poly1[j][1] * ny
                if p < min1: min1 = p
                if p > max1: max1 = p
            min2, max2 = float('inf'), float('-inf')
            for j in range(len(poly2)):
                p = poly2[j][0] * nx + poly2[j][1] * ny
                if p < min2: min2 = p
                if p > max2: max2 = p
            if max1 < min2 or max2 < min1:
                return False  # separating axis found
    return True

def _sat_gap(poly1, poly2):
    """Return minimum gap (negative if overlapping) between two convex polygons via SAT."""
    min_gap = float('inf')
    for poly in (poly1, poly2):
        n = len(poly)
        for i in range(n):
            ax, ay = poly[i]
            bx, by = poly[(i + 1) % n]
            dx, dy = bx - ax, by - ay
            length = math.sqrt(dx * dx + dy * dy)
            if length < 1e-12:
                continue
            nx, ny = -dy / length, dx / length
            # project
            min1, max1 = float('inf'), float('-inf')
            for j in range(len(poly1)):
                p = poly1[j][0] * nx + poly1[j][1] * ny
                if p < min1: min1 = p
                if p > max1: max1 = p
            min2, max2 = float('inf'), float('-inf')
            for j in range(len(poly2)):
                p = poly2[j][0] * nx + poly2[j][1] * ny
                if p < min2: min2 = p
                if p > max2: max2 = p
            gap = max(min2 - max1, min1 - max2)
            if gap < 0:
                return gap  # overlapping
            if gap < min_gap:
                min_gap = gap
    return min_gap

def _compute_side(centers, angles, outer_angle_deg):
    """Minimum outer hex side length to contain all inner hexagons."""
    a = math.radians(outer_angle_deg)
    normals = np.array([[math.cos(a + k * math.pi / 3.0),
                          math.sin(a + k * math.pi / 3.0)] for k in range(6)])
    max_proj = 0.0
    for i in range(N):
        verts = _hex_vertices(centers[i, 0], centers[i, 1], angles[i])
        proj = verts @ normals.T  # (6,6)
        max_proj = max(max_proj, np.max(np.abs(proj)))
    return max_proj / APOTHEM

def _overlap_penalty(centers, angles):
    """Sum of squared overlap gaps for all pairs (negative gap = overlap)."""
    polys = [_hex_vertices(centers[i, 0], centers[i, 1], angles[i]) for i in range(N)]
    pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            gap = _sat_gap(polys[i], polys[j])
            if gap < 0.005:
                pen += (0.005 - gap) ** 2
    return pen

def _is_valid(centers, angles, S, outer_angle_deg):
    """Check non-overlap and containment."""
    polys = [_hex_vertices(centers[i, 0], centers[i, 1], angles[i]) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if _sat_overlap(polys[i], polys[j]):
                return False
    # containment via half-plane test
    a = math.radians(outer_angle_deg)
    for k in range(6):
        theta = a + k * math.pi / 3.0
        nx, ny = math.cos(theta), math.sin(theta)
        limit = S * APOTHEM
        for i in range(N):
            for v in polys[i]:
                if v[0] * nx + v[1] * ny > limit + 1e-6:
                    return False
    return True

def _seed_hexagonal():
    """Standard hexagonal lattice seed."""
    d = math.sqrt(3.0)  # distance between centers of touching hexagons
    centers = np.zeros((N, 2))
    angles = np.zeros(N)
    # center
    centers[0] = [0, 0]
    # ring 1: 6 hexagons
    for i in range(6):
        theta = i * math.pi / 3.0
        centers[i + 1] = [d * math.cos(theta), d * math.sin(theta)]
    # ring 2: pick 4 well-separated
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * APOTHEM * r
                ring2.append([x, y])
    # pick 4 evenly spaced from the 12
    selected = [ring2[i] for i in [0, 3, 6, 9]]
    for k, pos in enumerate(selected):
        centers[7 + k] = pos
    S = _compute_side(centers, angles, 0.0) * 1.01
    return centers, angles, S

def _seed_rotated():
    """Seed with 30-degree rotation of inner hexagons for denser packing."""
    d = math.sqrt(3.0)
    centers = np.zeros((N, 2))
    angles = np.zeros(N)
    centers[0] = [0, 0]
    angles[0] = 0.0
    for i in range(6):
        theta = i * math.pi / 3.0
        centers[i + 1] = [d * math.cos(theta), d * math.sin(theta)]
        angles[i + 1] = 30.0
    # 4 outer in gaps
    for k in range(4):
        theta = (2 * k + 1) * math.pi / 12.0
        centers[7 + k] = [1.8 * d * math.cos(theta), 1.8 * d * math.sin(theta)]
        angles[7 + k] = 0.0
    S = _compute_side(centers, angles, 0.0) * 1.02
    return centers, angles, S

def construct():
    t0 = time.time()
    DEADLINE = 138.0  # 12s safety margin before 150s kill

    # Get seeds
    seeds = []
    for fn in [_seed_hexagonal, _seed_rotated]:
        try:
            c, a, S = fn()
            seeds.append((c, a, S, 0.0))
        except Exception:
            pass
    if not seeds:
        c = np.zeros((N, 2))
        a = np.zeros(N)
        d = math.sqrt(3.0) * 1.5
        for i in range(1, N):
            theta = i * 2 * math.pi / N
            c[i] = [d * math.cos(theta), d * math.sin(theta)]
        seeds.append((c, a, _compute_side(c, a, 0.0) * 1.1, 0.0))

    best_S = min(s[2] for s in seeds)
    best_c = seeds[0][0].copy()
    best_a = seeds[0][1].copy()
    best_oa = seeds[0][3]
    for s in seeds:
        if s[2] < best_S:
            best_S = s[2]
            best_c = s[0].copy()
            best_a = s[1].copy()
            best_oa = s[3]

    def objective(params):
        if time.time() > DEADLINE:
            return 1e10
        c = params[:2 * N].reshape(N, 2)
        a = params[2 * N:2 * N + N]
        oa = params[2 * N + N]
        S = _compute_side(c, a, oa)
        pen = _overlap_penalty(c, a)
        return S + 200.0 * pen

    bounds = []
    for _ in range(N):
        bounds.append((-4.0, 4.0))
        bounds.append((-4.0, 4.0))
    for _ in range(N):
        bounds.append((0.0, 180.0))
    bounds.append((0.0, 60.0))  # outer angle (hex symmetry period 60)

    # Phase 1: Differential Evolution for global search
    if time.time() < DEADLINE - 50:
        try:
            np.random.seed(42)
            init_pop = []
            for c, a, S, oa in seeds:
                init_pop.append(list(c.flatten()) + list(a) + [oa])
            for _ in range(6):
                c_r = np.random.uniform(-2.5, 2.5, size=(N, 2))
                a_r = np.random.uniform(0, 180, size=N)
                oa_r = np.random.uniform(0, 60)
                init_pop.append(list(c_r.flatten()) + list(a_r) + [oa_r])

            res = differential_evolution(
                objective, bounds, seed=42, maxiter=60, popsize=10,
                tol=1e-5, mutation=(0.5, 1.5), recombination=0.8,
                polish=False, init=init_pop, updating='deferred'
            )
            if res.fun < best_S:
                p = res.x
                c = p[:2 * N].reshape(N, 2)
                a = p[2 * N:2 * N + N]
                oa = p[2 * N + N]
                S_val = _compute_side(c, a, oa)
                pen_val = _overlap_penalty(c, a)
                if pen_val < 1e-4 and S_val < best_S:
                    best_S = S_val
                    best_c, best_a, best_oa = c, a, oa
        except Exception:
            pass

    # Phase 2: Nelder-Mead local refinement
    if time.time() < DEADLINE - 25:
        try:
            x0 = list(best_c.flatten()) + list(best_a) + [best_oa]
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'xatol': 1e-8, 'fatol': 1e-8,
                                    'maxiter': 5000, 'maxfev': 15000})
            if res.fun < best_S:
                p = res.x
                c = p[:2 * N].reshape(N, 2)
                a = p[2 * N:2 * N + N]
                oa = p[2 * N + N]
                S_val = _compute_side(c, a, oa)
                pen_val = _overlap_penalty(c, a)
                if pen_val < 1e-5 and S_val < best_S:
                    best_S = S_val
                    best_c, best_a, best_oa = c, a, oa
        except Exception:
            pass

    # Phase 3: Second DE round with perturbation
    if time.time() < DEADLINE - 35:
        try:
            np.random.seed(99)
            init_pop2 = [list(best_c.flatten()) + list(best_a) + [best_oa]]
            for _ in range(5):
                c_p = best_c + np.random.normal(0, 0.4, size=(N, 2))
                a_p = np.mod(best_a + np.random.normal(0, 20, size=N), 180)
                oa_p = np.mod(best_oa + np.random.normal(0, 10), 60)
                init_pop2.append(list(c_p.flatten()) + list(a_p) + [oa_p])
            res = differential_evolution(
                objective, bounds, seed=99, maxiter=40, popsize=8,
                tol=1e-5, mutation=(0.3, 1.0), recombination=0.7,
                polish=False, init=init_pop2, updating='deferred'
            )
            if res.fun < best_S:
                p = res.x
                c = p[:2 * N].reshape(N, 2)
                a = p[2 * N:2 * N + N]
                oa = p[2 * N + N]
                S_val = _compute_side(c, a, oa)
                pen_val = _overlap_penalty(c, a)
                if pen_val < 1e-4 and S_val < best_S:
                    best_S = S_val
                    best_c, best_a, best_oa = c, a, oa
        except Exception:
            pass

    # Phase 4: Final fine Nelder-Mead
    if time.time() < DEADLINE - 10:
        try:
            x0 = list(best_c.flatten()) + list(best_a) + [best_oa]
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'xatol': 1e-9, 'fatol': 1e-9,
                                    'maxiter': 3000, 'maxfev': 10000})
            if res.fun < best_S:
                p = res.x
                c = p[:2 * N].reshape(N, 2)
                a = p[2 * N:2 * N + N]
                oa = p[2 * N + N]
                S_val = _compute_side(c, a, oa)
                pen_val = _overlap_penalty(c, a)
                if pen_val < 1e-6 and S_val < best_S:
                    best_S = S_val
                    best_c, best_a, best_oa = c, a, oa
        except Exception:
            pass

    # Final validation: ensure validity, add margin if needed
    if not _is_valid(best_c, best_a, best_S, best_oa):
        # Recompute with margin
        S_val = _compute_side(best_c, best_a, best_oa)
        best_S = S_val * 1.02
        if not _is_valid(best_c, best_a, best_S, best_oa):
            # Fallback to seed
            c, a, S, oa = seeds[0]
            best_c, best_a, best_S, best_oa = c, a, S, oa

    inner = []
    for i in range(N):
        inner.append([float(best_c[i, 0]), float(best_c[i, 1]), float(best_a[i])])
    return inner, (0.0, 0.0), float(best_S), float(best_oa)
