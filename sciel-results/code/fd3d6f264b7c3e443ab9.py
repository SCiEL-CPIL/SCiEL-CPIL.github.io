"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons.

Combines:
- Analytical support-function computation of the minimal enclosing outer hexagon
  (fast, smooth, no iterative optimization for the outer parameters).
- A 3-4-3 rigid hexagonal-cluster analytical seed (optimal outer hexagon computed
  directly from the convex hull of the cluster).
- Robust numerical local search (L-BFGS-B) with a smooth distance-based overlap
  penalty for better gradient flow.
- A fast analytical refinement pass that re-optimizes only the outer hexagon
  parameters around a fixed cluster.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START

def hexagon_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon with side s, center (cx,cy), rotation angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.pi / 3 * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def polygons_overlap(v1, v2, tol=1e-9):
    """Check if two convex polygons overlap using SAT."""
    axes = []
    for v in (v1, v2):
        n = len(v)
        for i in range(n):
            edge = v[(i + 1) % n] - v[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                axes.append(normal / norm)

    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - tol or max2 < min1 - tol:
            return False, axis
    return True, None

def convex_hull(points):
    """Andrew's monotone chain. Returns hull vertices (counter-clockwise, no last=first repeat)."""
    pts = np.unique(np.round(points, 12), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.vstack([lower[:-1], upper[:-1]])

def compute_outer_side_length(inner_hexagons, outer_center, outer_angle):
    """Compute minimum outer hexagon side length that contains all inner hexagons."""
    ox, oy = outer_center
    oa = np.radians(outer_angle)

    normals = []
    for k in range(6):
        theta = oa + np.radians(30 + 60 * k)
        normals.append(np.array([np.cos(theta), np.sin(theta)]))

    min_S = 0.0
    oc = np.array([ox, oy])
    for cx, cy, ang in inner_hexagons:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for v in verts:
            for nrm in normals:
                d = np.dot(v - oc, nrm)
                if d > 0:
                    S_needed = 2.0 * d / np.sqrt(3.0)
                    if S_needed > min_S:
                        min_S = S_needed
    return min_S

def analytic_min_outer(inner_hexagons):
    """Analytically find the outer hexagon (center, angle, side) that minimally
    encloses the fixed cluster of inner hexagons, using convex-hull support functions.

    Returns (S, ox, oy, outer_angle_deg).
    """
    # Gather all vertices of all inner hexagons
    all_verts = []
    for cx, cy, ang in inner_hexagons:
        all_verts.append(hexagon_vertices(cx, cy, 1.0, ang))
    all_verts = np.vstack(all_verts)

    hull = convex_hull(all_verts)

    # For a regular hexagon of side S, the inradius (apothem) is r = S * sqrt(3)/2,
    # so S = 2 * r / sqrt(3). The 6 outward normals point at angles
    #   theta_k = oa + 30deg + 60deg * k.
    # For the hexagon to contain the hull, for each normal n_k the maximum
    # projection of the hull onto n_k must be <= r (with the center placed so the
    # projections are balanced). The tightest choice of center is the midpoint of
    # the projection range along each normal; the required apothem is half the
    # projection width. The minimal S over angle is the max over the 6 normals of
    # (projection width / 2) / (sqrt(3)/2) = max_k width_k / sqrt(3).

    best_S = float('inf')
    best_oa = 0.0
    best_center = None

    # Search over outer angle; the optimum is where the dominant support is balanced.
    # Sample finely then refine.
    for oa_deg in np.arange(0.0, 60.0, 0.5):
        oa = np.radians(oa_deg)
        widths = 0.0
        center = np.zeros(2)
        for k in range(6):
            theta = oa + np.radians(30 + 60 * k)
            nrm = np.array([np.cos(theta), np.sin(theta)])
            proj = np.dot(hull, nrm)
            w = proj.max() - proj.min()
            if w > widths:
                widths = w
        S_candidate = widths / np.sqrt(3.0)
        if S_candidate < best_S - 1e-12:
            best_S = S_candidate
            best_oa = oa

    # Refine the angle around the best sample
    lo, hi = best_oa - np.radians(0.5), best_oa + np.radians(0.5)
    for _ in range(30):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        def width_at(oa):
            widths = 0.0
            for k in range(6):
                theta = oa + np.radians(30 + 60 * k)
                nrm = np.array([np.cos(theta), np.sin(theta)])
                proj = np.dot(hull, nrm)
                w = proj.max() - proj.min()
                if w > widths:
                    widths = w
            return widths
        w1, w2 = width_at(m1), width_at(m2)
        if w1 < w2:
            hi = m2
        else:
            lo = m1
    best_oa = (lo + hi) / 2.0

    # Recompute best_S and center at refined angle
    oa = best_oa
    widths = 0.0
    for k in range(6):
        theta = oa + np.radians(30 + 60 * k)
        nrm = np.array([np.cos(theta), np.sin(theta)])
        proj = np.dot(hull, nrm)
        w = proj.max() - proj.min()
        if w > widths:
            widths = w
    best_S = widths / np.sqrt(3.0)

    # Center: place so the projection is balanced along each of the 6 normals.
    # Use the average of the per-normal midpoints projected back; simplest robust
    # choice is the centroid of the hull (the enclosing hexagon will be centered so
    # that all 6 faces are at distance r; the centroid works when the shape is
    # well-centered, which is the case for a near-symmetric cluster).
    center = hull.mean(axis=0)

    # Verify / tighten: shift center to minimize the max over normals of the
    # required apothem. The required apothem for normal n is max over hull of
    # dot(v - c, n); we want to choose c to minimize the max over the 6 normals.
    # This is a small 2-D convex problem; solve with a few gradient steps.
    c = center.copy()
    for _ in range(60):
        # subgradient: pick the worst normal
        worst = 0
        worst_val = -float('inf')
        worst_nrm = None
        for k in range(6):
            theta = oa + np.radians(30 + 60 * k)
            nrm = np.array([np.cos(theta), np.sin(theta)])
            proj = np.dot(hull - c, nrm)
            val = proj.max()
            if val > worst_val:
                worst_val = val
                worst = k
                worst_nrm = nrm
        # Move center in direction that reduces the worst projection.
        c = c + 0.02 * worst_nrm
    # Final S and center
    widths = 0.0
    for k in range(6):
        theta = oa + np.radians(30 + 60 * k)
        nrm = np.array([np.cos(theta), np.sin(theta)])
        proj = np.dot(hull - c, nrm)
        # need both + and - faces: apothem must cover max and -min
        w = proj.max() - proj.min()
        if w > widths:
            widths = w
    best_S = widths / np.sqrt(3.0)

    return best_S, float(c[0]), float(c[1]), float(np.degrees(oa) % 360)

def hexagon_distance(p1, a1, p2, a2):
    """Compute minimum distance between two unit hexagons (0 if overlapping)."""
    d = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if d > 4.0:
        return d - 2.0
    v1 = hexagon_vertices(p1[0], p1[1], 1.0, a1)
    v2 = hexagon_vertices(p2[0], p2[1], 1.0, a2)
    overlap, axis = polygons_overlap(v1, v2, tol=1e-10)
    if not overlap:
        min_dist = float('inf')
        for i in range(6):
            for j in range(6):
                d_ij = np.linalg.norm(v1[i] - v2[j])
                if d_ij < min_dist:
                    min_dist = d_ij
        return min_dist
    return 0.0

def check_validity(inner_hexagons, outer_center, outer_side, outer_angle):
    """Check if the configuration is valid: no overlaps, all inside outer hexagon."""
    ox, oy = outer_center

    for i in range(len(inner_hexagons)):
        for j in range(i + 1, len(inner_hexagons)):
            cx1, cy1, a1 = inner_hexagons[i]
            cx2, cy2, a2 = inner_hexagons[j]
            v1 = hexagon_vertices(cx1, cy1, 1.0, a1)
            v2 = hexagon_vertices(cx2, cy2, 1.0, a2)
            overlap, _ = polygons_overlap(v1, v2, tol=1e-8)
            if overlap:
                return False

    outer_verts = hexagon_vertices(ox, oy, outer_side, outer_angle)

    for cx, cy, ang in inner_hexagons:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for v in verts:
            px, py = v
            n = len(outer_verts)
            inside = True
            for i in range(n):
                v1 = outer_verts[i]
                v2 = outer_verts[(i + 1) % n]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                d1 = np.dot(normal, np.array([px, py]) - v1)
                d2 = np.dot(normal, outer_verts[(i + 2) % n] - v1)
                if d1 * d2 < 0:
                    inside = False
                    break
            if not inside:
                return False

    return True

def objective(params, deadline):
    """Objective function: outer side length + smooth penalty for overlaps."""
    if time.time() > deadline:
        return float('inf')

    n = N
    inner = []
    for i in range(n):
        cx = params[3 * i]
        cy = params[3 * i + 1]
        ang = np.degrees(params[3 * i + 2]) % 360
        inner.append((cx, cy, ang))
    ox = params[3 * n]
    oy = params[3 * n + 1]
    outer_angle = np.degrees(params[3 * n + 2]) % 360

    S = compute_outer_side_length(inner, (ox, oy), outer_angle)

    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            cx1, cy1, a1 = inner[i]
            cx2, cy2, a2 = inner[j]
            d = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
            if d < 2.0:
                dist = hexagon_distance(np.array([cx1, cy1]), a1, np.array([cx2, cy2]), a2)
                if dist < 1e-9:
                    penalty += (2.0 - d) ** 2 * 100.0
                else:
                    # smooth: penalize when distance is small (close to touching/overlap)
                    penalty += max(0.0, 1.0 - dist) ** 2 * 50.0

    return S + penalty

def _build_initial_conditions():
    """Build a set of diverse initial conditions."""
    ics = []

    # Hexagonal lattice positions for N=11: center + ring of 6 + 4 from ring 2
    scale = 1.001
    hex_positions = [(0.0, 0.0)]
    for k in range(6):
        angle = np.radians(60 * k)
        hex_positions.append((np.sqrt(3) * scale * np.cos(angle),
                              np.sqrt(3) * scale * np.sin(angle)))
    for k in range(6):
        angle = np.radians(60 * k)
        hex_positions.append((2 * np.sqrt(3) * scale * np.cos(angle),
                              2 * np.sqrt(3) * scale * np.sin(angle)))
        if len(hex_positions) >= N:
            break
    hex_positions = hex_positions[:N]

    # IC 1: hexagonal lattice, outer at origin, angle 0
    ic1 = np.zeros(3 * N + 3)
    for i, (px, py) in enumerate(hex_positions):
        ic1[3 * i] = px
        ic1[3 * i + 1] = py
    ics.append(ic1)

    # IC 2: perturbed hexagonal lattice
    np.random.seed(42)
    ics.append(ic1 + np.random.randn(3 * N + 3) * 0.1)

    # IC 3: different outer angle
    ic3 = ic1.copy()
    ic3[3 * N + 2] = np.radians(30)
    ics.append(ic3)

    # IC 4: shifted center
    ic4 = ic1.copy()
    ic4[3 * N] = 0.5
    ic4[3 * N + 1] = 0.5
    ics.append(ic4)

    # IC 5: tighter hexagonal lattice
    tight_scale = 0.99
    tight_positions = [(0.0, 0.0)]
    for k in range(6):
        angle = np.radians(60 * k)
        tight_positions.append((np.sqrt(3) * tight_scale * np.cos(angle),
                                np.sqrt(3) * tight_scale * np.sin(angle)))
    for k in range(6):
        angle = np.radians(60 * k)
        tight_positions.append((2 * np.sqrt(3) * tight_scale * np.cos(angle),
                                2 * np.sqrt(3) * tight_scale * np.sin(angle)))
        if len(tight_positions) >= N:
            break
    tight_positions = tight_positions[:N]
    ic5 = np.zeros(3 * N + 3)
    for i, (px, py) in enumerate(tight_positions):
        ic5[3 * i] = px
        ic5[3 * i + 1] = py
    ics.append(ic5)

    # IC 6: rotated inner hexagons
    ic6 = ic1.copy()
    for i in range(N):
        ic6[3 * i + 2] = np.radians(30)
    ics.append(ic6)

    # IC 7: 3-4-3 rigid cluster with analytically computed optimal outer hexagon
    # Rows of 3, 4, 3 hexagons stacked in a hexagonal arrangement.
    try:
        s3 = np.sqrt(3)
        row_y = [0.0, s3, 2 * s3]
        cluster = []
        # top row: 3 hexagons
        for m in range(-1, 2):
            cluster.append((m * 2.0, row_y[2], 0.0))
        # middle row: 4 hexagons, offset by 1 in x
        for m in range(-1, 3):
            cluster.append((m * 2.0 + 1.0, row_y[1], 0.0))
        # bottom row: 3 hexagons
        for m in range(-1, 2):
            cluster.append((m * 2.0, row_y[0], 0.0))
        # Take N of them (10 in this 3-4-3); add one more at a ring-2 position
        cluster = cluster[:N]
        if len(cluster) < N:
            cluster.append((0.0, 3 * s3, 0.0))
        # Center the cluster
        cx_mean = np.mean([c[0] for c in cluster])
        cy_mean = np.mean([c[1] for c in cluster])
        cluster = [(c[0] - cx_mean, c[1] - cy_mean, c[2]) for c in cluster]
        S_an, ox_an, oy_an, oa_an = analytic_min_outer(cluster)
        ic7 = np.zeros(3 * N + 3)
        for i, (px, py, ang) in enumerate(cluster):
            ic7[3 * i] = px
            ic7[3 * i + 1] = py
            ic7[3 * i + 2] = np.radians(ang)
        ic7[3 * N] = ox_an
        ic7[3 * N + 1] = oy_an
        ic7[3 * N + 2] = np.radians(oa_an)
        ics.append(ic7)
    except Exception:
        pass

    return ics, hex_positions

def construct():
    """Main entry point: find optimal packing of N unit hexagons in minimum outer hexagon."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    initial_conditions, hex_positions = _build_initial_conditions()

    best_result = None
    best_score = float('inf')

    def _extract(params):
        inner = []
        for i in range(N):
            cx = params[3 * i]
            cy = params[3 * i + 1]
            ang = np.degrees(params[3 * i + 2]) % 360
            inner.append([cx, cy, ang])
        ox = params[3 * N]
        oy = params[3 * N + 1]
        outer_angle = np.degrees(params[3 * N + 2]) % 360
        return inner, (ox, oy), outer_angle

    # Analytical seed: evaluate the 3-4-3 cluster's analytical outer hexagon
    for ic in initial_conditions:
        if time.time() > deadline - 25:
            break
        try:
            inner, (ox, oy), oa = _extract(ic)
            S = compute_outer_side_length(inner, (ox, oy), oa)
            # Check overlaps
            ok = True
            for i in range(N):
                for j in range(i + 1, N):
                    v1 = hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
                    v2 = hexagon_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
                    ov, _ = polygons_overlap(v1, v2, tol=1e-8)
                    if ov:
                        ok = False
                        break
                if not ok:
                    break
            if ok and S < best_score:
                if check_validity(inner, (ox, oy), S, oa):
                    best_score = S
                    best_result = (inner, (ox, oy), S, oa)
        except Exception:
            pass

    # Numerical local search from each initial condition
    for ic in initial_conditions:
        if time.time() > deadline - 25:
            break
        try:
            res = minimize(objective, ic, args=(deadline,), method='L-BFGS-B',
                           options={'maxiter': 250, 'ftol': 1e-12, 'gtol': 1e-10})

            if res.success or res.fun < best_score:
                inner, (ox, oy), oa = _extract(res.x)
                S = compute_outer_side_length(inner, (ox, oy), oa)

                if check_validity(inner, (ox, oy), S, oa):
                    if S < best_score:
                        best_score = S
                        best_result = (inner, (ox, oy), S, oa)
        except Exception:
            pass

    # Fallback
    if best_result is None:
        inner = [[px, py, 0.0] for px, py in hex_positions]
        S = compute_outer_side_length(inner, (0.0, 0.0), 0.0) + 0.1
        best_result = (inner, (0.0, 0.0), S, 0.0)

    # Fast analytical refinement of the outer hexagon around the fixed cluster
    if time.time() < deadline - 15:
        try:
            inner_fixed = best_result[0]
            S_an, ox_an, oy_an, oa_an = analytic_min_outer(inner_fixed)
            if check_validity(inner_fixed, (ox_an, oy_an), S_an, oa_an) and S_an < best_score:
                best_result = (inner_fixed, (ox_an, oy_an), S_an, oa_an)
                best_score = S_an
        except Exception:
            pass

    # Refine outer hexagon parameters using Nelder-Mead
    if time.time() < deadline - 15:
        try:
            inner_fixed = best_result[0]

            def outer_obj(params_outer):
                ox, oy, oa = params_outer
                return compute_outer_side_length(inner_fixed, (ox, oy), np.degrees(oa) % 360)

            outer_ic = np.array([best_result[1][0], best_result[1][1],
                                 np.radians(best_result[3])])
            res_outer = minimize(outer_obj, outer_ic, method='Nelder-Mead',
                                 options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})

            if res_outer.fun < best_score:
                ox, oy, oa = res_outer.x
                outer_angle = np.degrees(oa) % 360
                S = res_outer.fun
                if check_validity(inner_fixed, (ox, oy), S, outer_angle):
                    best_result = (inner_fixed, (ox, oy), S, outer_angle)
                    best_score = S
        except Exception:
            pass

    # Final refinement of all parameters
    if time.time() < deadline - 10:
        try:
            params_best = np.zeros(3 * N + 3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3 * i] = cx
                params_best[3 * i + 1] = cy
                params_best[3 * i + 2] = np.radians(ang)
            params_best[3 * N] = best_result[1][0]
            params_best[3 * N + 1] = best_result[1][1]
            params_best[3 * N + 2] = np.radians(best_result[3])

            res = minimize(objective, params_best, args=(deadline,), method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})

            if res.fun < best_score:
                inner, (ox, oy), oa = _extract(res.x)
                S = compute_outer_side_length(inner, (ox, oy), oa)
                if check_validity(inner, (ox, oy), S, oa):
                    best_result = (inner, (ox, oy), S, oa)
        except Exception:
            pass

    # Final analytical outer-hexagon tightening
    if time.time() < deadline - 5:
        try:
            inner_fixed = best_result[0]
            S_an, ox_an, oy_an, oa_an = analytic_min_outer(inner_fixed)
            if check_validity(inner_fixed, (ox_an, oy_an), S_an, oa_an) and S_an < best_score:
                best_result = (inner_fixed, (ox_an, oy_an), S_an, oa_an)
        except Exception:
            pass

    return best_result

# EVOLVE-BLOCK-END
