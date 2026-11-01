"""Improved packing: 12 unit hexagons in minimal outer hexagon."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    HEX_ANGLES = np.deg2rad(np.arange(0, 360, 60))
    # Outward normals for a unit hexagon (apothem = sqrt(3)/2)
    HEX_NORMALS = np.array([
        [np.cos(np.deg2rad(30 + k * 60)), np.sin(np.deg2rad(30 + k * 60))]
        for k in range(6)
    ])
    APOTHEM = np.sqrt(3) / 2

    def hex_verts(cx, cy, side, angle_deg):
        a = np.radians(angle_deg)
        angles = a + HEX_ANGLES
        return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

    def hex_support(cx, cy, side, angle_deg, nx, ny):
        """Maximum dot product of hex vertices with direction (nx,ny)."""
        a = np.radians(angle_deg)
        angles = a + HEX_ANGLES
        verts_x = cx + side * np.cos(angles)
        verts_y = cy + side * np.sin(angles)
        return np.max(verts_x * nx + verts_y * ny)

    def point_in_hex(px, py, cx, cy, side, angle_deg):
        """Check if point (px,py) is inside hexagon."""
        dx = px - cx
        dy = py - cy
        a = np.radians(angle_deg)
        cos_a = np.cos(-a)
        sin_a = np.sin(-a)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        for nx, ny in HEX_NORMALS:
            if lx * nx + ly * ny > side * APOTHEM + 1e-9:
                return False
        return True

    def all_points_in_hex(pts, cx, cy, side, angle_deg):
        """pts: (m,2) array. Returns True if all inside."""
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        a = np.radians(angle_deg)
        cos_a = np.cos(-a)
        sin_a = np.sin(-a)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        for nx, ny in HEX_NORMALS:
            if np.any(lx * nx + ly * ny > side * APOTHEM + 1e-9):
                return False
        return True

    def sat_disjoint(v1, v2, tol=1e-9):
        """SAT: returns True if polygons v1, v2 (6,2) are disjoint."""
        # Combined edge normals
        edges = np.vstack([np.roll(v1, -1, 0) - v1, np.roll(v2, -1, 0) - v2])
        normals = np.column_stack([-edges[:, 1], edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / norms
        for axis in normals:
            p1 = v1 @ axis
            p2 = v2 @ axis
            if np.max(p1) < np.min(p2) - tol or np.max(p2) < np.min(p1) - tol:
                return True
        return False

    def pairwise_disjoint(verts_list):
        for i in range(N):
            for j in range(i + 1, N):
                if not sat_disjoint(verts_list[i], verts_list[j]):
                    return False
        return True

    def compute_outer_side(inner, oa_deg):
        """Compute minimum outer side length for given inner config and outer angle."""
        all_verts = np.vstack([hex_verts(x, y, 1.0, a) for x, y, a in inner])
        oa = np.radians(oa_deg)
        normals = np.array([[np.cos(oa + np.deg2rad(30 + k * 60)),
                             np.sin(oa + np.deg2rad(30 + k * 60))] for k in range(6)])
        max_side = 0.0
        for nx, ny in normals:
            d = all_verts[:, 0] * nx + all_verts[:, 1] * ny
            m = np.max(d)
            if m > 0:
                s = m / APOTHEM
                if s > max_side:
                    max_side = s
        return max_side

    def objective(params):
        """Objective: minimize outer side with penalties for violations."""
        inner = [(params[3 * i], params[3 * i + 1], params[3 * i + 2]) for i in range(N)]
        ocx, ocy = params[3 * N], params[3 * N + 1]
        os = max(params[3 * N + 2], 0.5)
        oa = params[3 * N + 3]

        # Check inside outer
        verts_list = [hex_verts(x, y, 1.0, a) for x, y, a in inner]
        all_v = np.vstack(verts_list)
        oa_rad = np.radians(oa)
        o_normals = np.array([[np.cos(oa_rad + np.deg2rad(30 + k * 60)),
                               np.sin(oa_rad + np.deg2rad(30 + k * 60))] for k in range(6)])
        penalty_inside = 0.0
        for nx, ny in o_normals:
            d = (all_v[:, 0] - ocx) * nx + (all_v[:, 1] - ocy) * ny
            excess = np.maximum(d - os * APOTHEM, 0.0)
            penalty_inside += np.sum(excess ** 2)

        # Check disjoint
        penalty_disjoint = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                if not sat_disjoint(verts_list[i], verts_list[j]):
                    penalty_disjoint += 1.0

        return os + 50.0 * penalty_inside + 20.0 * penalty_disjoint

    def make_params(inner, oc, os, oa):
        p = []
        for x, y, a in inner:
            p.extend([x, y, a])
        p.extend([oc[0], oc[1], os, oa])
        return np.array(p, dtype=float)

    # --- Generate diverse initial configurations ---
    def init_cluster():
        """Hexagonal cluster: center + 6 neighbors + 5 from second ring."""
        pos = [(0.0, 0.0, 0.0)]
        for k in range(6):
            ang = np.radians(60 * k)
            pos.append((2.0 * np.cos(ang), 2.0 * np.sin(ang), 0.0))
        s3 = np.sqrt(3)
        ring2 = [(4, 0), (3, s3), (1, 2 * s3), (-1, 2 * s3), (-3, s3),
                 (-4, 0), (-3, -s3), (-1, -2 * s3), (1, -2 * s3), (3, -s3)]
        best_combo, best_s = None, float('inf')
        for combo in itertools.combinations(range(10), 5):
            tp = pos.copy()
            for idx in combo:
                tp.append((float(ring2[idx][0]), float(ring2[idx][1]), 0.0))
            s = compute_outer_side(tp, 30.0)
            if s < best_s:
                best_s, best_combo = s, combo
        for idx in best_combo:
            pos.append((float(ring2[idx][0]), float(ring2[idx][1]), 0.0))
        return pos

    def init_rows():
        """4 rows of 3, staggered."""
        pos = []
        s3 = np.sqrt(3)
        for row in range(4):
            y0 = row * 2 * s3 - 3 * s3
            x0 = (row % 2) * 2.0 - 4.0
            for col in range(3):
                pos.append((col * 4.0 + x0, y0, 0.0))
        return pos

    def init_grid():
        """3x4 grid."""
        pos = []
        s3 = np.sqrt(3)
        for i in range(3):
            for j in range(4):
                pos.append((j * 4.0 - 6.0, i * 2 * s3 - 2 * s3, 0.0))
        return pos

    def init_compact_ring():
        """Tighter arrangement: 2 center + 10 around."""
        pos = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        for k in range(10):
            ang = np.radians(36 * k + 18)
            r = 3.2
            pos.append((r * np.cos(ang), r * np.sin(ang), 0.0))
        return pos

    inits = [init_cluster(), init_rows(), init_grid(), init_compact_ring()]

    # Find best initial
    best_params = None
    best_score = float('inf')
    for inner in inits:
        for oa in [0.0, 15.0, 30.0, 45.0, 60.0]:
            os = compute_outer_side(inner, oa)
            p = make_params(inner, (0.0, 0.0), os, oa)
            s = objective(p)
            if s < best_score:
                best_score = s
                best_params = p

    # Differential evolution for global search
    if time.time() < deadline - 60:
        bounds = [(-10, 10), (-10, 10), (-180, 180)] * N + [(-5, 5), (-5, 5), (4, 12), (-90, 90)]
        try:
            de_res = differential_evolution(objective, bounds, maxiter=30, popsize=15,
                                             tol=0.01, seed=42, polish=False,
                                             update_interval=5)
            if de_res.fun < best_score:
                best_score = de_res.fun
                best_params = de_res.x.copy()
        except Exception:
            pass

    # Local refinement with multiple restarts
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 25:
            break
        if r == 0:
            p0 = best_params.copy()
        else:
            p0 = best_params + np.random.normal(0, 0.3, size=len(best_params))
        try:
            res = minimize(objective, p0, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 0.002, 'fatol': 0.002, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Final polish
    if time.time() < deadline - 10 and best_score < 9.0:
        try:
            res2 = minimize(objective, best_params, method='Nelder-Mead',
                            options={'maxiter': 400, 'xatol': 0.0005, 'fatol': 0.0005, 'adaptive': True})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass

    # Parse result
    fp = best_params
    inner = [(fp[3 * i], fp[3 * i + 1], fp[3 * i + 2]) for i in range(N)]
    oc = (fp[3 * N], fp[3 * N + 1])
    os = fp[3 * N + 2]
    oa = fp[3 * N + 3]

    # Validate; if invalid, expand outer side
    verts_list = [hex_verts(x, y, 1.0, a) for x, y, a in inner]
    if not pairwise_disjoint(verts_list):
        # Fallback to initial cluster
        inner = init_cluster()
        oa = 30.0
        os = compute_outer_side(inner, oa)
        oc = (0.0, 0.0)

    while not all_points_in_hex(np.vstack([hex_verts(x, y, 1.0, a) for x, y, a in inner]),
                                oc[0], oc[1], os, oa):
        os += 0.05
        if os > 15:
            break

    return inner, oc, os, oa
# EVOLVE-BLOCK-END
