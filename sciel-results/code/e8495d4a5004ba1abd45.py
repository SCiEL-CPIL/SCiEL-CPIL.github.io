import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    eps = 1e-10
    D = np.sqrt(3)

    # ---- Fixed inner layout: "flower" pattern (1 center + 6 ring + 4 ring) ----
    centers = np.array([[0.0, 0.0]])

    r1 = D + eps
    for k in range(6):
        a = np.deg2rad(k * 60)
        centers = np.vstack([centers, [r1 * np.cos(a), r1 * np.sin(a)]])

    r2 = 3.0 + 2 * eps
    for k in range(4):
        a = np.deg2rad(30 + k * 60)
        centers = np.vstack([centers, [r2 * np.cos(a), r2 * np.sin(a)]])

    n_hex = len(centers)
    assert n_hex == 11

    # Base vertices of a unit hexagon, pointy-right (angle 0)
    base_vertices = np.array(
        [[np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)]
    )

    def rot_mat(phi):
        c, s = np.cos(phi), np.sin(phi)
        return np.array([[c, -s], [s, c]])

    def all_vertices(centers_arr, phis):
        """Return (66, 2) array of all inner-hexagon vertices."""
        out = np.empty((n_hex * 6, 2))
        for i in range(n_hex):
            R = rot_mat(phis[i])
            out[i * 6:(i + 1) * 6] = base_vertices @ R.T + centers_arr[i]
        return out

    def convex_hull(pts):
        """Andrew's monotone chain. Returns (m,2) hull in CCW order (no dup last)."""
        pts = np.array(pts, dtype=float)
        pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
        n = len(pts)
        if n <= 2:
            return pts
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        hull = np.array(lower[:-1] + upper[:-1])
        return hull

    def outer_normals(theta_out):
        return np.array([
            [np.cos(np.deg2rad(30 + k * 60) + theta_out),
             np.sin(np.deg2rad(30 + k * 60) + theta_out)]
            for k in range(6)
        ])

    def compute_outer_side(centers_arr, theta_out, phis):
        """Exact minimum outer side length via convex hull of all inner vertices."""
        verts = all_vertices(centers_arr, phis)
        hull = convex_hull(verts)
        normals = outer_normals(theta_out)
        dots = hull @ normals.T          # (m, 6)
        max_proj = float(np.max(dots))
        return 2.0 * max_proj / D

    def check_overlap(centers_arr, phis, tol=1e-7):
        """Check if any two inner hexagons overlap. Returns True if overlap detected."""
        # For efficiency, only check pairs that could potentially overlap
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                # Quick distance check
                dist = np.linalg.norm(centers_arr[i] - centers_arr[j])
                if dist > 2 * D + tol:  # Max possible distance for touching hexagons
                    continue
                # Get vertices
                vi = all_vertices(centers_arr[i:i+1], phis[i:i+1])
                vj = all_vertices(centers_arr[j:j+1], phis[j:j+1])
                # Check if any vertex of i is inside j, or vice versa
                if _point_in_convex_polygon(vi[0], vj):
                    return True
                if _point_in_convex_polygon(vj[0], vi):
                    return True
                # Check edge intersections (simplified: check all vertex pairs)
                for v in vi:
                    if _point_in_convex_polygon(v, vj):
                        return True
                for v in vj:
                    if _point_in_convex_polygon(v, vi):
                        return True
        return False

    def _point_in_convex_polygon(pt, poly):
        """Check if point is inside or on boundary of convex polygon."""
        n = len(poly)
        sign = 0
        for i in range(n):
            a = poly[i]
            b = poly[(i + 1) % n]
            cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
            if cross > 1e-10:
                if sign < 0:
                    return False
                sign = 1
            elif cross < -1e-10:
                if sign > 0:
                    return False
                sign = -1
        return True

    def objective(params):
        """
        params: [theta_out, cx0, cy0, phi0, cx1, cy1, phi1, ..., cx10, cy10, phi10]
        Total length: 1 + 3*n_hex
        """
        theta_out = params[0]
        rest = params[1:]
        centers_arr = rest[::3].reshape(n_hex, 2)
        phis = rest[1::3]
        side = compute_outer_side(centers_arr, theta_out, phis)
        return side

    def objective_with_penalty(params, penalty_weight=1000.0):
        """Objective with overlap penalty."""
        theta_out = params[0]
        rest = params[1:]
        centers_arr = rest[::3].reshape(n_hex, 2)
        phis = rest[1::3]
        side = compute_outer_side(centers_arr, theta_out, phis)
        
        # Simple overlap check with penalty
        penalty = 0.0
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                dist = np.linalg.norm(centers_arr[i] - centers_arr[j])
                # Minimum center distance for non-overlapping hexagons is 2*D
                if dist < 2 * D - 1e-6:
                    penalty += penalty_weight * (2 * D - dist) ** 2
        return side + penalty

    # Initial parameters: theta_out=0, then for each hex: cx, cy, phi
    x0 = np.zeros(1 + 3 * n_hex)
    x0[0] = 0.0
    for i in range(n_hex):
        x0[1 + 3*i] = centers[i][0]
        x0[2 + 3*i] = centers[i][1]
        x0[3 + 3*i] = 0.0

    best_score = float('inf')
    best_params = None

    # ---- Phase 1: Optimize only rotations (fast, safe) ----
    def objective_rotations(params):
        theta_out = params[0]
        phis = params[1:]
        return compute_outer_side(centers, theta_out, phis)

    x0_rot = np.zeros(n_hex + 1)
    best_score_rot = float('inf')
    best_params_rot = None

    n_starts = 8
    for i in range(n_starts):
        if time.time() > deadline - 40:
            break
        if i == 0:
            x_start = x0_rot.copy()
        elif i == 1:
            x_start = np.array([np.pi / 6] + [0.0] * n_hex)
        elif i == 2:
            x_start = np.array([0.0] + [np.pi / 6] * n_hex)
        elif i == 3:
            x_start = np.array([np.pi / 6] + [np.pi / 6] * n_hex)
        elif i == 4:
            x_start = np.array([0.0] + [np.pi / 12] * n_hex)
        elif i == 5:
            x_start = np.array([np.pi / 12] + [np.pi / 12] * n_hex)
        else:
            x_start = np.random.uniform(0, np.pi / 3, n_hex + 1)
        try:
            res = minimize(objective_rotations, x_start, method='Powell',
                           options={'maxiter': 300, 'disp': False,
                                    'ftol': 1e-14, 'xtol': 1e-14})
            if res.fun < best_score_rot:
                best_score_rot = res.fun
                best_params_rot = res.x.copy()
        except Exception:
            pass

    if best_params_rot is not None:
        best_score = best_score_rot
        # Convert to full parameter format
        best_params = np.zeros(1 + 3 * n_hex)
        best_params[0] = best_params_rot[0]
        for i in range(n_hex):
            best_params[1 + 3*i] = centers[i][0]
            best_params[2 + 3*i] = centers[i][1]
            best_params[3 + 3*i] = best_params_rot[1 + i]

    # ---- Phase 2: Try to optimize positions too (risky, with penalty) ----
    if time.time() < deadline - 20:
        try:
            # Start from best rotation solution
            x_start_pos = best_params.copy() if best_params is not None else x0.copy()
            res2 = minimize(objective_with_penalty, x_start_pos, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12,
                                    'adaptive': True, 'disp': False})
            
            # Validate: check for overlaps
            theta_out = res2.x[0]
            rest = res2.x[1:]
            centers_test = rest[::3].reshape(n_hex, 2)
            phis_test = rest[1::3]
            
            if not check_overlap(centers_test, phis_test):
                true_score = compute_outer_side(centers_test, theta_out, phis_test)
                if true_score < best_score - 1e-10:
                    best_score = true_score
                    best_params = res2.x.copy()
        except Exception:
            pass

    # ---- Phase 3: Polish with rotation-only optimization from best solution ----
    if time.time() < deadline - 5 and best_params is not None:
        # Extract current rotations
        theta_cur = best_params[0]
        phis_cur = best_params[3::3]
        x0_polish = np.concatenate([[theta_cur], phis_cur])
        
        try:
            res3 = minimize(objective_rotations, x0_polish, method='Nelder-Mead',
                           options={'xatol': 1e-12, 'fatol': 1e-14,
                                    'maxiter': 5000, 'disp': False})
            if res3.fun < best_score - 1e-12:
                # Update best with improved rotations
                best_params[0] = res3.x[0]
                best_params[3::3] = res3.x[1:]
                best_score = res3.fun
        except Exception:
            pass

    # ---- Final validation and return ----
    if best_params is None:
        return _fallback_solution()

    theta_out = best_params[0]
    rest = best_params[1:]
    final_centers = rest[::3].reshape(n_hex, 2)
    final_phis = rest[1::3]

    # Final overlap check
    if check_overlap(final_centers, final_phis):
        # Fall back to rotation-only solution if overlaps detected
        if best_params_rot is not None:
            theta_out = best_params_rot[0]
            final_centers = centers
            final_phis = best_params_rot[1:]
            best_score = compute_outer_side(final_centers, theta_out, final_phis)
        else:
            return _fallback_solution()

    side = best_score + 1e-9

    inner = []
    for i in range(n_hex):
        inner.append([float(final_centers[i][0]), float(final_centers[i][1]),
                      float(np.rad2deg(final_phis[i]))])

    return inner, (0.0, 0.0), float(side), float(np.rad2deg(theta_out))


def _fallback_solution():
    eps = 1e-10
    D = np.sqrt(3)
    centers = [(0.0, 0.0)]
    r1 = D + eps
    for k in range(6):
        a = np.deg2rad(k * 60)
        centers.append((r1 * np.cos(a), r1 * np.sin(a)))
    r2 = 3.0 + 2 * eps
    for k in range(4):
        a = np.deg2rad(30 + k * 60)
        centers.append((r2 * np.cos(a), r2 * np.sin(a)))

    inner = [[x, y, 30.0] for x, y in centers]

    theta_out = np.deg2rad(30)
    normals = np.array([
        [np.cos(np.deg2rad(30 + k * 60) + theta_out),
         np.sin(np.deg2rad(30 + k * 60) + theta_out)]
        for k in range(6)
    ])
    base_v = np.array([[np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))]
                       for k in range(6)])
    phi = np.deg2rad(30)
    c, s = np.cos(phi), np.sin(phi)
    rot = np.array([[c, -s], [s, c]])
    local_verts = base_v @ rot.T

    max_proj = 0.0
    for x, y in centers:
        verts = local_verts + np.array([x, y])
        dots = verts @ normals.T
        max_proj = max(max_proj, float(np.max(dots)))

    side = 2.0 * max_proj / D + 1e-9
    return inner, (0.0, 0.0), float(side), 30.0


if __name__ == "__main__":
    construct()
