# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


def _hex_vertices(cx, cy, s, a_deg):
    """Vertices of a regular hexagon centered at (cx,cy) with side s and angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _edge_normals(hex_verts):
    """Return 6 outward unit edge normals for a hexagon given its 6 vertices in order."""
    normals = []
    centroid = hex_verts.mean(axis=0)
    for i in range(6):
        p1 = hex_verts[i]
        p2 = hex_verts[(i + 1) % 6]
        edge = p2 - p1
        n = np.array([-edge[1], edge[0]])
        mid = (p1 + p2) / 2.0
        if np.dot(n, mid - centroid) < 0:
            n = -n
        n = n / np.linalg.norm(n)
        normals.append(n)
    return np.array(normals)


def _separated(v1, n1, v2, n2):
    """
    Check if two hexagons with vertices v1 (6x2) and v2 (6x2) and normals n1, n2 (6x2)
    are separated. Returns True if separated by > 1e-6 on at least one of 12 normals.
    """
    all_normals = np.vstack([n1, n2])  # 12 x 2
    for k in range(12):
        n = all_normals[k]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if (max1 < min2 - 1e-6) or (max2 < min1 - 1e-6):
            return True
    return False


def _all_separated(inner_hexes):
    """Check all pairs of inner hexagons are separated."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _separated(inner_hexes[i][0], inner_hexes[i][1],
                              inner_hexes[j][0], inner_hexes[j][1]):
                return False
    return True


def _min_outer_side(inner_hexes):
    """
    Find the minimum side length S such that all inner hexagons fit inside
    a regular hexagon centered at origin with side S and some angle.
    Returns (S, outer_angle) or (inf, 0) if infeasible.
    """
    def _fits(S, angle_deg):
        outer_verts = _hex_vertices(0, 0, S, angle_deg)
        for hex_verts, normals in inner_hexes:
            for p in hex_verts:
                for i in range(6):
                    p1 = outer_verts[i]
                    p2 = outer_verts[(i + 1) % 6]
                    cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                    if cross < -1e-6:
                        return False
        return True

    # Compute bounding radius
    max_r = 0
    for hex_verts, normals in inner_hexes:
        r = np.max(np.linalg.norm(hex_verts, axis=1))
        max_r = max(max_r, r)

    # Binary search for minimum S with angle=0 first
    S_low = max_r
    S_high = S_low * 2.0

    for _ in range(50):
        S_mid = (S_low + S_high) / 2.0
        if _fits(S_mid, 0.0):
            S_high = S_mid
        else:
            S_low = S_mid

    best_S = S_high
    best_angle = 0.0

    # Try different angles to see if we can do better
    for angle_deg in [0, 10, 20, 30, 40, 50, 60]:
        sl = max_r * 0.95
        sh = max_r * 2.0
        for _ in range(50):
            sm = (sl + sh) / 2.0
            if _fits(sm, angle_deg):
                sh = sm
            else:
                sl = sm
        if sh < best_S:
            best_S = sh
            best_angle = angle_deg

    return best_S, best_angle


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    t_start = time.time()
    deadline = 600.0

    def _elapsed():
        return time.time() - t_start

    def _time_left():
        return deadline - _elapsed()

    # Seed arrangement: 1 center + 6 ring + 4 outer
    seed_positions = np.array(
        [
            [0, 0],
            [-2.5, 0],
            [2.5, 0],
            [-1.25, 2.165],
            [1.25, 2.165],
            [-1.25, -2.165],
            [1.25, -2.165],
            [-3.75, 2.165],
            [3.75, 2.165],
            [-3.75, -2.165],
            [3.75, -2.165],
        ]
    )

    def _build_inner_hexes(positions, angles):
        """Build list of (vertices, normals) for each inner hexagon."""
        hexes = []
        for i in range(11):
            verts = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            norms = _edge_normals(verts)
            hexes.append((verts, norms))
        return hexes

    def _objective(params):
        """Objective: minimize outer side length, with penalty for separation violations."""
        if _time_left() < 30:
            return 1e10

        positions = params[:22].reshape(11, 2)
        angles = params[22:]

        # Check for non-finite values
        if not np.all(np.isfinite(params)):
            return 1e10

        # Build hexes
        inner_hexes = _build_inner_hexes(positions, angles)

        # Check separation
        if not _all_separated(inner_hexes):
            # Penalty for violations
            penalty = 0.0
            for i in range(11):
                for j in range(i + 1, 11):
                    v1, n1 = inner_hexes[i]
                    v2, n2 = inner_hexes[j]
                    all_normals = np.vstack([n1, n2])
                    found_separation = False
                    for k in range(12):
                        n = all_normals[k]
                        proj1 = v1 @ n
                        proj2 = v2 @ n
                        gap = min(proj2.min() - proj1.max(), proj1.min() - proj2.max())
                        if gap > 1e-6:
                            found_separation = True
                            break
                    if not found_separation:
                        min_gap = min(
                            proj2.min() - proj1.max(),
                            proj1.min() - proj2.max()
                        )
                        penalty += max(0, 1e-6 - min_gap)
            if penalty > 0:
                return 1e10 + penalty

        # Find min outer side
        S, angle = _min_outer_side(inner_hexes)
        if not np.isfinite(S):
            return 1e10
        return S

    def _refine(params, method='Nelder-Mead', maxiter=500):
        """Refine parameters using the specified optimization method."""
        if _time_left() < 30:
            return params, _objective(params)
        try:
            if method == 'Nelder-Mead':
                res = minimize(
                    _objective,
                    params,
                    method='Nelder-Mead',
                    options={'maxiter': maxiter, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
                )
            elif method == 'Powell':
                res = minimize(
                    _objective,
                    params,
                    method='Powell',
                    options={'maxiter': maxiter, 'xatol': 1e-7, 'fatol': 1e-7}
                )
            else:
                res = minimize(
                    _objective,
                    params,
                    method='Nelder-Mead',
                    options={'maxiter': maxiter, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
                )
            if res.fun < _objective(params):
                return res.x, res.fun
        except Exception:
            pass
        return params, _objective(params)

    # Initial guess
    x0 = np.zeros(33)
    x0[:22] = seed_positions.flatten()
    x0[22:] = 0.0  # angles

    # Evaluate seed
    best_params = x0.copy()
    best_S, best_angle = _min_outer_side(_build_inner_hexes(seed_positions, np.zeros(11)))
    best_obj = best_S

    # Multi-start optimization with diverse methods
    rng = np.random.default_rng(42)

    # Define bounds for differential evolution
    bounds = [(-6, 6)] * 22 + [(-30, 30)] * 11

    # Phase 1: Differential evolution for global search
    if _time_left() > 300:
        try:
            de_res = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=50,
                tol=1e-6,
                popsize=15,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube'
            )
            if de_res.fun < best_obj:
                best_obj = de_res.fun
                best_params = de_res.x.copy()
        except Exception:
            pass

    # Phase 2: Multiple local refinements from best point
    if _time_left() > 60:
        # Powell refinement
        p, obj = _refine(best_params.copy(), method='Powell', maxiter=300)
        if obj < best_obj:
            best_obj = obj
            best_params = p

        # Nelder-Mead refinement
        p, obj = _refine(best_params.copy(), method='Nelder-Mead', maxiter=300)
        if obj < best_obj:
            best_obj = obj
            best_params = p

    # Phase 3: Perturbation-based multi-start from current best
    if _time_left() > 30:
        for pert_scale in [0.5, 0.2, 0.1]:
            if _time_left() < 30:
                break
            for trial in range(5):
                if _time_left() < 30:
                    break
                perturbed = best_params + rng.normal(0, pert_scale, size=33)
                # Clamp angles to reasonable range
                perturbed[22:] = np.clip(perturbed[22:], -30, 30)
                # Clamp positions
                perturbed[:22] = np.clip(perturbed[:22], -6, 6)

                # Try both Powell and Nelder-Mead
                for method in ['Powell', 'Nelder-Mead']:
                    if _time_left() < 30:
                        break
                    p, obj = _refine(perturbed.copy(), method=method, maxiter=200)
                    if obj < best_obj:
                        best_obj = obj
                        best_params = p

    # Final refinement
    if _time_left() > 10:
        p, obj = _refine(best_params.copy(), method='Nelder-Mead', maxiter=200)
        if obj < best_obj:
            best_obj = obj
            best_params = p

    # Compute final result
    positions = best_params[:22].reshape(11, 2)
    angles = best_params[22:]

    inner_hexes = _build_inner_hexes(positions, angles)
    final_S, final_angle = _min_outer_side(inner_hexes)

    inner_hex_data = np.column_stack([positions, angles])
    outer_hex_data = np.array([0.0, 0.0, final_angle])
    outer_hex_side_length = float(final_S)

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
