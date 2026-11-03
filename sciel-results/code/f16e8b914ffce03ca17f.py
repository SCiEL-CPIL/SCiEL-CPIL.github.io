import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _score(pts):
    """Compute packing radius: min(pairwise_dist/2, 1 - max_radius)."""
    if len(pts) != N:
        return -1.0
    if not np.all(np.isfinite(pts)):
        return -1.0
    # Distance from origin
    dists = np.sqrt(np.sum(pts ** 2, axis=1))
    min_boundary = 1.0 - np.max(dists)
    if min_boundary <= 0:
        return -1.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2)) / 2.0
    return min(min_boundary, min_pair)

def _ring_to_points(radii, angles_list):
    """Convert ring parameters to point array.
    radii: array of ring radii [r0, r1, r2, r3, r4]
    angles_list: list of angle arrays, one per ring (ring 0 has 1 point)
    """
    pts = [np.array([[0.0, 0.0]])]  # center
    for i in range(1, len(radii)):
        r = radii[i]
        a = angles_list[i]
        pts.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))
    return np.vstack(pts)

def _flatten_ring(radii, angles_list):
    """Flatten ring params to a 1D vector."""
    vec = list(radii)
    for i in range(1, len(angles_list)):
        vec.extend(list(angles_list[i]))
    return np.array(vec)

def _unflatten_ring(vec, ring_sizes):
    """Unflatten 1D vector to radii and angles_list."""
    radii = vec[:len(ring_sizes)].copy()
    angles_list = [np.array([0.0])]
    idx = len(ring_sizes)
    for i in range(1, len(ring_sizes)):
        n = ring_sizes[i]
        angles_list.append(vec[idx:idx + n].copy())
        idx += n
    return radii, angles_list

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 10s safety margin before 140s kill

    ring_sizes = [1, 6, 12, 18, 13]  # total 50

    # Initial ring radii estimates (based on known packing ~0.1258)
    # Outer ring at ~1 - r_circle ≈ 0.87, inner rings spaced accordingly
    init_radii = np.array([0.0, 0.252, 0.504, 0.756, 0.875])
    angles_list = []
    for i, n in enumerate(ring_sizes):
        if i == 0:
            angles_list.append(np.array([0.0]))
        else:
            # Offset alternating rings for better packing
            offset = np.pi / n if i % 2 == 0 else 0.0
            angles_list.append(np.linspace(0, 2 * np.pi, n, endpoint=False) + offset)

    x0 = _flatten_ring(init_radii, angles_list)
    best_points = None
    best_score = -1.0

    def update_best(pts):
        nonlocal best_points, best_score
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
        return s

    # --- Stage 1: L-BFGS-B on ring parameters ---
    if time.time() < deadline - 40:
        bounds = [(0.0, 0.95)] * len(ring_sizes)
        for i in range(1, len(ring_sizes)):
            bounds.extend([(0, 2 * np.pi)] * ring_sizes[i])

        def ring_objective(vec):
            if time.time() > deadline - 40:
                return 1e10
            radii, angles_list = _unflatten_ring(vec, ring_sizes)
            radii = np.clip(radii, 0.0, 0.95)
            pts = _ring_to_points(radii, angles_list)
            return -_score(pts)

        try:
            result = minimize(ring_objective, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if time.time() < deadline - 30:
                x_opt = result.x
                radii_opt, angles_opt = _unflatten_ring(x_opt, ring_sizes)
                radii_opt = np.clip(radii_opt, 0.0, 0.95)
                pts_opt = _ring_to_points(radii_opt, angles_opt)
                update_best(pts_opt)
        except Exception:
            pass

    # --- Stage 2: Nelder-Mead refinement on ring parameters ---
    if time.time() < deadline - 25 and best_points is not None:
        # Use current best as starting point if we have one
        if best_score > 0.1:
            # Reconstruct ring params from best points for better start
            # Actually, just continue from the last optimized ring params
            pass

        def ring_objective_nm(vec):
            if time.time() > deadline - 25:
                return 1e10
            radii, angles_list = _unflatten_ring(vec, ring_sizes)
            radii = np.clip(radii, 0.0, 0.95)
            pts = _ring_to_points(radii, angles_list)
            return -_score(pts)

        # Start from the result of stage 1 if available
        x_start = x0
        if 'result' in locals() and result.success:
            x_start = result.x

        try:
            result2 = minimize(ring_objective_nm, x_start, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if time.time() < deadline - 20:
                x_opt2 = result2.x
                radii_opt2, angles_opt2 = _unflatten_ring(x_opt2, ring_sizes)
                radii_opt2 = np.clip(radii_opt2, 0.0, 0.95)
                pts_opt2 = _ring_to_points(radii_opt2, angles_opt2)
                update_best(pts_opt2)
        except Exception:
            pass

    # --- Stage 3: Full point-based optimization (free positions) ---
    if time.time() < deadline - 10 and best_points is not None:
        def point_objective(vec):
            if time.time() > deadline - 10:
                return 1e10
            pts = vec.reshape(N, 2)
            s = _score(pts)
            return -s

        x0_pts = best_points.flatten()
        try:
            result3 = minimize(point_objective, x0_pts, method='Nelder-Mead',
                               options={'maxiter': 10000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True})
            if time.time() < deadline - 5:
                pts_opt3 = result3.x.reshape(N, 2)
                update_best(pts_opt3)
        except Exception:
            pass

    # --- Stage 4: Final tight polish ---
    if time.time() < deadline - 3 and best_points is not None:
        def point_objective_tight(vec):
            if time.time() > deadline - 3:
                return 1e10
            pts = vec.reshape(N, 2)
            s = _score(pts)
            return -s

        x0_final = best_points.flatten()
        try:
            result4 = minimize(point_objective_tight, x0_final, method='Nelder-Mead',
                               options={'maxiter': 20000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True})
            if time.time() < deadline - 1:
                pts_opt4 = result4.x.reshape(N, 2)
                update_best(pts_opt4)
        except Exception:
            pass

    # Fallback: hexagonal lattice
    if best_points is None or best_score < 0.05:
        spacing = 0.24
        points = []
        for j in range(-10, 11):
            for i in range(-10, 11):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.sqrt(x * x + y * y) <= 1.0 - spacing / 2.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) > N:
            # Sort by distance from center, keep closest N
            d = np.linalg.norm(points, axis=1)
            points = points[np.argsort(d, kind='stable')[:N]]
        best_points = points

    return best_points
# EVOLVE-BLOCK-END
