import numpy as np
from scipy.optimize import minimize
import time

N = 50


def _score(points):
    """Compute the packing radius for a set of points."""
    if len(points) != N:
        return -1.0
    if not np.all(np.isfinite(points)):
        return -1.0
    # Distance from origin
    dists = np.sqrt(np.sum(points ** 2, axis=1))
    min_boundary = 1.0 - np.max(dists)
    if min_boundary <= 0:
        return -1.0
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2)) / 2.0
    return min(min_boundary, min_pair)


def _soft_min(values, alpha=100.0):
    """Soft minimum using log-sum-exp trick for smoothness."""
    values = np.asarray(values)
    if len(values) == 0:
        return 0.0
    # Use softmin = -softmax(-values) approach
    # For numerical stability, shift by max
    v = -values
    v_max = np.max(v)
    if not np.isfinite(v_max):
        return np.min(values)
    v_shifted = v - v_max
    exp_v = np.exp(v_shifted)
    log_sum = v_max + np.log(np.sum(exp_v))
    return -log_sum


def _smooth_objective(points, alpha=100.0):
    """Smooth surrogate for the packing radius optimization.
    
    Returns a smooth approximation of -min(min_boundary, min_pair_dist/2).
    Lower is better (we minimize this).
    """
    points = points.reshape(-1, 2)
    if len(points) != N:
        return 1e10
    if not np.all(np.isfinite(points)):
        return 1e10
    
    # Distance from origin
    dists = np.sqrt(np.sum(points ** 2, axis=1))
    boundary_terms = 1.0 - dists  # Each point's distance to boundary
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    pair_dists = np.sqrt(d2) / 2.0
    # Only take upper triangle to avoid duplicates
    iu = np.triu_indices(N, k=1)
    pair_terms = pair_dists[iu]
    
    # Combine all constraints into a single smooth minimum
    all_terms = np.concatenate([boundary_terms, pair_terms])
    
    # If any term is very negative (violation), penalize heavily
    min_term = np.min(all_terms)
    if min_term < -0.01:
        # Heavy penalty for violations
        return 10.0 * min_term + 1e10 * (min_term ** 2)
    
    # Smooth min
    sm = _soft_min(all_terms, alpha)
    return -sm


def _build_ring_params(radii, angles):
    """Build point array from ring parameters.
    
    radii: array of ring radii
    angles: list of angle arrays, one per ring
    """
    pts = [np.zeros((0, 2))]
    for i, r in enumerate(radii):
        if i == 0:
            # Center
            pts.append(np.array([[0.0, 0.0]]))
        else:
            n = len(angles[i])
            a = np.array(angles[i])
            pts.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))
    return np.vstack(pts)


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit

    # Best packing found so far
    best_points = None
    best_score = -1.0

    def update_best(pts):
        nonlocal best_points, best_score
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
        return s

    # Initial packing: concentric rings
    ring_sizes = [1, 6, 12, 18, 13]
    init_radii = np.array([0.0, 0.25, 0.50, 0.70, 0.85])

    angles_list = []
    for i, n in enumerate(ring_sizes):
        if i == 0:
            angles_list.append(np.array([0.0]))
        else:
            angles_list.append(np.linspace(0, 2 * np.pi, n, endpoint=False))

    # Build initial points from ring structure
    init_points = _build_ring_params(init_radii, angles_list)
    update_best(init_points)

    # Phase 1: L-BFGS-B optimization with smooth objective on free points
    if time.time() - start_time < deadline - 30:
        def smooth_obj(vec):
            if time.time() - start_time > deadline:
                return 1e10
            pts = vec.reshape(N, 2)
            return _smooth_objective(pts, alpha=200.0)

        def smooth_grad(vec):
            """Numerical gradient of smooth objective."""
            pts = vec.reshape(N, 2)
            eps = 1e-7
            grad = np.zeros_like(vec)
            for i in range(len(vec)):
                vec_plus = vec.copy()
                vec_minus = vec.copy()
                vec_plus[i] += eps
                vec_minus[i] -= eps
                fp = _smooth_objective(vec_plus.reshape(N, 2), alpha=200.0)
                fm = _smooth_objective(vec_minus.reshape(N, 2), alpha=200.0)
                grad[i] = (fp - fm) / (2 * eps)
            return grad

        x0 = init_points.flatten()
        try:
            result = minimize(smooth_obj, x0, jac=smooth_grad, method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8,
                                       'maxfun': 5000})
            if time.time() - start_time < deadline - 20:
                pts_opt = result.x.reshape(N, 2)
                update_best(pts_opt)
        except Exception:
            pass

    # Phase 2: Nelder-Mead polish on the raw score (nonsmooth but local)
    if time.time() - start_time < deadline - 15 and best_points is not None:
        def nm_obj(vec):
            if time.time() - start_time > deadline:
                return 1e10
            pts = vec.reshape(N, 2)
            s = _score(pts)
            if s < 0:
                return 1e10 - s  # Penalize invalid
            return -s

        x0 = best_points.flatten()
        try:
            result = minimize(nm_obj, x0, method='Nelder-Mead',
                              options={'maxiter': 30000, 'xatol': 1e-10, 'fatol': 1e-12,
                                       'adaptive': True})
            if time.time() - start_time < deadline - 5:
                pts_opt = result.x.reshape(N, 2)
                update_best(pts_opt)
        except Exception:
            pass

    # Phase 3: Final tight polish
    if time.time() - start_time < deadline - 5 and best_points is not None:
        def nm_obj_tight(vec):
            if time.time() - start_time > deadline:
                return 1e10
            pts = vec.reshape(N, 2)
            s = _score(pts)
            if s < 0:
                return 1e10 - s
            return -s

        x0 = best_points.flatten()
        try:
            result = minimize(nm_obj_tight, x0, method='Nelder-Mead',
                              options={'maxiter': 50000, 'xatol': 1e-12, 'fatol': 1e-14,
                                       'adaptive': True})
            if time.time() - start_time < deadline - 2:
                pts_opt = result.x.reshape(N, 2)
                update_best(pts_opt)
        except Exception:
            pass

    # Final check and return
    if best_points is None:
        # Fallback: hexagonal grid
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
            points = points[:N]
        best_points = points

    return best_points
