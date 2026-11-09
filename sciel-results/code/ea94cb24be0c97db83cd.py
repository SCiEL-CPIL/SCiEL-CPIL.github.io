import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, L_BFGS_B

def _vertex_score(distances_sorted):
    d = distances_sorted
    n = len(d)
    if n != 19:
        return 0.0
    if np.any(d < 0.05):
        return 0.0
    diffs = d[1:] - d[:-1]
    min_diff = np.min(diffs)
    k = int(np.argmax(diffs == min_diff))
    D = (d[k] + d[k + 1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        four_d = d[0:4]
    elif k == 17:
        four_d = d[15:19]
    else:
        four_d = d[k-1:k+3]
    if np.any(four_d <= 0):
        return 0.0
    ratios = np.maximum(four_d / D, D / four_d)
    raw = np.mean(ratios)
    score = min(1.0, 1.0 / raw) if raw > 0 else 0.0
    return float(score)

def _is_valid_convex(p):
    n = len(p)
    if not np.all(np.isfinite(p)):
        return False
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(p[i] - p[j]) < 1e-10:
                return False
    cross_products = np.zeros(n)
    for i in range(n):
        e1 = p[(i+1) % n] - p[i]
        e2 = p[(i+2) % n] - p[(i+1) % n]
        cross_products[i] = e1[0] * e2[1] - e1[1] * e2[0]
    if np.any(np.abs(cross_products) < 1e-14):
        return False
    signs = np.sign(cross_products)
    if not np.all(signs == signs[0]):
        return False
    total_turn = 0.0
    for i in range(n):
        v1 = p[(i+1) % n] - p[i]
        v2 = p[(i+2) % n] - p[(i+1) % n]
        ang1 = np.arctan2(v1[1], v1[0])
        ang2 = np.arctan2(v2[1], v2[0])
        turn = ang2 - ang1
        while turn > np.pi:
            turn -= 2 * np.pi
        while turn < -np.pi:
            turn += 2 * np.pi
        total_turn += turn
    if not np.isclose(abs(total_turn), 2 * np.pi, atol=1e-6):
        return False
    return True

def _compute_min_vertex_score(polygon):
    n = len(polygon)
    if n != 20:
        return 0.0
    if not np.all(np.isfinite(polygon)):
        return 0.0
    centroid = np.mean(polygon, axis=0)
    p = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(p**2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    p = p / mean_dist

    if not _is_valid_convex(p):
        return 0.0

    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))

    min_score = 1.0
    for i in range(n):
        d = np.delete(dists[i], i)
        d_sorted = np.sort(d)
        score = _vertex_score(d_sorted)
        min_score = min(min_score, score)
    return min_score

def _regular_polygon():
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)

def _soft_min_vertex_score(polygon, beta=20.0):
    """
    Smooth surrogate for the minimum vertex score using log-sum-exp.
    """
    n = len(polygon)
    if n != 20:
        return 0.0
    if not np.all(np.isfinite(polygon)):
        return 0.0
    centroid = np.mean(polygon, axis=0)
    p = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(p**2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    p = p / mean_dist

    if not _is_valid_convex(p):
        return -1000.0

    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))

    scores = np.zeros(n)
    for i in range(n):
        d = np.delete(dists[i], i)
        d_sorted = np.sort(d)
        score = _vertex_score(d_sorted)
        scores[i] = score

    # Soft min: -log(sum(exp(-beta * s_i))) / beta
    # This is approximately min(s_i) when beta is large
    if np.any(scores < 0):
        return -1000.0
    
    # Use log-sum-exp trick for numerical stability
    shifted = -beta * scores
    max_shifted = np.max(shifted)
    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    soft_min = -log_sum_exp / beta
    
    return float(soft_min)

def _angular_to_poly(angle_params):
    """
    Convert angular perturbation parameters to polygon vertices.
    angle_params: 20-element array of angular perturbations.
    """
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    angles = base_angles + angle_params
    
    # Ensure angles are in increasing order (they should be if perturbations are small)
    # But let's normalize to ensure they span 2*pi
    angles = np.sort(angles)
    # Re-center
    angles = angles - np.mean(angles)
    
    # Radii: start with 1, can be perturbed too if we add more params
    radii = np.ones(n)
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)

def _fourier_to_poly(params):
    """
    Convert Fourier parameters to polygon vertices.
    params: [a1, b1, a2, b2, a3, b3, a4, b4] for harmonics 1-4
    """
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    radii = np.ones(n)
    # Add harmonic perturbations
    radii += params[0] * np.cos(base_angles) + params[1] * np.sin(base_angles)
    radii += params[2] * np.cos(2 * base_angles) + params[3] * np.sin(2 * base_angles)
    radii += params[4] * np.cos(3 * base_angles) + params[5] * np.sin(3 * base_angles)
    radii += params[6] * np.cos(4 * base_angles) + params[7] * np.sin(4 * base_angles)
    
    radii = np.maximum(radii, 0.1)
    
    x = radii * np.cos(base_angles)
    y = radii * np.sin(base_angles)
    return np.stack([x, y], axis=1)

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 620.0
    
    n = 20
    best_poly = _regular_polygon()
    best_score = _compute_min_vertex_score(best_poly)
    
    if time.time() > deadline - 10:
        return best_poly
    
    # Phase 1: Differential evolution in Fourier space (8 params for harmonics 1-4)
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    def _objective_fourier(params):
        poly = _fourier_to_poly(params)
        s = _soft_min_vertex_score(poly, beta=20.0)
        return -s
    
    bounds = [(-0.2, 0.2)] * 8
    
    best_de_score = best_score
    best_de_poly = best_poly.copy()
    
    # Run DE with multiple seeds
    for seed in [42, 123, 456]:
        if time.time() > deadline - 120:
            break
        try:
            result_de = differential_evolution(
                _objective_fourier,
                bounds,
                seed=seed,
                maxiter=150,
                tol=1e-12,
                polish=False,
                init='sobol',
                popsize=20
            )
            if result_de.fun is not None and np.isfinite(result_de.fun):
                de_score = -result_de.fun
                if de_score > best_de_score + 1e-9:
                    best_de_score = de_score
                    best_de_poly = _fourier_to_poly(result_de.x)
        except Exception:
            pass
    
    if best_de_score > best_score:
        best_score = best_de_score
        best_poly = best_de_poly
    
    if time.time() > deadline - 90:
        return best_poly
    
    # Phase 2: L-BFGS-B refinement in Fourier space
    if time.time() < deadline - 60:
        # Extract initial Fourier coefficients from best_poly
        # For simplicity, start from zero and a few other inits
        inits = [
            np.zeros(8),
            np.array([0.05, 0.03, 0.02, 0.01, 0.01, 0.005, 0.005, 0.003]),
            np.array([-0.05, 0.03, -0.02, 0.01, 0.01, -0.005, 0.005, 0.003]),
        ]
        
        for init in inits:
            if time.time() > deadline - 45:
                break
            try:
                res = minimize(
                    _objective_fourier,
                    init,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if res.fun is not None and np.isfinite(res.fun):
                    s = -res.fun
                    if s > best_score + 1e-9:
                        best_score = s
                        best_poly = _fourier_to_poly(res.x)
            except Exception:
                pass
    
    if time.time() > deadline - 45:
        return best_poly
    
    # Phase 3: Angular perturbation optimization
    def _objective_angular(params):
        poly = _angular_to_poly(params)
        s = _soft_min_vertex_score(poly, beta=20.0)
        return -s
    
    # Bounds for angular perturbations: keep them small to maintain convexity
    ang_bounds = [(-0.3, 0.3)] * 20
    
    best_ang_score = best_score
    best_ang_poly = best_poly.copy()
    
    for seed in [42, 789]:
        if time.time() > deadline - 30:
            break
        try:
            result_ang = differential_evolution(
                _objective_angular,
                ang_bounds,
                seed=seed,
                maxiter=100,
                tol=1e-12,
                polish=False,
                init='sobol',
                popsize=15
            )
            if result_ang.fun is not None and np.isfinite(result_ang.fun):
                ang_score = -result_ang.fun
                if ang_score > best_ang_score + 1e-9:
                    best_ang_score = ang_score
                    best_ang_poly = _angular_to_poly(result_ang.x)
        except Exception:
            pass
    
    if best_ang_score > best_score:
        best_score = best_ang_score
        best_poly = best_ang_poly
    
    if time.time() > deadline - 30:
        return best_poly
    
    # Phase 4: Full coordinate optimization in Cartesian space with L-BFGS-B
    def _objective_full(flat_params):
        poly = flat_params.reshape(20, 2)
        s = _soft_min_vertex_score(poly, beta=20.0)
        return -s
    
    x0 = best_poly.flatten()
    try:
        result = minimize(
            _objective_full,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        if result.fun is not None and np.isfinite(result.fun):
            s = -result.fun
            if s > best_score + 1e-9:
                best_score = s
                best_poly = result.x.reshape(20, 2)
    except Exception:
        pass
    
    if time.time() > deadline - 20:
        return best_poly
    
    # Phase 5: Nelder-Mead polish
    def _objective_exact(flat_params):
        poly = flat_params.reshape(20, 2)
        return -_compute_min_vertex_score(poly)
    
    x0 = best_poly.flatten()
    try:
        result = minimize(
            _objective_exact,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        if result.fun is not None and np.isfinite(result.fun):
            s = -result.fun
            if s > best_score + 1e-9:
                best_score = s
                best_poly = result.x.reshape(20, 2)
    except Exception:
        pass
    
    return best_poly
