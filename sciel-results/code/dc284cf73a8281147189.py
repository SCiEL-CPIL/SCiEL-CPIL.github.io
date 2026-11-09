# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

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
    
    # Check no repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.allclose(p[i], p[j], atol=1e-10):
                return False

    # Check strict convexity (all cross products same sign and non-zero)
    cross_products = []
    for i in range(n):
        v1 = p[(i+1) % n] - p[i]
        v2 = p[(i+2) % n] - p[(i+1) % n]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        cross_products.append(cross)
    cross_products = np.array(cross_products)
    if np.any(np.abs(cross_products) < 1e-12):
        return False
    signs = np.sign(cross_products)
    if not np.all(signs == signs[0]):
        return False
    
    # Check total turning angle is +/- 2*pi
    total_turn = 0.0
    for i in range(n):
        v1 = p[(i+1) % n] - p[i]
        v2 = p[(i+2) % n] - p[(i+1) % n]
        angle1 = np.arctan2(v1[1], v1[0])
        angle2 = np.arctan2(v2[1], v2[0])
        turn = angle2 - angle1
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

    min_score = 1.0
    for i in range(n):
        diffs = p - p[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        dists = np.delete(dists, i)
        dists_sorted = np.sort(dists)
        score = _vertex_score(dists_sorted)
        min_score = min(min_score, score)
    return min_score

def _regular_polygon():
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.stack([x, y], axis=1)

def _fourier_polygon(fourier_coeffs):
    """
    Reconstruct polygon from Fourier coefficients.
    Supports harmonics 1, 2, and 3.
    coeffs: [a1, b1, a2, b2, a3, b3]
    """
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radii = np.ones(n)
    if len(fourier_coeffs) >= 2:
        radii += fourier_coeffs[0] * np.cos(base_angles) + fourier_coeffs[1] * np.sin(base_angles)
    if len(fourier_coeffs) >= 4:
        radii += fourier_coeffs[2] * np.cos(2 * base_angles) + fourier_coeffs[3] * np.sin(2 * base_angles)
    if len(fourier_coeffs) >= 6:
        radii += fourier_coeffs[4] * np.cos(3 * base_angles) + fourier_coeffs[5] * np.sin(3 * base_angles)
    
    x = radii * np.cos(base_angles)
    y = radii * np.sin(base_angles)
    return np.stack([x, y], axis=1)

def _soft_min_vertex_score(polygon, temperature=0.02):
    """
    Computes a smooth approximation of the minimum vertex score.
    Uses log-sum-exp trick for soft-min.
    """
    n = len(polygon)
    if not np.all(np.isfinite(polygon)):
        return -1.0
        
    centroid = np.mean(polygon, axis=0)
    p = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(p**2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return -1.0
    p = p / mean_dist
    
    if not _is_valid_convex(p):
        return -1.0

    scores = np.zeros(n)
    for i in range(n):
        diffs = p - p[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        dists = np.delete(dists, i)
        dists_sorted = np.sort(dists)
        scores[i] = _vertex_score(dists_sorted)
        
    # Soft min: -T * log( (1/n) * sum(exp(-s_i/T)) )
    neg_scores_over_T = -scores / temperature
    max_neg = np.max(neg_scores_over_T)
    if max_neg > 100:
         return -1.0
         
    sum_exp = np.sum(np.exp(neg_scores_over_T - max_neg))
    log_sum = max_neg + np.log(sum_exp)
    soft_min_val = -temperature * (log_sum - np.log(n))
    
    return soft_min_val

def _optimize_with_scipy(initial_poly, max_time=580.0):
    """
    Optimize polygon vertices using scipy.optimize.minimize with a soft-min objective.
    """
    start_time = time.time()
    current_poly = initial_poly.copy()
    
    # Flatten the polygon for scipy
    x0 = current_poly.flatten()
    
    def objective(x):
        poly = x.reshape(20, 2)
        score = _soft_min_vertex_score(poly, temperature=0.02)
        if score < 0:
            return 1e6
        return -score
    
    best_poly = current_poly.copy()
    best_obj = objective(x0)
    
    # Run Nelder-Mead
    options = {
        'maxiter': 1000,
        'maxfev': 1000,
        'xatol': 1e-5,
        'fatol': 1e-5,
        'adaptive': True
    }
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options=options)
        if res.success or res.fun < best_obj:
            best_poly = res.x.reshape(20, 2)
            best_obj = res.fun
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > max_time:
        return best_poly, -best_obj if best_obj < 1e5 else 0.0
        
    # Refine with L-BFGS-B using numerical gradients
    def objective_lf(x):
        poly = x.reshape(20, 2)
        score = _soft_min_vertex_score(poly, temperature=0.01)
        if score < 0:
            return 1e6
        return -score
        
    try:
        res2 = minimize(objective_lf, best_poly.flatten(), method='L-BFGS-B', 
                        options={'maxiter': 500, 'ftol': 1e-8})
        if res2.fun < best_obj:
            best_poly = res2.x.reshape(20, 2)
            best_obj = res2.fun
    except Exception:
        pass
        
    final_score = _compute_min_vertex_score(best_poly)
    return best_poly, final_score

def _try_radial_modulation():
    """
    Try polygons with modulated radii to improve distance uniformity.
    """
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    best_poly = _regular_polygon()
    best_score = _compute_min_vertex_score(best_poly)
    
    for a in [0.01, 0.02, 0.03, 0.05, 0.08, 0.1]:
        for m in [1, 2, 3, 4, 5]:
            radii = 1.0 + a * np.cos(m * base_angles)
            x = radii * np.cos(base_angles)
            y = radii * np.sin(base_angles)
            poly = np.stack([x, y], axis=1)
            score = _compute_min_vertex_score(poly)
            if score > best_score:
                best_score = score
                best_poly = poly.copy()
                
    for a in [0.01, 0.02, 0.03, 0.05]:
        for m in [1, 2, 3]:
            angles = base_angles + a * np.sin(m * base_angles)
            order = np.argsort(angles)
            angles_sorted = angles[order]
            x = np.cos(angles_sorted)
            y = np.sin(angles_sorted)
            poly = np.stack([x, y], axis=1)
            score = _compute_min_vertex_score(poly)
            if score > best_score:
                best_score = score
                best_poly = poly.copy()
                
    return best_poly, best_score

def _optimize_fourier_global():
    """Use differential evolution to find good Fourier coefficients."""
    start_time = time.time()
    deadline = start_time + 150.0
    
    # Bounds for a1, b1, a2, b2, a3, b3
    bounds = [(-0.3, 0.3), (-0.3, 0.3), (-0.2, 0.2), (-0.2, 0.2), (-0.1, 0.1), (-0.1, 0.1)]
    
    def _negative_score_fourier(fourier_coeffs):
        poly = _fourier_polygon(fourier_coeffs)
        score = _compute_min_vertex_score(poly)
        return -score

    try:
        result = differential_evolution(
            _negative_score_fourier,
            bounds,
            maxiter=80,
            tol=1e-8,
            popsize=20,
            seed=42,
            polish=True
        )
        
        if time.time() > deadline:
            return _regular_polygon(), 0.0
        
        best_fourier = result.x
        best_score = -result.fun
        best_poly = _fourier_polygon(best_fourier)
        
        return best_poly, best_score
    except Exception:
        return _regular_polygon(), 0.0

def _optimize_fourier_local(initial_coeffs, max_time=60.0):
    """Refine Fourier coefficients using Nelder-Mead."""
    start_time = time.time()
    
    def neg_score(coeffs):
        poly = _fourier_polygon(coeffs)
        return -_compute_min_vertex_score(poly)
    
    try:
        result = minimize(
            neg_score,
            initial_coeffs,
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
        )
        
        if time.time() > start_time + max_time:
            return initial_coeffs, -neg_score(initial_coeffs)
            
        best_coeffs = result.x
        best_score = -result.fun
        return best_coeffs, best_score
    except Exception:
        return initial_coeffs, -neg_score(initial_coeffs)

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0
    
    best_poly = _regular_polygon()
    best_score = _compute_min_vertex_score(best_poly)
    
    if time.time() > deadline:
        return best_poly
        
    # Try radial modulation
    mod_poly, mod_score = _try_radial_modulation()
    if mod_score > best_score:
        best_score = mod_score
        best_poly = mod_poly
        
    if time.time() > deadline:
        return best_poly
        
    # Try Fourier optimization (global search in lower dimensional space)
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        fourier_poly, fourier_score = _optimize_fourier_global()
        if fourier_score > best_score:
            best_score = fourier_score
            best_poly = fourier_poly
            
    if time.time() > deadline:
        return best_poly
        
    # Refine with Cartesian optimization
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        opt_poly, opt_score = _optimize_with_scipy(best_poly, max_time=remaining_time)
        if opt_score > best_score:
            best_score = opt_score
            best_poly = opt_poly
            
    return best_poly

# EVOLVE-BLOCK-END
