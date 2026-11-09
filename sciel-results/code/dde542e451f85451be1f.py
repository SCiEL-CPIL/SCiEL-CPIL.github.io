# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

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
    # Cross products all same sign
    cross_products = np.array([
        (p[(i+1)%n] - p[i])[0] * (p[(i+2)%n] - p[(i+1)%n])[1] -
        (p[(i+1)%n] - p[i])[1] * (p[(i+2)%n] - p[(i+1)%n])[0]
        for i in range(n)
    ])
    if np.any(np.abs(cross_products) < 1e-12):
        return False
    signs = np.sign(cross_products)
    if not np.all(signs == signs[0]):
        return False
    # Total turning angle
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
    
    # Vectorized distance computation
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    min_score = 1.0
    for i in range(n):
        d = np.delete(dists[i], i)
        d_sorted = np.sort(d)
        score = _vertex_score(d_sorted)
        min_score = min(min_score, score)
    return min_score

def _fourier_polygon(fourier_coeffs):
    """
    Generate a polygon from Fourier coefficients.
    fourier_coeffs: array of shape (max_order+1, 2) where [m, 0] is cos coeff, [m, 1] is sin coeff
    Returns 20x2 array of points.
    """
    n = 20
    max_order = len(fourier_coeffs) - 1
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = np.zeros(n)
    y = np.zeros(n)
    for m in range(1, max_order + 1):
        x += fourier_coeffs[m, 0] * np.cos(m * angles)
        y += fourier_coeffs[m, 1] * np.sin(m * angles)
    return np.stack([x, y], axis=1)

def _negative_fourier_score(flat_coeffs):
    """
    Objective for Fourier-space optimization.
    flat_coeffs: flattened array of shape (max_order+1)*2
    Returns negative min vertex score.
    """
    max_order = int(np.sqrt(len(flat_coeffs) / 2))
    fourier_coeffs = flat_coeffs.reshape(max_order + 1, 2)
    poly = _fourier_polygon(fourier_coeffs)
    score = _compute_min_vertex_score(poly)
    return -score

def _optimize_fourier_global(max_order=5, max_time=120.0):
    """
    Global optimization in Fourier space using differential evolution.
    """
    from scipy.optimize import differential_evolution
    
    start_time = time.time()
    
    # Bounds for Fourier coefficients
    # m=0 is the mean (should be 0 for centered polygon)
    # For m>=1, coefficients should be small to maintain convexity
    bounds = [(0, 0)]  # m=0 fixed at origin
    for m in range(1, max_order + 1):
        # Larger m -> smaller amplitude to maintain convexity
        bound = 0.5 / (m * m + 1)
        bounds.append((-bound, bound))
        bounds.append((-bound, bound))
    
    result = differential_evolution(
        _negative_fourier_score,
        bounds,
        maxiter=50,
        popsize=15,
        tol=1e-8,
        seed=42,
        polish=False,
        callback=lambda x, convergence: None
    )
    
    # Check time
    if time.time() - start_time > max_time:
        # Return best found so far (result.x should be available)
        pass
    
    best_coeffs = result.x
    best_poly = _fourier_polygon(best_coeffs.reshape(max_order + 1, 2))
    best_score = _compute_min_vertex_score(best_poly)
    
    return best_poly, best_score, best_coeffs

def _optimize_fourier_local(initial_coeffs, max_order=5, max_time=120.0):
    """
    Local refinement in Fourier space using Nelder-Mead.
    """
    start_time = time.time()
    
    x0 = initial_coeffs.flatten()
    
    result = minimize(
        _negative_fourier_score,
        x0,
        method='Nelder-Mead',
        options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
    )
    
    best_coeffs = result.x.reshape(max_order + 1, 2)
    best_poly = _fourier_polygon(best_coeffs)
    best_score = _compute_min_vertex_score(best_poly)
    
    return best_poly, best_score, best_coeffs

def _optimize_scipy_from_poly(initial_poly, max_time=100.0):
    """
    Local refinement in Cartesian space using Nelder-Mead.
    """
    start_time = time.time()
    
    best_poly = initial_poly.copy()
    best_score = _compute_min_vertex_score(best_poly)
    
    x0 = initial_poly.flatten()
    
    def neg_score(x):
        poly = x.reshape(20, 2)
        return -_compute_min_vertex_score(poly)
    
    result = minimize(
        neg_score,
        x0,
        method='Nelder-Mead',
        options={'maxiter': 150, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
    )
    
    if result.fun < -best_score:
        best_poly = result.x.reshape(20, 2)
        best_score = -result.fun
    
    return best_poly, best_score

def _coordinate_descent(poly, max_time=60.0):
    """
    Fine-grained coordinate descent in Cartesian space.
    """
    start_time = time.time()
    current_poly = poly.copy()
    current_score = _compute_min_vertex_score(current_poly)
    best_poly = current_poly.copy()
    best_score = current_score
    
    step_size = 0.01
    no_improve = 0
    
    while time.time() - start_time < max_time and no_improve < 30:
        improved = False
        for i in range(20):
            if time.time() - start_time > max_time:
                break
            old_v = current_poly[i].copy()
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    new_poly = current_poly.copy()
                    new_poly[i] = old_v + np.array([dx, dy])
                    score = _compute_min_vertex_score(new_poly)
                    if score > current_score + 1e-13:
                        current_poly = new_poly
                        current_score = score
                        improved = True
                        if score > best_score:
                            best_score = score
                            best_poly = current_poly.copy()
                        break
                if improved:
                    break
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            step_size *= 0.5
    
    return best_poly, best_score

def _regular_polygon():
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.stack([x, y], axis=1)

def _try_symmetric_patterns():
    """
    Try various symmetric patterns to find good starting points.
    """
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    best_poly = _regular_polygon()
    best_score = _compute_min_vertex_score(best_poly)
    
    # Radial modulation: r_i = 1 + a * cos(m * theta_i)
    for a in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15]:
        for m in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            radii = 1.0 + a * np.cos(m * base_angles)
            x = radii * np.cos(base_angles)
            y = radii * np.sin(base_angles)
            poly = np.stack([x, y], axis=1)
            score = _compute_min_vertex_score(poly)
            if score > best_score:
                best_score = score
                best_poly = poly.copy()
    
    # Sinusoidal radial modulation
    for a in [0.01, 0.02, 0.03, 0.05]:
        for m in [1, 2, 3, 4, 5]:
            radii = 1.0 + a * np.sin(m * base_angles)
            x = radii * np.cos(base_angles)
            y = radii * np.sin(base_angles)
            poly = np.stack([x, y], axis=1)
            score = _compute_min_vertex_score(poly)
            if score > best_score:
                best_score = score
                best_poly = poly.copy()
    
    return best_poly, best_score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin before 630s limit
    
    # Start with regular polygon
    best_poly = _regular_polygon()
    best_score = _compute_min_vertex_score(best_poly)
    
    if time.time() > deadline:
        return best_poly
    
    # Try symmetric patterns to find good starting point
    sym_poly, sym_score = _try_symmetric_patterns()
    if sym_score > best_score:
        best_score = sym_score
        best_poly = sym_poly
    
    if time.time() > deadline:
        return best_poly
    
    # Global Fourier optimization
    remaining = deadline - time.time()
    if remaining > 30:
        fourier_time = min(120.0, remaining * 0.4)
        if fourier_time > 20:
            f_poly, f_score, f_coeffs = _optimize_fourier_global(max_order=5, max_time=fourier_time)
            if f_score > best_score:
                best_score = f_score
                best_poly = f_poly
    
    if time.time() > deadline:
        return best_poly
    
    # Local Fourier refinement
    remaining = deadline - time.time()
    if remaining > 20:
        local_fourier_time = min(80.0, remaining * 0.3)
        if local_fourier_time > 10:
            # Use the best Fourier coefficients if available, else derive from best_poly
            # For simplicity, just do local optimization from current best in Cartesian
            l_poly, l_score = _optimize_scipy_from_poly(best_poly, max_time=local_fourier_time)
            if l_score > best_score:
                best_score = l_score
                best_poly = l_poly
    
    if time.time() > deadline:
        return best_poly
    
    # Coordinate descent refinement
    remaining = deadline - time.time()
    if remaining > 10:
        cd_time = min(60.0, remaining - 5.0)
        if cd_time > 5:
            cd_poly, cd_score = _coordinate_descent(best_poly, max_time=cd_time)
            if cd_score > best_score:
                best_score = cd_score
                best_poly = cd_poly
    
    return best_poly

# EVOLVE-BLOCK-END
