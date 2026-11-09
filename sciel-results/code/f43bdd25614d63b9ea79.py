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

def _soft_min_vertex_score(polygon, beta=30.0):
    """Smooth surrogate using soft-min (log-sum-exp) of vertex scores."""
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

    scores = np.zeros(n)
    for i in range(n):
        d = np.delete(dists[i], i)
        d_sorted = np.sort(d)
        scores[i] = _vertex_score(d_sorted)
    
    # Soft min: -log(sum(exp(-beta * s_i))) / beta
    # This approximates min(s_i) as beta -> inf
    max_s = np.max(scores)
    if max_s <= 0:
        return 0.0
    exp_terms = np.exp(-beta * (scores - max_s))
    soft_min = max_s - np.log(np.sum(exp_terms)) / beta
    return float(soft_min)

def _regular_polygon():
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0
    
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    best_poly = _regular_polygon()
    best_score = _compute_min_vertex_score(best_poly)
    
    # Radial Fourier: r_i = 1 + sum_{k=1}^{9} (a_k cos(k*theta_i) + b_k sin(k*theta_i))
    # 18 parameters
    def _radial_poly_from_params(params):
        radii = np.ones(n)
        for k in range(1, 10):
            radii += params[2*(k-1)] * np.cos(k * base_angles)
            radii += params[2*(k-1)+1] * np.sin(k * base_angles)
        radii = np.maximum(radii, 0.1)
        x = radii * np.cos(base_angles)
        y = radii * np.sin(base_angles)
        return np.stack([x, y], axis=1)
    
    def _objective_soft(params):
        poly = _radial_poly_from_params(params)
        return -_soft_min_vertex_score(poly, beta=30.0)
    
    def _objective_exact(params):
        poly = _radial_poly_from_params(params)
        return -_compute_min_vertex_score(poly)
    
    # Multiple restarts with different initial conditions
    rng = np.random.RandomState(42)
    best_radial_score = best_score
    best_radial_poly = best_poly.copy()
    
    num_restarts = 15
    for r in range(num_restarts):
        if time.time() > deadline - 90.0:
            break
        
        # Initial parameters
        if r == 0:
            ip = np.zeros(18)
        elif r < 5:
            # Small random perturbations
            ip = rng.uniform(-0.03, 0.03, 18)
        elif r < 10:
            # Low-frequency dominant
            ip = np.zeros(18)
            k = rng.randint(1, 5)
            ip[2*(k-1)] = rng.choice([-1, 1]) * 0.04
            ip[2*(k-1)+1] = rng.choice([-1, 1]) * 0.04
        else:
            # Medium frequency
            ip = np.zeros(18)
            k = rng.randint(2, 8)
            ip[2*(k-1)] = rng.choice([-1, 1]) * 0.02
            ip[2*(k-1)+1] = rng.choice([-1, 1]) * 0.02
        
        # Phase 1: Optimize with soft objective
        try:
            result = minimize(
                _objective_soft,
                ip,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
        except Exception:
            continue
        
        # Evaluate exact score
        poly = _radial_poly_from_params(result.x)
        exact_score = _compute_min_vertex_score(poly)
        
        if exact_score > best_radial_score + 1e-9:
            best_radial_score = exact_score
            best_radial_poly = poly.copy()
        
        # Phase 2: Refine with exact objective
        if time.time() < deadline - 60.0:
            try:
                result2 = minimize(
                    _objective_exact,
                    result.x,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                poly2 = _radial_poly_from_params(result2.x)
                exact_score2 = _compute_min_vertex_score(poly2)
                if exact_score2 > best_radial_score + 1e-9:
                    best_radial_score = exact_score2
                    best_radial_poly = poly2.copy()
            except Exception:
                pass
    
    if best_radial_score > best_score:
        best_score = best_radial_score
        best_poly = best_radial_poly
    
    # Phase 3: Full coordinate optimization from best radial result
    if time.time() < deadline - 45.0:
        def _objective_full(flat_params):
            poly = flat_params.reshape(20, 2)
            return -_soft_min_vertex_score(poly, beta=30.0)
        
        x0 = best_poly.flatten()
        try:
            result = minimize(
                _objective_full,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 400, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            poly = result.x.reshape(20, 2)
            exact_score = _compute_min_vertex_score(poly)
            if exact_score > best_score + 1e-9:
                best_score = exact_score
                best_poly = poly.copy()
        except Exception:
            pass
        
        # Refine with exact
        if time.time() < deadline - 30.0:
            def _objective_full_exact(flat_params):
                poly = flat_params.reshape(20, 2)
                return -_compute_min_vertex_score(poly)
            
            try:
                result2 = minimize(
                    _objective_full_exact,
                    best_poly.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                poly2 = result2.x.reshape(20, 2)
                exact_score2 = _compute_min_vertex_score(poly2)
                if exact_score2 > best_score + 1e-9:
                    best_score = exact_score2
                    best_poly = poly2.copy()
            except Exception:
                pass
    
    # Phase 4: Coordinate descent refinement
    if time.time() < deadline - 10.0:
        current_poly = best_poly.copy()
        current_score = best_score
        step_size = 0.005
        no_improve = 0
        
        while time.time() < deadline - 5.0 and no_improve < 20:
            improved = False
            for i in range(20):
                if time.time() > deadline - 5.0:
                    break
                old_v = current_poly[i].copy()
                best_local_score = current_score
                best_new_poly = None
                for dx in [-step_size, 0.0, step_size]:
                    for dy in [-step_size, 0.0, step_size]:
                        if dx == 0.0 and dy == 0.0:
                            continue
                        new_poly = current_poly.copy()
                        new_poly[i] = old_v + np.array([dx, dy])
                        s = _compute_min_vertex_score(new_poly)
                        if s > best_local_score + 1e-12:
                            best_local_score = s
                            best_new_poly = new_poly
                if best_new_poly is not None:
                    current_poly = best_new_poly
                    current_score = best_local_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_poly = current_poly.copy()
            if improved:
                no_improve = 0
            else:
                no_improve += 1
                step_size *= 0.5
                if step_size < 1e-7:
                    break
    
    return best_poly
