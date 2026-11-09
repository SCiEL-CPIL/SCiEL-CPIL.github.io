import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

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
            if np.linalg.norm(p[i] - p[j]) < 1e-10:
                return False
    # Cross products all same sign and nonzero
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
    # Total turning
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

    # Vectorized pairwise distances
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

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin
    
    n = 20
    best_poly = _regular_polygon()
    best_score = _compute_min_vertex_score(best_poly)
    
    if time.time() > deadline:
        return best_poly

    # --- Strategy 1: Global search in Fourier radial parameter space ---
    # r_i = 1 + a1*cos(θ_i) + a2*cos(2θ_i) + ... + a4*cos(4θ_i)
    # Using low harmonics (1-4) captures most structural variations
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    def _radial_poly_4(params):
        radii = np.ones(n)
        for k in range(1, 5):
            radii += params[k-1] * np.cos(k * base_angles)
        radii = np.maximum(radii, 0.1)
        x = radii * np.cos(base_angles)
        y = radii * np.sin(base_angles)
        return np.stack([x, y], axis=1)

    def _objective_radial_4(params):
        poly = _radial_poly_4(params)
        s = _compute_min_vertex_score(poly)
        # Add small penalty for invalid to help DE
        if s == 0.0:
            return 1.0 - 0.01 * np.sum(np.abs(params))
        return -s

    bounds = [(-0.3, 0.3)] * 4
    
    try:
        result_de = differential_evolution(
            _objective_radial_4,
            bounds,
            maxiter=100,
            popsize=20,
            tol=1e-8,
            seed=42,
            polish=True
        )
        score = -result_de.fun if result_de.fun < 0 else 0.0
        if score > best_score + 1e-9:
            best_score = score
            best_poly = _radial_poly_4(result_de.x)
    except Exception:
        pass

    if time.time() > deadline - 120:
        return best_poly

    # --- Strategy 2: Nelder-Mead refinement in Cartesian space ---
    def _objective_cart(flat_params):
        poly = flat_params.reshape(20, 2)
        return -_compute_min_vertex_score(poly)

    x0 = best_poly.flatten()
    try:
        result = minimize(
            _objective_cart,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
        )
        score = -result.fun
        if score > best_score + 1e-9:
            best_score = score
            best_poly = result.x.reshape(20, 2)
    except Exception:
        pass

    if time.time() > deadline - 60:
        return best_poly

    # --- Strategy 3: Local coordinate descent with adaptive step ---
    current_poly = best_poly.copy()
    current_score = best_score
    step_size = 0.02
    no_improve = 0
    
    while time.time() < deadline - 10 and no_improve < 40:
        improved = False
        for i in range(20):
            if time.time() > deadline - 10:
                break
            old_v = current_poly[i].copy()
            best_local = current_score
            best_new = None
            # Try 8 directions + smaller steps
            for dx in [-step_size, -step_size*0.5, 0.0, step_size*0.5, step_size]:
                for dy in [-step_size, -step_size*0.5, 0.0, step_size*0.5, step_size]:
                    if dx == 0.0 and dy == 0.0:
                        continue
                    new_poly = current_poly.copy()
                    new_poly[i] = old_v + np.array([dx, dy])
                    s = _compute_min_vertex_score(new_poly)
                    if s > best_local + 1e-12:
                        best_local = s
                        best_new = new_poly
            if best_new is not None:
                current_poly = best_new
                current_score = best_local
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
