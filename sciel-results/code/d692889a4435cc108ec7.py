# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _vertex_score(distances_sorted):
    """
    Compute the score for a single vertex given sorted distances to other vertices.
    """
    d = distances_sorted
    n = len(d)
    if n != 19:
        return 0.0
    if np.any(d < 0.05):
        return 0.0
    
    diffs = d[1:] - d[:-1]
    min_diff = np.min(diffs)
    # Find the first index k minimizing d_{k+1} - d_k
    # np.argmin returns the first occurrence
    k = int(np.argmin(diffs))
    
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
    if raw == 0:
        return 0.0
    score = min(1.0, 1.0 / raw)
    return float(score)

def _is_valid_convex(p):
    """
    Check if the polygon is strictly convex and has no repeated points.
    """
    n = len(p)
    if not np.all(np.isfinite(p)):
        return False
        
    # Check for repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.allclose(p[i], p[j], atol=1e-10):
                return False
                
    # Check convexity via cross products of consecutive edges
    cross_products = np.zeros(n)
    for i in range(n):
        v1 = p[(i+1) % n] - p[i]
        v2 = p[(i+2) % n] - p[(i+1) % n]
        cross_products[i] = v1[0] * v2[1] - v1[1] * v2[0]
        
    if np.any(np.abs(cross_products) < 1e-12):
        return False
        
    signs = np.sign(cross_products)
    if not np.all(signs == signs[0]):
        return False
        
    # Check total turning angle
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
    """
    Compute the exact minimum vertex score for the polygon.
    Handles normalization.
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
        return 0.0
        
    # Compute pairwise distances
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    min_score = 1.0
    for i in range(n):
        d = np.delete(dists[i], i)
        d_sorted = np.sort(d)
        score = _vertex_score(d_sorted)
        min_score = min(min_score, score)
        
    return min_score

def _soft_score_with_penalty(polygon, beta=50.0, penalty_weight=1000.0):
    """
    Smooth surrogate for the objective function.
    Uses soft-min for vertex scores and adds a penalty for convexity violations.
    """
    n = len(polygon)
    if n != 20:
        return -penalty_weight
    if not np.all(np.isfinite(polygon)):
        return -penalty_weight
        
    centroid = np.mean(polygon, axis=0)
    p = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(p**2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return -penalty_weight
    p = p / mean_dist
    
    # Check convexity and calculate penalty
    cross_products = np.zeros(n)
    is_convex = True
    for i in range(n):
        v1 = p[(i+1) % n] - p[i]
        v2 = p[(i+2) % n] - p[(i+1) % n]
        cross_products[i] = v1[0] * v2[1] - v1[1] * v2[0]
        
    # If any cross product is close to 0 or signs differ, it's not strictly convex
    min_abs_cross = np.min(np.abs(cross_products))
    if min_abs_cross < 1e-12:
        is_convex = False
    
    signs = np.sign(cross_products)
    if not np.all(signs == signs[0]):
        is_convex = False
        
    # Penalty for non-convexity
    # We want to maximize the score, so minimize -score + penalty
    # If not convex, return a large negative score (since we minimize -score)
    # Or simply return -penalty_weight
    
    # Compute vertex scores
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    scores = np.zeros(n)
    for i in range(n):
        d = np.delete(dists[i], i)
        d_sorted = np.sort(d)
        scores[i] = _vertex_score(d_sorted)
        
    # Soft min
    min_s = np.min(scores)
    if min_s <= 0:
        # If any score is 0, the soft min will be close to 0, but we can penalize hard
        # Actually, if min_s is 0, the objective is 0.
        # But we also need to ensure convexity.
        pass
        
    exp_scores = np.exp(-beta * (scores - min_s))
    soft_min = min_s - np.log(np.sum(exp_scores)) / beta
    
    if not is_convex:
        # Return a bad score to discourage this region
        # The penalty should be proportional to how bad the convexity is
        # For simplicity, a large negative value
        return -penalty_weight
        
    return float(soft_min)

def _regular_polygon():
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.stack([x, y], axis=1)

def _fourier_polygon(fourier_coeffs):
    """
    Reconstruct polygon from Fourier coefficients.
    Supports harmonics 1-4.
    coeffs: [a1, b1, a2, b2, a3, b3, a4, b4]
    """
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radii = np.ones(n)
    for h in range(1, 5):
        idx_a = 2 * (h - 1)
        idx_b = 2 * (h - 1) + 1
        if idx_a < len(fourier_coeffs):
            radii += fourier_coeffs[idx_a] * np.cos(h * base_angles) + fourier_coeffs[idx_b] * np.sin(h * base_angles)
            
    # Ensure radii are positive to maintain star-shaped property which helps convexity
    radii = np.maximum(radii, 0.1)
    
    x = radii * np.cos(base_angles)
    y = radii * np.sin(base_angles)
    return np.stack([x, y], axis=1)

def _neg_soft_score_fourier(fourier_coeffs):
    poly = _fourier_polygon(fourier_coeffs)
    score = _soft_score_with_penalty(poly, beta=50.0, penalty_weight=1000.0)
    return -score

def _neg_exact_score_cartesian(flat):
    poly = flat.reshape(20, 2)
    score = _compute_min_vertex_score(poly)
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin
    
    # 1. Global Search in Fourier Space
    # Use differential evolution with the soft penalty score
    bounds = [(-0.3, 0.3), (-0.3, 0.3),
              (-0.2, 0.2), (-0.2, 0.2),
              (-0.1, 0.1), (-0.1, 0.1),
              (-0.05, 0.05), (-0.05, 0.05)]
    
    best_poly = _regular_polygon()
    best_score = _compute_min_vertex_score(best_poly)
    
    if time.time() < deadline - 200:
        try:
            result_de = differential_evolution(
                _neg_soft_score_fourier,
                bounds,
                maxiter=100,
                tol=1e-6,
                popsize=20,
                seed=42,
                polish=False # We will do local polish later
            )
            
            best_fourier = result_de.x
            best_poly_fe = _fourier_polygon(best_fourier)
            best_score_fe = _compute_min_vertex_score(best_poly_fe)
            
            if best_score_fe > best_score:
                best_score = best_score_fe
                best_poly = best_poly_fe
        except Exception as e:
            pass

    # 2. Local Refinement in Fourier Space (Soft Score)
    if time.time() < deadline - 100:
        # Extract initial Fourier coefficients from best_poly
        n = 20
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        centroid = np.mean(best_poly, axis=0)
        p = best_poly - centroid
        radii = np.sqrt(np.sum(p**2, axis=1))
        mean_r = np.mean(radii)
        radii_norm = (radii - mean_r) / mean_r
        
        coeffs = np.zeros(8)
        for h in range(1, 5):
            coeffs[2*(h-1)] = np.mean(radii_norm * np.cos(h * angles))
            coeffs[2*(h-1)+1] = np.mean(radii_norm * np.sin(h * angles))
            
        try:
            result_min = minimize(
                _neg_soft_score_fourier,
                coeffs,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            best_fourier_local = result_min.x
            best_poly_fl = _fourier_polygon(best_fourier_local)
            best_score_fl = _compute_min_vertex_score(best_poly_fl)
            
            if best_score_fl > best_score:
                best_score = best_score_fl
                best_poly = best_poly_fl
        except Exception as e:
            pass

    # 3. Local Refinement in Cartesian Space (Exact Score)
    # This step is crucial for satisfying the strict constraints and maximizing the exact metric
    if time.time() < deadline - 50:
        x0 = best_poly.flatten()
        try:
            result_cart = minimize(
                _neg_exact_score_cartesian,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            
            best_poly_cart = result_cart.x.reshape(20, 2)
            best_score_cart = _compute_min_vertex_score(best_poly_cart)
            
            if best_score_cart > best_score:
                best_score = best_score_cart
                best_poly = best_poly_cart
        except Exception as e:
            pass

    # 4. Fine Coordinate Descent Polish (Exact Score)
    if time.time() < deadline - 10:
        current_poly = best_poly.copy()
        current_score = _compute_min_vertex_score(current_poly)
        best_poly_cd = current_poly.copy()
        best_score_cd = current_score
        
        step_size = 0.01
        no_improve = 0
        
        while time.time() < deadline - 5 and no_improve < 20:
            improved = False
            for i in range(20):
                if time.time() >= deadline - 5:
                    break
                old_v = current_poly[i].copy()
                # Try small perturbations
                for dx in [-step_size, 0, step_size]:
                    for dy in [-step_size, 0, step_size]:
                        if dx == 0 and dy == 0:
                            continue
                        new_poly = current_poly.copy()
                        new_poly[i] = old_v + np.array([dx, dy])
                        score = _compute_min_vertex_score(new_poly)
                        if score > current_score + 1e-12:
                            current_poly = new_poly
                            current_score = score
                            improved = True
                            if score > best_score_cd:
                                best_score_cd = score
                                best_poly_cd = current_poly.copy()
                            break
                    if improved:
                        break
            if improved:
                no_improve = 0
            else:
                no_improve += 1
                step_size *= 0.5
                
        if best_score_cd > best_score:
            best_score = best_score_cd
            best_poly = best_poly_cd

    return best_poly

# EVOLVE-BLOCK-END
