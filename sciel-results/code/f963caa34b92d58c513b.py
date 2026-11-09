# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _superellipse_points(n_exp, n_pts=20, scale=1.0):
    """Generate n_pts points on a superellipse |x|^n + |y|^n = 1, evenly spaced in angle."""
    angles = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    # Superellipse parameterization: x = sign(cos)|cos|^(2/n), y = sign(sin)|sin|^(2/n)
    x = np.sign(cos_a) * np.abs(cos_a) ** (2.0 / n_exp)
    y = np.sign(sin_a) * np.abs(sin_a) ** (2.0 / n_exp)
    return np.stack([x * scale, y * scale], axis=1)


def _compute_vertex_scores(polygon):
    """Compute per-vertex scores for a polygon. Returns array of 20 scores."""
    n = len(polygon)
    scores = np.zeros(n)
    for i in range(n):
        # Distances from vertex i to all other vertices
        diffs = polygon - polygon[i]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        # Remove self-distance
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        dists = dists[mask]
        dists = np.sort(dists)  # d_0 <= d_1 <= ... <= d_18
        
        # Check: if any distance < 0.05, score = 0
        if dists[0] < 0.05:
            scores[i] = 0.0
            continue
        
        # Find k: first index minimizing d_{k+1} - d_k
        gaps = np.diff(dists)  # length 18
        k = np.argmin(gaps)
        
        D = (dists[k] + dists[k + 1]) / 2.0
        if D < 0.1:
            scores[i] = 0.0
            continue
        
        # Select four distances
        if k == 0:
            four_d = dists[0:4]
        elif k == 17:
            four_d = dists[15:19]
        else:
            four_d = dists[k - 1:k + 3]
        
        # raw = mean of max(d/D, D/d) over the four distances
        ratios = np.maximum(four_d / D, D / four_d)
        raw = np.mean(ratios)
        scores[i] = min(1.0, 1.0 / raw)
    
    return scores


def _score_polygon(polygon):
    """Compute the overall score (minimum vertex score) for a polygon."""
    if not np.all(np.isfinite(polygon)):
        return 0.0
    
    # Check strictly convex
    n = len(polygon)
    # Check no repeated points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(polygon[i] - polygon[j]) < 1e-10:
                return 0.0
    
    # Check convexity: all cross products should have same sign
    cross_products = []
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        p2 = polygon[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cp = v1[0] * v2[1] - v1[1] * v2[0]
        cross_products.append(cp)
    
    cross_products = np.array(cross_products)
    if np.any(np.abs(cross_products) < 1e-12):
        return 0.0
    
    # All cross products must have same sign
    if not (np.all(cross_products > 0) or np.all(cross_products < 0)):
        return 0.0
    
    # Total turning should be ±2π
    # For a convex polygon traversed in order, total turning is 2π
    # Check by summing angles
    angles = []
    for i in range(n):
        p0 = polygon[(i - 1) % n]
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
        # Normalize to [-pi, pi]
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        angles.append(angle)
    
    total_turning = np.sum(angles)
    if abs(abs(total_turning) - 2 * np.pi) > 0.1:
        return 0.0
    
    scores = _compute_vertex_scores(polygon)
    return np.min(scores)


def _soft_min(scores, beta=50.0):
    """Soft minimum using log-sum-exp."""
    # -log(sum(exp(-beta * x))) / beta
    min_val = np.min(scores)
    shifted = scores - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return min_val - np.log(np.sum(np.exp(-beta * shifted))) / beta


def _objective(params):
    """Objective function for optimization: negative soft-min of vertex scores."""
    n_exp = params[0]
    # Constrain n_exp to reasonable range
    n_exp = np.clip(n_exp, 1.0, 50.0)
    
    polygon = _superellipse_points(n_exp)
    
    # Normalize: subtract centroid, divide by mean distance
    centroid = np.mean(polygon, axis=0)
    polygon_norm = polygon - centroid
    mean_dist = np.mean(np.linalg.norm(polygon_norm, axis=1))
    if mean_dist <= 0.01:
        return 0.0
    polygon_norm = polygon_norm / mean_dist
    
    scores = _compute_vertex_scores(polygon_norm)
    
    # Use soft-min for smoothness
    soft_min_val = _soft_min(scores, beta=50.0)
    
    return -soft_min_val


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon optimized to maximize the minimum vertex score.
    Uses superellipse parameterization with optimized exponent.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s limit
    
    # Initial guess: regular polygon (n_exp = 1 is circle)
    # Try a range of exponents
    best_score = 0.0
    best_polygon = None
    
    # First, try a grid of exponents to find a good starting point
    exponents = np.arange(1.0, 30.0, 0.5)
    for n_exp in exponents:
        if time.time() > deadline:
            break
        polygon = _superellipse_points(n_exp)
        centroid = np.mean(polygon, axis=0)
        polygon_norm = polygon - centroid
        mean_dist = np.mean(np.linalg.norm(polygon_norm, axis=1))
        if mean_dist <= 0.01:
            continue
        polygon_norm = polygon_norm / mean_dist
        score = _score_polygon(polygon_norm)
        if score > best_score:
            best_score = score
            best_polygon = polygon_norm.copy()
    
    # Refine with local optimization around the best exponent
    if best_polygon is not None:
        # Find the exponent that gave the best score
        # Re-run to find it
        best_exp = 1.0
        for n_exp in np.arange(1.0, 30.0, 0.1):
            if time.time() > deadline:
                break
            polygon = _superellipse_points(n_exp)
            centroid = np.mean(polygon, axis=0)
            polygon_norm = polygon - centroid
            mean_dist = np.mean(np.linalg.norm(polygon_norm, axis=1))
            if mean_dist <= 0.01:
                continue
            polygon_norm = polygon_norm / mean_dist
            score = _score_polygon(polygon_norm)
            if score > best_score:
                best_score = score
                best_polygon = polygon_norm.copy()
                best_exp = n_exp
        
        # Try local optimization with L-BFGS-B
        if time.time() < deadline - 100:
            try:
                x0 = np.array([best_exp])
                result = minimize(
                    _objective, x0, method='L-BFGS-B',
                    bounds=[(1.0, 50.0)],
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if result.success or result.fun < _objective(x0):
                    n_exp_opt = np.clip(result.x[0], 1.0, 50.0)
                    polygon = _superellipse_points(n_exp_opt)
                    centroid = np.mean(polygon, axis=0)
                    polygon_norm = polygon - centroid
                    mean_dist = np.mean(np.linalg.norm(polygon_norm, axis=1))
                    if mean_dist > 0.01:
                        polygon_norm = polygon_norm / mean_dist
                        score = _score_polygon(polygon_norm)
                        if score > best_score:
                            best_score = score
                            best_polygon = polygon_norm.copy()
            except Exception:
                pass
    
    # If we didn't find anything good, return regular polygon
    if best_polygon is None:
        angles = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
        best_polygon = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    
    return best_polygon


# EVOLVE-BLOCK-END
