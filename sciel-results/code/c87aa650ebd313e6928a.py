# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_scores(polygon):
    """Compute per-vertex scores and return the minimum."""
    n = len(polygon)
    # Normalize: subtract centroid, divide by mean distance
    centroid = polygon.mean(axis=0)
    p = polygon - centroid
    dists_to_c = np.linalg.norm(p, axis=1)
    mean_dist = dists_to_c.mean()
    if mean_dist <= 0.01:
        return 0.0
    p = p / mean_dist
    
    # Check strictly convex
    # Compute cross products of consecutive edges
    cross_products = []
    for i in range(n):
        v1 = p[(i + 1) % n] - p[i]
        v2 = p[(i + 2) % n] - p[(i + 1) % n]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        cross_products.append(cross)
    cross_products = np.array(cross_products)
    if np.any(np.abs(cross_products) < 1e-12):
        return 0.0
    if not (np.all(cross_products > 0) or np.all(cross_products < 0)):
        return 0.0
    
    # Check no repeated points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(p[i] - p[j]) < 1e-10:
                return 0.0
    
    # Total turning should be +-2*pi
    # For a closed convex polygon, this is automatically satisfied if strictly convex
    
    # Compute vertex scores
    min_score = 1.0
    for i in range(n):
        # Distances from vertex i to all others
        d = np.linalg.norm(p - p[i], axis=1)
        d[i] = np.inf  # Exclude self
        d_sorted = np.sort(d)
        
        # Check if any distance < 0.05
        if d_sorted[0] < 0.05:
            min_score = 0.0
            break
        
        # Find k: first index minimizing d_{k+1} - d_k
        gaps = d_sorted[1:] - d_sorted[:-1]
        k = np.argmin(gaps)
        
        D = (d_sorted[k] + d_sorted[k + 1]) / 2
        if D < 0.1:
            min_score = 0.0
            break
        
        # Select 4 distances
        if k == 0:
            selected = d_sorted[0:4]
        elif k == 17:
            selected = d_sorted[15:19]
        else:
            selected = d_sorted[k-1:k+3]
        
        # raw = mean of max(d/D, D/d)
        ratios = np.maximum(selected / D, D / selected)
        raw = ratios.mean()
        score = min(1.0, 1.0 / raw)
        min_score = min(min_score, score)
    
    return min_score

def _soft_min_score(polygon, temperature=0.1):
    """Soft-min surrogate for optimization."""
    n = len(polygon)
    centroid = polygon.mean(axis=0)
    p = polygon - centroid
    dists_to_c = np.linalg.norm(p, axis=1)
    mean_dist = dists_to_c.mean()
    if mean_dist <= 0.01:
        return 0.0
    p = p / mean_dist
    
    scores = []
    for i in range(n):
        d = np.linalg.norm(p - p[i], axis=1)
        d[i] = np.inf
        d_sorted = np.sort(d)
        
        if d_sorted[0] < 0.05:
            scores.append(0.0)
            continue
        
        gaps = d_sorted[1:] - d_sorted[:-1]
        k = np.argmin(gaps)
        
        D = (d_sorted[k] + d_sorted[k + 1]) / 2
        if D < 0.1:
            scores.append(0.0)
            continue
        
        if k == 0:
            selected = d_sorted[0:4]
        elif k == 17:
            selected = d_sorted[15:19]
        else:
            selected = d_sorted[k-1:k+3]
        
        ratios = np.maximum(selected / D, D / selected)
        raw = ratios.mean()
        score = min(1.0, 1.0 / raw)
        scores.append(score)
    
    scores = np.array(scores)
    # Soft min using log-sum-exp
    if np.any(scores < 1e-10):
        return 0.0
    log_scores = np.log(scores + 1e-15)
    min_log = np.min(log_scores)
    soft_min = min_log - np.log(np.sum(np.exp(log_scores - min_log)) / len(scores))
    return np.exp(soft_min)

def _regular_polygon():
    n = 20
    radius = 1.0
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.stack([x, y], axis=1)

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon optimized for high minimum vertex score.
    """
    start_time = time.time()
    deadline = 600.0
    
    # Start with regular polygon
    best_polygon = _regular_polygon()
    best_score = _compute_scores(best_polygon)
    
    # Try to improve using gradient-based optimization
    # Parametrize using small perturbations from regular polygon
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Use radial perturbation: r_i = 1 + sum of Fourier components
    # Keep it simple: try radial perturbations with even harmonics
    def _radial_to_polygon(coeffs):
        # coeffs: [a1, a2, a3, a4, a5] for cos(2k*theta) terms
        r = np.ones(n)
        for k, a in enumerate(coeffs, start=1):
            r += a * np.cos(2 * k * angles)
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        return np.stack([x, y], axis=1)
    
    def _objective(coeffs):
        polygon = _radial_to_polygon(coeffs)
        return -_soft_min_score(polygon)
    
    # Try optimization with a few initial points
    np.random.seed(42)
    for trial in range(5):
        if time.time() > deadline - 60:
            break
        
        if trial == 0:
            x0 = np.zeros(5)
        else:
            x0 = np.random.randn(5) * 0.05
        
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 200, 'ftol': 1e-10})
            if result.success or result.fun < -best_score:
                candidate = _radial_to_polygon(result.x)
                score = _compute_scores(candidate)
                if score > best_score:
                    best_score = score
                    best_polygon = candidate
        except Exception:
            pass
    
    # Also try direct vertex optimization
    def _direct_objective(flat):
        polygon = flat.reshape(20, 2)
        return -_soft_min_score(polygon)
    
    x0_direct = best_polygon.flatten()
    try:
        result = minimize(_direct_objective, x0_direct, method='L-BFGS-B',
                         options={'maxiter': 300, 'ftol': 1e-10})
        if result.fun < -best_score:
            candidate = result.x.reshape(20, 2)
            score = _compute_scores(candidate)
            if score > best_score:
                best_score = score
                best_polygon = candidate
    except Exception:
        pass
    
    return best_polygon

# EVOLVE-BLOCK-END
