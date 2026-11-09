# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _is_strictly_convex(points: np.ndarray) -> bool:
    """Check if the polygon defined by points is strictly convex."""
    n = len(points)
    if n < 3:
        return False
    
    # Check for repeated points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(points[i] - points[j]) < 1e-8:
                return False

    # Check cross products for consistent sign (convexity)
    # For a CCW convex polygon, cross product of consecutive edges should be positive
    cross_products = []
    for i in range(n):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        p2 = points[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        cross_products.append(cross)
        
    cross_products = np.array(cross_products)
    
    # All cross products must be non-zero and of the same sign
    if np.any(cross_products == 0):
        return False
        
    if np.all(cross_products > 0) or np.all(cross_products < 0):
        return True
    else:
        return False

def _score_polygon(polygon: np.ndarray) -> float:
    """Compute the minimum vertex score for a given polygon."""
    n = len(polygon)
    if n != 20:
        return 0.0
        
    # Check convexity first to avoid wasting time on invalid shapes
    if not _is_strictly_convex(polygon):
        return 0.0

    # Normalize: subtract centroid, divide by mean distance
    centroid = polygon.mean(axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.sqrt((centered ** 2).sum(axis=1))
    mean_dist = dists_to_centroid.mean()
    if mean_dist <= 0.01:
        return 0.0
    normalized = centered / mean_dist
    
    min_vertex_score = 1.0
    for i in range(n):
        # Distances from vertex i to all other vertices
        diffs = normalized - normalized[i]
        d = np.sqrt((diffs ** 2).sum(axis=1))
        # Remove self-distance
        d = np.delete(d, i)
        d = np.sort(d)
        
        # Check if any distance < 0.05
        if d[0] < 0.05:
            return 0.0
        
        # Find k: first index minimizing d_{k+1} - d_k
        gaps = np.diff(d)
        k = np.argmin(gaps)
        D = (d[k] + d[k + 1]) / 2
        if D < 0.1:
            return 0.0
        
        # Select four distances
        if k == 0:
            selected = d[0:4]
        elif k == 17:  # k=17 means d[17] and d[18] are the closest gap
            selected = d[15:19]
        else:
            selected = d[k-1:k+3]
        
        # raw = mean over these four d of max(d/D, D/d)
        ratios = np.maximum(selected / D, D / selected)
        raw = ratios.mean()
        vertex_score = min(1.0, 1.0 / raw)
        min_vertex_score = min(min_vertex_score, vertex_score)
    
    return min_vertex_score


def _neg_score(params: np.ndarray) -> float:
    """Negative score for minimization. params = flattened (20, 2) coordinates."""
    polygon = params.reshape(20, 2)
    
    # Check for finite values
    if not np.all(np.isfinite(polygon)):
        return 1000.0
        
    # Quick convexity check with penalty
    if not _is_strictly_convex(polygon):
        # Calculate how "non-convex" it is to guide the optimizer back
        # Simple penalty: count negative cross products or zero
        n = 20
        cross_products = []
        for i in range(n):
            p0 = polygon[i]
            p1 = polygon[(i + 1) % n]
            p2 = polygon[(i + 2) % n]
            v1 = p1 - p0
            v2 = p2 - p1
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            cross_products.append(cross)
        cross_products = np.array(cross_products)
        # Penalty proportional to violation
        violation = np.sum(np.abs(cross_products[cross_products <= 0]))
        return 100.0 + violation * 10.0

    score = _score_polygon(polygon)
    return -score


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices.
    Uses direct coordinate optimization starting from a regular polygon.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Initial guess: Regular 20-gon
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    regular_polygon = np.stack([x, y], axis=1)
    
    # Verify regular polygon is valid
    reg_score = _score_polygon(regular_polygon)
    
    best_score = reg_score
    best_polygon = regular_polygon
    
    # Try to improve using optimization
    initial_params = regular_polygon.flatten()
    
    try:
        # Use Nelder-Mead for local optimization
        # We want to maximize score, so minimize negative score
        result = minimize(_neg_score, initial_params, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        
        if result.fun < -best_score:
            candidate = result.x.reshape(20, 2)
            # Final validation
            if _is_strictly_convex(candidate) and np.all(np.isfinite(candidate)):
                cand_score = _score_polygon(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_polygon = candidate
    except Exception:
        pass
    
    # If time allows, try a few random perturbations of the regular polygon
    if time.time() < deadline - 100:
        np.random.seed(42)
        for _ in range(20):
            if time.time() > deadline - 50:
                break
            # Small random perturbation
            noise = np.random.normal(0, 0.05, (20, 2))
            perturbed = regular_polygon + noise
            # Ensure it's still roughly convex by checking
            if _is_strictly_convex(perturbed):
                p_score = _score_polygon(perturbed)
                if p_score > best_score:
                    best_score = p_score
                    best_polygon = perturbed
                    # Try to optimize from this new point
                    try:
                        res2 = minimize(_neg_score, perturbed.flatten(), method='Nelder-Mead',
                                     options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
                        if res2.fun < -best_score:
                            cand2 = res2.x.reshape(20, 2)
                            if _is_strictly_convex(cand2):
                                s2 = _score_polygon(cand2)
                                if s2 > best_score:
                                    best_score = s2
                                    best_polygon = cand2
                    except Exception:
                        pass

    return best_polygon


# EVOLVE-BLOCK-END
