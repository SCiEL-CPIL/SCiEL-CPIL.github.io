# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

# EVOLVE-BLOCK-END

def _normalize(polygon):
    """Subtract centroid, divide by mean distance to centroid."""
    c = polygon.mean(axis=0)
    p = polygon - c
    dists = np.linalg.norm(p, axis=1)
    mean_d = dists.mean()
    if mean_d <= 0.01:
        return None
    return p / mean_d

def _is_strictly_convex(p):
    """Check strictly convex: no repeated points, all turns same sign nonzero, total turn +-2pi."""
    n = len(p)
    # Check no repeated points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(p[i] - p[j]) < 1e-10:
                return False
    # Compute cross products of consecutive edges
    cross_products = []
    for i in range(n):
        e1 = p[(i + 1) % n] - p[i]
        e2 = p[(i + 2) % n] - p[(i + 1) % n]
        cp = e1[0] * e2[1] - e1[1] * e2[0]
        cross_products.append(cp)
    cross_products = np.array(cross_products)
    # All must be nonzero and same sign
    if np.any(np.abs(cross_products) < 1e-10):
        return False
    signs = np.sign(cross_products)
    if len(set(signs)) != 1:
        return False
    # Total turning should be +-2*pi
    angles = np.arctan2(cross_products, 
                        np.array([(p[(i+1)%n] - p[i]) @ (p[(i+2)%n] - p[(i+1)%n]) for i in range(n)]))
    total_turn = np.sum(angles)
    if abs(abs(total_turn) - 2 * np.pi) > 1e-5:
        return False
    return True

def _vertex_score(v, polygon):
    """Compute the score for a single vertex v in the polygon."""
    dists = np.linalg.norm(polygon - v, axis=1)
    other_dists = np.sort(dists[dists > 1e-10])
    
    if len(other_dists) < 19:
        return 0.0
    
    d = other_dists[:19]
    
    if d[0] < 0.05:
        return 0.0
    
    gaps = np.diff(d)
    k = np.argmin(gaps)
    
    D = (d[k] + d[k + 1]) / 2
    
    if D < 0.1:
        return 0.0
    
    if k == 0:
        four_d = d[0:4]
    elif k == 17:
        four_d = d[15:19]
    else:
        four_d = d[k-1:k+3]
    
    ratios = np.maximum(four_d / D, D / four_d)
    raw = np.mean(ratios)
    
    return min(1.0, 1.0 / raw)

def _polygon_score(polygon):
    """Compute the overall score: minimum vertex score over all vertices."""
    n = len(polygon)
    scores = np.zeros(n)
    for i in range(n):
        scores[i] = _vertex_score(polygon[i], polygon)
    return np.min(scores)

def _evaluate(polygon):
    """Full evaluation: normalize, check convexity, compute score."""
    normed = _normalize(polygon)
    if normed is None:
        return 0.0
    if not _is_strictly_convex(normed):
        return 0.0
    return _polygon_score(normed)

def _build_polygon(params):
    """
    Build polygon from 40 parameters: 20 radii, 20 angular offsets.
    Base angles are uniform. Radii are centered around 1.
    """
    n = 20
    radii = params[:n]
    angles_offset = params[n:]
    
    # Base angles
    base_angles = 2 * np.pi * np.arange(n) / n
    
    # Apply offsets
    angles = base_angles + angles_offset
    
    # Sort by angle to ensure cyclic order (assuming offsets are small enough not to cross)
    # If offsets are large, sorting might break the "convex" assumption if it creates self-intersection,
    # but the convexity check will catch it. However, for DE to work, we want the mapping to be continuous.
    # Small offsets preserve the order.
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    
    return np.stack([x, y], axis=1)

def _objective(params):
    """
    Objective function for Differential Evolution.
    Returns negative score (to be minimized).
    """
    poly = _build_polygon(params)
    score = _evaluate(poly)
    
    # Penalize invalid polygons heavily
    if score <= 0:
        return 1.0
    
    return -score

def _regular_polygon(n=20, radius=1.0):
    """Create a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices.
    Uses Differential Evolution on a 40-dimensional space (20 radii, 20 angles).
    """
    start_time = time.time()
    deadline = start_time + 580  # 20s margin
    
    # Define bounds for Differential Evolution
    # Radii: [0.5, 1.5] to ensure convexity potential (not too flat, not too spiky)
    # Angles: [-0.1, 0.1] radians. 2*pi/20 is approx 0.314. 0.1 is about 1/3 of that, safe for order preservation.
    bounds = [(0.5, 1.5)] * 20 + [(-0.1, 0.1)] * 20
    
    # Initial population: centered around regular polygon
    # radii ~ 1, angles ~ 0
    init_bounds = [(0.9, 1.1)] * 20 + [(-0.01, 0.01)] * 20
    
    try:
        result = differential_evolution(
            _objective,
            bounds,
            maxiter=200,
            popsize=15,  # 15 * 40 = 600 individuals per generation. 200 gens = 120k evals.
            tol=1e-6,
            seed=42,
            polish=False, # We'll do our own polish or just rely on DE
            updating='immediate',
            init='sobol' # Sobol sequence for better initial coverage
        )
        
        best_params = result.x
        best_score = -result.fun
        
        # Check time
        if time.time() > deadline:
            return _build_polygon(best_params)
            
        # Local refinement using Nelder-Mead or L-BFGS-B on the best result
        # Since the objective is non-smooth (min of scores), Nelder-Mead might be more robust
        from scipy.optimize import minimize
        
        def _refine_obj(params):
            poly = _build_polygon(params)
            score = _evaluate(poly)
            if score <= 0:
                return 1.0
            return -score
        
        res = minimize(
            _refine_obj,
            best_params,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'fatol': 1e-10, 'xatol': 1e-10}
        )
        
        if res.fun < -best_score: # res.fun is negative score, so lower is better
            best_params = res.x
            best_score = -res.fun
            
    except Exception:
        # Fallback
        pass
    
    # If time allows, try a few random restarts of DE? No, DE is already global.
    # Just return the best found.
    
    if time.time() < deadline - 5:
         # Try to verify if we can improve with a second short DE run on a tighter box around best
         if best_score > 0.5: # Only if we have a decent solution
             tight_bounds = []
             for i in range(20):
                 tight_bounds.append((max(0.5, best_params[i] - 0.1), min(1.5, best_params[i] + 0.1)))
             for i in range(20, 40):
                 tight_bounds.append((max(-0.1, best_params[i] - 0.05), min(0.1, best_params[i] + 0.05)))
             
             try:
                 result2 = differential_evolution(
                     _objective,
                     tight_bounds,
                     maxiter=100,
                     popsize=10,
                     seed=123,
                     polish=False
                 )
                 if -result2.fun > best_score:
                     best_params = result2.x
                     best_score = -result2.fun
             except Exception:
                 pass
    
    final_poly = _build_polygon(best_params)
    
    # Final validation
    if _evaluate(final_poly) <= 0:
        return _regular_polygon(20)
        
    return final_poly

# EVOLVE-BLOCK-START
import numpy as np
import time

# EVOLVE-BLOCK-END
