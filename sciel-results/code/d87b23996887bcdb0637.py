# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _compute_vertex_score(vertices):
    """
    Compute the per-vertex score for a single vertex of a polygon.
    vertices: (n, 2) array of polygon vertices in cyclic order.
    Returns the score for vertex 0.
    """
    n = len(vertices)
    diffs = vertices - vertices[0]
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    other_dists = dists[1:]
    d = np.sort(other_dists)
    
    if d[0] < 0.05:
        return 0.0
    
    gaps = d[1:] - d[:-1]
    k = np.argmin(gaps)
    D = (d[k] + d[k+1]) / 2.0
    
    if D < 0.1:
        return 0.0
    
    if k == 0:
        sel = d[0:4]
    elif k == 17:
        sel = d[15:19]
    else:
        sel = d[k-1:k+3]
        
    raw = np.mean(np.maximum(sel / D, D / sel))
    score = min(1.0, 1.0 / raw)
    return score

def _polygon_min_score(polygon):
    """
    Compute the minimum vertex score over all vertices of a polygon.
    polygon: (n, 2) array.
    """
    n = len(polygon)
    centroid = np.mean(polygon, axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(centered ** 2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    normalized = centered / mean_dist
    
    # Check convexity
    edges = np.diff(np.vstack([normalized, normalized[:1]]), axis=0)
    if np.any(np.linalg.norm(edges, axis=1) < 1e-10):
        return 0.0
        
    cross = edges[:, 0] * edges[:, 1] - edges[:, 1] * edges[:, 0]
    if np.any(np.abs(cross) < 1e-10):
        return 0.0
    signs = np.sign(cross)
    if np.any(signs != signs[0]):
        return 0.0
        
    min_score = 1.0
    for i in range(n):
        rotated = np.roll(normalized, -i, axis=0)
        v_score = _compute_vertex_score(rotated)
        if v_score < min_score:
            min_score = v_score
    return min_score

def _objective(params):
    """
    Objective function for optimization.
    Minimizes the negative of the minimum vertex score.
    params: 40-d array representing 20 (x,y) coordinates.
    """
    n = 20
    poly = params.reshape(n, 2)
    
    # Quick convexity check to penalize invalid shapes heavily
    # Centroid subtraction and normalization
    centroid = np.mean(poly, axis=0)
    centered = poly - centroid
    dists_to_centroid = np.sqrt(np.sum(centered ** 2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 1.0 # Penalty
    
    normalized = centered / mean_dist
    
    # Check convexity
    edges = np.diff(np.vstack([normalized, normalized[:1]]), axis=0)
    if np.any(np.linalg.norm(edges, axis=1) < 1e-10):
        return 1.0
        
    cross = edges[:, 0] * edges[:, 1] - edges[:, 1] * edges[:, 0]
    if np.any(np.abs(cross) < 1e-10):
        return 1.0
    signs = np.sign(cross)
    if np.any(signs != signs[0]):
        return 1.0
        
    # Compute score
    score = _polygon_min_score(normalized)
    return -score

def _regular_polygon_params(n=20, radius=1.0):
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.stack([x, y], axis=1).flatten()

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices.
    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    n = 20
    
    # Initial best: Regular polygon
    best_poly = _regular_polygon_params(n).reshape(n, 2)
    best_score = _polygon_min_score(best_poly)
    
    # Phase 1: Global Search using Differential Evolution
    # Bounds: x,y in [-2, 2] is usually sufficient for normalized shapes
    bounds = [(-2.0, 2.0)] * (2 * n)
    
    # DE parameters: popsize=15, maxiter=100, tol=1e-8
    # This might be slow, so we monitor time.
    try:
        # Use a smaller popsize and fewer iterations to fit in time
        # 40 dimensions. DE is population based.
        # Let's try popsize=20, maxiter=50
        result_de = differential_evolution(
            _objective, 
            bounds, 
            maxiter=50, 
            popsize=20, 
            tol=1e-6,
            seed=42,
            polish=False, # We will polish manually
            init='sobol' # Sobol sequence for better initial spread
        )
        
        if result_de.fun < -best_score:
            candidate = result_de.x.reshape(n, 2)
            score = _polygon_min_score(candidate)
            if score > best_score:
                best_score = score
                best_poly = candidate
                
    except Exception as e:
        print(f"DE failed: {e}")
        
    if time.time() > deadline - 60.0:
        # Not enough time for local optimization, return best global result
        return best_poly

    # Phase 2: Local Optimization using Nelder-Mead
    # Start from the best candidate found by DE, or regular polygon if DE didn't improve
    x0 = best_poly.flatten()
    
    # We can try multiple local optimizations starting from slightly perturbed versions of x0
    # to avoid local minima, but Nelder-Mead on 40D is tricky.
    # Let's just do one robust Nelder-Mead run.
    
    try:
        # Nelder-Mead options
        options = {
            'maxiter': 1000,
            'xatol': 1e-8,
            'fatol': 1e-10,
            'adaptive': True
        }
        
        result_nm = minimize(
            _objective, 
            x0, 
            method='Nelder-Mead', 
            options=options
        )
        
        if result_nm.fun < -best_score:
            candidate = result_nm.x.reshape(n, 2)
            score = _polygon_min_score(candidate)
            if score > best_score:
                best_score = score
                best_poly = candidate
                
    except Exception as e:
        print(f"NM failed: {e}")
        
    # Phase 3: Coordinate Descent Polish
    # Iterate over each vertex, and optimize its x and y coordinate independently
    # using a small grid or golden section search?
    # Given time constraints, a simple coordinate-wise 1D minimization using scipy.optimize.minimize_scalar
    # might be too slow for 40 vars * iterations.
    # Instead, let's do a few more Nelder-Mead runs from perturbed starts if time permits.
    
    if time.time() < deadline - 30.0:
        # Try a few random restarts of local optimization from the best point
        for restart in range(5):
            if time.time() > deadline - 30.0:
                break
            
            # Perturb best_poly
            perturbation_scale = 0.05 * (1.0 / (restart + 1))
            x0_perturbed = best_poly.flatten() + np.random.normal(0, perturbation_scale, size=2*n)
            
            try:
                options = {
                    'maxiter': 500,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
                result_nm = minimize(
                    _objective, 
                    x0_perturbed, 
                    method='Nelder-Mead', 
                    options=options
                )
                
                if result_nm.fun < -best_score:
                    candidate = result_nm.x.reshape(n, 2)
                    score = _polygon_min_score(candidate)
                    if score > best_score:
                        best_score = score
                        best_poly = candidate
                        
            except Exception:
                pass

    return best_poly

# EVOLVE-BLOCK-END
