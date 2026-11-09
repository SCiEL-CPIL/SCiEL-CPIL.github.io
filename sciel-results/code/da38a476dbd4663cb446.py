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

def _make_polygon_from_fourier(params):
    """
    Generate a polygon from 11 Fourier parameters.
    params: [a1, b1, a2, b2, ..., a5, b5]
    r(theta) = 1 + sum_{k=1}^{5} (a_k cos(k*theta) + b_k sin(k*theta))
    """
    n = 20
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    r = np.ones(n)
    for k in range(1, 6):
        a = params[2*(k-1)]
        b = params[2*(k-1)+1]
        r += a * np.cos(k * theta) + b * np.sin(k * theta)
        
    # Ensure r > 0 to maintain star-shaped/convex potential
    # If r goes negative or too close to 0, convexity might fail, 
    # but we rely on the objective function to penalize invalid shapes.
    # However, to be safe against self-intersection or non-convexity 
    # caused by r <= 0, we can clamp or let the evaluator handle it.
    # The evaluator checks convexity strictly.
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y], axis=1)

def _objective_fourier(params):
    """
    Objective function for Fourier parameter optimization.
    Minimizes the negative of the minimum vertex score.
    """
    n = 20
    try:
        poly = _make_polygon_from_fourier(params)
        score = _polygon_min_score(poly)
        return -score
    except Exception:
        return 1.0 # Penalty for invalid shapes

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
    
    # Initial best: Regular polygon (all Fourier coeffs 0)
    best_params = np.zeros(10)
    best_poly = _make_polygon_from_fourier(best_params)
    best_score = _polygon_min_score(best_poly)
    
    # Phase 1: Global Search using Differential Evolution on 10D space
    # Bounds: [-0.5, 0.5] to ensure r stays positive-ish and shape is reasonable
    # If bounds are too large, r can go negative.
    # r_min = 1 - sum |coeffs|. If sum |coeffs| < 1, r > 0.
    # With 10 coeffs, if each is in [-0.1, 0.1], sum is at most 1.0.
    # Let's use [-0.2, 0.2] to be safe and allow some modulation.
    bounds = [(-0.2, 0.2)] * 10
    
    try:
        result_de = differential_evolution(
            _objective_fourier, 
            bounds, 
            maxiter=100, 
            popsize=30, 
            tol=1e-8,
            seed=42,
            polish=False,
            init='sobol'
        )
        
        if result_de.fun < -best_score:
            candidate_params = result_de.x
            candidate_poly = _make_polygon_from_fourier(candidate_params)
            score = _polygon_min_score(candidate_poly)
            if score > best_score:
                best_score = score
                best_poly = candidate_poly
                best_params = candidate_params
                
    except Exception as e:
        print(f"DE failed: {e}")
        
    if time.time() > deadline - 120.0:
        return best_poly

    # Phase 2: Local Optimization using Nelder-Mead on Fourier params
    # Start from the best candidate found by DE
    x0 = best_params.copy()
    
    try:
        options = {
            'maxiter': 2000,
            'xatol': 1e-10,
            'fatol': 1e-12,
            'adaptive': True
        }
        
        result_nm = minimize(
            _objective_fourier, 
            x0, 
            method='Nelder-Mead', 
            options=options
        )
        
        if result_nm.fun < -best_score:
            candidate_params = result_nm.x
            candidate_poly = _make_polygon_from_fourier(candidate_params)
            score = _polygon_min_score(candidate_poly)
            if score > best_score:
                best_score = score
                best_poly = candidate_poly
                best_params = candidate_params
                
    except Exception as e:
        print(f"NM failed: {e}")
        
    # Phase 3: Coordinate Descent / Local Refinement on Cartesian coords?
    # The Fourier space is smooth. Nelder-Mead should have done a good job.
    # We can try a few restarts from perturbed Fourier params if time allows.
    
    if time.time() < deadline - 60.0:
        for restart in range(5):
            if time.time() > deadline - 60.0:
                break
            
            # Perturb best_params
            perturbation_scale = 0.05 * (1.0 / (restart + 1))
            x0_perturbed = best_params + np.random.normal(0, perturbation_scale, size=10)
            
            try:
                options = {
                    'maxiter': 1000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
                result_nm = minimize(
                    _objective_fourier, 
                    x0_perturbed, 
                    method='Nelder-Mead', 
                    options=options
                )
                
                if result_nm.fun < -best_score:
                    candidate_params = result_nm.x
                    candidate_poly = _make_polygon_from_fourier(candidate_params)
                    score = _polygon_min_score(candidate_poly)
                    if score > best_score:
                        best_score = score
                        best_poly = candidate_poly
                        best_params = candidate_params
                        
            except Exception:
                pass

    return best_poly

# EVOLVE-BLOCK-END
