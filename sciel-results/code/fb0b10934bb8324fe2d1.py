# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

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

def _polygon_from_params(params):
    """
    Convert parameters to polygon vertices.
    params: array of shape (40,).
    First 20: log(radii)
    Next 19: log(angles) - the 20th angle is determined by sum=2*pi
    """
    n = 20
    log_r = params[:n]
    log_theta = params[n:n+19]
    
    r = np.exp(log_r)
    # Ensure angles are positive and sum to 2*pi
    # Use softmax-like or just normalize positive values
    theta_raw = np.exp(log_theta)
    theta_20 = 2.0 * np.pi - np.sum(theta_raw)
    
    # If theta_20 is not positive, the parameterization is invalid for a simple star-shaped polygon
    # We handle this in the objective by returning a penalty
    if theta_20 <= 1e-6:
        return None
        
    theta = np.concatenate([theta_raw, [theta_20]])
    
    # Cumulative angles
    cum_theta = np.cumsum(theta)
    # Subtract the first angle to start at 0? No, just use cumsum.
    # Vertices are (r_i * cos(cum_theta_i), r_i * sin(cum_theta_i))
    x = r * np.cos(cum_theta)
    y = r * np.sin(cum_theta)
    
    return np.stack([x, y], axis=1)

def _objective(params):
    """
    Objective function for optimization.
    Minimizes the negative of the minimum vertex score.
    """
    n = 20
    poly = _polygon_from_params(params)
    if poly is None:
        return 100.0
        
    # Quick convexity check
    centroid = np.mean(poly, axis=0)
    centered = poly - centroid
    dists_to_centroid = np.sqrt(np.sum(centered ** 2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 100.0
        
    normalized = centered / mean_dist
    
    edges = np.diff(np.vstack([normalized, normalized[:1]]), axis=0)
    if np.any(np.linalg.norm(edges, axis=1) < 1e-10):
        return 100.0
        
    cross = edges[:, 0] * edges[:, 1] - edges[:, 1] * edges[:, 0]
    if np.any(np.abs(cross) < 1e-10):
        return 100.0
    signs = np.sign(cross)
    if np.any(signs != signs[0]):
        # Penalty for non-convexity
        # Scale penalty by how much it violates
        min_cross = np.min(np.abs(cross))
        return 100.0 + (1.0 / (min_cross + 1e-6))
        
    # Compute score
    score = _polygon_min_score(normalized)
    if score == 0.0:
        return 50.0
    return -score

def _regular_polygon_params(n=20, radius=1.0):
    """
    Generate initial parameters for a regular polygon.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    # r_i = radius
    # theta_i = 2*pi/n
    log_r = np.full(n, np.log(radius))
    log_theta = np.full(n-1, np.log(2.0 * np.pi / n))
    
    return np.concatenate([log_r, log_theta])

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices.
    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    start_time = time.time()
    deadline = start_time + 590.0 # Safety margin
    n = 20
    
    # Initial best: Regular polygon
    best_params = _regular_polygon_params(n)
    best_poly = _polygon_from_params(best_params)
    best_score = _polygon_min_score(best_poly)
    
    # Optimization using L-BFGS-B
    # Bounds for log_r: [log(0.1), log(5.0)] -> [-2.3, 1.6]
    # Bounds for log_theta: [log(0.1), log(2*pi)] -> [-2.3, 1.8]
    # But we need to ensure the sum constraint doesn't break.
    # We'll use broad bounds.
    bounds_r = [(-2.0, 2.0)] * n
    bounds_theta = [(-2.0, 2.0)] * (n-1)
    bounds = bounds_r + bounds_theta
    
    # Try multiple restarts with different initial perturbations if time permits
    # Start with the regular polygon
    x0 = best_params
    
    # Run L-BFGS-B
    try:
        result = minimize(
            _objective, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.fun < -best_score:
            candidate_params = result.x
            candidate_poly = _polygon_from_params(candidate_params)
            if candidate_poly is not None:
                score = _polygon_min_score(candidate_poly)
                if score > best_score:
                    best_score = score
                    best_poly = candidate_poly
                    best_params = candidate_params
                    
    except Exception as e:
        print(f"L-BFGS-B failed: {e}")
        
    # If time remains, try a second optimization from a slightly perturbed start
    if time.time() < deadline - 60.0:
        # Perturb the best params
        np.random.seed(42)
        x0_perturbed = best_params + np.random.normal(0, 0.1, size=best_params.shape)
        
        try:
            result2 = minimize(
                _objective, 
                x0_perturbed, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result2.fun < -best_score:
                candidate_params = result2.x
                candidate_poly = _polygon_from_params(candidate_params)
                if candidate_poly is not None:
                    score = _polygon_min_score(candidate_poly)
                    if score > best_score:
                        best_score = score
                        best_poly = candidate_poly
                        best_params = candidate_params
        except Exception as e:
            print(f"Second L-BFGS-B failed: {e}")
            
    # Final check: if best_poly is None (shouldn't happen), return regular
    if best_poly is None:
        best_poly = _polygon_from_params(_regular_polygon_params(n))
        
    return best_poly

# EVOLVE-BLOCK-END
