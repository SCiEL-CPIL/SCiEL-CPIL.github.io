# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

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


def _regular_polygon(n=20, radius=1.0):
    """Create a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)


def _parametric_polygon_2mode(coeffs):
    """
    Generate a convex 20-gon from 2-mode Fourier radial perturbation.
    coeffs = [a1, b1, a2, b2]
    r_i = 1 + a1*cos(theta_i) + b1*sin(theta_i) + a2*cos(2*theta_i) + b2*sin(2*theta_i)
    theta_i = 2*pi*i/20
    
    This restricts the shape to very smooth deformations (ellipse + 4-lobed shape).
    """
    n = 20
    theta = 2 * np.pi * np.arange(n) / n
    r = np.ones(n)
    
    # Mode 1: Elliptical deformation
    r += coeffs[0] * np.cos(theta) + coeffs[1] * np.sin(theta)
    
    # Mode 2: 4-lobed deformation (symmetric under 90 deg rotation effectively, but breaks 20-gon symmetry slightly)
    r += coeffs[2] * np.cos(2 * theta) + coeffs[3] * np.sin(2 * theta)
    
    # Ensure positive radii to maintain convexity/validity
    # If r becomes too small, the polygon might become non-convex or self-intersecting.
    # A lower bound of 0.5 is usually safe for small perturbations.
    r = np.maximum(r, 0.5)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y], axis=1)


def _objective_2mode(coeffs):
    """
    Objective function for 2-mode parametric optimization.
    Returns negative of the minimum vertex score.
    """
    poly = _parametric_polygon_2mode(coeffs)
    score = _evaluate(poly)
    
    # If invalid, return a large penalty
    if score <= 0:
        # Add a small penalty based on how far we are from valid? 
        # Or just a constant penalty. 
        # Since the space is small, we can use a constant.
        return 100.0
    
    return -score


def _search_2mode(time_budget=600, deadline_margin=30):
    """
    Search for best polygon using 2-mode parametric optimization.
    """
    start_time = time.time()
    deadline = start_time + time_budget - deadline_margin
    
    # Initial guess: Regular polygon (all coeffs 0)
    best_coeffs = np.zeros(4)
    best_poly = _regular_polygon(20)
    best_score = _evaluate(best_poly)
    
    # Generate several random starting points in the low-dimensional space
    # The coefficients should be small to maintain convexity and avoid wild shapes
    # Let's sample from a normal distribution with sigma=0.1
    n_restarts = 50
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
            
        if i == 0:
            init = np.zeros(4)
        else:
            init = np.random.normal(0, 0.15, 4)
        
        # Evaluate initial
        init_poly = _parametric_polygon_2mode(init)
        init_score = _evaluate(init_poly)
        
        if init_score > best_score:
            best_score = init_score
            best_poly = init_poly.copy()
            best_coeffs = init.copy()
            
        # Run optimization from this start
        try:
            result = minimize(
                _objective_2mode,
                init,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            result_poly = _parametric_polygon_2mode(result.x)
            result_score = _evaluate(result_poly)
            
            if result_score > best_score:
                best_score = result_score
                best_poly = result_poly.copy()
                best_coeffs = result.x.copy()
                
        except Exception:
            pass
        
    return best_poly, best_score


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices using a 2-mode Fourier radial parametric search.
    
    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    result, score = _search_2mode(time_budget=600, deadline_margin=30)
    
    # Validate the result
    final_score = _evaluate(result)
    if final_score <= 0:
        # Fallback to regular polygon
        return _regular_polygon(20)
    
    return result

# EVOLVE-BLOCK-START
import numpy as np
import time

# EVOLVE-BLOCK-END
