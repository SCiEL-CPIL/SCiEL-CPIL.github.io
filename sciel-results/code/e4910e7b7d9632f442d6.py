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


def _fourier_angular_polygon(coeffs):
    """
    Generate a convex 20-gon using a Fourier series for angular deviations.
    coeffs: array of shape (20,) representing Fourier coefficients for angular deviations.
    
    The base angle for vertex i is theta_i = 2*pi*i/20.
    The deviation delta_theta_i is given by:
    delta_theta_i = sum_{k=1}^{10} (a_k * cos(2*pi*k*i/20) + b_k * sin(2*pi*k*i/20))
    
    Here coeffs[0:10] are a_k and coeffs[10:20] are b_k.
    Radius is fixed at 1.0 for simplicity in this parameterization, 
    relying on angular deviations to create the necessary distance structure.
    """
    n = 20
    base_angles = 2 * np.pi * np.arange(n) / n
    
    # Compute deviations
    deviations = np.zeros(n)
    for k in range(1, 11):
        a_k = coeffs[k-1]
        b_k = coeffs[10 + k-1]
        deviations += a_k * np.cos(2 * np.pi * k * np.arange(n) / n) + b_k * np.sin(2 * np.pi * k * np.arange(n) / n)
    
    final_angles = base_angles + deviations
    
    # Ensure angles are in increasing order for convexity? 
    # Actually, for a star-shaped polygon with r=1, if angles are monotonic, it's convex.
    # Fourier series might create non-monotonic angles if coefficients are large.
    # We need to check convexity in _evaluate, but for the optimizer to work well,
    # we should try to keep deviations small.
    
    x = np.cos(final_angles)
    y = np.sin(final_angles)
    return np.stack([x, y], axis=1)


def _objective_fourier(coeffs):
    """
    Objective function for Fourier angular optimization.
    Returns negative of the minimum vertex score.
    """
    poly = _fourier_angular_polygon(coeffs)
    score = _evaluate(poly)
    
    # If invalid, return a large penalty
    if score <= 0:
        # Penalize based on how close we are to validity? 
        # For Nelder-Mead, a constant penalty is often sufficient if the search space is constrained.
        # However, we can add a small penalty for large coefficients to encourage smoothness.
        penalty = np.sum(np.abs(coeffs)) * 0.1
        return 10.0 + penalty
    
    return -score


def _search_fourier_angular(time_budget=600, deadline_margin=30):
    """
    Search for best polygon using Fourier angular deviation optimization.
    """
    start_time = time.time()
    deadline = start_time + time_budget - deadline_margin
    
    best_coeffs = np.zeros(20)
    best_poly = _regular_polygon(20)
    best_score = _evaluate(best_poly)
    
    # Nelder-Mead is derivative-free and good for low-to-mid dimensional non-smooth problems.
    # We will do multiple restarts with small random perturbations.
    
    n_restarts = 50
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
            
        if i == 0:
            init = np.zeros(20)
        else:
            # Small random perturbations
            # Standard deviation 0.05 for angles (radians) is about 3 degrees max deviation
            init = np.random.normal(0, 0.05, 20)
        
        # Evaluate initial
        init_poly = _fourier_angular_polygon(init)
        init_score = _evaluate(init_poly)
        
        if init_score > best_score:
            best_score = init_score
            best_poly = init_poly.copy()
            best_coeffs = init.copy()
            
        # Run optimization
        try:
            # Nelder-Mead options
            options = {
                'maxiter': 500,
                'xatol': 1e-4,
                'fatol': 1e-6,
                'adaptive': True
            }
            
            result = minimize(
                _objective_fourier,
                init,
                method='Nelder-Mead',
                options=options
            )
            
            result_poly = _fourier_angular_polygon(result.x)
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
    Build a convex polygon with n = 20 vertices using Fourier angular deviation search.
    
    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    result, score = _search_fourier_angular(time_budget=600, deadline_margin=30)
    
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
