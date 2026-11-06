"""Program for eight planar points maximizing the product of distances at fixed diameter."""
import numpy as np
from scipy.optimize import minimize
from itertools import product

# EVOLVE-BLOCK-START
def _score_from_points(points):
    """Compute Delta / n^n for a set of n points, scaled to diameter 2."""
    pts = np.array(points, dtype=float)
    n = len(pts)
    
    # Compute all pairwise distances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    
    # Diameter is the max pairwise distance
    diameter = np.max(dists)
    if diameter < 1e-15:
        return 0.0
    
    # Scale so diameter is 2
    scale = 2.0 / diameter
    dists_scaled = dists * scale
    
    # Product over ordered pairs (i != j)
    # For n points, there are n*(n-1) ordered pairs
    # Delta = product of all ordered pair distances
    # log(Delta) = sum over i != j of log(d_ij)
    mask = ~np.eye(n, dtype=bool)
    log_delta = np.sum(np.log(dists_scaled[mask]))
    
    # Score = Delta / n^n = exp(log_delta) / n^n = exp(log_delta - n*log(n))
    score = np.exp(log_delta - n * np.log(n))
    
    return float(score)

def _generate_two_squares(r1, r2, theta):
    """Generate 8 points: two concentric squares with radii r1, r2 and relative rotation theta."""
    # First square: radius r1, rotation 0
    angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    
    # Second square: radius r2, rotation theta
    angles2 = angles1 + theta
    pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
    
    return np.vstack([pts1, pts2])

def _negative_score(params):
    """Negative score for minimization."""
    r1, r2, theta = params
    if r1 < 0 or r2 < 0:
        return 1e10
    points = _generate_two_squares(r1, r2, theta)
    return -_score_from_points(points)

def construct():
    """Construct 8 points maximizing the product of distances at fixed diameter.
    
    Uses two concentric squares with independent radii and relative rotation.
    Performs global grid search followed by local refinement with multiple restarts.
    """
    best_score = -np.inf
    best_params = None
    
    # The regular octagon corresponds to r1=r2=1, theta=pi/4 (score 1.0)
    # We search around this region.
    # r1, r2 can vary. Let's search [0.2, 1.8] to be safe.
    # theta in [0, pi/2] due to symmetry.
    
    # Coarse grid search
    r1_vals = np.linspace(0.2, 1.8, 25)
    r2_vals = np.linspace(0.2, 1.8, 25)
    theta_vals = np.linspace(0, np.pi/2, 25)
    
    for r1, r2, theta in product(r1_vals, r2_vals, theta_vals):
        points = _generate_two_squares(r1, r2, theta)
        score = _score_from_points(points)
        if score > best_score:
            best_score = score
            best_params = (r1, r2, theta)
            
    # Local refinement with multiple restarts from the best grid point and nearby points
    # We will run minimize from the best grid point and a few random perturbations
    
    candidates = [best_params]
    
    # Add some random points near the best found
    for _ in range(20):
        r1_p = best_params[0] + np.random.uniform(-0.2, 0.2)
        r2_p = best_params[1] + np.random.uniform(-0.2, 0.2)
        theta_p = (best_params[2] + np.random.uniform(-0.2, 0.2)) % (np.pi/2)
        candidates.append((r1_p, r2_p, theta_p))
        
    # Also include the regular octagon specifically in case grid missed it
    candidates.append((1.0, 1.0, np.pi/4))
    
    for init in candidates:
        result = minimize(
            _negative_score,
            init,
            method='Nelder-Mead',
            options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 5000, 'maxfev': 5000}
        )
        if result.fun < -best_score:
            best_score = -result.fun
            best_params = result.x
            
    # Generate final points
    r1, r2, theta = best_params
    points = _generate_two_squares(r1, r2, theta)
    
    # Ensure distinct points
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if np.linalg.norm(points[i] - points[j]) < 1e-10:
                points[j][0] += 1e-8
    
    # Convert to list of lists of floats
    return [[float(p[0]), float(p[1])] for p in points]
# EVOLVE-BLOCK-END
