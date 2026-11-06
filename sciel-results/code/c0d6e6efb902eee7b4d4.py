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
    Performs global grid search followed by local refinement.
    """
    best_score = -np.inf
    best_params = None
    
    # Global grid search over the 3 parameters
    # r1, r2 in [0.1, 2.0], theta in [0, pi/2]
    # (By symmetry, we can restrict theta to [0, pi/2])
    
    # Coarse grid
    r1_vals = np.linspace(0.1, 2.0, 20)
    r2_vals = np.linspace(0.1, 2.0, 20)
    theta_vals = np.linspace(0, np.pi/2, 20)
    
    for r1, r2, theta in product(r1_vals, r2_vals, theta_vals):
        points = _generate_two_squares(r1, r2, theta)
        score = _score_from_points(points)
        if score > best_score:
            best_score = score
            best_params = (r1, r2, theta)
    
    # Local refinement using scipy.optimize
    if best_params is not None:
        initial = np.array(best_params)
        result = minimize(
            _negative_score,
            initial,
            method='Nelder-Mead',
            options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 10000, 'maxfev': 10000}
        )
        if result.fun < -best_score:
            best_params = result.x
            best_score = -result.fun
    
    # Generate final points
    r1, r2, theta = best_params
    points = _generate_two_squares(r1, r2, theta)
    
    # Ensure distinct points
    # Check for duplicates and perturb if necessary
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if np.linalg.norm(points[i] - points[j]) < 1e-10:
                points[j][0] += 1e-8
    
    # Convert to list of lists of floats
    return [[float(p[0]), float(p[1])] for p in points]
# EVOLVE-BLOCK-END
