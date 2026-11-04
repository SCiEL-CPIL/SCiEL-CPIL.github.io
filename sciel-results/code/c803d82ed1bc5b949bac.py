import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

# Precompute triple indices for N=14
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    if len(points) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    # Calculate Minimum Triangle Area
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]

    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])

    min_area = 0.5 * np.min(np.abs(cross))

    return min_area / hull_area

def _soft_min_score(points, beta=100.0):
    """
    Soft-min surrogate for the minimum triangle area.
    Uses log-sum-exp for numerical stability.
    """
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]

    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])

    areas = 0.5 * np.abs(cross)

    # Soft-min: -1/beta * log(sum(exp(-beta * areas)))
    max_area = np.max(areas)
    shifted = -beta * (areas - max_area)
    log_sum = np.log(np.sum(np.exp(shifted))) + beta * max_area
    soft_min = max_area - log_sum / beta

    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    return soft_min / hull_area

def _regular_polygon(n, radius=1.0, rotation=0.0):
    """Generate a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _optimize_coordinates(initial_points, deadline):
    """
    Optimize points in free 2D space using soft-min L-BFGS-B and exact Nelder-Mead.
    """
    current_points = initial_points.copy()
    current_score = _calculate_score(current_points)
    
    # Phase 1: L-BFGS-B with soft-min surrogate
    # Use a sequence of betas to anneal from smooth to sharp
    betas = [10.0, 50.0, 100.0, 200.0, 500.0]
    
    # Bounds: keep points within a reasonable box around the origin
    # The regular polygon has radius 1, so [-2, 2] is safe
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
            
        def obj_soft(x):
            pts = x.reshape(N, 2)
            return -_soft_min_score(pts, beta)
        
        try:
            res = minimize(
                obj_soft,
                current_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-12, 'gtol': 1e-10, 'maxiter': 200}
            )
            if -res.fun > current_score:
                new_points = res.x.reshape(N, 2)
                new_score = _calculate_score(new_points)
                if new_score > current_score:
                    current_points = new_points
                    current_score = new_score
        except Exception:
            pass

    # Phase 2: Nelder-Mead polish on exact score
    if time.time() < deadline - 5:
        def obj_exact(x):
            pts = x.reshape(N, 2)
            return -_calculate_score(pts)
        
        try:
            res_nm = minimize(
                obj_exact,
                current_points.flatten(),
                method='Nelder-Mead',
                options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 500}
            )
            if -res_nm.fun > current_score:
                new_points = res_nm.x.reshape(N, 2)
                new_score = _calculate_score(new_points)
                if new_score > current_score:
                    current_points = new_points
                    current_score = new_score
        except Exception:
            pass

    return current_points, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    best_overall_score = 0.0
    best_overall_points = None

    # Strategy: Multiple random restarts from regular polygon with perturbations
    # The regular 14-gon is a strong starting point.
    rng = np.random.default_rng(42)
    
    # Number of restarts based on time
    # Each optimization cycle takes roughly 5-10 seconds.
    # We have ~140s. Let's do 10-15 restarts.
    n_restarts = 12
    
    for i in range(n_restarts):
        if time.time() > deadline - 10:
            break
            
        # Generate initial points
        # Start with regular polygon
        # Add random perturbation to break symmetry and escape local minima
        rotation = rng.uniform(0, 2 * np.pi)
        base_points = _regular_polygon(N, radius=1.0, rotation=rotation)
        
        # Perturbation magnitude decreases with restart index to focus on fine-tuning
        # Or keep it constant for exploration
        noise_scale = 0.1
        perturbed_points = base_points + rng.normal(0, noise_scale, (N, 2))
        
        # Normalize to unit area? 
        # The score is invariant to scaling, but convex hull area changes.
        # Actually, score = min_area / hull_area. 
        # Scaling points by k scales min_area by k^2 and hull_area by k^2. 
        # So score is scale invariant.
        # We don't need to normalize, but it helps keep numbers stable.
        
        # Run optimization
        try:
            improved_points, improved_score = _optimize_coordinates(perturbed_points, deadline)
            if improved_score > best_overall_score:
                best_overall_score = improved_score
                best_overall_points = improved_points
        except Exception:
            pass

    # Fallback if no valid points found
    if best_overall_points is None:
        best_overall_points = _regular_polygon(N)
        best_overall_score = _calculate_score(best_overall_points)

    return best_overall_points
