import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    Returns 0 if invalid.
    """
    n = len(points)
    if n != 13:
        return 0.0
    
    # Check for NaN/Inf
    if not np.all(np.isfinite(points)):
        return 0.0

    # Calculate min triangle area
    min_area = float('inf')
    
    for i in range(13):
        for j in range(i + 1, 13):
            for k in range(j + 1, 13):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
                if area < min_area:
                    min_area = area

    # Calculate Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0

    if hull_area <= 1e-10:
        return 0.0

    return min_area / hull_area

def _negative_score(points):
    """
    Objective function for minimization (maximizing the score).
    """
    s = _score(points)
    if s <= 0:
        return 1.0 # Penalize invalid/collinear
    return -s

def _generate_points_from_params(params):
    """
    Reconstruct 13 points from a low-dimensional parameter vector.
    Params structure:
    - 13 radial deviations (delta_r)
    - 13 angular deviations (delta_theta)
    - 1 global aspect ratio (a/b)
    - 1 global rotation
    
    Total: 28 parameters.
    Base: Regular 13-gon.
    """
    n = 13
    delta_r = params[0:13]
    delta_theta = params[13:26]
    aspect = params[26]
    rotation = params[27]
    
    # Base angles for regular 13-gon
    base_theta = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    
    # Base radius 1.0
    base_r = np.ones(n)
    
    # Apply deviations
    r = base_r + delta_r
    theta = base_theta + delta_theta
    
    # Convert to Cartesian
    x = r * np.cos(theta)
    y = r * np.sin(theta) * aspect
    
    return np.column_stack((x, y))

def _random_params(rng, scale=0.1):
    """
    Generate random perturbation parameters around the regular 13-gon.
    """
    n = 13
    delta_r = rng.normal(0, scale, n)
    delta_theta = rng.normal(0, scale * 0.5, n)
    aspect = 1.0 + rng.normal(0, 0.1)
    rotation = rng.uniform(0, 2 * np.pi)
    
    return np.concatenate([delta_r, delta_theta, [aspect, rotation]])

def heilbronn_convex13() -> np.ndarray:
    """
    Uses structured parameterization of a regular 13-gon with Nelder-Mead optimization.
    """
    start_time = time.time()
    deadline = start_time + 580 # Safety margin
    
    best_score = 0.0
    best_points = None
    
    rng = np.random.default_rng(123)
    
    # Initial best guess: Regular 13-gon
    theta0 = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    init_points = np.column_stack((np.cos(theta0), np.sin(theta0)))
    best_points = init_points.copy()
    best_score = _score(best_points)
    
    # We will run Nelder-Mead on the parameter space from multiple random starts.
    # Nelder-Mead is derivative-free and handles non-smooth objectives reasonably well
    # when the search space is constrained.
    
    num_restarts = 20
    max_iters_per_restart = 5000
    
    # Pre-allocate bounds? Nelder-Mead doesn't use bounds directly, 
    # but we can clip parameters inside the objective if needed, 
    # or just let the optimizer explore. 
    # To keep points "convex-ish", we limit the magnitude of deviations.
    # We'll implement a soft penalty or just rely on the optimizer not going crazy 
    # if we start close to the regular polygon.
    # Actually, let's add a hard constraint check in the objective if points become degenerate.
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Generate initial parameters
        # First restart is the regular polygon (zero deviations, aspect 1, rot 0)
        if restart == 0:
            params = np.zeros(28)
        else:
            # Random perturbation. Scale decreases as we get more restarts?
            # Or keep it constant to explore different basins.
            # Let's try a mix: some small, some larger.
            scale = 0.05 if restart < 5 else 0.2
            params = _random_params(rng, scale=scale)
        
        # Run Nelder-Mead
        # We wrap the objective to ensure we always return a finite value
        def obj_func(p):
            pts = _generate_points_from_params(p)
            if not np.all(np.isfinite(pts)):
                return 1e6
            s = _score(pts)
            if s <= 0:
                return 1e6
            return -s
        
        try:
            result = minimize(
                obj_func, 
                params, 
                method='Nelder-Mead',
                options={'maxiter': max_iters_per_restart, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
        except Exception:
            continue
            
        if result.success or result.fun < 0: # If it found a valid solution
            final_params = result.x
            final_points = _generate_points_from_params(final_params)
            
            # Sanity check
            if np.all(np.isfinite(final_points)) and final_points.shape == (13, 2):
                final_score = _score(final_points)
                if final_score > best_score:
                    best_score = final_score
                    best_points = final_points.copy()
                    
    # If Nelder-Mead didn't find anything better than the regular polygon,
    # we might try a quick local refinement on the best points found using the old subgradient method
    # just in case the parameterization is too restrictive.
    # But for this "different strategy" task, we stick to the parameterization result.
    
    # Final sanity check
    if best_points is None or not np.all(np.isfinite(best_points)) or best_points.shape != (13, 2):
        theta = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_points = np.column_stack((np.cos(theta), np.sin(theta)))
        
    return best_points

# To ensure the function is standalone and ready
if __name__ == "__main__":
    pts = heilbronn_convex13()
    print(_score(pts))
