import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _triangle_areas(points):
    """Compute areas of all triangles formed by 3 points. Returns array of shape (286,)."""
    n = len(points)
    # Get all combinations of 3 points
    indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                indices.append((i, j, k))
    indices = np.array(indices)
    
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    areas = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) +
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) +
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return areas

def _score(points):
    """Compute A_min / A_hull. Returns 0 if invalid."""
    try:
        if np.any(~np.isfinite(points)):
            return 0.0
        if len(points) != 13 or points.shape != (13, 2):
            return 0.0
        hull = ConvexHull(points)
        a_hull = hull.volume
        if a_hull <= 0:
            return 0.0
        areas = _triangle_areas(points)
        a_min = np.min(areas)
        if a_min <= 0:
            return 0.0
        return a_min / a_hull
    except Exception:
        return 0.0

def _soft_min(areas, beta):
    """Soft minimum using log-sum-exp: -log(sum(exp(-beta*areas)))/beta approximates min for large beta."""
    # For numerical stability
    min_val = np.min(areas)
    exp_terms = np.exp(-beta * (areas - min_val))
    log_sum = np.log(np.sum(exp_terms))
    return min_val - np.log(log_sum) / beta

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    deadline = time.time() + 600
    
    # Start with regular 13-gon as initial guess
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    best_points = np.column_stack((np.cos(angles), np.sin(angles)))
    best_score = _score(best_points)
    
    # Use gradient-free optimization to improve the arrangement
    # We maximize A_min / A_hull. To handle the nonsmooth min, we use a smooth surrogate.
    
    def _objective_flat(x):
        """Objective function for optimization (to minimize negative score)."""
        pts = x.reshape(n, 2)
        # Add a small regularization to keep points within a reasonable bound
        if np.any(~np.isfinite(pts)):
            return 1e10
        # Normalize to unit circle scale to prevent scale drift
        # Actually, let's work in a fixed domain [0,1]^2 and normalize
        # But the ratio is scale-invariant, so let's just optimize directly
        
        areas = _triangle_areas(pts)
        a_min = np.min(areas)
        if a_min < 1e-15:
            return 1e10
        
        try:
            hull = ConvexHull(pts)
            a_hull = hull.volume
            if a_hull < 1e-15:
                return 1e10
            score = a_min / a_hull
            return -score
        except Exception:
            return 1e10
    
    def _objective_smooth(x, beta):
        """Smooth objective using soft-min."""
        pts = x.reshape(n, 2)
        if np.any(~np.isfinite(pts)):
            return 1e10
        
        areas = _triangle_areas(pts)
        # Use soft min for smoothness
        a_min_soft = _soft_min(areas, beta)
        if a_min_soft < 1e-15:
            return 1e10
        
        try:
            hull = ConvexHull(pts)
            a_hull = hull.volume
            if a_hull < 1e-15:
                return 1e10
            score = a_min_soft / a_hull
            return -score
        except Exception:
            return 1e10
    
    # Try multiple optimization strategies
    strategies = [
        ('nelder_mead', {'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}),
        ('powell', {'maxiter': 3000, 'xtol': 1e-10, 'ftol': 1e-10}),
    ]
    
    # Initial points: regular polygon
    x0 = best_points.flatten()
    
    # Try Nelder-Mead with the exact (nonsmooth) objective first
    for method, options in strategies:
        if time.time() > deadline - 30:
            break
        
        try:
            result = minimize(_objective_flat, x0, method=method, options=options)
            if result.success or result.fun < -best_score:
                new_points = result.x.reshape(n, 2)
                new_score = _score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points
                    x0 = result.x
        except Exception:
            pass
    
    # Now try with increasing beta for soft-min approach
    betas = [10, 50, 100, 200, 500]
    for beta in betas:
        if time.time() > deadline - 60:
            break
        
        x0 = best_points.flatten()
        try:
            result = minimize(
                lambda x: _objective_smooth(x, beta),
                x0,
                method='nelder-mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            new_points = result.x.reshape(n, 2)
            new_score = _score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points
        except Exception:
            pass
    
    # Final polish with high-precision Nelder-Mead
    if time.time() < deadline - 30:
        x0 = best_points.flatten()
        try:
            result = minimize(
                _objective_flat,
                x0,
                method='nelder-mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            new_points = result.x.reshape(n, 2)
            new_score = _score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points
        except Exception:
            pass
    
    # Try a few random restarts if time permits
    rng = np.random.default_rng(42)
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 120:
            break
        
        # Random perturbation of best points
        x0 = best_points.flatten() + rng.normal(0, 0.1, size=2 * n)
        try:
            result = minimize(
                _objective_flat,
                x0,
                method='nelder-mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            new_points = result.x.reshape(n, 2)
            new_score = _score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points
        except Exception:
            pass
    
    # Ensure the result is valid
    if _score(best_points) <= 0:
        # Fall back to regular polygon
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return best_points
