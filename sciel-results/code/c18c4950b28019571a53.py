# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def _compute_ratio(points: np.ndarray) -> float:
    """Compute R = (d_min / d_max)^2 for the given points."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _energy(points: np.ndarray) -> float:
    """
    Compute the energy sum of 1/d^2 for all pairs.
    Lower energy means points are more spread out.
    """
    n = len(points)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            d2 = np.dot(diff, diff)
            if d2 < 1e-12:
                return 1e12 # Penalize coincident points heavily
            energy += 1.0 / d2
    return energy

def _energy_grad(points: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of the energy function.
    """
    n = len(points)
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            d2 = np.dot(diff, diff)
            if d2 < 1e-12:
                # Arbitrary large gradient if coincident
                grad[i] += diff * 1e12
                grad[j] -= diff * 1e12
                continue
            # d/dx_i (1/d2) = -1/d2^2 * 2*diff
            coeff = -2.0 / (d2 * d2)
            grad[i] += coeff * diff
            grad[j] -= coeff * diff
    return grad

def _refine_points(points: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    """
    Refine points using gradient descent on the inverse-squared distance energy.
    """
    x0 = points.flatten()
    
    def fun(x):
        pts = x.reshape(-1, 2)
        return _energy(pts)
        
    def jac(x):
        pts = x.reshape(-1, 2)
        return _energy_grad(pts).flatten()

    try:
        res = minimize(fun, x0, jac=jac, method='L-BFGS-B', options={'maxiter': max_iter, 'gtol': 1e-8})
        if res.success:
            return res.x.reshape(-1, 2)
    except Exception:
        pass
    
    return points

def _normalize(points: np.ndarray) -> np.ndarray:
    """
    Normalize points so that the maximum distance is 1.
    Also center them at the origin.
    """
    # Center
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # Scale
    d = pdist(points)
    d_max = np.max(d)
    if d_max > 0:
        points = points / d_max
    return points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates of the points.
    """
    start_time = time.time()
    deadline = 580.0 # Safety margin for 600s limit
    
    n = 16
    best_points = None
    best_score = -1.0

    # Generate initial configurations
    inits = []
    
    # 1. Regular polygon (circle)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circle_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(circle_pts)
    
    # 2. Grid-like (4x4)
    grid_x = np.linspace(-1, 1, 4)
    grid_y = np.linspace(-1, 1, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    grid_pts = np.column_stack([gx.flatten(), gy.flatten()])
    inits.append(grid_pts)
    
    # 3. Random starts
    np.random.seed(42)
    for _ in range(20):
        rand_pts = np.random.randn(n, 2) * 10
        inits.append(rand_pts)

    # Process each initialization
    for i, init_pts in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Normalize before optimization to help with scaling
        pts = _normalize(init_pts)
        
        # Refine
        refined_pts = _refine_points(pts, max_iter=500)
        
        # Normalize again after refinement
        refined_pts = _normalize(refined_pts)
        
        # Score
        score = _compute_ratio(refined_pts)
        
        if score > best_score:
            best_score = score
            best_points = refined_pts.copy()
            
            # If we find a very good score, we might stop early? 
            # Theoretical max for 16 points in 2D is not 1, but likely around 0.1-0.2 for squared ratio?
            # Actually, for N points, max min/max ratio is related to packing density.
            # For 16 points, a good packing might have d_min/d_max ~ 0.3-0.4? R ~ 0.1-0.16.
            # The score 48 mentioned in the prompt is likely normalized differently or I am misinterpreting "score".
            # The prompt says "score is proportional to R". And "1.0 equals the seed program".
            # Seed program is random. Random points have very low d_min/d_max.
            # So 48 is a huge improvement.
            
    # Fallback if no points found
    if best_points is None:
        best_points = _normalize(inits[0])
        
    return best_points

# EVOLVE-BLOCK-END
