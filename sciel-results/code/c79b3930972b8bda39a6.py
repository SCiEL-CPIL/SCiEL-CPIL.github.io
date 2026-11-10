# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
from scipy.optimize import minimize

def _compute_score(points):
    """Compute R = (d_min/d_max)^2 for a set of points."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _pairwise_indices(n):
    """Precompute indices for all unique pairs."""
    i, j = np.triu_indices(n, k=1)
    return i, j

def _objective_and_grad(x, i, j):
    """
    Objective: Minimize -min_dist (i.e., maximize min_dist).
    Constraints: max_dist <= 1.0 (handled via penalty or trust-constr constraints).
    
    For trust-constr, we define the objective as -min_dist.
    We need gradients.
    min_dist = min_k ||x_k - x_l||
    Let m be the index of the minimum distance pair.
    grad of -min_dist w.r.t x is -grad of ||x_m - x_l||.
    """
    pts = x.reshape(-1, 2)
    n = pts.shape[0]
    
    # Compute all distances
    # Using broadcasting for speed in gradient calculation context
    # d[i, j] = distance between point i and point j
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    
    # Mask out diagonal
    mask = ~np.eye(n, dtype=bool)
    dist_sq_masked = np.where(mask, dist_sq, np.inf)
    
    min_dist_sq = np.min(dist_sq_masked)
    min_dist = np.sqrt(min_dist_sq)
    
    # Find the pair (i, j) that achieves the minimum distance
    # We need the gradient of this specific distance
    min_idx = np.argmin(dist_sq_masked)
    i_min = min_idx // n
    j_min = min_idx % n
    
    # Gradient of objective f(x) = -||x_i - x_j||
    # d/dx_k ||x_i - x_j||
    # If k == i: (x_i - x_j) / ||x_i - x_j||
    # If k == j: (x_j - x_i) / ||x_i - x_j||
    # Else: 0
    
    vec = pts[i_min] - pts[j_min]
    norm = min_dist if min_dist > 1e-12 else 1e-12
    
    grad = np.zeros_like(x)
    # Chain rule: we are minimizing -dist, so grad is - (grad of dist)
    grad[2*i_min:2*i_min+2] -= vec / norm
    grad[2*j_min:2*j_min+2] += vec / norm  # Because d/dx_j ||x_i-x_j|| = (x_j-x_i)/norm = -vec/norm.
                                           # So -d/dx_j = +vec/norm
    
    return -min_dist, grad

def _constraints_fun(x, i, j):
    """
    Constraints for trust-constr.
    We want to enforce max_dist <= 1.0.
    This is a single constraint g(x) = 1.0 - max_dist >= 0.
    Also, to prevent points from flying off to infinity (which would keep ratio low but might cause numerical issues),
    we can bound the points to a disk of radius R_max, e.g., 10.
    However, trust-constr handles bounds natively.
    
    Constraint: max_dist - 1.0 <= 0
    Let m be the index of the maximum distance pair.
    g(x) = ||x_m - x_l|| - 1.0
    """
    pts = x.reshape(-1, 2)
    n = pts.shape[0]
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    mask = ~np.eye(n, dtype=bool)
    dist_sq_masked = np.where(mask, dist_sq, -np.inf)
    
    max_dist_sq = np.max(dist_sq_masked)
    max_dist = np.sqrt(max_dist_sq)
    
    # Value of constraint: max_dist - 1.0
    # We want this to be <= 0.
    val = max_dist - 1.0
    
    # Gradient of max_dist
    max_idx = np.argmax(dist_sq_masked)
    i_max = max_idx // n
    j_max = max_idx % n
    
    vec = pts[i_max] - pts[j_max]
    norm = max_dist if max_dist > 1e-12 else 1e-12
    
    grad = np.zeros_like(x)
    grad[2*i_max:2*i_max+2] += vec / norm
    grad[2*j_max:2*j_max+2] -= vec / norm
    
    return val, grad

def _minimize_min_dist(x0, i, j, time_budget):
    """
    Run trust-constr to maximize min_dist subject to max_dist <= 1.
    """
    start = time.time()
    
    # Define constraints
    # trust-constr expects constraints as dictionaries or LinearConstraint/NonlinearConstraint
    # We have a nonlinear inequality: max_dist - 1.0 <= 0
    
    class MaxDistConstraint:
        def __init__(self):
            self.type = 'ineq'
        
        def fun(self, x):
            val, _ = _constraints_fun(x, i, j)
            return -val # trust-constr ineq constraint is g(x) >= 0. We want max_dist-1 <= 0 => 1-max_dist >= 0.
        
        def jac(self, x):
            _, grad = _constraints_fun(x, i, j)
            return -grad

    cons = MaxDistConstraint()
    
    # Bounds: Keep points within a reasonable range to help the optimizer.
    # Since max_dist <= 1, the points are in a bounded region (diameter 1).
    # So coordinates should be in [-1, 1] roughly.
    bounds = [(-1.5, 1.5)] * (2 * 16)
    
    try:
        res = minimize(
            _objective_and_grad, 
            x0, 
            jac=True, 
            method='trust-constr', 
            constraints=cons, 
            bounds=bounds, 
            options={'maxiter': 500, 'gtol': 1e-8, 'xtol': 1e-8, 'verbose': 0}
        )
        if time.time() - start > time_budget:
             return res.x
        return res.x
    except Exception:
        return x0

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    start_time = time.time()
    deadline = start_time + 600  # safety margin before 630s kill
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = 0.0
    
    def _update(pts, sc):
        nonlocal best_points, best_score
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
    
    n = 16
    i, j = _pairwise_indices(n)
    
    # Initial guesses
    # 1. Regular grid
    x_grid = np.linspace(-0.5, 0.5, 4)
    y_grid = np.linspace(-0.5, 0.5, 4)
    xx, yy = np.meshgrid(x_grid, y_grid)
    init_grid = np.column_stack([xx.ravel(), yy.ravel()]).ravel()
    
    # 2. Random points in unit disk
    init_random = np.zeros(32)
    for k in range(16):
        r = np.sqrt(rng.random())
        theta = rng.uniform(0, 2*np.pi)
        init_random[2*k] = r * np.cos(theta)
        init_random[2*k+1] = r * np.sin(theta)
        
    # 3. Hexagonal packing approximation (scaled)
    init_hex = np.zeros(32)
    # Just use a few random restarts from the grid and random
    inits = [init_grid, init_random]
    
    # Add a few more random restarts
    for _ in range(3):
        r_pts = np.zeros(32)
        for k in range(16):
            r = np.sqrt(rng.random())
            theta = rng.uniform(0, 2*np.pi)
            r_pts[2*k] = r * np.cos(theta)
            r_pts[2*k+1] = r * np.sin(theta)
        inits.append(r_pts)
        
    # Optimize each initial guess
    for init in inits:
        if time.time() > deadline - 30:
            break
            
        # Ensure initial points satisfy max_dist <= 1 roughly?
        # If not, trust-constr might struggle or take many steps.
        # Let's scale down if max_dist > 1
        pts_init = init.reshape(-1, 2)
        d_init = pdist(pts_init)
        max_d = np.max(d_init)
        if max_d > 1.0:
            pts_init = pts_init / max_d
            init = pts_init.ravel()
            
        # Run optimization
        # Budget per run: 60s
        opt_result = _minimize_min_dist(init, i, j, time_budget=60)
        
        pts = opt_result.reshape(-1, 2)
        
        # Calculate score
        # Note: The optimization maximizes min_dist subject to max_dist <= 1.
        # The score is (min_dist / max_dist)^2.
        # If max_dist < 1, we can scale up the points to make max_dist = 1, which increases min_dist proportionally.
        # So we should normalize the result to max_dist = 1.
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        
        if d_max > 0:
            pts_normalized = pts / d_max
            # Recalculate score on normalized points (which will be d_min/d_max of original)
            # Actually, R is scale invariant.
            score = (d_min / d_max) ** 2
        else:
            score = 0.0
            pts_normalized = pts
            
        _update(pts_normalized, score)
        
    # If we have time, do a final polish on the best point using Nelder-Mead on the full 32D space
    # This can help if trust-constr got stuck in a local minimum of the smooth approximation
    if best_points is not None and time.time() < deadline - 10:
        def _neg_score(flat):
            pts = flat.reshape(-1, 2)
            return -_compute_score(pts)
        
        try:
            res = minimize(_neg_score, best_points.ravel(), method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-9})
            if res.fun < -best_score:
                _update(res.x.reshape(-1, 2), -res.fun)
        except Exception:
            pass
            
    if best_points is None:
        # Fallback: 4x4 grid
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        xx, yy = np.meshgrid(x, y)
        best_points = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Ensure finite and correct shape
    if not np.all(np.isfinite(best_points)):
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        xx, yy = np.meshgrid(x, y)
        best_points = np.column_stack([xx.ravel(), yy.ravel()])
        
    # Final check: ensure shape is (16, 2)
    if best_points.shape != (16, 2):
        best_points = best_points.reshape(16, 2)

    return best_points

# EVOLVE-BLOCK-END
