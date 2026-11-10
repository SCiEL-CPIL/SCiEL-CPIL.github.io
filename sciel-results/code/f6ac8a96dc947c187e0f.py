# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def _compute_ratio(points: np.ndarray) -> float:
    """Compute R = (d_min / d_max)^2 for the given points."""
    if len(points) < 2:
        return 0.0
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _solve_for_lb(lb: float, x0_init: np.ndarray = None) -> tuple:
    """
    Try to find 16 points such that all pairwise distances >= lb.
    Minimizes the maximum pairwise distance.
    Returns (success, points, d_max).
    """
    n = 16
    
    # Generate initializations
    inits = []
    
    # 1. Regular polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Grid
    grid_x = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], dtype=float)
    grid_y = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], dtype=float)
    inits.append(np.column_stack([grid_x, grid_y]))
    
    # 3. Hexagonal-like packing
    # 4 rows of 4
    hex_x = []
    hex_y = []
    for i in range(4):
        for j in range(4):
            hex_x.append(j + (0.5 if i % 2 == 1 else 0))
            hex_y.append(i * 0.866) # sqrt(3)/2
    inits.append(np.column_stack([hex_x, hex_y]))
    
    # 4. Random
    np.random.seed(42)
    inits.append(np.random.randn(n, 2) * 2.0)
    
    # 5. Use provided initial if available
    if x0_init is not None:
        inits.insert(0, x0_init)
        
    best_success = False
    best_points = None
    best_dmax = np.inf

    def objective(x):
        pts = x.reshape(n, 2)
        dists = pdist(pts)
        return np.max(dists)

    def constraint_min_dist(x):
        pts = x.reshape(n, 2)
        dists = pdist(pts)
        return np.min(dists) - lb

    cons = [{'type': 'ineq', 'fun': constraint_min_dist}]

    for x0 in inits:
        # Scale initial point to roughly match lb to help convergence
        # If lb is large, we need a larger spread
        d0 = pdist(x0)
        if np.min(d0) > 0 and np.max(d0) > 0:
             # Try to scale such that min dist is close to lb
             # This is a heuristic
             pass 
             
        try:
            res = minimize(
                objective,
                x0.flatten(),
                method='SLSQP',
                constraints=cons,
                options={'maxiter': 1000, 'ftol': 1e-12}
            )

            if res.success:
                pts = res.x.reshape(n, 2)
                d = pdist(pts)
                d_max = np.max(d)
                d_min = np.min(d)
                if d_min >= lb - 1e-6:  # constraint satisfied with tolerance
                    if not best_success or d_max < best_dmax:
                        best_success = True
                        best_points = pts
                        best_dmax = d_max
        except Exception:
            pass

    return best_success, best_points, best_dmax


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    n = 16
    best_points = None
    best_ratio = -1.0
    last_points = None

    # We search for lb in a reasonable range.
    # Since the ratio is scale invariant, we can assume d_max is around 1-2.
    # A good lb will be around 0.3-0.5 for d_max=1.
    
    # Coarse search
    lb_values = np.linspace(0.1, 0.8, 15)
    
    for lb in lb_values:
        success, pts, dmax = _solve_for_lb(lb, last_points)
        if success and dmax > 0:
            ratio = (lb / dmax) ** 2
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = pts
                last_points = pts

    # Fine search around the best lb
    if best_points is not None:
        # Find the lb that was used for best_points? 
        # We didn't store it. Let's just re-run a finer grid.
        # Or we can estimate.
        # Let's just do a finer grid over the whole range to be safe, but use last_points as warm start.
        # Actually, let's identify the best lb from the coarse search.
        # We can just do a finer linspace.
        
        # To save time, we can narrow the range.
        # Let's assume the best lb is in the middle.
        # We'll do a finer search from 0.2 to 0.7
        lb_fine = np.linspace(0.2, 0.7, 20)
        
        for lb in lb_fine:
            success, pts, dmax = _solve_for_lb(lb, last_points)
            if success and dmax > 0:
                ratio = (lb / dmax) ** 2
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = pts
                    last_points = pts

    if best_points is None:
        # Fallback
        np.random.seed(42)
        return np.random.randn(n, 2)

    return best_points

# EVOLVE-BLOCK-END
