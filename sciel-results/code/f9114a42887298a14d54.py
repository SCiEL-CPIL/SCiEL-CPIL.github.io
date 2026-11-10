# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.
    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 500  # Keep a safety margin before 630s

    best_score = -1.0
    best_points = np.random.randn(n, d)
    
    # Helper to compute score
    def compute_score(pts):
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    # Update best if current is better
    curr_score = compute_score(best_points)
    if curr_score > best_score:
        best_score = curr_score

    # Define the optimization objective and constraints
    # We want to maximize d_min subject to d_max <= 1
    # This is equivalent to minimizing -d_min subject to d_ij <= 1
    
    def objective(x):
        pts = x.reshape(n, d)
        dists = pdist(pts)
        # d_min is the minimum distance
        return -np.min(dists)

    # Constraints: all pairwise distances <= 1
    # We will generate constraints for all pairs
    # This is a lot of constraints, but n=16 is small.
    # To speed up, we can use a sparse representation or just list them.
    # scipy.optimize.minimize with SLSQP handles inequality constraints of the form c(x) >= 0
    # So we define c(x) = 1 - d_ij >= 0
    
    def make_constraint(i, j):
        def constraint(x):
            pts = x.reshape(n, d)
            di, dj = pts[i], pts[j]
            dist = np.linalg.norm(di - dj)
            return 1.0 - dist
        constraint.name = f"dist_{i}_{j}"
        return constraint

    constraints = []
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})

    # Multi-start strategy
    # 1. Regular polygon
    # 2. Random points
    # 3. Grid-like
    
    initializations = []
    
    # 1. Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    poly_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    initializations.append(poly_pts.flatten())
    
    # 2. Two concentric 8-gons
    angles_inner = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi/8
    angles_outer = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    r_inner = 0.4
    r_outer = 1.0
    inner_pts = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    outer_pts = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
    two_ring_pts = np.vstack([inner_pts, outer_pts])
    initializations.append(two_ring_pts.flatten())
    
    # 3. Random points
    np.random.seed(42)
    for _ in range(5):
        rand_pts = np.random.randn(n, d)
        # Normalize to unit circle roughly
        max_dist = np.max(pdist(rand_pts))
        if max_dist > 0:
            rand_pts = rand_pts * (1.0 / max_dist)
        initializations.append(rand_pts.flatten())

    # Run optimization for each initialization
    for x0 in initializations:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, x0, method='SLSQP', constraints=constraints, 
                           options={'maxiter': 500, 'ftol': 1e-8})
            
            if res.success or res.fun < best_score * -1: # res.fun is -d_min, so lower is better
                # Check if constraints are satisfied
                pts = res.x.reshape(n, d)
                dists = pdist(pts)
                if np.max(dists) <= 1.0 + 1e-6: # Allow small numerical error
                    score = compute_score(pts)
                    if score > best_score:
                        best_score = score
                        best_points = pts
        except Exception:
            pass

    # Final check and return
    # The problem asks for points. The score is (d_min/d_max)^2.
    # My optimization maximized d_min subject to d_max <= 1.
    # This is equivalent to maximizing d_min/d_max.
    
    return best_points

# EVOLVE-BLOCK-END
