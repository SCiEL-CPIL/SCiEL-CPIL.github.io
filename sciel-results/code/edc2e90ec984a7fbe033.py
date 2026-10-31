import numpy as np
from scipy.optimize import minimize
import time

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    # --- Objective Function ---
    # Minimize (max_d / min_d)^2.
    # Using log-space helps with numerical stability and gradient scaling.
    # f = 2 * (log(max_d) - log(min_d))
    # This is equivalent to minimizing log(ratio^2).
    
    def objective(pts_flat):
        pts = pts_flat.reshape(N, D)
        # Compute pairwise squared distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d_sq = np.sum(diff**2, axis=2)
        
        # Mask diagonal
        iu = np.triu_indices(N, 1)
        d_sq_valid = d_sq[iu]
        
        # Check for coincident points
        min_d_sq = np.min(d_sq_valid)
        if min_d_sq < 1e-12:
            return 1e10
            
        max_d_sq = np.max(d_sq_valid)
        
        # Ratio of squared distances is (max_d_sq / min_d_sq)
        # The task asks for (max_d / min_d)^2 which is exactly max_d_sq / min_d_sq
        return max_d_sq / min_d_sq

    def objective_log(pts_flat):
        """
        Log-space objective for better conditioning in L-BFGS-B.
        Minimizing log(max_d_sq/min_d_sq) = log(max_d_sq) - log(min_d_sq)
        is equivalent to minimizing the ratio.
        """
        pts = pts_flat.reshape(N, D)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d_sq = np.sum(diff**2, axis=2)
        
        iu = np.triu_indices(N, 1)
        d_sq_valid = d_sq[iu]
        
        min_d_sq = np.min(d_sq_valid)
        if min_d_sq < 1e-12:
            return 1e10
            
        max_d_sq = np.max(d_sq_valid)
        
        return np.log(max_d_sq) - np.log(min_d_sq)

    # --- Initializers ---
    
    def make_hexagonal_cluster(n):
        """
        Generate a hexagonal lattice cluster of n points centered at origin.
        """
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.5, np.sqrt(3) / 2])
        
        pts = []
        for i in range(-5, 6):
            for j in range(-5, 6):
                p = i * v1 + j * v2
                pts.append(p)
        pts = np.array(pts)
        
        # Center the cluster
        pts -= pts.mean(axis=0)
        
        # Sort by distance to center and pick the closest n
        dists = np.linalg.norm(pts, axis=1)
        order = np.argsort(dists)
        return pts[order[:n]]

    def make_circular_config(n):
        """
        Points on concentric rings.
        1 center + 6 inner + 9 outer? Or 1+5+10?
        Let's try 1 center, 6 on r=1, 9 on r=1.8 (approx hex packing).
        """
        pts = [np.array([0.0, 0.0])]
        r1 = 1.0
        r2 = 1.8
        # 6 points on inner ring
        for k in range(6):
            theta = 2 * np.pi * k / 6
            pts.append(r1 * np.array([np.cos(theta), np.sin(theta)]))
        # 9 points on outer ring
        for k in range(9):
            theta = 2 * np.pi * k / 9
            pts.append(r2 * np.array([np.cos(theta), np.sin(theta)]))
        return np.array(pts)

    def make_grid_config(n):
        """4x4 grid, centered."""
        x, y = np.meshgrid(np.arange(4), np.arange(4))
        pts = np.column_stack([x.ravel(), y.ravel()]).astype(float)
        pts -= pts.mean(axis=0)
        return pts

    def make_random_config(n, seed=0):
        """Random points in unit square, centered."""
        rng = np.random.default_rng(seed)
        pts = rng.random((n, 2))
        pts -= pts.mean(axis=0)
        return pts

    # --- Optimization Loop ---
    
    # 1. Generate diverse initial candidates
    candidates = [
        make_hexagonal_cluster(N),
        make_circular_config(N),
        make_grid_config(N),
        make_random_config(N, seed=1),
        make_random_config(N, seed=2),
        make_random_config(N, seed=3),
    ]

    best_pts = None
    best_score = np.inf

    def update_best(pts, score):
        nonlocal best_pts, best_score
        if score < best_score:
            best_score = score
            best_pts = pts.copy()

    # Evaluate all candidates
    for c in candidates:
        s = objective(c.flatten())
        update_best(c, s)

    # 2. Local Refinement using L-BFGS-B
    # L-BFGS-B is much faster than Powell for this smooth-ish problem and
    # converges to sharper minima. We run it on all initial candidates
    # to ensure we find the global basin.
    
    # Bounds: The optimal configuration is compact.
    # Hexagonal cluster of 16 points has radius ~3.
    # Let's use [-5, 5] to be safe.
    bounds = [(-5.0, 5.0)] * (N * D)

    for c in candidates:
        if time.time() > deadline - 10:
            break
            
        try:
            # Use log-objective for L-BFGS-B as it handles scale better
            # and the gradient is more well-behaved.
            res = minimize(
                objective_log,
                c.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'maxls': 50}
            )
            
            # Evaluate the actual score (not log) for comparison
            final_pts = res.x.reshape(N, D)
            final_score = objective(final_pts.flatten())
            update_best(final_pts, final_score)
            
        except Exception:
            pass

    # 3. If we have time, try a few more random restarts with L-BFGS-B
    # to increase probability of finding a better local minimum.
    num_restarts = 5
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Random start near the best point found so far, or purely random
        if i % 2 == 0 and best_pts is not None:
            # Perturbation of best
            rng = np.random.default_rng(42 + i)
            c = best_pts + rng.normal(0, 0.5, (N, D))
        else:
            c = make_random_config(N, seed=100 + i)
            
        try:
            res = minimize(
                objective_log,
                c.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'maxls': 50}
            )
            
            final_pts = res.x.reshape(N, D)
            final_score = objective(final_pts.flatten())
            update_best(final_pts, final_score)
            
        except Exception:
            pass

    # Final check
    if best_pts is None:
        # Fallback
        best_pts = make_hexagonal_cluster(N)

    return best_pts
