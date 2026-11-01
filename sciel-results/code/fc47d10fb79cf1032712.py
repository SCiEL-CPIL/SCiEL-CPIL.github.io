import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Initialize with ring structure
    # Known optimal topology for N=50: 1, 6, 12, 12, 19
    ring_sizes = [1, 6, 12, 12, 19]
    # Approximate radial positions for centers
    # These are guesses, the optimizer will refine them
    ring_radii = [0.0, 0.25, 0.45, 0.65, 0.85]
    
    points = []
    for r, n in zip(ring_radii, ring_sizes):
        if n == 1:
            points.append([0.0, 0.0])
        else:
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            # Add a small offset to break symmetry if needed, or just start with 0
            for a in angles:
                points.append([r * np.cos(a), r * np.sin(a)])
    
    points = np.array(points)
    
    # Helper to compute the score (min radius)
    def get_min_radius(pts):
        # Distance to boundary
        dists_bound = 1.0 - np.linalg.norm(pts, axis=1)
        min_bound = np.min(dists_bound)
        
        # Pairwise distances
        # To be efficient, we can compute all pairwise distances
        # N=50, so 50*49/2 = 1225 pairs. This is small.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Mask diagonal
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        return min(min_bound, min_pair)
    
    best_radius = get_min_radius(points)
    best_points = points.copy()
    
    # Objective function for optimizer
    # We want to maximize min_radius.
    # Minimize -log-sum-exp(-lambda * gaps)
    # Gaps are: 1 - |c_i| and |c_i - c_j|/2
    # Let's define a function that takes flat array of points and returns a scalar to minimize.
    
    def objective(flat_pts, lambda_val):
        pts = flat_pts.reshape(-1, 2)
        
        # Boundary gaps
        dists_bound = 1.0 - np.linalg.norm(pts, axis=1)
        
        # Pairwise gaps
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        pair_gaps = dists / 2.0
        
        # Combine all gaps into one array
        # Note: We want to maximize the MINIMUM gap.
        # So we minimize -min(gaps).
        # Smooth approximation: -log(sum(exp(-lambda * gaps))) / lambda
        # As lambda -> inf, this approaches -min(gaps).
        
        all_gaps = np.concatenate([dists_bound, pair_gaps.ravel()])
        
        # Clip gaps to avoid overflow in exp
        all_gaps = np.clip(all_gaps, -100, 100)
        
        # Log-sum-exp trick
        max_gap = np.max(all_gaps)
        if max_gap == 0:
             max_gap = 1e-10
        lse = max_gap + np.log(np.sum(np.exp(-lambda_val * (all_gaps - max_gap))))
        
        return -lse / lambda_val
        
    # Optimization loop
    # Start with small lambda, increase it
    lambda_vals = [10, 50, 100, 500, 1000, 2000, 5000]
    
    current_points = points.copy()
    
    for lam in lambda_vals:
        if time.time() > deadline:
            break
            
        # Reshape to flat
        x0 = current_points.flatten()
        
        # Use L-BFGS-B
        try:
            res = minimize(objective, x0, args=(lam,), method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
            if res.success:
                new_points = res.x.reshape(-1, 2)
                new_radius = get_min_radius(new_points)
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_points = new_points
                    current_points = new_points
        except Exception as e:
            pass
            
        # Check time
        if time.time() > deadline:
            break

    return best_points
