import numpy as np
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    # Distance to walls
    # centers shape (N, 3)
    # dist to wall k for center i is 0.5 - abs(centers[i, k])
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances
    # Compute all pairwise differences
    # Use broadcasting
    # diff shape (N, N, 3)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective(centers_flat, beta):
    """
    Objective function for L-BFGS-B.
    Minimizes the soft-min approximation of the radius.
    r(p) approx -1/beta * ln(sum(exp(-beta * x_k)))
    where x_k are the constraints (distances).
    We want to maximize r, so we minimize -r, which is 1/beta * ln(sum(exp(-beta * x_k))).
    """
    centers = centers_flat.reshape(-1, 3)
    
    # 1. Wall distances
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()
    
    # 2. Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    # Get upper triangle to avoid double counting, though summing all is fine for LSE
    # But LSE of duplicates is slightly different scaling. 
    # Let's just use all unique pairs to be consistent with the min.
    iu = np.triu_indices(N, k=1)
    pair_vals_unique = pair_vals[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    # Calculate LSE: 1/beta * ln(sum(exp(-beta * x)))
    # To avoid overflow/underflow, use log-sum-exp trick
    # max_x = np.max(all_vals)
    # sum_exp = np.sum(np.exp(-beta * (all_vals - max_x)))
    # val = max_x - (1.0/beta) * np.log(sum_exp)
    
    # Simple version, beta is usually not huge enough to cause immediate overflow if centers are reasonable
    # But let's be safe.
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    if max_val > 700: # exp(700) is near limit
        # This implies some x are very small (close to 0 or negative?)
        # x should be positive if valid.
        pass
        
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    # Objective is 1/beta * ln(sum(exp(-beta * x)))
    # = 1/beta * (max_val + ln(sum_exp))
    obj = (max_val + np.log(sum_exp)) / beta
    
    # Add a small penalty if centers are outside or coincident to help gradient?
    # The LSE term naturally penalizes small distances.
    
    return obj

def construct():
    start_time = time.time()
    deadline = 120.0 # Safety margin for 120s limit, keep it 115s maybe?
    # The prompt says 120s limit. Let's use 115s to be safe.
    deadline = start_time + 115.0
    
    # Initial guess: Grid
    # 100 spheres. 4x5x5 = 100.
    m = 5
    # We need 100 points.
    # Let's use a grid that fits 100 points roughly evenly.
    # 5x5x4 = 100.
    # x: 5 points, y: 5 points, z: 4 points.
    nx, ny, nz = 5, 5, 4
    
    xs = np.linspace(-0.45, 0.45, nx)
    ys = np.linspace(-0.45, 0.45, ny)
    zs = np.linspace(-0.45, 0.45, nz)
    
    # Create meshgrid
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Shuffle to break symmetry?
    # np.random.shuffle(centers) # Maybe not, keep structure for initial LSE
    # Add small noise to break perfect symmetry which might cause LSE to get stuck in local minima
    centers += np.random.normal(0, 0.01, centers.shape)
    
    # Clip to valid range to ensure initial validity
    centers = np.clip(centers, -0.49, 0.49)
    
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Optimization loop
    # Start with low beta, increase it
    beta = 10.0
    max_beta = 1000.0
    
    while time.time() < deadline and beta < max_beta:
        # Flatten for scipy
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            
            if res.success or res.fun < _soft_min_objective(x0, beta):
                new_centers = res.x.reshape(-1, 3)
                # Ensure validity (clip just in case optimizer went slightly out)
                new_centers = np.clip(new_centers, -0.5, 0.5)
                
                r = _compute_radius(new_centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = new_centers
        except Exception as e:
            pass
            
        # Increase beta
        beta *= 1.5
        
        # Check time
        if time.time() > deadline:
            break
            
    return best_centers

# EVOLVE-BLOCK-END
