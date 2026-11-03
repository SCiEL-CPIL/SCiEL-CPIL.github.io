import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # 10s margin from 150s limit

def compute_score(centers):
    """Compute the actual score (min radius)."""
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists) / 2.0
    
    # Distances to boundaries
    # Cube is [-0.5, 0.5]^3
    # Dist to x_min: 0.5 - x, x_max: 0.5 + x, etc.
    dist_x_min = 0.5 - centers[:, 0]
    dist_x_max = 0.5 + centers[:, 0]
    dist_y_min = 0.5 - centers[:, 1]
    dist_y_max = 0.5 + centers[:, 1]
    dist_z_min = 0.5 - centers[:, 2]
    dist_z_max = 0.5 + centers[:, 2]
    
    min_boundary = np.min(np.concatenate([dist_x_min, dist_x_max, dist_y_min, dist_y_max, dist_z_min, dist_z_max]))
    
    return min(min_pairwise, min_boundary)

def soft_min_objective(x_flat, beta):
    """
    Objective function for L-BFGS-B.
    Minimize -soft_min(distances, beta).
    """
    centers = x_flat.reshape(N, 3)
    
    # Calculate all distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    
    # Boundary distances
    dist_x_min = 0.5 - centers[:, 0]
    dist_x_max = 0.5 + centers[:, 0]
    dist_y_min = 0.5 - centers[:, 1]
    dist_y_max = 0.5 + centers[:, 1]
    dist_z_min = 0.5 - centers[:, 2]
    dist_z_max = 0.5 + centers[:, 2]
    
    all_dists = np.concatenate([dists.ravel(), dist_x_min, dist_x_max, dist_y_min, dist_y_max, dist_z_min, dist_z_max])
    
    # Filter out any non-finite or negative distances (shouldn't happen with bounds, but safe)
    all_dists = all_dists[all_dists > 0]
    
    if len(all_dists) == 0:
        return 0.0
        
    d_min = np.min(all_dists)
    # Log-sum-exp for stability: -1/beta * log(sum(exp(-beta * (d - d_min)))) + d_min
    # We want to minimize -soft_min = - (d_min - 1/beta * log(sum(exp(-beta*(d-d_min)))))
    # = -d_min + 1/beta * log(sum(exp(-beta*(d-d_min))))
    
    exponents = -beta * (all_dists - d_min)
    # Clip exponents to avoid underflow/overflow issues if beta is very large, though log-sum-exp is stable
    # max exponent is 0, so exp(0)=1. Others are negative.
    sum_exp = np.sum(np.exp(exponents))
    
    if sum_exp == 0:
        return 0.0
        
    soft_min_val = d_min - (1.0 / beta) * np.log(sum_exp)
    return -soft_min_val

def construct():
    start_time = time.time()
    
    # Initializations
    initializations = []
    
    # 1. Jittered Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:N]
    # Add small jitter
    jittered = grid + np.random.normal(0, 0.01, grid.shape)
    initializations.append(jittered)
    
    # 2. Random Points
    for _ in range(5):
        random_pts = np.random.uniform(-0.45, 0.45, (N, 3))
        initializations.append(random_pts)
        
    # 3. "Spectral" or structured? Let's try a few more jittered grids with different seeds
    for _ in range(5):
        jittered2 = grid + np.random.normal(0, 0.05, grid.shape)
        initializations.append(jittered2)

    best_score = -1.0
    best_centers = initializations[0]
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    for init_idx, init_centers in enumerate(initializations):
        if time.time() - start_time > DEADLINE:
            break
            
        x0 = init_centers.flatten()
        
        # Annealing schedule for beta
        betas = [10, 50, 100, 200, 500, 1000, 2000]
        
        current_x = x0
        for beta in betas:
            if time.time() - start_time > DEADLINE:
                break
                
            try:
                res = minimize(
                    soft_min_objective, 
                    current_x, 
                    args=(beta,), 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                current_x = res.x
            except Exception:
                # If optimization fails, keep current_x
                pass
                
        # Evaluate the hard score for this result
        final_centers = current_x.reshape(N, 3)
        # Ensure strict bounds (clip)
        final_centers = np.clip(final_centers, -0.5, 0.5)
        
        score = compute_score(final_centers)
        
        if score > best_score:
            best_score = score
            best_centers = final_centers
            
        # Check time again
        if time.time() - start_time > DEADLINE - 5:
            break

    return best_centers
