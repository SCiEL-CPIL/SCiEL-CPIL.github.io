import numpy as np
import scipy.optimize
import time

N = 100

def _compute_min_radius(c):
    """
    Compute the exact minimum radius for configuration c.
    c: (N, 3) array of centers.
    """
    # Pairwise distances
    # c[:, np.newaxis, :] - c[np.newaxis, :, :]
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair = np.sqrt(np.min(dist_sq)) / 2.0
    
    # Wall distances
    # Distance to x_min (-0.5) is c_x + 0.5
    # Distance to x_max (0.5) is 0.5 - c_x
    # Min is 0.5 - |c_x|
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    return min(min_pair, min_wall)

def _softmin_radius(c, beta=100.0):
    """
    Compute a smooth approximation of the minimum radius.
    Returns the soft-min value and the weights for gradient calculation if needed.
    Here we just return the value for optimization.
    """
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    pair_dists = dist / 2.0
    
    wall_dists = 0.5 - np.abs(c)
    min_wall_per_sphere = np.min(wall_dists, axis=1)
    
    # Extract upper triangle for pairs
    idx = np.triu_indices(N, 1)
    pair_flat = pair_dists[idx]
    
    # Combine all distances
    all_dists = np.concatenate([pair_flat, min_wall_per_sphere])
    
    # Soft min: -1/beta * log(sum(exp(-beta * d)))
    # Use log-sum-exp trick for stability
    max_d = np.max(all_dists)
    # If all_dists are very large, exp(-beta * (d - max_d)) is safe
    log_sum = np.log(np.sum(np.exp(-beta * (all_dists - max_d))))
    softmin_val = max_d - (1.0 / beta) * log_sum
    
    return softmin_val

def construct():
    # 1. Initialization: Jittered Grid
    # 5x5x5 grid = 125 points. We need 100.
    # Create grid points
    m = 5
    # Center at 0.
    # Points: -0.4, -0.2, 0.0, 0.2, 0.4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Randomly select 100 points
    indices = np.random.choice(len(grid), N, replace=False)
    c = grid[indices].copy()
    
    # Add small jitter to break symmetry
    c += np.random.normal(0, 0.01, c.shape)
    
    # Ensure inside bounds (jitter might push out)
    # Clip to [-0.49, 0.49] to be safe, but let SA handle it
    c = np.clip(c, -0.49, 0.49)
    
    best_c = c.copy()
    best_r = _compute_min_radius(c)
    
    # 2. Simulated Annealing
    start_time = time.time()
    time_limit_sa = 130.0
    T = 0.05
    T_min = 1e-6
    cooling_rate = 0.999
    current_r = best_r
    current_c = c.copy()
    
    while time.time() - start_time < time_limit_sa:
        # Adaptive step size
        # As T drops, sigma drops
        sigma = 0.02 * (T / 0.05) + 0.001
        
        # Pick a random sphere
        i = np.random.randint(N)
        
        # Propose move
        c_new = current_c.copy()
        c_new[i] += np.random.normal(0, sigma, 3)
        
        # Calculate new radius
        r_new = _compute_min_radius(c_new)
        
        # Update best
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
            
        # Metropolis criterion
        delta = r_new - current_r
        if delta > 0:
            current_c = c_new
            current_r = r_new
        else:
            # Accept with probability exp(delta / T)
            if np.random.random() < np.exp(delta / T):
                current_c = c_new
                current_r = r_new
        
        # Cool
        T *= cooling_rate
        if T < T_min:
            break
            
    # 3. Polish with Nelder-Mead on Soft-Min Surrogate
    # Reserve 15s for polish
    time_left = 150.0 - (time.time() - start_time) - 5.0 # 5s safety margin
    if time_left > 10.0:
        # Define objective for minimization
        # We want to maximize softmin, so minimize -softmin
        def objective(c_flat):
            c = c_flat.reshape(N, 3)
            # Soft min with high beta for sharpness
            val = _softmin_radius(c, beta=100.0)
            return -val
        
        # Use Nelder-Mead
        # Initial simplex size
        x0 = best_c.flatten()
        
        # Run minimization
        # maxiter is a hard stop, but we also monitor time
        # SciPy doesn't have a time limit callback, so we rely on maxiter
        # or a wrapper.
        # For N=100, 300 vars, NM might take many iterations.
        # Let's limit iterations to avoid timeout.
        try:
            res = scipy.optimize.minimize(
                objective, 
                x0, 
                method='Nelder-Mead',
                options={
                    'maxiter': 5000, 
                    'xatol': 1e-8, 
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            c_polished = res.x.reshape(N, 3)
            r_polished = _compute_min_radius(c_polished)
            
            if r_polished > best_r:
                best_c = c_polished
                best_r = r_polished
        except Exception:
            pass # Fallback to SA result if polish fails

    return best_c
