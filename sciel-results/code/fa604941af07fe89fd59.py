"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as opt

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

N = 50

def _compute_radius(centers):
    """
    Compute the true maximum radius for a given configuration.
    """
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_pair, min_wall)

def _soft_min_obj(x, tau):
    """
    Objective function for optimization: Minimize -soft_min(distances).
    x: flattened array of centers (2*N)
    tau: temperature for soft-min
    """
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    # We want min(dists)/2, so we can just take dists and divide by 2 at the end or scale tau
    # Let's work with raw distances and divide by 2 at the end of the soft min
    # Soft min of distances: -1/tau * log(sum(exp(-d_i/tau)))
    # To avoid overflow, subtract max
    d_flat = dists[~np.eye(N, dtype=bool)]
    
    all_dists = np.concatenate([walls, d_flat])
    
    # Clip distances to be positive to avoid log issues if they cross
    # But in a valid packing they should be positive.
    # If a circle is outside, wall dist is negative. We want to penalize that heavily.
    # The soft min will naturally pick the smallest (most negative) one.
    
    m = np.min(all_dists)
    # If m is very negative, exp(-d/tau) might overflow for large d.
    # Standard trick: exp(-(d-m)/tau)
    # soft_min = m - tau * log(sum(exp(-(d-m)/tau)))
    
    # To keep numbers stable, we can just use the formula directly if tau is not too small
    # But for optimization, we just need the gradient direction.
    
    # Calculate soft min
    # Using log-sum-exp trick
    vals = -all_dists / tau
    vals_max = np.max(vals)
    sum_exp = np.sum(np.exp(vals - vals_max))
    soft_min_val = m - tau * (np.log(sum_exp) + vals_max)
    
    # We want to MAXIMIZE soft_min_val, so MINIMIZE -soft_min_val
    return -soft_min_val / 2.0 # Divide by 2 because pair dist is radius*2

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    # 1. Initialization
    # Try a few different initializations and pick the best
    best_centers = None
    best_radius = 0.0
    
    rng = np.random.default_rng(42)
    
    # Initialization 1: Jittered Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    init1 = grid[:N].copy()
    init1 += rng.normal(0, 0.02, init1.shape)
    init1 = np.clip(init1, -0.49, 0.49)
    
    # Initialization 2: Hexagonal-ish
    # Generate points in a hexagonal pattern
    points = []
    spacing = 0.2 # Initial guess
    for i in range(10):
        for j in range(10):
            x = -0.5 + j * spacing + (i % 2) * spacing / 2
            y = -0.5 + i * spacing * 0.866
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
    # Shuffle and take N
    rng.shuffle(points)
    init2 = np.array(points[:N])
    init2 += rng.normal(0, 0.02, init2.shape)
    init2 = np.clip(init2, -0.49, 0.49)
    
    # Evaluate inits
    r1 = _compute_radius(init1)
    r2 = _compute_radius(init2)
    
    if r1 >= r2:
        current_centers = init1
        current_radius = r1
    else:
        current_centers = init2
        current_radius = r2
        
    best_centers = current_centers.copy()
    best_radius = current_radius
    
    # 2. Simulated Annealing
    T_start = 0.05
    T_end = 1e-6
    max_steps = 1000000 # Upper bound, limited by time
    
    # Cooling rate: T_k = T_start * (T_end/T_start)^(k/max_steps)
    # Or simple geometric decay
    alpha = (T_end / T_start) ** (1.0 / max_steps)
    
    for step in range(max_steps):
        if time.time() > deadline - 40.0: # Reserve 40s for polishing
            break
            
        T = T_start * (alpha ** step)
        
        # Move: perturb 1 to 3 random circles
        num_movers = rng.integers(1, 4) # 1, 2, or 3
        indices = rng.choice(N, size=num_movers, replace=False)
        
        # Step size scales with T? Or fixed?
        # Fixed step size that decays slightly is often robust
        step_size = 0.05 * (0.5 + 0.5 * T / T_start)
        
        delta = rng.normal(0, step_size, (num_movers, 2))
        
        new_centers = current_centers.copy()
        new_centers[indices] += delta
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_radius = _compute_radius(new_centers)
        
        if new_radius > current_radius:
            current_centers = new_centers
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = new_centers.copy()
        else:
            # Metropolis criterion
            delta_r = new_radius - current_radius
            # If delta_r is very negative, prob is 0
            if delta_r < -10 * T:
                prob = 0.0
            else:
                prob = np.exp(delta_r / T)
                
            if rng.random() < prob:
                current_centers = new_centers
                current_radius = new_radius
                
    # 3. Polishing Phase
    # Use L-BFGS-B to maximize the soft-min radius
    # We optimize -soft_min_obj
    
    if time.time() < deadline - 5.0:
        x0 = best_centers.flatten()
        
        # Try a few tau values, decreasing
        taus = [0.05, 0.01, 0.001, 0.0001]
        
        for tau in taus:
            if time.time() > deadline - 5.0:
                break
                
            # Define objective for this tau
            def obj(x):
                return _soft_min_obj(x, tau)
            
            # Bounds: each coordinate in [-0.5, 0.5]
            bounds = [(-0.5, 0.5)] * (2 * N)
            
            try:
                res = opt.minimize(
                    obj, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.success or res.fun < obj(x0):
                    x_new = res.x
                    centers_new = x_new.reshape(-1, 2)
                    # Verify it's valid and calculate true radius
                    r_new = _compute_radius(centers_new)
                    if r_new > best_radius:
                        best_radius = r_new
                        best_centers = centers_new
                        x0 = x_new
            except Exception:
                pass
                
    return best_centers
# EVOLVE-BLOCK-END
