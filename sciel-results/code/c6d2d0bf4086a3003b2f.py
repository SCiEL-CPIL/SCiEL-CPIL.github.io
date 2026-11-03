import numpy as np
import time
import scipy.optimize as opt

N = 50

def compute_score(centers):
    """
    Compute the score (radius) for a given set of centers.
    Returns 0 if invalid.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Check bounds
    # Cube is [-0.5, 0.5]
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
        
    # Calculate min distance between centers
    # dists[i, j] = |c_i - c_j|
    # Efficient calculation using broadcasting
    dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    # Calculate min distance to walls
    # Distance to nearest wall for each point
    # dist_to_wall_i = min(0.5 - x, x + 0.5, 0.5 - y, y + 0.5, 0.5 - z, z + 0.5)
    # Simplified: min(0.5 - |x|, 0.5 - |y|, 0.5 - |z|)
    abs_centers = np.abs(centers)
    wall_dists = 0.5 - np.max(abs_centers, axis=1)
    min_wall_dist = np.min(wall_dists)
    
    radius = min(min_dist / 2.0, min_wall_dist)
    
    if radius < 0:
        return 0.0
        
    return radius

def md_step(centers, velocities, dt, damping=0.99, shake=0.0, box_limit=0.5):
    """
    Perform one step of molecular dynamics.
    Potential: U = sum 1/d^2
    Force: F = 2/d^3 * r_vec
    """
    # Calculate forces
    # dists[i, j]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Avoid division by zero
    dists[~np.isfinite(dists)] = 1e-6
    dists[range(N), range(N)] = 1.0 # Diagonal, will be masked
    
    # Force magnitude factor: 2 / d^3
    # F_ij = (x_i - x_j) / d^3 * 2
    force_factor = 2.0 / (dists ** 3)
    # Mask diagonal
    force_factor[range(N), range(N)] = 0.0
    
    # Force on i from j is sum over j of force_factor[i,j] * diff[i,j]
    # diff[i,j] is x_i - x_j
    forces = np.sum(force_factor[:, :, np.newaxis] * diff, axis=1)
    
    # Add wall forces
    # If x > limit - margin, push back
    margin = 0.01
    wall_force = np.zeros_like(centers)
    # Strong spring force near walls
    k_wall = 100.0
    
    for k in range(3):
        # Right wall
        mask_r = centers[:, k] > box_limit - margin
        wall_force[mask_r, k] -= k_wall * (centers[mask_r, k] - (box_limit - margin))
        # Left wall
        mask_l = centers[:, k] < -box_limit + margin
        wall_force[mask_l, k] += k_wall * ((-box_limit + margin) - centers[mask_l, k])
        
    total_forces = forces + wall_force
    
    # Update velocities
    velocities = velocities * damping + total_forces * dt
    if shake > 0:
        velocities += np.random.normal(0, shake, size=centers.shape)
        
    # Update positions
    new_centers = centers + velocities * dt
    
    # Hard constraint: If still outside, clamp and zero velocity component
    for k in range(3):
        out_r = new_centers[:, k] > box_limit
        if np.any(out_r):
            new_centers[out_r, k] = box_limit
            velocities[out_r, k] = 0.0
        out_l = new_centers[:, k] < -box_limit
        if np.any(out_l):
            new_centers[out_l, k] = -box_limit
            velocities[out_l, k] = 0.0
            
    return new_centers, velocities

def run_md(initial_centers, steps=5000, dt=0.005, start_shake=0.05, end_shake=0.001):
    centers = initial_centers.copy()
    velocities = np.zeros_like(centers)
    
    best_score = compute_score(centers)
    best_centers = centers.copy()
    
    for i in range(steps):
        # Anneal shake
        progress = i / steps
        shake = start_shake * (1 - progress) + end_shake * progress
        
        centers, velocities = md_step(centers, velocities, dt, shake=shake)
        
        # Check score periodically
        if i % 100 == 0:
            score = compute_score(centers)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                
    return best_centers, best_score

def soft_min_obj(x, beta=1000.0):
    """
    Objective to minimize: -soft_min(distances)
    x is flattened centers (N*3)
    """
    centers = x.reshape(N, 3)
    
    # Check bounds penalty
    # If any center is outside [-0.5, 0.5], return large penalty
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 1e10
        
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0) # Ignore self
    
    # Soft min: -1/beta * log(sum(exp(-beta * dist)))
    # To avoid overflow, subtract max (which is 0 since exp(-beta*d) <= 1)
    # Actually, exp(-beta * dist) can underflow if dist is large, which is fine.
    # But if dist is 0, exp(0)=1.
    # We want to maximize min dist, so minimize -soft_min.
    
    # Use log-sum-exp trick for stability
    # log(sum(exp(-beta * d)))
    # Let m = max(-beta * d) = -beta * min(d)
    # sum(exp(-beta * d)) = exp(m) * sum(exp(-beta * d - m))
    # log(sum(...)) = m + log(sum(exp(...)))
    
    exponents = -beta * dists
    # Mask diagonal
    exponents[range(N), range(N)] = -1e10 # -inf
    
    max_exp = np.max(exponents)
    if not np.isfinite(max_exp):
        return 1e10
        
    sum_exp = np.sum(np.exp(exponents - max_exp))
    log_sum_exp = max_exp + np.log(sum_exp)
    
    soft_min_val = -log_sum_exp / beta
    
    return -soft_min_val

def polish(centers, time_limit=30.0):
    """
    Use L-BFGS-B to polish the centers.
    """
    start_time = time.time()
    x0 = centers.flatten()
    
    # Bounds
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    def obj(x):
        if time.time() - start_time > time_limit:
            return 1e10
        return soft_min_obj(x, beta=500.0)
        
    try:
        res = opt.minimize(obj, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
        if res.success:
            new_centers = res.x.reshape(N, 3)
            # Ensure strictly inside
            new_centers = np.clip(new_centers, -0.499999, 0.499999)
            return new_centers
    except Exception:
        pass
        
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_score = 0.0
    best_centers = None
    
    # Initializations
    inits = []
    
    # 1. Random
    for i in range(5):
        np.random.seed(i)
        inits.append(np.random.uniform(-0.4, 0.4, (N, 3)))
        
    # 2. Grid (Baseline)
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    inits.append(grid[:N])
    
    # 3. FCC-like
    # Generate FCC points in a larger grid and pick closest N to origin?
    # Or just scale a standard FCC chunk.
    # Let's try a simple structured random
    for i in range(3):
        np.random.seed(100 + i)
        # Start with grid and perturb
        perturbed = grid[:N].copy()
        perturbed += np.random.normal(0, 0.05, (N, 3))
        perturbed = np.clip(perturbed, -0.4, 0.4)
        inits.append(perturbed)

    # Run MD for each init
    for idx, init in enumerate(inits):
        if time.time() > deadline - 40: # Leave time for polish
            break
            
        centers, score = run_md(init, steps=2000, dt=0.005)
        
        if score > best_score:
            best_score = score
            best_centers = centers
            
    # Polish the best candidate
    if best_centers is not None and time.time() < deadline - 10:
        polished = polish(best_centers, time_limit=30.0)
        polished_score = compute_score(polished)
        if polished_score > best_score:
            best_score = polished_score
            best_centers = polished
            
    # Final check
    if best_centers is None:
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers

# EVOLVE-BLOCK-END
