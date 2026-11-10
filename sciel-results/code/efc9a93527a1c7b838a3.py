"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if centers.shape != (N, 3):
        return 0.0
    
    # Distance from each center to nearest wall
    # Wall distance is min(0.5 - x, x + 0.5) for each coordinate
    # which is equivalent to 0.5 - abs(x)
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    if min_wall < -1e-10:
        return 0.0
    
    # Pairwise distances
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # The radius is half the minimum distance between any two centers
    # and also limited by the distance to the walls
    r = 0.5 * min(min_pair, min_wall)
    return r if r > 0 else 0.0

def _force_directed_relax(centers, max_iters=1000, step_size=0.005, decay=0.999, start_time=0, deadline=140.0):
    """
    Vectorized force-directed relaxation to resolve overlaps and push away from walls.
    This is a coarse optimization step.
    """
    centers = centers.copy()
    n = len(centers)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        # Calculate pairwise differences and distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, 1.0) # Avoid division by zero
        
        # Force magnitude: Inverse square law for repulsion
        # We cap the force to prevent instability
        force_mag = 0.05 / dist_sq
        force_mag = np.minimum(force_mag, 10.0)
        np.fill_diagonal(force_mag, 0.0)
        
        # Unit vectors
        dist = np.sqrt(dist_sq)
        unit = diff / dist[:, :, np.newaxis]
        
        # Sum forces
        forces = np.sum(force_mag[:, :, np.newaxis] * unit, axis=1)
        
        # Wall repulsion: Push spheres away from walls if they are too close
        # If x > 0.4, push left. If x < -0.4, push right.
        # We use a soft potential to make it differentiable-ish for the step
        for k in range(3):
            # Distance to wall
            wall_dist = 0.5 - np.abs(centers[:, k])
            # If wall_dist is small, force is large
            # Only apply if close to wall (e.g., < 0.1)
            close_to_wall = wall_dist < 0.1
            if np.any(close_to_wall):
                # Force direction: away from wall
                # sign(centers[:, k]) points away from center towards wall? No, sign is + if x>0.
                # If x > 0, wall is at 0.5. Force should be negative (towards center).
                # So force = -sign(x) * strength
                strength = 0.05 / (np.maximum(wall_dist, 1e-6)**2)
                forces[:, k] -= strength * np.sign(centers[:, k])
                # Only apply to those close
                forces[~close_to_wall, k] = forces[~close_to_wall, k] # No change for others, but we already added to all? 
                # Actually, the vectorized addition above added to all. 
                # Let's refine: only add wall force if close.
                # The line above `forces[:, k] -= ...` added to all.
                # This is okay if strength is 0 for far walls, but 1/d^2 is large.
                # Let's mask it properly.
        
        # Better wall force calculation:
        # Reset wall part of forces? No, just calculate correctly.
        # Let's redo wall force cleanly.
        for k in range(3):
            pos = centers[:, k]
            dist_to_wall = 0.5 - np.abs(pos)
            # Force is only active if dist_to_wall < threshold
            # Force direction: -sign(pos)
            # Magnitude: proportional to 1/d^2
            mag = np.zeros(n)
            mask = dist_to_wall < 0.1
            if np.any(mask):
                mag[mask] = 0.05 / (dist_to_wall[mask]**2 + 1e-8)
            forces[:, k] -= mag * np.sign(pos)
            
        # Update positions
        centers += step_size * forces
        centers = np.clip(centers, -0.499, 0.499)
        step_size *= decay
        
    return centers

def _smooth_min_objective(x, beta=100.0):
    """
    Objective function for L-BFGS-B optimization.
    Minimizes the smooth approximation of the negative minimum distance.
    We want to MAXIMIZE min(distances).
    So we MINIMIZE -min(distances).
    Using log-sum-exp to approximate min:
    min(a_i) approx -1/beta * log(sum(exp(-beta * a_i)))
    So we want to minimize: -(-1/beta * log(sum(exp(-beta * a_i)))) = 1/beta * log(sum(exp(-beta * a_i)))
    Wait, if we want to maximize min(d), we minimize -min(d).
    -min(d) approx 1/beta * log(sum(exp(-beta * d)))
    So objective = 1/beta * log(sum(exp(-beta * d)))
    """
    c = x.reshape(N, 3)
    c = np.clip(c, -0.5, 0.5)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Wall distances
    wall_d = 0.5 - np.abs(c)
    wall_d_flat = wall_d.flatten()
    
    # Combine all distances (pairwise / 2 because radius is d/2, but for relative optimization, scaling doesn't matter much, 
    # but we want to match the definition of r. 
    # r = min( min_pair/2, min_wall )
    # So we are optimizing min( min_pair/2, min_wall ).
    # Let's define d_i as the "radius constraint" for each constraint.
    # For pair (i,j), constraint is dist_ij / 2.
    # For wall (i,k), constraint is wall_dist_ik.
    
    pair_dists = dist[~np.isinf(dist)] / 2.0
    all_dists = np.concatenate([pair_dists, wall_d_flat])
    
    # Smooth min of all_dists
    # We want to maximize this. So minimize -smooth_min.
    # smooth_min(x) = -1/beta * log(sum(exp(-beta * x)))
    # Objective = - smooth_min(x) = 1/beta * log(sum(exp(-beta * x)))
    
    # Numerical stability: subtract max
    max_d = np.max(all_dists)
    shifted = all_dists - max_d
    shifted = np.clip(shifted, -50.0, 0.0) # Avoid overflow in exp
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    
    # The smooth min value is max_d - log_sum / beta
    # We want to maximize this, so we return the negative:
    return -(max_d - log_sum / beta)

def _local_refine(centers, maxiter=100, beta=100.0, start_time=0, deadline=140.0):
    """
    Refine centers using L-BFGS-B to maximize the minimum distance (radius).
    """
    if time.time() > deadline:
        return centers
        
    try:
        res = minimize(
            _smooth_min_objective, 
            centers.flatten(), 
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (N*3),
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        
        if res.success:
            c_opt = res.x.reshape(N, 3)
            c_opt = np.clip(c_opt, -0.5, 0.5)
            return c_opt
        else:
            return centers
    except Exception:
        return centers

def construct():
    """Construct centers for N equal spheres in unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # 1. Initial Grid
    m = 5 # 5x5x5 = 125 points, we take 100
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Randomly select 100 from the 125 grid points to start
    rng = np.random.RandomState(42)
    idx = rng.choice(len(grid), N, replace=False)
    best_centers = grid[idx].astype(float).copy()
    best_radius = _compute_radius(best_centers)
    
    # 2. Phase 1: Multiple restarts with Force-Directed Relaxation
    # We will try a few random restarts, relax them, and keep the best.
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline - 30.0: # Reserve 30s for final refinement
            break
            
        # Random initial positions
        centers = rng.uniform(-0.4, 0.4, size=(N, 3))
        
        # Force-directed relaxation to resolve overlaps
        centers = _force_directed_relax(
            centers, 
            max_iters=500, 
            step_size=0.005, 
            decay=0.999,
            start_time=start_time,
            deadline=deadline - 30.0
        )
        
        # Quick local refinement to get a better estimate
        centers_refined = _local_refine(
            centers, 
            maxiter=20, 
            beta=50.0,
            start_time=start_time,
            deadline=deadline - 30.0
        )
        
        radius = _compute_radius(centers_refined)
        if radius > best_radius:
            best_radius = radius
            best_centers = centers_refined.copy()
            
    # 3. Phase 2: Deep Refinement on the Best Configuration
    # Use a higher beta for more precise minimization of the smooth min
    if time.time() < deadline - 5.0:
        # First, a bit more force-directed to ensure no hard overlaps remain that L-BFGS might struggle with
        best_centers = _force_directed_relax(
            best_centers, 
            max_iters=200, 
            step_size=0.001, 
            decay=0.999,
            start_time=start_time,
            deadline=deadline - 5.0
        )
        
        # High precision L-BFGS
        best_centers = _local_refine(
            best_centers, 
            maxiter=200, 
            beta=200.0,
            start_time=start_time,
            deadline=deadline - 2.0
        )
        
        final_radius = _compute_radius(best_centers)
        if final_radius > best_radius:
            best_radius = final_radius
            best_centers = best_centers.copy()
            
    # 4. Final Safety Check
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
# EVOLVE-BLOCK-END
