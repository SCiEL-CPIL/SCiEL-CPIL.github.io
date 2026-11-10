"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _compute_radius(centers):
    """Compute the valid radius for a set of centers."""
    if centers is None or centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Distance to walls
    min_dist_wall = 0.5 - np.max(np.abs(centers), axis=1)
    r_wall = np.min(min_dist_wall)
    
    # Distance between spheres
    if N < 2:
        return r_wall
        
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist_pair = np.min(dist)
    r_pair = min_dist_pair / 2.0
    
    return min(r_wall, r_pair)

def _smooth_min_obj(centers, beta=50.0):
    """
    Objective function to minimize.
    We want to maximize r.
    r = min( min_wall_dist, min_pair_dist/2 )
    
    We approximate min using log-sum-exp (soft-min).
    To maximize min, we minimize -min.
    
    Let d_pairs be all pair distances.
    Let d_walls be all wall distances.
    
    We combine them into a single set of 'constraints' that must be >= 2r.
    Actually, it's easier to just minimize the soft-min of all relevant distances.
    
    Constraints:
    1. Pair distances d_ij >= 2r
    2. Wall distances w_i >= r
    
    To treat them equally, we can scale wall distances by 2?
    If we define effective distance D:
    For pairs: D_ij = d_ij
    For walls: D_i = 2 * w_i
    
    Then r = min( min(D_ij)/2, min(D_i)/2 ) = min( all D ) / 2.
    
    So we want to maximize min(D).
    Minimize -log(sum(exp(-beta * D))) approximates -min(D).
    """
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Get all pair distances (upper triangle to avoid duplicates, but for logsumexp order doesn't matter much)
    iu = np.triu_indices(N, k=1)
    pair_dists = dist[iu]
    
    # Wall distances
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    
    # Effective distances
    # Pairs contribute directly to 2r constraint
    # Walls contribute to r constraint, so 2*wall_dist is the equivalent 'pair-like' distance
    eff_dists = np.concatenate([pair_dists, 2.0 * wall_dists])
    
    # Soft min
    # min(x) ~ -1/beta * log(sum(exp(-beta * x)))
    # We want to maximize this, so we minimize its negative.
    # Let's return the value to minimize: -soft_min
    # To avoid overflow, subtract max
    m = np.max(eff_dists)
    log_sum_exp = m + np.log(np.sum(np.exp(-beta * (eff_dists - m))))
    soft_min_val = -log_sum_exp / beta
    
    return -soft_min_val

def _push_algorithm(centers, max_iter=5000, tol=1e-8):
    """
    Structural Change: Hard-constraint push algorithm.
    Instead of smooth forces, identify the exact bottleneck and fix it.
    """
    centers = centers.copy()
    
    for iter in range(max_iter):
        # 1. Find current bottleneck
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Pair bottleneck
        min_pair_idx = np.unravel_index(np.argmin(dist), dist.shape)
        i, j = min_pair_idx
        d_pair = dist[i, j]
        
        # Wall bottleneck
        wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
        min_wall_idx = np.argmin(wall_dists)
        d_wall = wall_dists[min_wall_idx]
        
        # Determine which is the limiting factor
        # r_pair = d_pair / 2
        # r_wall = d_wall
        # If d_pair/2 < d_wall, pair is bottleneck.
        # Else wall is bottleneck.
        
        if d_pair / 2.0 < d_wall:
            # Push pair apart
            if d_pair < 1e-6:
                # Random nudge if coincident
                centers[i] += np.random.normal(0, 0.01, 3)
                centers[j] -= np.random.normal(0, 0.01, 3)
                continue
            
            dir_vec = centers[i] - centers[j]
            dir_norm = np.linalg.norm(dir_vec)
            if dir_norm < 1e-9:
                continue
            dir_vec /= dir_norm
            
            # We want to increase d_pair.
            # Move i in dir_vec, j in -dir_vec.
            # How much? 
            # If we just push by a small epsilon, we might not fix it if it's deep.
            # But pushing too much might break other constraints.
            # A safe bet: push by half the "deficit" relative to some target? 
            # We don't have a target. 
            # Let's just push by a fixed small step, or proportional to distance?
            # Actually, to maximize min, we should push until it's no longer the minimum, 
            # or by a small amount to allow other pairs to become minimum.
            # Let's use a step size that is a fraction of the current min distance?
            # No, that's dangerous.
            # Let's use a small fixed step, but adaptive if stuck.
            
            step = 0.005
            # Check if this move would violate wall constraints for i or j
            # Simple check: just move and clip
            centers[i] += step * dir_vec
            centers[j] -= step * dir_vec
        else:
            # Push wall point inward
            c = centers[min_wall_idx]
            abs_c = np.abs(c)
            max_coord = np.argmax(abs_c)
            sign = np.sign(c[max_coord])
            
            # Move inward
            step = 0.005
            centers[min_wall_idx, max_coord] -= sign * step
            
        # Clip to box (strictly inside)
        centers = np.clip(centers, -0.4999, 0.4999)
        
        # Convergence check: if min radius hasn't improved in a while, break?
        # For now, just run fixed iterations.
        
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_radius = 0.0
    best_centers = None
    
    # Strategy: 
    # 1. Generate diverse initializations (Grid, Random, Lattice)
    # 2. For each, run a "Hybrid" optimization:
    #    a. Smooth-min gradient ascent (using finite differences or analytic if possible? Finite diff is slow but robust. Let's use a simple coordinate descent or Nelder-Mead on the smooth objective? Nelder-Mead on 300 dims is bad.
    #    b. Let's stick to the Push Algorithm but with "Jitter" to escape local minima.
    #    
    #    The Push Algorithm is deterministic and can get stuck.
    #    Let's combine:
    #    1. Random start.
    #    2. Run Push Algorithm for T1 iterations.
    #    3. Add noise.
    #    4. Run Push Algorithm for T2 iterations.
    #    5. Evaluate.
    
    # Initialization 1: Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    init_centers = grid[:N].copy()
    
    # Initialization 2: Random
    np.random.seed(42)
    rand_centers = np.random.uniform(-0.4, 0.4, (N, 3))
    
    # Initialization 3: Face-centered cubic lattice subset?
    # FCC is dense. 
    # Let's just use the grid and random for now to keep it simple and fast.
    
    initializations = [init_centers, rand_centers]
    
    for init in initializations:
        if time.time() > deadline - 30.0:
            break
            
        centers = init.copy()
        
        # Phase 1: Coarse Push
        # Run push algorithm with larger steps? The function uses fixed 0.005.
        # Let's run it for a while.
        centers = _push_algorithm(centers, max_iter=2000)
        
        # Phase 2: Jitter and Fine Push
        # Add small noise to break symmetry/stuck states
        centers += np.random.normal(0, 0.01, centers.shape)
        centers = np.clip(centers, -0.49, 0.49)
        
        centers = _push_algorithm(centers, max_iter=2000)
        
        # Phase 3: Final Polish with very small steps
        # We can modify _push_algorithm to take step size, or just trust the fixed one.
        # Let's do a few manual micro-pushes
        for _ in range(500):
            if time.time() > deadline - 10.0:
                break
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist, np.inf)
            
            min_pair_idx = np.unravel_index(np.argmin(dist), dist.shape)
            i, j = min_pair_idx
            d_pair = dist[i, j]
            
            wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
            min_wall_idx = np.argmin(wall_dists)
            d_wall = wall_dists[min_wall_idx]
            
            if d_pair / 2.0 < d_wall:
                if d_pair < 1e-6: continue
                dir_vec = centers[i] - centers[j]
                dir_norm = np.linalg.norm(dir_vec)
                if dir_norm < 1e-9: continue
                dir_vec /= dir_norm
                step = 0.001
                centers[i] += step * dir_vec
                centers[j] -= step * dir_vec
            else:
                c = centers[min_wall_idx]
                abs_c = np.abs(c)
                max_coord = np.argmax(abs_c)
                sign = np.sign(c[max_coord])
                step = 0.001
                centers[min_wall_idx, max_coord] -= sign * step
                
            centers = np.clip(centers, -0.4999, 0.4999)
            
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
            
    # If we have time, try a few more random restarts
    while time.time() < deadline - 20.0:
        centers = np.random.uniform(-0.45, 0.45, (N, 3))
        centers = _push_algorithm(centers, max_iter=1000)
        centers += np.random.normal(0, 0.005, centers.shape)
        centers = np.clip(centers, -0.49, 0.49)
        centers = _push_algorithm(centers, max_iter=1000)
        
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
            
    if best_centers is None:
        # Fallback to grid
        best_centers = init_centers.copy()
        
    return best_centers

# EVOLVE-BLOCK-END
