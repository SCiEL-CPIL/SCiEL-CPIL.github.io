"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_radius(centers):
    """Compute the maximum valid radius for given centers."""
    centers = np.asarray(centers)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Wall constraints: distance to nearest wall
    # Cube is [-0.5, 0.5]^3
    # Distance from center c to wall is 0.5 - |c|
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    if min_wall < 0:
        return 0.0
    
    # Pairwise constraints
    # We only need the minimum pairwise distance / 2
    # Vectorized computation
    # diff[i, j, k] = centers[i, k] - centers[j, k]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    min_pair_dist = np.sqrt(min_dist_sq)
    
    return min(min_wall, min_pair_dist / 2.0)

def _fast_radius(centers):
    """Faster radius computation for tight loops, avoids some overhead."""
    # Wall
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    # Pairwise - use broadcasting but be careful with memory if N is large
    # N=50 is small, so full matrix is fine (50x50)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    min_pair_dist = np.sqrt(min_dist_sq)
    
    return min(min_wall, min_pair_dist / 2.0)

def _simulated_annealing(centers, start_time, deadline, target_score=0.135):
    """
    Perform Simulated Annealing to maximize the minimum radius.
    """
    current_centers = centers.copy()
    current_r = _compute_radius(current_centers)
    best_centers = current_centers.copy()
    best_r = current_r
    
    # Initial temperature
    T = 0.01
    cooling_rate = 0.9995
    min_T = 1e-6
    
    rng = np.random.RandomState(42)
    
    # We will perturb one or two spheres at a time
    while time.time() < deadline - 20.0: # Leave 20s for polish
        if T < min_T:
            break
            
        # Choose a sphere to move
        i = rng.randint(0, N)
        
        # Perturbation magnitude
        # Scale perturbation with temperature
        step_size = T * 0.5
        # Move in random direction
        delta = rng.normal(0, step_size, size=3)
        
        new_centers = current_centers.copy()
        new_centers[i] += delta
        
        # Keep inside cube (soft constraint in SA, hard in final)
        # Actually, let's allow going slightly outside to explore, 
        # but the radius calculation will penalize it heavily.
        # To keep it simple and valid, we can clip, but that creates a flat region.
        # Better: allow out of bounds, radius becomes negative or very small.
        # But the task requires valid output. 
        # Let's clip to [-0.5, 0.5] to maintain validity of intermediate states 
        # for the sake of the radius function not returning 0 for invalid positions.
        new_centers = np.clip(new_centers, -0.4999, 0.4999)
        
        new_r = _compute_radius(new_centers)
        
        # Acceptance criterion
        if new_r > current_r:
            current_centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = current_centers.copy()
        else:
            # Delta is negative
            delta_r = new_r - current_r
            # Probability of accepting worse solution
            # We want to maximize r, so we accept if exp(delta_r / T) > random
            # Note: delta_r is negative.
            # If T is large, we accept more.
            # Use a scaled delta to make the exponent reasonable
            # r is around 0.1, so delta_r is small.
            # Let's scale T to be comparable to delta_r.
            # If T ~ 0.01, delta_r ~ -0.001, exp(-0.1) ~ 0.9
            accept_prob = np.exp(delta_r / (T * 0.1)) 
            # T*0.1 makes T effectively smaller, so we cool faster in terms of acceptance
            # Let's just use T directly but scale delta_r
            # Standard SA: exp(-delta_E / T). Here E = -r. delta_E = -delta_r.
            # So exp(delta_r / T).
            # If delta_r is -0.001 and T is 0.01, exp(-0.1) = 0.90.
            # If T is 0.001, exp(-1) = 0.36.
            
            if rng.rand() < accept_prob:
                current_centers = new_centers
                current_r = new_r
        
        # Cool down
        T *= cooling_rate
        
        # Adaptive cooling: if stuck, cool faster
        # (Simple version: just linear cooling)
        
    return best_centers, best_r

def _coordinate_descent_polish(centers, start_time, deadline):
    """
    Refine the solution using coordinate descent.
    Move each sphere along each axis to maximize the minimum radius.
    """
    current_centers = centers.copy()
    current_r = _compute_radius(current_centers)
    
    # Pre-compute some data for speed?
    # For N=50, recomputing is fast enough.
    
    max_iters = 1000
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
            
        improved = False
        
        for i in range(N):
            if time.time() > deadline - 5.0:
                break
                
            for d in range(3):
                # Try moving sphere i in dimension d
                # We want to increase the minimum radius.
                # The radius is limited by:
                # 1. Wall distance in dim d: 0.5 - |c_i_d|
                # 2. Wall distances in other dims (unchanged)
                # 3. Pairwise distances involving i
                
                # Let's do a small search in the neighborhood of c_i_d
                # Or use binary search for the best position?
                # Simpler: try a few small steps.
                
                base_pos = current_centers[i, d]
                
                # Try moving left and right
                for direction in [-1, 1]:
                    # Try a small step
                    step = 0.001
                    new_pos = base_pos + direction * step
                    
                    # Check bounds
                    if new_pos < -0.5 or new_pos > 0.5:
                        continue
                    
                    new_centers = current_centers.copy()
                    new_centers[i, d] = new_pos
                    
                    new_r = _compute_radius(new_centers)
                    
                    if new_r > current_r + 1e-12:
                        current_centers[i, d] = new_pos
                        current_r = new_r
                        improved = True
                    else:
                        # Try smaller step or opposite? 
                        # If moving in this direction decreases r, stop for this direction.
                        # This is a greedy hill climb.
                        pass
        
        if not improved:
            # Try smaller steps?
            # For now, if no improvement, break.
            # Or do a random perturbation to escape local min?
            # Coordinate descent is prone to local minima.
            # Let's add a small random jitter if stuck.
            if it % 10 == 9:
                # Jitter
                current_centers += np.random.normal(0, 0.001, size=(N, 3))
                current_centers = np.clip(current_centers, -0.4999, 0.4999)
                current_r = _compute_radius(current_centers)
            else:
                break
                
    return current_centers, current_r

def _make_structured_initial(seed=42):
    """
    Create a good initial configuration.
    A 4x4x4 grid has 64 points. We need 50.
    We can take the first 50 points from a sorted grid, or randomize the selection.
    Better: Start with a 3x4x5 grid (60 points) or similar?
    50 is close to 5^3=125 or 4^3=64.
    Let's use a 4x4x4 grid, remove 14 points randomly, and jitter.
    """
    rng = np.random.RandomState(seed)
    
    # 4x4x4 grid
    m = 4
    # Spacing
    # To maximize radius, we want to spread them out.
    # Center of cube is 0.
    # Points at -0.375, -0.125, 0.125, 0.375?
    # Distance between neighbors = 0.25. Radius = 0.125.
    # Wall distance for outer points: 0.5 - 0.375 = 0.125.
    # So a perfect 4x4x4 grid has r=0.125.
    # We need 50 points, so we remove 14.
    # Removing points allows the remaining ones to move apart?
    # No, they are fixed in the grid initially.
    # But SA will move them.
    
    ticks = np.array([-0.375, -0.125, 0.125, 0.375])
    
    # Generate all grid points
    points = []
    for x in ticks:
        for y in ticks:
            for z in ticks:
                points.append([x, y, z])
    points = np.array(points)
    
    # Shuffle and pick 50
    idx = rng.permutation(len(points))[:N]
    centers = points[idx].copy()
    
    # Add small jitter to break symmetry
    centers += rng.uniform(-0.01, 0.01, size=centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 1. Generate initial configuration
    centers = _make_structured_initial(42)
    
    # 2. Simulated Annealing
    # Run SA for most of the time
    sa_deadline = deadline - 25.0
    centers, r_sa = _simulated_annealing(centers, start_time, sa_deadline)
    
    # 3. Coordinate Descent Polish
    # Use remaining time
    centers, r_cd = _coordinate_descent_polish(centers, start_time, deadline)
    
    # 4. Final validation and clipping
    centers = np.clip(centers, -0.499999, 0.499999)
    
    # Check validity
    final_r = _compute_radius(centers)
    
    # If something went wrong and final_r is 0, fallback to grid
    if final_r < 0.1:
        m = 4
        ticks = np.array([-0.375, -0.125, 0.125, 0.375])
        points = []
        for x in ticks:
            for y in ticks:
                for z in ticks:
                    points.append([x, y, z])
        points = np.array(points)
        return points[:N]
        
    return centers

# EVOLVE-BLOCK-END
