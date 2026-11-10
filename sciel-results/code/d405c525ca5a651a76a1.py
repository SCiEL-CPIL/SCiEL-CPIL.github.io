"""
Strategy: Force-Directed Relaxation (Molecular Dynamics)
Instead of stochastic global search or greedy coordinate ascent, we model the spheres
as particles with repulsive forces. We minimize the potential energy defined by
overlaps with other spheres and walls. This allows simultaneous adjustment of all
spheres, effectively "pushing" them apart to maximize the minimum distance.
"""

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _radius_score(centres):
    """Compute the packing radius for a given set of centres."""
    n = centres.shape[0]
    if n != N:
        return -1.0
    if not np.all(np.isfinite(centres)):
        return -1.0
    # Check bounds: all coordinates must be in [-0.5, 0.5]
    if np.any(centres < -0.5 - 1e-10) or np.any(centres > 0.5 + 1e-10):
        return -1.0
    # Clip to valid range for distance calculation (in case of tiny violations)
    c = np.clip(centres, -0.5, 0.5)
    # Wall distance
    wall_dists = np.min(0.5 - np.abs(c), axis=1)
    min_wall = np.min(wall_dists)
    if min_wall < -1e-10:
        return -1.0
    # Pairwise distances
    # Use cdist for efficiency if available, otherwise manual loop
    try:
        from scipy.spatial.distance import cdist
        dists = cdist(c, c)
    except ImportError:
        # Fallback for environments without scipy (though task says it's available)
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dists[i, j] = np.linalg.norm(c[i] - c[j])
    
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    r = min(min_wall, min_pair)
    return r

def _force_directed_relaxation(centres, max_iters=2000, dt_initial=0.01, dt_min=1e-6, 
                                temperature=1.0, cooling=0.999):
    """
    Simulates a system of repulsive spheres.
    Force = Gradient of potential energy.
    Potential is high when spheres overlap or are close to walls.
    We use a soft repulsion model: F ~ 1/d^2 for overlaps, and similar for walls.
    """
    c = centres.copy()
    dt = dt_initial
    current_temp = temperature
    
    # Precompute indices for neighbor checking if N is small enough, 
    # but for N=100, full matrix is 100x100 which is trivial.
    
    for it in range(max_iters):
        # Check time budget
        if it % 100 == 0:
            pass # Placeholder for time check if needed, but loop is fast
        
        # 1. Calculate Forces
        forces = np.zeros_like(c)
        
        # Pairwise Repulsion
        # d_ij = ||c_i - c_j||
        # We want to push them apart if d < 2r_target. 
        # Since we don't know r_target, we maximize min distance.
        # A common trick: Force is proportional to (1 - d/d_ref)^k or similar.
        # Let's use a potential V = 1/d^2 for d < d_max, 0 otherwise.
        # Actually, simpler: Just push apart based on current min distance?
        # No, that's greedy. 
        # Better: Use a "soft" collision detection. 
        # Let's define a target radius r_current estimated from current min dist.
        # But that changes dynamics.
        
        # Alternative: Fixed repulsion core.
        # If we assume a radius r, force is strong if d < 2r.
        # We can iteratively estimate r.
        
        # Let's use a standard MD approach with a fixed "interaction radius" 
        # that we tune, or just use the current minimum distance as a proxy.
        
        # More robust: 
        # Calculate all pairwise distances.
        # For each pair i,j:
        #   if dist < threshold: apply force
        # Threshold should be larger than current min dist to allow spreading.
        # Let's set threshold to be roughly the current best radius * 2 * 1.1
        # But we don't have "best radius" easily accessible without full calc.
        
        # Let's compute min pair dist and min wall dist first.
        d_min_pair = 1e9
        d_min_wall = 1e9
        
        # Efficient distance calculation
        # Using scipy if available
        try:
            from scipy.spatial.distance import cdist
            dist_mat = cdist(c, c)
            np.fill_diagonal(dist_mat, 1e9)
            d_min_pair = np.min(dist_mat)
        except ImportError:
            dist_mat = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    dist_mat[i, j] = np.linalg.norm(c[i] - c[j])
            np.fill_diagonal(dist_mat, 1e9)
            d_min_pair = np.min(dist_mat)
            
        # Wall distances
        wall_dists = np.min(0.5 - np.abs(c), axis=1)
        d_min_wall = np.min(wall_dists)
        
        current_r = min(d_min_pair / 2.0, d_min_wall)
        
        if current_r < 1e-8:
            # If completely overlapping or on wall, use a small default
            r_ref = 0.1
        else:
            r_ref = current_r * 1.05 # Slightly larger to encourage spreading
        
        # Interaction range
        r_cut = r_ref * 2.2 # Only interact if closer than this
        
        # Apply Pairwise Forces
        # F_ij = k * (1 - d/r_cut)^2 * (c_i - c_j)/d  for d < r_cut
        k = 1.0
        
        for i in range(N):
            for j in range(i + 1, N):
                d = dist_mat[i, j]
                if d < r_cut and d > 1e-6:
                    # Force magnitude
                    f_mag = k * (1.0 - d / r_cut) ** 2
                    # Direction
                    dir_vec = (c[i] - c[j]) / d
                    forces[i] += f_mag * dir_vec
                    forces[j] -= f_mag * dir_vec
                elif d <= 1e-6:
                    # Random kick if exactly on top
                    forces[i] += np.random.normal(0, 0.01, 3)
                    forces[j] -= np.random.normal(0, 0.01, 3)
                    
        # Apply Wall Forces
        # Push away from walls if close
        wall_cut = r_ref * 1.5
        for i in range(N):
            for axis in range(3):
                pos = c[i, axis]
                dist_to_neg_wall = pos - (-0.5) # pos + 0.5
                dist_to_pos_wall = 0.5 - pos
                
                # Force from negative wall (pushes positive)
                if dist_to_neg_wall < wall_cut:
                    f = k * (1.0 - dist_to_neg_wall / wall_cut) ** 2
                    forces[i, axis] += f
                # Force from positive wall (pushes negative)
                if dist_to_pos_wall < wall_cut:
                    f = k * (1.0 - dist_to_pos_wall / wall_cut) ** 2
                    forces[i, axis] -= f

        # 2. Update Positions
        # Velocity Verlet or simple Euler? 
        # Simple Euler with damping is often more stable for packing.
        # c_new = c + dt * forces
        # Add some noise to escape local minima? 
        # "Temperature" can be added as random force.
        noise = np.random.normal(0, current_temp * 0.01, size=(N, 3))
        
        c += dt * (forces + noise)
        
        # 3. Enforce Hard Bounds (Clipping)
        # If a sphere goes outside, clip it back. This acts as an inelastic collision with the wall.
        c = np.clip(c, -0.5, 0.5)
        
        # 4. Cool down
        dt *= cooling
        current_temp *= cooling
        
        if dt < dt_min:
            break
            
    return c

def _local_refine_gradient_free(centres, max_iters=100):
    """
    A very light local polish to fix any tiny numerical issues or fine-tune.
    """
    c = centres.copy()
    for _ in range(max_iters):
        r_old = _radius_score(c)
        if r_old < 0: break
        improved = False
        # Try moving the sphere with the smallest margin
        # Calculate margins
        try:
            from scipy.spatial.distance import cdist
            dists = cdist(c, c)
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists) / 2.0
            wall_dists = np.min(0.5 - np.abs(c), axis=1)
            min_wall = np.min(wall_dists)
        except ImportError:
            min_pair = 0
            min_wall = 0
            
        r = min(min_pair, min_wall)
        
        # Identify critical spheres
        # This is complex, so just do random small steps
        for _ in range(10):
            i = np.random.randint(N)
            k = np.random.randint(3)
            step = 0.001
            for sign in [-1, 1]:
                c2 = c.copy()
                c2[i, k] += sign * step
                if np.any(c2 < -0.5) or np.any(c2 > 0.5): continue
                r2 = _radius_score(c2)
                if r2 > r + 1e-10:
                    c = c2
                    improved = True
                    break
            if improved: break
        if not improved: break
    return c

def construct():
    start_time = time.time()
    deadline = start_time + 140 # 10s safety margin
    
    np.random.seed(42)
    
    best_centres = None
    best_r = -1.0
    
    # Strategy 1: Grid Initialization
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:N]
    
    # Run Force Directed Relaxation on Grid
    c_relaxed = _force_directed_relaxation(grid, max_iters=1000, dt_initial=0.02)
    r_relaxed = _radius_score(c_relaxed)
    
    if r_relaxed > best_r:
        best_r = r_relaxed
        best_centres = c_relaxed.copy()
        
    # Local polish
    c_polished = _local_refine_gradient_free(best_centres, max_iters=50)
    r_polished = _radius_score(c_polished)
    if r_polished > best_r:
        best_r = r_polished
        best_centres = c_polished.copy()

    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_centres

    # Strategy 2: Multiple Random Restarts with Force Directed Relaxation
    # The force directed method is deterministic given the start. 
    # Different starts lead to different local minima.
    
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() - start_time > deadline - 5:
            break
            
        # Random start, slightly perturbed from grid or purely random
        # Purely random is often too chaotic for MD to settle well without long runs.
        # Perturbed grid is better.
        c_start = grid + np.random.normal(0, 0.05, size=(N, 3))
        c_start = np.clip(c_start, -0.5, 0.5)
        
        # Run MD
        # Use shorter run for restarts to keep within time budget
        c_relaxed = _force_directed_relaxation(c_start, max_iters=500, dt_initial=0.01)
        r_relaxed = _radius_score(c_relaxed)
        
        if r_relaxed > best_r:
            # Polish
            c_polished = _local_refine_gradient_free(c_relaxed, max_iters=30)
            r_polished = _radius_score(c_polished)
            if r_polished > best_r:
                best_r = r_polished
                best_centres = c_polished.copy()

    # Final Polish if time permits
    if time.time() - start_time < deadline - 2 and best_centres is not None:
        c_final = _local_refine_gradient_free(best_centres, max_iters=100)
        r_final = _radius_score(c_final)
        if r_final > best_r:
            best_r = r_final
            best_centres = c_final.copy()

    return best_centres
# EVOLVE-BLOCK-END
