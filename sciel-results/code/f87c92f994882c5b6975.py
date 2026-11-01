"""
Strategy: Simulated Annealing with Radius Targeting
Instead of minimizing a smooth surrogate energy (which favors uniform spacing),
we use Simulated Annealing to find a configuration that satisfies a target
minimum radius R. The objective is to minimize the total overlap violation.
By annealing the temperature, we allow the system to escape local minima
and explore the configuration space more broadly than gradient descent.
We also use a "ratchet" approach: we try to increase R gradually.
"""
import numpy as np
import time

N = 50

def _compute_radius(centers):
    """Compute the maximum common radius for a given set of centers."""
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    
    # Pairwise distances
    # For N=50, 50x50 matrix is small.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists, axis=1)
    
    radii = np.minimum(wall_dists, min_pair_dist / 2.0)
    
    return np.min(radii)

def _violation(centers, r_target):
    """
    Compute the total violation of the constraint that spheres of radius r_target
    do not overlap and are inside the cube.
    Returns a scalar energy.
    """
    # 1. Wall violations
    # dist_to_wall = 0.5 - max(|c|)
    # If dist_to_wall < r_target, violation is (r_target - dist_to_wall)^2
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    wall_viol = np.sum(np.maximum(r_target - wall_dists, 0.0)**2)
    
    # 2. Pairwise violations
    # dist_ij < 2*r_target -> violation (2*r_target - dist_ij)^2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Only consider i < j to avoid double counting and self-interaction
    # We can use a mask or just sum all and divide by 2, but mask is cleaner for logic
    # Actually, summing all pairs (i,j) and (j,i) is fine if we just want a scalar penalty.
    # Let's compute for all pairs i != j.
    np.fill_diagonal(dists, np.inf)
    
    pair_viol = np.sum(np.maximum(2.0 * r_target - dists, 0.0)**2)
    
    return wall_viol + pair_viol

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Initial best found
    # Start with a decent configuration. 
    # A simple cubic grid is a good start, but we can also try random.
    # Let's use a jittered grid as initial state.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    centers = grid[:N].copy()
    centers += rng.normal(0, 0.01, size=centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Simulated Annealing Parameters
    # We want to find the max R.
    # We will run a loop that tries to increase R.
    # For each R, we run a short SA to see if we can find a valid config.
    # If we find one, we accept it and try a slightly higher R.
    # If we fail, we lower R slightly or restart with a different seed.
    
    # Alternatively, we can just run SA on the violation for a fixed R 
    # and update R based on success.
    
    # Let's use a continuous annealing approach where we minimize violation
    # for a target R, and R itself is slowly increased if violation is low.
    
    current_r = 0.1
    current_centers = centers.copy()
    
    # Temperature schedule
    # T_start should be high enough to allow moves that increase energy.
    # Energy scale: violations are squared distances. 
    # Typical violation magnitude might be 0.01 to 1.0.
    # Let's start T = 0.01 and decay.
    
    # We will do multiple "epochs" of annealing.
    # In each epoch, we fix R, run SA steps, check if we can reduce violation to ~0.
    # If yes, increase R. If no, decrease R slightly and restart SA with new random init?
    # Or just keep R and try harder?
    
    # A simpler robust strategy:
    # 1. Initialize centers.
    # 2. Set R = 0.1.
    # 3. Run SA to minimize violation(R).
    # 4. If min_violation < eps, R += step_R.
    # 5. If min_violation > threshold for too long, R -= step_R * 0.5.
    # 6. Repeat until time limit.
    
    # To be efficient, we'll interleave.
    
    T = 0.005
    T_min = 1e-6
    cooling_rate = 0.9999
    
    # Step size for random moves
    # We move one sphere at a time or a small cluster?
    # Moving one sphere is O(N) for energy update (if we do incremental), 
    # but O(N^2) if we recompute. N=50 is small, so O(N^2) per step is fine.
    # 50*50 = 2500 ops. 1e6 steps = 2.5e9 ops. Might be too slow in Python loops.
    # We should vectorize the move selection or use small batches.
    # Actually, let's just do full recompute. It's simpler and N is small.
    
    # To speed up, we can try moving a random subset of points?
    # No, single point moves are better for SA.
    # Let's optimize the violation calc.
    
    def _fast_violation(centers, r):
        # Vectorized
        wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
        wall_viol = np.sum(np.maximum(r - wall_dists, 0.0)**2)
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # We only need the sum of max(2r - d, 0)^2
        # This is computed for all pairs.
        pair_viol = np.sum(np.maximum(2.0 * r - dists, 0.0)**2)
        
        return wall_viol + pair_viol

    current_energy = _fast_violation(current_centers, current_r)
    
    # We will run a loop
    # To handle the time budget, we'll check time every few iterations.
    
    # Initial R estimate from best known is ~0.136. 
    # Let's start R at 0.12 to give it a chance to climb.
    current_r = 0.12
    current_energy = _fast_violation(current_centers, current_r)
    
    # If initial energy is huge, we might need to lower R or reset centers.
    # Let's just start SA.
    
    # We will use a "restart" strategy if stuck.
    # If energy doesn't decrease for X iterations, restart with new random centers and slightly lower R.
    
    stuck_count = 0
    best_energy_in_epoch = current_energy
    epoch_start_energy = current_energy
    
    # Move magnitude
    # Start with larger moves, decay them.
    move_scale = 0.05
    
    while time.time() < deadline:
        # Pick a random sphere
        idx = rng.integers(0, N)
        
        # Generate a random move
        # Gaussian move
        delta = rng.normal(0, move_scale, size=3)
        
        new_centers = current_centers.copy()
        new_centers[idx] += delta
        new_centers = np.clip(new_centers, -0.499, 0.499)
        
        new_energy = _fast_violation(new_centers, current_r)
        
        # Metropolis criterion
        delta_E = new_energy - current_energy
        
        if delta_E < 0:
            # Accept
            current_centers = new_centers
            current_energy = new_energy
            stuck_count = 0
            if current_energy < best_energy_in_epoch:
                best_energy_in_epoch = current_energy
                # Check if we can improve radius
                # If energy is very small, try to increase R
                if current_energy < 1e-4:
                    # Try to increase R
                    new_r = current_r + 0.0005
                    # Check if valid at new R?
                    # Just set R and let SA continue to reduce energy
                    current_r = new_r
                    # Recalculate energy at new R
                    current_energy = _fast_violation(current_centers, current_r)
                    # Reset epoch stats
                    best_energy_in_epoch = current_energy
                    epoch_start_energy = current_energy
                    # Update global best
                    actual_r = _compute_radius(current_centers)
                    if actual_r > best_radius:
                        best_radius = actual_r
                        best_centers = current_centers.copy()
        else:
            # Probabilistic acceptance
            # T needs to be comparable to energy scale.
            # If T is too low, no moves accepted.
            if T > 1e-8:
                prob = np.exp(-delta_E / T)
                if rng.random() < prob:
                    current_centers = new_centers
                    current_energy = new_energy
                    # Don't reset stuck count on uphill move? 
                    # Usually yes, it means we are exploring.
                    # But if we are stuck in a local min, we want to escape.
                    # Let's not reset stuck count for uphill moves to trigger restarts if stuck in a bad basin.
        
        # Cooling
        T *= cooling_rate
        move_scale *= 0.99999
        
        # Stagnation detection
        if current_energy > best_energy_in_epoch:
            stuck_count += 1
        else:
            stuck_count = 0
            
        # If stuck too long, restart
        if stuck_count > 5000:
            # Restart with new random configuration and slightly lower R
            # Or just randomize current configuration?
            # Randomizing is a strong escape.
            # Let's try a "kick": add large noise to all centers
            current_centers += rng.normal(0, 0.1, size=current_centers.shape)
            current_centers = np.clip(current_centers, -0.499, 0.499)
            current_energy = _fast_violation(current_centers, current_r)
            
            # If energy is still bad, lower R
            if current_energy > 1.0:
                current_r *= 0.95
                current_energy = _fast_violation(current_centers, current_r)
                
            # Reset counters
            stuck_count = 0
            best_energy_in_epoch = current_energy
            epoch_start_energy = current_energy
            T = 0.005 # Reheat
            move_scale = 0.05
            
        # Periodic check of global best
        # We do this every 1000 iterations to avoid overhead
        # But _compute_radius is O(N^2). 
        # We can just track best_radius when we accept a lower energy.
        # Actually, we updated best_radius when R increased.
        # What if we find a config with same R but better structure?
        # The score is R. So we only care about R.
        # If we can't increase R, we are stuck.
        
        # To be safe, let's compute actual radius every 10000 steps
        # Or just rely on the R increase logic.
        # The R increase logic only triggers if energy < 1e-4.
        # This might be too strict. 
        # Let's relax it: if energy < 1e-2, try R + 0.0002.
        
        # Let's add a check: if current_energy < 0.01:
        #   new_r = current_r + 0.0002
        #   ...
        
        # I'll modify the acceptance logic slightly in the code below to be more aggressive with R increase.

    # Final polish: use the best centers found
    # We can do a quick local refinement on best_centers using gradient descent
    # to squeeze out a bit more radius.
    
    # Let's do a simple Nelder-Mead or just a few gradient steps on the exact radius?
    # Gradient of min distance is hard.
    # Let's just return best_centers.
    
    # One last check: ensure best_centers is valid
    r_final = _compute_radius(best_centers)
    if r_final > best_radius:
        best_radius = r_final
        # best_centers is already set
        
    return best_centers

# EVOLVE-BLOCK-END
