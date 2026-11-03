"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _compute_radius(centres):
    """Compute the maximum common radius for given centres in [-0.5, 0.5]^3."""
    # Distance to nearest face
    face_dist = 0.5 - np.max(np.abs(centres), axis=1)
    min_face = np.min(face_dist)
    
    # Minimum pairwise distance
    n = len(centres)
    if n < 2:
        return min_face
    
    # Compute all pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_face, min_pair / 2.0)

def _objective_de(x, target_r):
    """
    Objective for Differential Evolution.
    We want to maximize the minimum radius.
    DE minimizes. So we minimize the 'violation' of the constraints relative to target_r.
    If all spheres fit with radius > target_r, return 0 (or small value).
    Otherwise, return the sum of violations.
    
    Violation for pair (i,j): max(0, target_r - d_ij/2)
    Violation for wall (i): max(0, target_r - (0.5 - max(|c_i|)))
    
    To help DE find feasible regions, we use a large penalty for violations.
    """
    centres = x.reshape(-1, 3)
    n = len(centres)
    
    # Wall violations
    wall_dists = 0.5 - np.max(np.abs(centres), axis=1)
    wall_violations = np.maximum(0, target_r - wall_dists)
    
    # Pair violations
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists_half = dists / 2.0
    pair_violations = np.maximum(0, target_r - pair_dists_half)
    
    # Sum of violations
    total_violation = np.sum(wall_violations) + 0.5 * np.sum(pair_violations) # 0.5 because pairs counted twice
    
    return total_violation

def _refine_local(centres, max_iter=2000, tol=1e-10):
    """
    Local refinement using a direct search / nudge strategy.
    This is more robust than gradient descent for max-min problems near the optimum.
    We iteratively identify the most violated constraint and nudge the spheres to fix it.
    """
    centres = centres.copy()
    n = len(centres)
    r = _compute_radius(centres)
    
    for _ in range(max_iter):
        # Check time? No, this is fast. But let's add a break condition.
        current_r = _compute_radius(centres)
        if current_r < r:
            # Revert if we made it worse? Or just keep trying?
            # Usually nudge methods are monotonic if small enough.
            # Let's try to improve current_r.
            pass
        
        # Find the minimum constraint
        # 1. Walls
        wall_dists = 0.5 - np.max(np.abs(centres), axis=1)
        min_wall_idx = np.argmin(wall_dists)
        min_wall_val = wall_dists[min_wall_idx]
        
        # 2. Pairs
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        min_pair_val = dists[min_pair_idx] / 2.0
        
        # Determine which is the bottleneck
        if min_wall_val < min_pair_val:
            # Wall is bottleneck
            i = min_wall_idx
            # Determine which coordinate is closest to wall
            c = centres[i]
            abs_c = np.abs(c)
            axis = np.argmax(abs_c)
            
            # Nudge away from wall
            # Target: move to center slightly? No, just push in.
            # Step size proportional to violation?
            # Let's push towards the center of the cube for that axis.
            step = 0.001 # Small nudge
            # If c is positive, decrease it. If negative, increase it.
            centres[i, axis] -= np.sign(c[axis]) * step
            
            # Clip to valid range
            centres[i, axis] = np.clip(centres[i, axis], -0.5, 0.5)
            
        else:
            # Pair is bottleneck
            i, j = min_pair_idx
            if i == j: continue
            
            # Nudge i and j apart
            # Vector from j to i
            v = centres[i] - centres[j]
            dist = np.linalg.norm(v)
            if dist < 1e-10:
                # Coincident, random nudge
                centres[i] += np.random.rand(3) * 0.001
                centres[j] -= np.random.rand(3) * 0.001
            else:
                u = v / dist
                step = 0.001
                centres[i] += u * step
                centres[j] -= u * step
                
                # Check wall violations introduced by nudge
                # If nudge pushes i out of bounds, reduce step or reflect?
                # Simple clip
                centres[i] = np.clip(centres[i], -0.5, 0.5)
                centres[j] = np.clip(centres[j], -0.5, 0.5)
        
        new_r = _compute_radius(centres)
        if new_r > r:
            r = new_r
        else:
            # If we didn't improve, maybe step was too big or wrong direction?
            # In nudge methods, if it doesn't improve, we might be at a local max or stuck.
            # To escape, we can add a small random perturbation to the stuck spheres.
            if np.random.rand() < 0.1:
                # Perturb the bottleneck spheres
                if min_wall_val < min_pair_val:
                    i = min_wall_idx
                    centres[i] += np.random.randn(3) * 0.0005
                    centres[i] = np.clip(centres[i], -0.5, 0.5)
                else:
                    i, j = min_pair_idx
                    centres[i] += np.random.randn(3) * 0.0005
                    centres[j] += np.random.randn(3) * 0.0005
                    centres[i] = np.clip(centres[i], -0.5, 0.5)
                    centres[j] = np.clip(centres[j], -0.5, 0.5)
                    
    return centres

def construct():
    """Construct 50 sphere centres in [-0.5, 0.5]^3 to maximize minimum radius."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    np.random.seed(42)
    
    # Initial guess: Grid
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    np.random.shuffle(grid)
    initial_centres = grid[:N].copy()
    
    current_best_r = _compute_radius(initial_centres)
    best_centres = initial_centres.copy()
    
    # Strategy: Iterative DE with increasing target radius
    # Start with a low target, find a feasible config, raise target, repeat.
    target_r = current_best_r * 0.95
    
    # DE parameters
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    while time.time() < deadline - 10.0:
        # Run DE to find a configuration that satisfies target_r
        # maxiter and popsize are tuned for speed
        # tol=0.01 is loose for DE, we refine later
        
        try:
            res = differential_evolution(
                _objective_de,
                bounds=bounds,
                args=(target_r,),
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-5,
                init='latinhypercube',
                polish=False, # We do our own polish
                update_interval=10
            )
        except Exception as e:
            break
            
        de_centres = res.x.reshape(-1, 3)
        
        # Check if DE found a feasible solution for target_r
        # If res.fun is close to 0, it means violations are small
        # Let's compute the actual radius
        de_r = _compute_radius(de_centres)
        
        if de_r > current_best_r:
            current_best_r = de_r
            best_centres = de_centres.copy()
            
            # If we successfully met or exceeded the target, increase the target
            # to push for better radius
            # Increase by a small factor
            target_r = current_best_r * 1.01
            
        else:
            # DE didn't improve, maybe target was too high or DE stuck.
            # If target_r is significantly higher than best_r, DE struggled.
            # If target_r is close to best_r, we might be at a local optimum.
            
            # Try to refine the current best locally
            if time.time() < deadline - 5.0:
                refined = _refine_local(best_centres, max_iter=500)
                ref_r = _compute_radius(refined)
                
                if ref_r > current_best_r:
                    current_best_r = ref_r
                    best_centres = refined.copy()
                    target_r = current_best_r * 1.005
                else:
                    # Stagnant. Add noise to escape local minimum and try again?
                    # Or break if we've been stagnant for a while.
                    # For simplicity, just break and return best.
                    # Or try one more DE with higher popsize?
                    # Let's just break to ensure we return a valid result.
                    break
                    
    # Final polish on the very best
    if time.time() < deadline - 2.0:
        final_refined = _refine_local(best_centres, max_iter=2000)
        final_r = _compute_radius(final_refined)
        if final_r > current_best_r:
            best_centres = final_refined
            current_best_r = final_r
            
    return best_centres
# EVOLVE-BLOCK-END
