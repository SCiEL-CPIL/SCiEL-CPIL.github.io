"""
Strategy: Local Iterative Best-Pair Displacement (Discrete/Coordinate Ascent)
Instead of global gradient optimization (NLP/SA), we treat the problem as a
sequence of local repairs. We identify the pair of circles (or circle-wall)
with the smallest "slack" (distance - 2r or dist_to_wall - r). We then move
these entities apart by the minimum amount required to restore feasibility
for a target radius r_target. This is repeated until no violation exists or
time runs out. This avoids the need for global gradient information and
exploits the local nature of packing constraints.
"""

import numpy as np
import time
import random
from scipy.spatial import cKDTree

N = 200
HALF_SIDE = 0.5

def compute_radius(centers):
    """Compute the maximum valid radius for the given centers."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > HALF_SIDE + 1e-12):
        return 0.0
    
    # Distance to boundary
    min_dist_to_wall = np.min(HALF_SIDE - np.abs(centers), axis=None)
    if min_dist_to_wall <= 0:
        return 0.0
    
    # Min pairwise distance
    tree = cKDTree(centers)
    # Use a generous radius for query_pairs to ensure we catch the minimum
    pairs = tree.query_pairs(r=0.1) 
    if len(pairs) == 0:
        min_pair_dist = 1e9
    else:
        min_pair_dist = min(np.linalg.norm(centers[i] - centers[j]) for i, j in pairs)
    
    r = min(min_dist_to_wall, min_pair_dist / 2)
    return r

def _simple_grid():
    """Fallback simple grid."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]

def local_repair(centers, r_target, max_iter=5000, time_limit=10.0):
    """
    Attempt to make the configuration feasible for r_target by moving circles.
    Returns (success, new_centers).
    """
    centers = centers.copy()
    start_time = time.time()
    
    # We need to find the most violated constraint and fix it.
    # Violation types:
    # 1. Pair (i, j): dist < 2 * r_target. Move i and j apart along (c_i - c_j).
    # 2. Wall (i, k): dist_to_wall < r_target. Move i away from wall.
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # 1. Check pairwise constraints
        tree = cKDTree(centers)
        # Query pairs that are too close
        # We look for pairs with dist < 2 * r_target
        # cKDTree.query_pairs(r) returns pairs with distance < r
        # So we query with r = 2 * r_target
        pairs = list(tree.query_pairs(r=2 * r_target))
        
        if not pairs:
            # Check walls
            # dist_to_wall = HALF_SIDE - |x|
            # Violation if dist_to_wall < r_target
            # i.e., |x| > HALF_SIDE - r_target
            dists_to_wall = HALF_SIDE - np.abs(centers)
            min_wall_dist = np.min(dists_to_wall)
            
            if min_wall_dist >= r_target:
                # All constraints satisfied
                return True, centers
            
            # Find the circle with min wall distance
            flat_min_idx = np.argmin(dists_to_wall)
            i = flat_min_idx // 2
            k = flat_min_idx % 2
            
            # Move circle i away from the wall
            # If x_i > 0, move left (decrease x). If x_i < 0, move right (increase x).
            # If x_i == 0, move in the direction that increases dist? 
            # Actually, dist = HALF_SIDE - |x|. To increase dist, we need to decrease |x|.
            # So move towards 0.
            # Wait, the constraint is HALF_SIDE - |x| >= r.
            # If HALF_SIDE - |x| < r, then |x| > HALF_SIDE - r.
            # We need to reduce |x| to HALF_SIDE - r.
            
            current_coord = centers[i, k]
            target_abs = HALF_SIDE - r_target
            if abs(current_coord) > target_abs:
                # Move towards 0
                if current_coord > 0:
                    centers[i, k] = target_abs
                else:
                    centers[i, k] = -target_abs
            else:
                # Should not happen if we found a violation, but safety check
                pass
                
            continue

        # 2. If there are close pairs, pick the one with the smallest distance (most violated)
        # Calculate distances for these pairs
        min_dist = np.inf
        min_i, min_j = -1, -1
        for i, j in pairs:
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist:
                min_dist = d
                min_i, min_j = i, j
        
        # We need dist >= 2 * r_target
        # Currently dist = min_dist.
        # Deficit = 2 * r_target - min_dist
        # We move i and j apart by deficit/2 each.
        
        vec = centers[min_i] - centers[min_j]
        norm_vec = np.linalg.norm(vec)
        
        if norm_vec == 0:
            # Coincident points. Pick random direction.
            vec = np.array([1.0, 0.0])
            norm_vec = 1.0
        else:
            vec = vec / norm_vec
            
        required_dist = 2 * r_target
        current_dist = norm_vec
        
        if current_dist < required_dist:
            # Move apart
            shift = (required_dist - current_dist) / 2.0
            centers[min_i] += vec * shift
            centers[min_j] -= vec * shift
            
        # Ensure we don't push out of bounds immediately? 
        # The wall check in the next iteration will handle it.
        # However, pushing out of bounds might make it harder to recover.
        # A soft penalty or clipping might be better, but let's stick to hard repair.
        
    # After loop, check if feasible
    # Quick check
    tree = cKDTree(centers)
    pairs = list(tree.query_pairs(r=2 * r_target))
    if pairs:
        for i, j in pairs:
            if np.linalg.norm(centers[i] - centers[j]) < 2 * r_target - 1e-12:
                return False, centers
                
    dists_to_wall = HALF_SIDE - np.abs(centers)
    if np.min(dists_to_wall) < r_target - 1e-12:
        return False, centers
        
    return True, centers

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with a good grid
    centers = _simple_grid()
    current_r = compute_radius(centers)
    
    # We want to push r up.
    # Strategy: Binary search or linear scan for r_target.
    # Since we have a repair function, we can try to achieve higher r.
    
    # Let's estimate a reasonable upper bound.
    # Grid r is approx 1/(2*sqrt(N)) ~ 1/(2*14) ~ 0.035
    # Best known is ~0.0366.
    
    best_config = centers.copy()
    best_r = current_r
    
    # Try to increase r in small steps
    # Step size 0.0001 might be too slow if we do many.
    # Let's try a few target values.
    
    # Initialize targets around current best
    # If current_r is ~0.035, try 0.036, 0.0365, 0.0366
    
    targets = []
    # Generate a range of targets to try
    # We'll try to jump to a higher value and repair.
    # If repair fails, we might have moved things around, which could be useful.
    
    # Let's do a linear search from current_r + 0.001 up to 0.037
    # Step 0.0002
    
    r_start = current_r + 0.0005
    r_end = 0.0370
    step = 0.0002
    
    r_target = r_start
    
    while r_target <= r_end:
        if time.time() > deadline - 10.0:
            break
            
        # Try to repair current config for this r_target
        success, new_centers = local_repair(centers, r_target, max_iter=2000, time_limit=5.0)
        
        if success:
            # Check if it's actually valid (repair might have introduced subtle issues or numerical errors)
            actual_r = compute_radius(new_centers)
            if actual_r >= r_target - 1e-9:
                centers = new_centers
                if actual_r > best_r:
                    best_r = actual_r
                    best_config = new_centers.copy()
                # If successful, we can try a slightly higher r
                # But local_repair modifies centers in place (or returns copy).
                # We should continue from this new configuration.
            else:
                # Repair claimed success but score is lower. 
                # This can happen if repair moves points but doesn't fully satisfy constraints 
                # due to numerical issues or if the "success" check was too loose.
                # We should probably revert or try a smaller step.
                # For now, we accept the configuration if it's valid, even if r < target,
                # but we only update best if it's better.
                # Actually, if actual_r < r_target, the config is NOT feasible for r_target.
                # We should not use it for higher targets.
                # Let's revert to the last known good config if this one is worse?
                # No, the movement might have helped for a *different* r.
                # But for binary search logic, we need monotonicity.
                # If it fails, we should probably stop increasing r.
                # However, the movement might have improved the packing for a *lower* r?
                # Unlikely to improve max r if it failed to reach target.
                
                # Let's just break if we can't reach the target, as higher targets will likely fail too.
                # Unless the failure was due to local minima in the repair process.
                # Let's try a slightly lower target if we fail?
                # For simplicity, let's just stop increasing if we fail.
                # But we can try a few random restarts or different inits?
                # Let's just break.
                break
        else:
            # Repair failed. 
            # The configuration might still be valid for current_r, but not r_target.
            # We might have moved points in a way that is worse for current_r?
            # local_repair moves points to satisfy r_target. If it fails, the points are moved 
            # but constraints are still violated.
            # We should revert to the last valid configuration.
            # We don't have a "last valid" easily accessible if we modify `centers` in place.
            # Let's keep a backup.
            
            # Actually, local_repair returns new_centers. If success is False, 
            # we should NOT update `centers` to new_centers because it's invalid.
            # So `centers` remains the previous valid one.
            # We can try a different r_target or stop.
            # Let's stop increasing r.
            break
            
        # If successful, we can try to push a bit more
        # The loop increments r_target by step
        
    # After the main loop, do a final polish with a very tight target
    # Try to squeeze out the last bit of precision
    if time.time() < deadline - 5.0:
        # Try a few very small increments around best_r
        for _ in range(10):
            if time.time() > deadline - 2.0:
                break
            r_target = best_r + 0.00001
            success, new_centers = local_repair(best_config, r_target, max_iter=1000, time_limit=2.0)
            if success:
                actual_r = compute_radius(new_centers)
                if actual_r > best_r:
                    best_r = actual_r
                    best_config = new_centers
            else:
                break

    # Final validation
    final_r = compute_radius(best_config)
    if final_r > best_r:
        best_r = final_r
        
    # Ensure validity
    if not np.all(np.isfinite(best_config)) or np.any(np.abs(best_config) > HALF_SIDE + 1e-12):
        # Fallback to grid if something went wrong
        return _simple_grid()
        
    return best_config

# EVOLVE-BLOCK-END
