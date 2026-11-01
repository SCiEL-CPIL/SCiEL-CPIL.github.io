"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.spatial import cKDTree

N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.spatial import cKDTree

N_CIRCLES = 200
HALF_SIDE = 0.5
KNOWN_BEST_R = 0.036612798904


def compute_radius(centers):
    """Compute the maximum valid radius for the given centers."""
    if len(centers) != N_CIRCLES:
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
    # Use a generous radius to ensure we catch all close pairs
    # In a valid packing, min dist is 2r. We check up to 2*(r+epsilon)
    # But we don't know r yet. A safe upper bound for search is ~0.05
    # Actually, to be strictly correct for the score, we just need the min distance.
    # query_pairs with r=0.1 is safe for N=200 in unit square (max dist ~1.4)
    # It's faster than full pairwise for dense clusters.
    pairs = tree.query_pairs(r=0.1) 
    if len(pairs) == 0:
        min_pair_dist = 1e9
    else:
        min_pair_dist = min(np.linalg.norm(centers[i] - centers[j]) for i, j in pairs)
    
    r = min(min_dist_to_wall, min_pair_dist / 2)
    return r


def score_config(centers):
    """Return the radius if valid, else 0."""
    if centers.shape != (N_CIRCLES, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > HALF_SIDE + 1e-10):
        return 0.0
    
    # Check for coincident centers
    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=1e-9)
    if len(pairs) > 0:
        return 0.0
    
    return compute_radius(centers)


def _simple_grid():
    """Fallback simple grid."""
    m = int(np.ceil(np.sqrt(N_CIRCLES)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N_CIRCLES]


def random_grid_initialization():
    """Generate a random grid-like initialization."""
    best_config = None
    best_r = 0.0
    
    for m in range(14, 16):
        for n in range(14, 16):
            if m * n < N_CIRCLES:
                continue
            
            x_ticks = (np.arange(m) + 0.5) / m - 0.5
            y_ticks = (np.arange(n) + 0.5) / n - 0.5
            grid = np.array([(x, y) for y in y_ticks for x in x_ticks])
            
            idx = np.random.permutation(len(grid))
            centers = grid[idx[:N_CIRCLES]]
            
            r = score_config(centers)
            if r > best_r:
                best_r = r
                best_config = centers.copy()
    
    return best_config if best_config is not None else _simple_grid()


def relaxation_loop(centers, max_iters=2000, target_r=0.0365):
    """
    Physics-based relaxation:
    1. Identify violations (overlaps > target_r, boundary violations).
    2. Apply repulsive forces to resolve them.
    3. Iterate until stable or max_iters.
    """
    current = centers.copy()
    current_r = compute_radius(current)
    
    # We want to maximize r. 
    # Strategy: Try to maintain a 'virtual' radius that is slightly larger than current_r 
    # to push them apart, or just resolve all violations for a fixed r.
    # A common effective method: 
    # - Pick a target radius R_target.
    # - If any pair is closer than 2*R_target, push them apart.
    # - If any point is closer than R_target to wall, push it in.
    # - If no violations, R_target can be increased.
    
    # However, simply increasing R_target might get stuck.
    # Alternative: Just relax until no violations for current best r, then nudge.
    
    # Let's use a "soft" repulsion that scales with overlap.
    
    for i_iter in range(max_iters):
        # 1. Check boundary violations
        # Points outside [-0.5, 0.5]
        violations = np.any(np.abs(current) > HALF_SIDE, axis=1)
        if np.any(violations):
            # Push inward
            # Simple clamp is robust, but we can add a small inward push for those on boundary
            # Actually, just clamping is the fastest way to fix hard constraints.
            current = np.clip(current, -HALF_SIDE, HALF_SIDE)
            # If we just clamped, we might have created overlaps. 
            # We will check overlaps next.
        
        # 2. Check pairwise overlaps
        # We want to ensure distance >= 2 * r_target
        # Let's try to maintain a radius slightly above the known best to force improvement?
        # No, let's just resolve the *current* minimum distance violations relative to a target.
        
        # A robust approach:
        # Let r_current = compute_radius(current)
        # If we want to improve, we need to increase the minimum distance.
        # We can simulate a pressure.
        
        # Find all pairs with distance < 2 * (current_r + epsilon)
        # This is tricky because current_r changes.
        
        # Let's use a fixed "desired" separation that we slowly increase.
        # Or, simpler: 
        # Identify the pair with minimum distance d_min.
        # If d_min/2 < target_r, push them apart.
        # Also push boundary violators.
        
        # Efficient check:
        tree = cKDTree(current)
        # Search for pairs closer than 2 * (current_r + 0.001)
        # If none, we are stable at current_r.
        # If some, push them.
        
        # To speed up, we can just check pairs within a small radius.
        # Let's use a dynamic target.
        # If we just want to improve the packing, we can try to "inflate" the circles.
        
        # Method: 
        # 1. Compute current min dist / 2 = r_curr
        # 2. Set target_r = r_curr + 0.0001 (small increment)
        # 3. Find pairs with dist < 2 * target_r
        # 4. Push them apart by (2*target_r - dist) / 2
        # 5. Repeat.
        
        r_curr = compute_radius(current)
        
        # If we are already very close to known best, increment smaller
        if r_curr > 0.0366:
            increment = 0.00001
        else:
            increment = 0.0005
            
        target_r = r_curr + increment
        
        # Find violations
        # Pairs with dist < 2 * target_r
        pairs = tree.query_pairs(r=2 * target_r)
        
        moved = False
        if len(pairs) > 0:
            for i, j in pairs:
                diff = current[i] - current[j]
                dist = np.linalg.norm(diff)
                if dist < 2 * target_r:
                    if dist < 1e-10:
                        # Random nudge
                        current[i] += np.random.uniform(-0.001, 0.001, 2)
                        current[j] -= np.random.uniform(-0.001, 0.001, 2)
                    else:
                        diff_norm = diff / dist
                        # Push apart
                        # We want to increase distance to 2*target_r
                        # So move each by (2*target_r - dist)/2
                        push = (2 * target_r - dist) / 2
                        # Damping factor to prevent oscillation
                        push *= 0.5 
                        current[i] += push * diff_norm
                        current[j] -= push * diff_norm
                    moved = True
        
        # Boundary check and push
        # If any point is too close to boundary (dist < target_r), push it in
        # dist_to_wall = HALF_SIDE - abs(coord)
        # If dist_to_wall < target_r, push towards center
        
        # Vectorized boundary check
        # For x: if x > HALF_SIDE - target_r, push left. if x < -(HALF_SIDE - target_r), push right.
        # Same for y.
        
        # This is a bit complex to vectorize perfectly with the pair pushes, 
        # but we can do a simple clamp + push for boundary.
        
        # Check x
        x_viol_high = current[:, 0] > HALF_SIDE - target_r
        x_viol_low = current[:, 0] < -(HALF_SIDE - target_r)
        y_viol_high = current[:, 1] > HALF_SIDE - target_r
        y_viol_low = current[:, 1] < -(HALF_SIDE - target_r)
        
        if np.any(x_viol_high):
            # Push left
            # Amount: (current - (HALF_SIDE - target_r))
            # We want to move it to HALF_SIDE - target_r
            # But this might cause overlaps. 
            # Let's just clamp to HALF_SIDE - target_r for now, 
            # and let the pair relaxation handle the rest.
            # Actually, hard clamping to boundary is often better for stability.
            # Let's clamp to the valid region [-0.5, 0.5].
            # The "target_r" logic for boundary is:
            # If we want radius r, the center must be in [-0.5+r, 0.5-r].
            # So we should clamp to that.
            
            # Let's refine: 
            # The valid region for centers given radius r is [-0.5+r, 0.5-r].
            # If a center is outside this, it's a violation.
            
            # Let's re-evaluate the boundary constraint.
            # If we clamp to [-0.5, 0.5], the radius is limited by the distance to the edge.
            # The relaxation loop tries to increase r.
            # So, effectively, we should push points that are "too close" to the edge.
            
            # Let's just clamp to [-0.5, 0.5] first to ensure validity.
            current = np.clip(current, -HALF_SIDE, HALF_SIDE)
            
            # Then, if we want to increase r, we need to move points away from edges.
            # But if they are at the edge, they limit r.
            # So, if a point is at x > 0.5 - target_r, it limits r to x - 0.5? No, 0.5 - x.
            # So we should push it towards 0.
            
            # Let's apply a soft push for boundary violations relative to target_r.
            # If x > 0.5 - target_r:
            #   push = (x - (0.5 - target_r)) * 0.5
            #   x -= push
            # If x < -0.5 + target_r:
            #   push = (x - (-0.5 + target_r)) * 0.5
            #   x += push
            
            mask_x_high = current[:, 0] > HALF_SIDE - target_r
            if np.any(mask_x_high):
                push = (current[mask_x_high, 0] - (HALF_SIDE - target_r)) * 0.5
                current[mask_x_high, 0] -= push
                moved = True
                
            mask_x_low = current[:, 0] < -(HALF_SIDE - target_r)
            if np.any(mask_x_low):
                push = (current[mask_x_low, 0] - (-(HALF_SIDE - target_r))) * 0.5
                current[mask_x_low, 0] += push
                moved = True
                
            mask_y_high = current[:, 1] > HALF_SIDE - target_r
            if np.any(mask_y_high):
                push = (current[mask_y_high, 1] - (HALF_SIDE - target_r)) * 0.5
                current[mask_y_high, 1] -= push
                moved = True
                
            mask_y_low = current[:, 1] < -(HALF_SIDE - target_r)
            if np.any(mask_y_low):
                push = (current[mask_y_low, 1] - (-(HALF_SIDE - target_r))) * 0.5
                current[mask_y_low, 1] += push
                moved = True
        
        # Final clamp to ensure we don't go out of bounds due to numerical issues
        current = np.clip(current, -HALF_SIDE, HALF_SIDE)
        
        if not moved:
            # No violations for target_r, so we can try to increase target_r further
            # The loop will re-calculate r_curr which should be >= target_r
            pass

    return current, compute_radius(current)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin from 150s
    
    best_config = _simple_grid()
    best_r = compute_radius(best_config)
    
    # Try several initializations
    inits = [
        random_grid_initialization(),
        _simple_grid(),
    ]
    
    for init in inits:
        r = compute_radius(init)
        if r > best_r:
            best_r = r
            best_config = init.copy()
    
    # Run relaxation loops
    # We will try multiple times with different random perturbations to escape local minima
    
    np.random.seed(42)
    
    for trial in range(30):
        if time.time() > deadline - 5.0:
            break
        
        # Start from best config with some noise to escape local minima
        noise_level = 0.005 * (1 - trial / 30) + 0.001
        start_config = best_config + np.random.uniform(-noise_level, noise_level, size=(N_CIRCLES, 2))
        start_config = np.clip(start_config, -HALF_SIDE, HALF_SIDE)
        
        # Run relaxation
        # We need to be careful with time. 
        # Each iteration of relaxation_loop involves a KDTree build and query.
        # For N=200, this is fast (~1ms). 
        # 2000 iterations might take 2-3 seconds.
        # 30 trials * 3s = 90s. This fits in the budget.
        
        result, r = relaxation_loop(start_config, max_iters=1500)
        
        if r > best_r:
            best_r = r
            best_config = result.copy()
            
    # Final polish: run relaxation on the best config found
    if time.time() < deadline - 3.0:
        final_result, final_r = relaxation_loop(best_config, max_iters=1000)
        if final_r > best_r:
            best_r = final_r
            best_config = final_result.copy()
    
    return best_config
# EVOLVE-BLOCK-END
