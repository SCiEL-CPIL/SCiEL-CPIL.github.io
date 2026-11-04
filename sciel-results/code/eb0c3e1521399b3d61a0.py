"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    # Check walls
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    # Check pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True

def _max_fit_radius(x, y, xy, r, i, n):
    """Calculate the maximum radius for circle i at position (x, y) given others."""
    # Wall constraint
    max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
    
    # Pairwise constraints
    max_r_pairs = max_r_wall
    for j in range(n):
        if i == j:
            continue
        dx = x - xy[j, 0]
        dy = y - xy[j, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        # dist >= r_i + r_j  => r_i <= dist - r_j
        max_r_pairs = min(max_r_pairs, dist - r[j])
        
    return max(0.0, max_r_pairs)

def _greedy_init(n, side, seed=0):
    """
    Initialize with a grid and compute the max possible radius for each.
    This ensures a valid start with reasonably large radii.
    """
    rng = np.random.RandomState(seed)
    # Create grid positions
    xs = (np.arange(side) + 0.5) / side
    ys = (np.arange(side) + 0.5) / side
    gx, gy = np.meshgrid(xs, ys, indexing='ij')
    # Flatten and take first n
    px = gx.ravel()[:n]
    py = gy.ravel()[:n]
    
    # Add small random jitter to break symmetry
    px = px + rng.uniform(-0.01, 0.01, n)
    py = py + rng.uniform(-0.01, 0.01, n)
    # Clip to [0.05, 0.95] to avoid sticking to walls immediately
    px = np.clip(px, 0.05, 0.95)
    py = np.clip(py, 0.05, 0.95)
    
    xy = np.column_stack([px, py])
    r = np.zeros(n)
    
    # Greedy max-fit pass
    # Order circles by initial radius potential (center of square first? or just random)
    # Let's just do a few passes of max-fit
    for _ in range(10):
        for i in range(n):
            r[i] = _max_fit_radius(xy[i, 0], xy[i, 1], xy, r, i, n)
            
    return xy, r

def _sa_packing(xy, r, deadline, rng, T0=0.1, T_end=0.001, alpha=0.9995):
    """
    Simulated Annealing to optimize packing.
    State: xy (n,2), r (n,)
    Moves: 
      1. Move a random circle to a new random position (within bounds)
      2. Move a random circle by a small delta
      3. Adjust radius of a random circle (try to increase)
    Cost: -sum(r) + penalty * violations
    """
    n = len(xy)
    current_xy = xy.copy()
    current_r = r.copy()
    best_xy = current_xy.copy()
    best_r = current_r.copy()
    
    # Initial evaluation
    cur_total, cur_valid = _eval_packing(current_xy, current_r)
    cur_cost = -cur_total if cur_valid else 1000.0
    best_total, best_valid = _eval_packing(best_xy, best_r)
    best_cost = -best_total if best_valid else 1000.0
    
    T = T0
    
    # Precompute wall bounds for valid moves
    # We will move circles to random positions within [0,1]
    # But we must ensure r is updated to be valid or penalize.
    
    # Strategy: 
    # 1. Pick a random circle i.
    # 2. Generate a new position (x, y).
    # 3. Calculate max possible radius r_new at that position given OTHERS.
    # 4. If r_new > current r[i], accept (improvement).
    # 5. If r_new < current r[i], accept with probability exp((cost_new - cost_old)/T).
    #    Note: cost is -sum(r). So cost_new = -sum(r with r_i=r_new).
    #    delta_cost = r_old - r_new.
    #    Accept if delta_cost < 0 OR random() < exp(-delta_cost/T).
    
    # To speed up, we only calculate the new radius for the moved circle.
    # The rest of the packing remains valid IF the new radius doesn't cause overlap?
    # Wait, if we set r_i to max_fit, it might overlap with others if others moved?
    # No, max_fit guarantees no overlap with others at the current positions.
    # So the packing remains valid!
    # This is a key insight: if we always set r_i to the max possible given others,
    # the packing is always valid (assuming others were valid).
    # Therefore, we only need to accept if the new radius is better, OR if we allow
    # "temporary" invalid states? 
    # Actually, if we always keep it valid, we are doing a greedy hill-climbing with
    # random restarts of positions. This might get stuck.
    # To escape, we need to allow "worse" states.
    # Let's allow the radius to be *less* than max_fit? No, that's dumb.
    # Let's allow the *position* to be such that it forces a smaller radius,
    # but maybe opens up space for *other* circles later?
    
    # Better SA Strategy:
    # State: positions and radii.
    # Move: Pick i. New position (x', y').
    # New radius r'_i = max_fit(x', y').
    # If r'_i > r_i: Accept (improvement).
    # If r'_i < r_i: Accept with prob exp((r_i - r'_i)/T).
    # This allows moving to a "worse" spot for circle i, hoping that later moves
    # of other circles will allow i (or others) to grow.
    
    steps = 0
    while time.time() < deadline and T > T_end:
        i = rng.randint(n)
        
        # Random move: jump to a new location in [0,1]^2
        # Or small step? Jumps are better for global search.
        # Let's mix: 50% jump, 50% small step
        if rng.random() < 0.5:
            new_x = rng.uniform(0.0, 1.0)
            new_y = rng.random()  # Wait, rng.uniform for y
            new_y = rng.uniform(0.0, 1.0)
        else:
            step_size = 0.1 * T / T0 + 0.01
            new_x = current_xy[i, 0] + rng.uniform(-step_size, step_size)
            new_y = current_xy[i, 1] + rng.uniform(-step_size, step_size)
            
        # Ensure within [0,1]
        new_x = np.clip(new_x, 0.0, 1.0)
        new_y = np.clip(new_y, 0.0, 1.0)
        
        # Calculate max radius for i at new position
        new_r = _max_fit_radius(new_x, new_y, current_xy, current_r, i, n)
        
        old_r = current_r[i]
        delta_r = new_r - old_r
        
        # Accept?
        if delta_r > 0:
            accept = True
        else:
            # delta_r is negative
            # Probability of accepting worse state
            # We use the change in total sum of radii
            # Cost = -Sum(r). Delta Cost = -Delta Sum = -delta_r.
            # P = exp(-Delta Cost / T) = exp(delta_r / T)
            # Since delta_r < 0, exp(delta_r/T) < 1
            if rng.random() < np.exp(delta_r / T):
                accept = True
            else:
                accept = False
        
        if accept:
            current_xy[i, 0] = new_x
            current_xy[i, 1] = new_y
            current_r[i] = new_r
            
            # Update best
            # We don't need to re-eval whole packing for validity because
            # _max_fit ensures no overlap with others.
            # However, we must ensure the circle is inside the square.
            # _max_fit handles walls too.
            # So the state is always valid.
            total = np.sum(current_r)
            if total > best_total:
                best_total = total
                best_xy = current_xy.copy()
                best_r = current_r.copy()
                
        T *= alpha
        steps += 1
        
        # Periodically try to "push" radii up? 
        # Actually, _max_fit already pushes them to the limit.
        # The only way to increase total radius is to move a circle to a spot 
        # where it can be larger, or move another circle out of the way.
        
    return best_xy, best_r, best_total

def _local_polish(xy, r, deadline):
    """
    Simple local search: for each circle, try to find a better position 
    on a fine grid.
    """
    n = len(xy)
    best_xy = xy.copy()
    best_r = r.copy()
    best_total = np.sum(best_r)
    
    grid_size = 50
    gx = np.linspace(0, 1, grid_size)
    gy = np.linspace(0, 1, grid_size)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(n):
            best_local_r = best_r[i]
            best_local_x = best_xy[i, 0]
            best_local_y = best_xy[i, 1]
            
            # Scan grid
            for x in gx:
                for y in gy:
                    # Quick check: is this point far enough from others?
                    # We can just compute max_fit
                    r_try = _max_fit_radius(x, y, best_xy, best_r, i, n)
                    if r_try > best_local_r + 1e-6:
                        best_local_r = r_try
                        best_local_x = x
                        best_local_y = y
                        
            if best_local_r > best_r[i] + 1e-6:
                best_xy[i, 0] = best_local_x
                best_xy[i, 1] = best_local_y
                best_r[i] = best_local_r
                best_total = np.sum(best_r)
                improved = True
                
    return best_xy, best_r, best_total

def construct():
    start_time = time.time()
    # 150s limit. Use 140s deadline.
    deadline = start_time + 140.0
    
    best_xy = None
    best_r = None
    best_total = 0.0
    
    rng = np.random.RandomState(12345)
    
    # Phase 1: Multiple greedy inits + SA
    # Try different grid sizes
    for side in [5, 6, 7]:
        if time.time() > deadline - 10:
            break
            
        xy, r = _greedy_init(N, side, seed=side)
        
        # Run SA
        # Allocate time based on remaining
        sa_deadline = min(deadline - 20, start_time + 30) # Give 30s per SA run initially
        out_xy, out_r, out_total = _sa_packing(xy, r, sa_deadline, rng)
        
        # Polish
        if time.time() < deadline - 10:
            pol_xy, pol_r, pol_total = _local_polish(out_xy, out_r, deadline)
            if pol_total > out_total:
                out_xy, out_r, out_total = pol_xy, pol_r, pol_total
                
        if out_total > best_total:
            best_total = out_total
            best_xy = out_xy
            best_r = out_r
            
    # Phase 2: If time remains, do more SA from the best solution found
    if best_xy is not None and time.time() < deadline - 5:
        # Perturb best solution
        pert_xy = best_xy + rng.uniform(-0.02, 0.02, size=(N, 2))
        pert_xy = np.clip(pert_xy, 0.0, 1.0)
        pert_r = best_r.copy()
        
        # Re-evaluate max fits for perturbed positions
        for i in range(N):
            pert_r[i] = _max_fit_radius(pert_xy[i, 0], pert_xy[i, 1], pert_xy, pert_r, i, N)
            
        out_xy, out_r, out_total = _sa_packing(pert_xy, pert_r, deadline - 2, rng, T0=0.05, T_end=0.0005, alpha=0.9998)
        
        if time.time() < deadline - 1:
            pol_xy, pol_r, pol_total = _local_polish(out_xy, out_r, deadline)
            if pol_total > out_total:
                out_xy, out_r, out_total = pol_xy, pol_r, pol_total
                
        if out_total > best_total:
            best_total = out_total
            best_xy = out_xy
            best_r = out_r

    # Fallback
    if best_xy is None:
        xy, r = _greedy_init(N, 5, seed=0)
        best_xy = xy
        best_r = r
        best_total = np.sum(r)

    # Final Validation
    result = np.zeros((N, 3))
    for i in range(N):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        # Ensure valid
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
        result[i, 2] = ri
        
    # One last check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        # Should not happen, but fallback to grid
        side = 5
        xy, r = _greedy_init(N, side, seed=0)
        result = np.zeros((N, 3))
        for i in range(N):
            x, y, ri = xy[i, 0], xy[i, 1], r[i]
            ri = min(ri, x, 1.0 - x, y, 1.0 - y)
            result[i, 0] = np.clip(x, ri, 1.0 - ri)
            result[i, 1] = np.clip(y, ri, 1.0 - ri)
            result[i, 2] = ri

    return result
# EVOLVE-BLOCK-END
