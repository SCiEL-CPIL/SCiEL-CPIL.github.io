# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 (q = 49, p = 7).
    Strategy: Simulated Annealing on the complement set E.
    N = F_49^2 \ E.
    Validity: For every point x, there exists a direction v such that
    all points on line(x, v) \ {x} are in N.
    Equivalently: For every x, min_{v} (count of points in E on line(x, v) \ {x}) == 0.
    
    We want to maximize |E| (minimize |N|).
    """
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401

    # Deadline guard
    deadline = time.time() + 570.0

    # --- Field Arithmetic Setup ---
    # F_49 = F_7[alpha], alpha^2 = 3
    k_const = 3
    
    idx_to_elem = [(i // p, i % p) for i in range(q)]
    
    # Precompute multiplication table
    mul_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ai, bi = idx_to_elem[i]
        for j in range(q):
            aj, bj = idx_to_elem[j]
            aa = (ai * aj + bi * bj * k_const) % p
            bb = (ai * bj + bi * aj) % p
            mul_table[i, j] = aa * p + bb

    # Precompute addition table
    add_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ai, bi = idx_to_elem[i]
        for j in range(q):
            aj, bj = idx_to_elem[j]
            aa = (ai + aj) % p
            bb = (bi + bj) % p
            add_table[i, j] = aa * p + bb

    # --- Directions ---
    # v = (0, 1) and v = (1, m) for m in F_q
    directions = []
    directions.append((0, 1))
    for m in range(q):
        directions.append((1, m))
    num_dirs = len(directions) # 50

    # --- Precompute line_other ---
    # line_other[z, v, t] = index of point x = z + t*v, t != 0
    zx_arr = np.array([z // q for z in range(total_points)], dtype=np.int32)
    zy_arr = np.array([z % q for z in range(total_points)], dtype=np.int32)
    t_vals = np.arange(1, q, dtype=np.int32)
    
    line_other = np.zeros((total_points, num_dirs, q - 1), dtype=np.int32)
    
    for v_idx, (dx, dy) in enumerate(directions):
        tdx = mul_table[t_vals, dx]
        tdy = mul_table[t_vals, dy]
        
        nx = add_table[zx_arr[:, None], tdx[None, :]]
        ny = add_table[zy_arr[:, None], tdy[None, :]]
        npidx = nx * q + ny
        line_other[:, v_idx, :] = npidx

    # --- State Representation ---
    # in_E: bool array of size total_points
    # cnt: int array of shape (total_points, num_dirs)
    # cnt[x, v] = number of points in E on line(x, v) excluding x.
    
    # Precompute unique neighbors for each point to speed up validity checks?
    # Actually, we need min over v.
    
    # Helper to initialize cnt from in_E
    def compute_cnt(in_E):
        cnt = np.zeros((total_points, num_dirs), dtype=np.int32)
        for z in np.where(in_E)[0]:
            for v in range(num_dirs):
                pts = line_other[z, v, :]
                cnt[pts, v] += 1
        return cnt

    # Helper to check validity
    # Valid if for all x, min(cnt[x, :]) == 0
    def is_valid(cnt):
        return np.all(np.min(cnt, axis=1) == 0)

    # --- Initial State ---
    # Start with a small valid E? Or empty E?
    # If E is empty, N is everything. Is N a Nikodym set?
    # Yes, because for any x, we can pick any v, and all points on line are in N (since E is empty).
    # So E = {} is a valid starting point with size 0.
    # We want to grow E.
    
    in_E = np.zeros(total_points, dtype=bool)
    cnt = np.zeros((total_points, num_dirs), dtype=np.int32)
    
    # We will use a "greedy start" to get a decent initial size, then anneal.
    # Greedy start:
    order = list(range(total_points))
    random.Random(42).shuffle(order)
    
    for z in order:
        if time.time() > deadline - 300: break # Reserve time for annealing
        
        # Try adding z
        # Check if adding z keeps validity.
        # Adding z increments cnt[x, v] for all x in line_other[z, v].
        # This might make some x invalid if min(cnt[x, :]) was 0 and we incremented the only 0.
        
        # Quick check: 
        # For each v, let pts = line_other[z, v].
        # We increment cnt[pts, v].
        # A point x becomes invalid if:
        # 1. x is in pts (for some v) AND
        # 2. min(cnt[x, :]) == 0 BEFORE update AND
        # 3. The specific direction v we are incrementing was the ONLY direction with 0 for x.
        
        # Efficient check:
        # Let valid_dirs_before = [v for v in range(num_dirs) if cnt[x, v] == 0]
        # If we increment cnt[x, v], and len(valid_dirs_before) == 1 and valid_dirs_before[0] == v, then x becomes invalid.
        
        # We need to check this for all affected x.
        
        can_add = True
        revert_info = []
        
        for v in range(num_dirs):
            pts = line_other[z, v, :]
            # For each x in pts, check if v was the only zero.
            # cnt[pts, v] is the current count.
            # We need to know if any other direction has 0.
            # This is hard to vectorize easily without iterating or complex masks.
            
            # Since num_dirs=50, we can check:
            # min_other = min(cnt[pts, u] for u in range(num_dirs) if u != v)
            # If min_other == 0, then x remains valid even if cnt[x, v] becomes 1.
            # If min_other > 0, then x was ONLY valid via direction v. If we increment v, x becomes invalid.
            
            # Vectorized check for a specific v:
            # Create a copy of cnt[pts, :]? No, too slow.
            
            # Alternative: 
            # zero_counts = (cnt == 0)
            # num_zeros = np.sum(zero_counts, axis=1)
            # For a point x, if num_zeros[x] == 1, it is "critical".
            # If we increment the specific direction that is 0, it becomes invalid.
            
            # Let's maintain num_zeros array.
            # num_zeros[x] = number of directions v such that cnt[x, v] == 0.
            
            pass # We will implement num_zeros tracking below.

    # Let's restart the loop with proper num_zeros tracking.
    in_E = np.zeros(total_points, dtype=bool)
    cnt = np.zeros((total_points, num_dirs), dtype=np.int32)
    num_zeros = np.full(total_points, num_dirs, dtype=np.int32) # Initially all 0s, so 50 zeros
    
    for z in order:
        if time.time() > deadline - 300: break
        
        # Tentatively add z
        # Update cnt and num_zeros
        changes = []
        valid = True
        
        for v in range(num_dirs):
            pts = line_other[z, v, :]
            old_vals = cnt[pts, v].copy()
            
            # Increment
            cnt[pts, v] += 1
            
            # Update num_zeros for pts where old_val was 0
            mask = (old_vals == 0)
            if np.any(mask):
                affected_pts = pts[mask]
                num_zeros[affected_pts] -= 1
                # Check if any became -1 (impossible) or if we broke validity?
                # Validity is broken if num_zeros[x] becomes 0? No, num_zeros==0 means NO direction has 0 count.
                # Wait. Validity condition: min(cnt[x, :]) == 0.
                # This is equivalent to num_zeros[x] > 0.
                # So if num_zeros[x] becomes 0, x is invalid.
                if np.any(num_zeros[affected_pts] == 0):
                    valid = False
                    # We need to revert everything.
                    break
            
            changes.append((pts, old_vals, v, mask, affected_pts))
        
        if not valid:
            # Revert
            for (pts, old_vals, v, mask, affected_pts) in changes:
                cnt[pts, v] = old_vals
                if np.any(mask):
                    num_zeros[affected_pts] += 1
            continue
        
        # If valid, commit
        in_E[z] = True
        # Note: The changes list was already applied. We just need to ensure we didn't revert.
        # Wait, the logic above breaks out of the v-loop if invalid.
        # If valid, we have already updated cnt and num_zeros.
        # But we stored old_vals. We don't need to revert.
        # However, we need to make sure we didn't partially update if we broke early?
        # Yes, if we break, we revert. If we don't break, we are done.
        
        # Optimization: The "changes" list is only needed for reverting.
        # If valid, we can discard it.

    # --- Simulated Annealing ---
    # Current best
    best_E = in_E.copy()
    best_size = np.sum(best_E)
    
    current_E = in_E.copy()
    current_cnt = cnt.copy()
    current_num_zeros = num_zeros.copy()
    current_size = best_size
    
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    # We need a way to add a point to E (increase size) and remove a point from E (decrease size).
    # Adding is the "growth" move. Removing is the "repair" move.
    
    # To speed up, we will precompute a list of "candidate points to add" 
    # and "candidate points to remove".
    
    # Actually, a standard SA move:
    # 1. Pick a random point z.
    # 2. If z not in E: try to add it.
    # 3. If z in E: try to remove it.
    
    # But "try to add" requires the complex check above.
    # "Try to remove" is always valid (monotonicity).
    
    # So:
    # Move 1: Add random z. If valid, accept. If invalid, reject (or accept with prob exp(-delta/T)? No, invalid is hard constraint).
    # Actually, we can't accept invalid states. So we must repair.
    # Repair strategy: If adding z causes invalidity, we can try to remove some other points to fix it?
    # That's complicated.
    
    # Alternative SA strategy:
    # State is always valid.
    # Move: 
    #   - With prob p_add: Try to add a random point. If valid, accept. Else, try to remove a random point from E (to "make room" or just explore)? 
    #   - No, if add fails, we just stay put? That's slow.
    
    # Better: 
    #   - Generate a new valid state by:
    #     1. Remove a random point from E (always valid).
    #     2. Add a random point to E (might be invalid).
    #     If invalid after step 2, try to fix by removing another point?
    
    # Let's stick to:
    # 1. Pick random z.
    # 2. If z in E: Remove it. (Always valid, size decreases).
    # 3. If z not in E: Try to add it. 
    #    - If valid: Accept (size increases).
    #    - If invalid: Reject (size stays same).
    
    # To escape local optima, we need to sometimes remove points even if we can't add?
    # Standard SA accepts worse states with prob exp(-delta/T).
    # Here, "worse" means smaller size.
    # So:
    # - Add move: delta = +1. Always accept if valid.
    # - Remove move: delta = -1. Accept with prob exp(-1/T).
    
    # This allows the search to shrink E to escape a trap, then grow it differently.
    
    steps = 0
    max_steps = 2000000 # Large number, limited by time
    
    while time.time() < deadline - 10:
        steps += 1
        
        z = random.randint(0, total_points - 1)
        
        if current_E[z]:
            # Remove z
            # Update cnt and num_zeros
            for v in range(num_dirs):
                pts = line_other[z, v, :]
                old_vals = current_cnt[pts, v].copy()
                current_cnt[pts, v] -= 1
                
                mask = (old_vals == 1) # Becomes 0
                if np.any(mask):
                    affected_pts = pts[mask]
                    current_num_zeros[affected_pts] += 1
            
            current_E[z] = False
            current_size -= 1
            
            # Acceptance: prob = exp(-1/T)
            if random.random() < np.exp(-1.0 / T):
                pass # Accepted
            else:
                # Revert
                for v in range(num_dirs):
                    pts = line_other[z, v, :]
                    old_vals = current_cnt[pts, v].copy() # This is the NEW value (decremented)
                    # We need the PREVIOUS value.
                    # We didn't store it. 
                    # Let's re-store in the remove block.
                    pass
                # This revert logic is messy. Let's use a helper function for moves.
                
        else:
            # Add z
            # Check validity
            valid = True
            changes = []
            
            for v in range(num_dirs):
                pts = line_other[z, v, :]
                old_vals = current_cnt[pts, v].copy()
                
                # Check if any point in pts has num_zeros == 1 and this v is the only zero?
                # We need to know if v is the only zero for x.
                # current_num_zeros[x] == 1 and current_cnt[x, v] == 0.
                
                # Let's check before updating.
                # For x in pts:
                # if current_num_zeros[x] == 1 and current_cnt[x, v] == 0:
                #    Then adding z will make current_cnt[x, v] = 1.
                #    Then num_zeros[x] becomes 0.
                #    Invalid.
                
                mask_critical = (current_num_zeros[pts] == 1) & (current_cnt[pts, v] == 0)
                
                if np.any(mask_critical):
                    valid = False
                    # Revert changes made in previous v's
                    for (pts_p, old_vals_p, v_p) in changes:
                        current_cnt[pts_p, v_p] = old_vals_p
                        # Revert num_zeros? 
                        # We didn't update num_zeros yet in the "check" phase.
                        # So no num_zeros revert needed.
                    break
                
                # If valid so far, record change
                changes.append((pts, old_vals, v))
            
            if valid:
                # Commit changes
                for (pts, old_vals, v) in changes:
                    current_cnt[pts, v] = old_vals + 1
                    # Update num_zeros
                    mask = (old_vals == 0)
                    if np.any(mask):
                        affected_pts = pts[mask]
                        current_num_zeros[affected_pts] -= 1
                
                current_E[z] = True
                current_size += 1
                
                if current_size > best_size:
                    best_size = current_size
                    best_E = current_E.copy()
            else:
                # Reject (do nothing)
                pass
        
        # Cool down
        T *= cooling
        if T < min_T:
            T = min_T

    # Return best_E
    points_list = []
    for pidx in range(total_points):
        if not best_E[pidx]:
            xi = pidx // q
            yi = pidx % q
            xa, xb = idx_to_elem[xi]
            ya, yb = idx_to_elem[yi]
            points_list.append([[xa, xb], [ya, yb]])

    if not points_list:
        points_list = []
        for pidx in range(total_points):
            xi = pidx // q
            yi = pidx % q
            xa, xb = idx_to_elem[xi]
            ya, yb = idx_to_elem[yi]
            points_list.append([[xa, xb], [ya, yb]])

    return np.array(points_list, dtype=np.int64)

# EVOLVE-BLOCK-END
