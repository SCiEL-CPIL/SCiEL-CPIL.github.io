import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    num_dirs = q + 1 # 50 directions: v=(0,1) and v=(1,m) for m in F_q

    # Deadline guard
    deadline = time.time() + 570.0

    # --- Field Arithmetic Setup ---
    # F_49 = F_7[alpha] with alpha^2 = 3
    k_const = 3
    
    # Map (a, b) -> index a*p + b
    idx_to_elem = [(i // p, i % p) for i in range(q)]
    
    # Precompute multiplication table for F_q
    # mul[i, j] = index of i * j
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

    # --- Direction and Line Precomputation ---
    # Directions: index 0 is (0,1), index 1..49 are (1, m) where m is index 0..48
    # Note: In the task description, v=(1,m) for m in F_49.
    # My indexing: 
    # dir_idx 0 -> (0, 1)
    # dir_idx i (1<=i<=49) -> (1, i-1)
    
    dir_dx = np.zeros(num_dirs, dtype=np.int32)
    dir_dy = np.zeros(num_dirs, dtype=np.int32)
    
    dir_dx[0] = 0
    dir_dy[0] = 1 # element 1 is (0,1)? No.
    # Element 1 in F_49 is (0, 1) if idx = a*p+b. 
    # Wait. idx = a*p+b. 
    # 0 = (0,0)
    # 1 = (0,1) -> alpha
    # 7 = (1,0) -> 1
    # Let's check the problem statement: "pair (a, b) ... stands for a + b*alpha".
    # So index 1 is 0 + 1*alpha = alpha.
    # Index 7 is 1 + 0*alpha = 1.
    
    # Direction v = (0, 1) in the field sense means dx=0, dy=1 (the field element 1).
    # Field element 1 is index 7.
    # Direction v = (1, m) means dx=1 (index 7), dy=m.
    
    # Let's re-verify standard vector notation.
    # Points are [[x_a, x_b], [y_a, y_b]].
    # Vector v = (1, m). This usually means dx = 1, dy = m.
    # In F_49, the element '1' is (1,0) which is index 7.
    # The element '0' is (0,0) which is index 0.
    
    # So:
    # v_vertical: dx=0 (idx 0), dy=1 (idx 7)
    # v_slant_m: dx=1 (idx 7), dy=m (idx m)
    
    dir_dx[0] = 0
    dir_dy[0] = 7
    
    for i in range(1, num_dirs):
        m = i - 1
        dir_dx[i] = 7
        dir_dy[i] = m

    # Precompute line_members: shape (total_points, num_dirs, q)
    # line_members[x, v, t] = index of point x + t*v
    zx_arr = np.array([z // q for z in range(total_points)], dtype=np.int32)
    zy_arr = np.array([z % q for z in range(total_points)], dtype=np.int32)
    
    t_vals = np.arange(q, dtype=np.int32)
    
    line_members = np.zeros((total_points, num_dirs, q), dtype=np.int32)
    
    for v_idx in range(num_dirs):
        dx = dir_dx[v_idx]
        dy = dir_dy[v_idx]
        
        # t * dx and t * dy
        tdx = mul_table[t_vals, dx]
        tdy = mul_table[t_vals, dy]
        
        # x + t*dx, y + t*dy
        nx = add_table[zx_arr[:, None], tdx[None, :]]
        ny = add_table[zy_arr[:, None], tdy[None, :]]
        
        line_members[:, v_idx, :] = nx * q + ny

    # --- Set Cover / Max-SAT Heuristic ---
    # Goal: Maximize |E| such that for all x, not all directions v are "blocked".
    # A direction v is blocked for x if there exists y in E such that y is on line(x, v) and y != x.
    # Actually, the condition is: line(x, v) \ {x} subset N.
    # This is equivalent to: (line(x, v) \ {x}) intersect E is empty.
    # So v is "blocked" for x if (line(x, v) \ {x}) intersect E is NOT empty.
    # We need for every x, at least one v is NOT blocked.
    
    # State:
    # in_E: bool array (total_points)
    # blocked_count[x, v]: number of points in E on line(x, v) excluding x.
    # If blocked_count[x, v] > 0, direction v is blocked for x.
    # If sum(blocked_count[x, :]) == num_dirs, x is "isolated" (invalid).
    # We want to avoid any x being isolated.
    
    # Optimization: 
    # When we add z to E, it increments blocked_count[x, v] for all x such that z is on line(x, v) \ {x}.
    # This is equivalent to x being on line(z, v) \ {z}.
    # So affected (x, v) pairs are those where x in line_members[z, v] and x != z.
    
    # Let's precompute for each z, the list of (x, v) pairs that would be incremented if z is added to E.
    # Actually, we can just compute it on the fly or precompute a sparse structure.
    # Given total_points=2401 and num_dirs=50, line_members[z, v] has 49 other points.
    # Total pairs per z is 50 * 49 = 2450.
    # Storing this for all z is 2401 * 2450 ~ 6M entries. Manageable.
    
    # Let's precompute affected_pairs[z] as a list of (x, v) tuples.
    # Or better, two arrays: affected_x[z], affected_v[z]
    
    affected_x = []
    affected_v = []
    
    for z in range(total_points):
        xs = []
        vs = []
        for v in range(num_dirs):
            # Get points on line(z, v) excluding z
            # line_members[z, v, :] contains z at some index t_z.
            # We can just iterate all t, and skip if point == z.
            # Or find t_z. Since line is a group action, z corresponds to t=0?
            # line_members[z, v, t] = z + t*v.
            # If t=0, point is z.
            # So we can just take t=1..48.
            # Wait, t_vals is 0..48.
            # So line_members[z, v, 0] is z.
            # The others are line_members[z, v, 1:]
            
            other_pts = line_members[z, v, 1:]
            xs.extend(other_pts)
            vs.extend([v] * len(other_pts))
            
        affected_x.append(np.array(xs, dtype=np.int32))
        affected_v.append(np.array(vs, dtype=np.int32))

    # Search
    best_E = np.zeros(total_points, dtype=bool)
    best_size = 0
    
    num_restarts = 50
    rng = random.Random(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        in_E = np.zeros(total_points, dtype=bool)
        blocked = np.zeros((total_points, num_dirs), dtype=np.int32)
        
        # Random order
        order = list(range(total_points))
        rng.shuffle(order)
        
        size = 0
        
        # Heuristic: Try to add points. 
        # To speed up, we can check if adding z would isolate any x.
        # x is isolated if all blocked[x, :] > 0.
        # Before adding z, let valid_x = [x for x if any(blocked[x, :]==0)]
        # If we add z, we increment blocked[x, v] for (x, v) in affected[z].
        # x becomes invalid if it was valid (had some 0s) and ALL its 0s are turned to >0 by this update.
        # i.e., for all v where blocked[x, v]==0, (x, v) is in affected[z].
        
        for z in order:
            if time.time() > deadline:
                break
            if in_E[z]:
                continue
                
            # Check if adding z is safe
            # Get affected pairs
            ax = affected_x[z]
            av = affected_v[z]
            
            # Unique x's affected
            # We need to check if any x becomes fully blocked.
            # An x is at risk if it currently has min(blocked[x, :]) == 0.
            # Let's find unique x in ax.
            # For each unique x, check if all its currently zero directions are covered by av.
            
            # Efficient check:
            # Create a temporary view? No, just iterate unique x's.
            # Number of unique x's is at most 2450, but usually much less (lines overlap).
            # Actually, for a fixed z, the affected x's are all points on the 50 lines through z, excluding z.
            # That's 50 * 49 = 2450 points.
            # Most of the 2401 points are affected!
            # So we have to check almost all points.
            
            # Alternative: 
            # Instead of checking all x, we can just apply the update, check for violations, and revert if bad.
            # Update is vectorized: blocked[ax, av] += 1
            # Check: min(blocked, axis=1). If any min > 0, revert.
            
            # Let's try this. It's O(2450) updates + O(2401) check.
            # 2450 + 2401 ~ 5k ops per point. 2401 points * 5k = 12M ops per restart.
            # 50 restarts = 600M ops. Might be too slow in Python/Numpy overhead?
            # Numpy vectorization should handle 2450 increments fast.
            
            # Apply update
            blocked[ax, av] += 1
            in_E[z] = True
            size += 1
            
            # Check validity
            # min_blocked = np.min(blocked, axis=1)
            # if np.any(min_blocked > 0):
            #    Revert
            
            # Optimization: Only check x's that were affected?
            # No, if an x was not affected, its blocked counts didn't change, so if it was valid, it stays valid.
            # If it was invalid, it was already invalid (but we maintain invariant that no x is invalid).
            # So we only need to check x in ax.
            
            # Get unique x in ax
            # np.unique(ax)
            
            # If we use unique, it's a bit expensive.
            # But ax has 2450 elements.
            
            # Let's just check all points? 2401 is small.
            # min_blocked = np.min(blocked, axis=1)
            # if np.any(min_blocked > 0):
            #     blocked[ax, av] -= 1
            #     in_E[z] = False
            #     size -= 1
            #     continue
            
            # Let's try checking only affected unique points to save time.
            unique_affected = np.unique(ax)
            min_blocked = np.min(blocked[unique_affected, :], axis=1)
            
            if np.any(min_blocked > 0):
                blocked[ax, av] -= 1
                in_E[z] = False
                size -= 1
            else:
                pass # Keep z in E
        
        if size > best_size:
            best_size = size
            best_E = in_E.copy()
            
    # Build N
    points_list = []
    for pidx in range(total_points):
        if not best_E[pidx]:
            xi = pidx // q
            yi = pidx % q
            xa, xb = idx_to_elem[xi]
            ya, yb = idx_to_elem[yi]
            points_list.append([[xa, xb], [ya, yb]])

    if not points_list:
        # Fallback to full space if something went wrong (shouldn't happen)
        for pidx in range(total_points):
            xi = pidx // q
            yi = pidx % q
            xa, xb = idx_to_elem[xi]
            ya, yb = idx_to_elem[yi]
            points_list.append([[xa, xb], [ya, yb]])

    return np.array(points_list, dtype=np.int64)
