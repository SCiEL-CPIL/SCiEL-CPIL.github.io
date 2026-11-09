import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    num_dirs = q + 1 # 50 directions: v=(0,1) and v=(1,m) for m in F_q

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 580.0

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
    # Field element 1 is (1,0) -> index 7
    # Field element 0 is (0,0) -> index 0
    
    dir_dx = np.zeros(num_dirs, dtype=np.int32)
    dir_dy = np.zeros(num_dirs, dtype=np.int32)
    
    dir_dx[0] = 0
    dir_dy[0] = 7 # dy = 1 (field element)
    
    for i in range(1, num_dirs):
        m = i - 1
        dir_dx[i] = 7 # dx = 1 (field element)
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

    # --- Initial Construction: Randomized Unital ---
    # A unital is a set of points such that every line intersects it in exactly 1 or q points.
    # If every line through a point x in N intersects N in q points (i.e., the whole line is in N), 
    # then x is covered.
    # Actually, the condition is: for every x in F_q^2, there exists a line L containing x such that L \ {x} subset N.
    # This is equivalent to: for every x, there is a direction v such that line(x, v) \ {x} has no points in E (excluded set).
    # Wait, the condition is: points OTHER than x must be in N.
    # So line(x, v) \ {x} must be disjoint from E.
    # This means: For every x, there must be at least one direction v such that (line(x, v) \ {x}) ∩ E = ∅.
    
    # Let's use the property of finite field unital constructions.
    # Consider the set E_0 = { (x, x^2) : x in F_q }? No, that's a parabola.
    # A standard unital in PG(2, q) or affine plane?
    # Let's try a simpler structural initialization:
    # Pick a random subset of points to be in E.
    # To ensure validity, we can start with E = ∅ (N = all points). This is valid but score 0.
    # We want to MAXIMIZE |E| (minimize |N|).
    # Wait, the metric is "complement size 2401 - |N|". So we want to MAXIMIZE |E|.
    # The current program tries to build E greedily.
    
    # Let's try a "randomized greedy" for E.
    # Start with E = ∅.
    # While we can add points to E without violating the condition, add them.
    # Condition violation: A point x is "bad" if for ALL v, (line(x, v) \ {x}) ∩ E != ∅.
    # We want to avoid any x being bad.
    
    # Strategy:
    # 1. Randomly shuffle all points.
    # 2. Try to add each point z to E.
    # 3. Check if adding z makes any x bad.
    #    x becomes bad if:
    #    - x was not bad before.
    #    - For all v, (line(x, v) \ {x}) ∩ E_new != ∅.
    #    Since E_new = E_old ∪ {z}, the only change is for lines passing through z.
    #    For x to become bad, it must be that for every v:
    #       Either (line(x, v) \ {x}) ∩ E_old != ∅ (already covered by old E)
    #       OR z ∈ (line(x, v) \ {x}) (covered by new point z)
    #    This implies that the set of directions v for which (line(x, v) \ {x}) ∩ E_old = ∅
    #    must be a subset of the directions v for which z ∈ line(x, v) \ {x}.
    #    Let S_old(x) = { v : (line(x, v) \ {x}) ∩ E_old = ∅ }
    #    Let S_z(x) = { v : z ∈ line(x, v) \ {x} }
    #    x becomes bad if S_old(x) ⊆ S_z(x).
    #    Note: S_z(x) is the set of directions of lines passing through both x and z.
    #    If x != z, there is exactly ONE direction v such that z is on line(x, v).
    #    So S_z(x) has size 1 (unless x=z, but z is being added to E, and x is checked).
    #    Wait, if x=z, z is in E. The condition is about line(x, v) \ {x}.
    #    If x=z, z is not in line(x, v) \ {x}. So adding z to E does NOT affect the coverage of x=z by E.
    #    So we only need to check x != z.
    #    For x != z, S_z(x) is a singleton {v_xz}.
    #    So x becomes bad if S_old(x) = {v_xz}.
    #    This means: Before adding z, x was "almost bad" (only one direction v_xz was not covered by E_old).
    #    And z lies on the line in direction v_xz through x.
    
    # This allows a very fast check!
    # Maintain for each x: count_uncovered[x] = number of directions v such that (line(x, v) \ {x}) ∩ E = ∅.
    # Initially count_uncovered[x] = 50 for all x.
    # When we add z to E:
    #    For each x != z, let v_xz be the direction of line(x, z).
    #    If z is the FIRST point of E on line(x, v_xz) \ {x}, then count_uncovered[x] decreases by 1.
    #    How to check if z is the first?
    #    We need to know if (line(x, v_xz) \ {x}) ∩ E_old = ∅.
    #    This is true iff count_uncovered[x] > 0 AND v_xz is one of the uncovered directions.
    #    We can maintain a bitmask or a set of uncovered directions for each x?
    #    50 directions. 2401 points.
    #    Bitmask is 50 bits. Fits in uint64.
    #    uncovered_mask[x]: bit v is 1 if direction v is uncovered for x.
    
    # Let's use this bitmask approach. It's O(1) per (x, z) pair update?
    # No, adding z affects all x != z. That's 2400 points.
    # For each x, we check if v_xz is in uncovered_mask[x].
    # If yes, we clear bit v_xz in uncovered_mask[x].
    # If uncovered_mask[x] becomes 0, x is bad. We must NOT add z.
    
    # So the check for adding z is:
    # 1. For all x != z, compute v_xz.
    # 2. Check if any x has bit v_xz set in uncovered_mask[x] AND clearing it makes the mask 0.
    #    i.e., uncovered_mask[x] == (1 << v_xz).
    # 3. If such an x exists, adding z is invalid.
    # 4. Else, add z, and for all x where bit v_xz was set, clear it.
    
    # Precomputation:
    # For each pair (x, z), what is v_xz?
    # We can precompute a matrix dir_xz[x, z] = v_idx.
    # Size 2401 x 2401. 5.7M entries. Int32. ~23MB. OK.
    
    dir_xz = np.zeros((total_points, total_points), dtype=np.int32)
    # For each x, z, find v such that z is on line(x, v).
    # line(x, v) contains x + t*v for t in F_q.
    # z = x + t*v.
    # If v=(0,1), z_x = x_x, z_y = x_y + t*1. So t = z_y - x_y.
    # If v=(1,m), z_x = x_x + t*1 => t = z_x - x_x. Then z_y = x_y + t*m.
    # Check consistency.
    
    # Faster:
    # For each x, we have 50 lines. Each line has 48 other points.
    # We can just iterate x, then v, then t (1..48), and set dir_xz[x, point] = v.
    
    for x in range(total_points):
        for v in range(num_dirs):
            # line_members[x, v, 1:] are the other points
            others = line_members[x, v, 1:]
            dir_xz[x, others] = v
    
    # Search: Randomized Greedy with Bitmask
    best_E = np.zeros(total_points, dtype=bool)
    best_size = 0
    
    rng = random.Random(42)
    
    # We can try multiple restarts with different random orders.
    # Or just one very good greedy pass?
    # The order matters. Random order is usually okay.
    # Let's do 20 restarts.
    
    for restart in range(20):
        if time.time() > deadline - 10.0:
            break
            
        in_E = np.zeros(total_points, dtype=bool)
        # uncovered_mask[x]: bit v is 1 if direction v is uncovered for x.
        # Initially all 50 bits are 1.
        # 50 bits fits in uint64.
        full_mask = (1 << num_dirs) - 1
        uncovered_mask = np.full(total_points, full_mask, dtype=np.uint64)
        
        order = list(range(total_points))
        rng.shuffle(order)
        
        size = 0
        
        for z in order:
            if time.time() > deadline - 5.0:
                break
            if in_E[z]:
                continue
                
            # Check if adding z is safe
            # We need to check all x != z.
            # For each x, v = dir_xz[x, z].
            # If (uncovered_mask[x] & (1 << v)) != 0:
            #     new_mask = uncovered_mask[x] ^ (1 << v)
            #     if new_mask == 0:
            #         return invalid
            
            # Vectorized check?
            # v_arr = dir_xz[:, z]
            # bits = 1 << v_arr
            # # We only care about x where bit is set.
            # # And specifically if it's the ONLY bit set.
            
            # Let's try to do this efficiently.
            # v_arr = dir_xz[:, z] # shape (2401,)
            # bits = 1 << v_arr # shape (2401,)
            
            # mask_is_only_bit = (uncovered_mask == bits)
            # # But we also need to ensure the bit is actually set.
            # # If uncovered_mask == bits, the bit is set (since bits has exactly one bit set).
            # # Wait, if bits=0? v_arr is 0..49. 1<<v is never 0.
            
            # So if any mask_is_only_bit[x] is True for x != z, adding z is invalid.
            # Note: for x=z, dir_xz[z, z] is undefined (0 by default).
            # We should exclude x=z.
            
            v_arr = dir_xz[:, z]
            bits = 1 << v_arr
            
            # Check for x != z
            # We can just check all, and ignore x=z if it happens to match.
            # But dir_xz[z, z] is 0. bits[z] = 1.
            # uncovered_mask[z] is likely not 1 unless z is very constrained.
            # To be safe, set bits[z] = 0 so it never matches.
            bits[z] = 0
            
            # If any (uncovered_mask == bits) is True, invalid.
            if np.any(uncovered_mask == bits):
                continue # Cannot add z
            
            # Safe to add z.
            in_E[z] = True
            size += 1
            
            # Update uncovered_mask
            # For all x where (uncovered_mask[x] & bits[x]) != 0, clear the bit.
            # affected = (uncovered_mask & bits) != 0
            # uncovered_mask[affected] ^= bits[affected]
            
            affected = (uncovered_mask & bits) != 0
            # Exclude z? z is in E now.
            # Does z's mask matter?
            # The condition for x is based on lines through x.
            # z is in E.
            # For x=z, the lines through z...
            # The condition "line(x, v) \ {x} subset N" for x=z means line(z, v) \ {z} has no points in E.
            # Adding z to E does not put z in line(z, v) \ {z}.
            # So z's mask should NOT be updated when z is added.
            # However, our update logic "clear bit v if z is on line(x, v)"
            # For x=z, z is NOT on line(z, v) \ {z}.
            # So affected[z] should be False.
            # Let's check: bits[z] was set to 0.
            # So affected[z] = (uncovered_mask[z] & 0) != 0 = False.
            # Correct.
            
            uncovered_mask[affected] ^= bits[affected]
            
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
        for pidx in range(total_points):
            xi = pidx // q
            yi = pidx % q
            xa, xb = idx_to_elem[xi]
            ya, yb = idx_to_elem[yi]
            points_list.append([[xa, xb], [ya, yb]])

    return np.array(points_list, dtype=np.int64)

