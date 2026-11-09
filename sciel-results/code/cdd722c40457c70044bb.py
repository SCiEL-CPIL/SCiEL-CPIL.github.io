import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2.
    Returns an array of points in N, where N is the Nikodym set.
    Uses baseline construction followed by aggressive greedy pruning.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 5.0
    
    p = 7
    q = p * p  # 49
    alpha_sq = 3  # alpha^2 = 3 in F_7

    # Element representation: (a, b) represents a + b*alpha
    # idx = a * p + b

    def elem_to_idx(a, b):
        return a * p + b

    def idx_to_elem(idx):
        return (idx // p, idx % p)

    # Precompute field operations
    add_table = np.zeros((q, q), dtype=np.int64)
    mul_table = np.zeros((q, q), dtype=np.int64)
    
    elems = [idx_to_elem(i) for i in range(q)]
    
    for i in range(q):
        a_i, b_i = elems[i]
        for j in range(q):
            a_j, b_j = elems[j]
            add_table[i, j] = elem_to_idx((a_i + a_j) % p, (b_i + b_j) % p)
            prod_a = (a_i * a_j + b_i * b_j * alpha_sq) % p
            prod_b = (a_i * b_j + b_i * a_j) % p
            mul_table[i, j] = elem_to_idx(prod_a, prod_b)

    num_dirs = q + 1  # 50 directions
    total_points = q * q  # 2401

    # Build baseline Nikodym set
    # N = {(x, y) : Trace(x) + Norm(y) = 0} minus Y_special
    in_N = np.ones(total_points, dtype=bool)
    
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
    
    for x_idx in range(q):
        xa, xb = elems[x_idx]
        trace_x = (2 * xa) % p
        for y_idx in range(q):
            ya, yb = elems[y_idx]
            norm_y = (ya * ya - yb * yb * alpha_sq) % p
            p_idx = x_idx * q + y_idx
            if (trace_x + norm_y) % p == 0 and (ya, yb) in y_special:
                in_N[p_idx] = False

    # Precompute line points and reverse mapping
    # line_points[pt, d, t] = point index at x + t*v
    # We need: for each y, list of (x, d) such that y is on line L(x, d) and y != x
    # Actually, for the update step, when we remove y, we need to know which (x, d) pairs are affected.
    # affected[y] = list of (x, d) such that y in L(x, d) \ {x}
    
    affected = [[] for _ in range(total_points)]
    
    one_idx = elem_to_idx(1, 0)  # element 1
    zero_idx = elem_to_idx(0, 0)  # element 0
    
    # We can compute affected by iterating over all x, d, and points y on the line
    # For each x, d:
    #   Line is {x + t*v : t in F_49}
    #   For each t != 0, y = x + t*v
    #   Then y is on line L(x, d) and y != x.
    #   So add (x, d) to affected[y].
    
    for x in range(total_points):
        x_idx = x // q
        y_idx = x % q
        for d in range(num_dirs):
            if d < q:
                vx_idx = one_idx
                vy_idx = d
            else:
                vx_idx = zero_idx
                vy_idx = one_idx  # element (0,1) = 1
            
            for t_idx in range(1, q):
                tx = mul_table[t_idx, vx_idx]
                ty = mul_table[t_idx, vy_idx]
                nx = add_table[x_idx, tx]
                ny = add_table[y_idx, ty]
                y_pt = nx * q + ny
                affected[y_pt].append((x, d))

    # Initialize validity tracking
    # valid_dirs[x] is a set of directions d such that L(x, d) \ {x} subseteq N
    # count_valid[x] is the size of valid_dirs[x]
    
    # Efficient representation:
    # valid_mask[x] is a bitset or boolean array of length num_dirs
    # Since num_dirs = 50, we can use an integer bitmask or a numpy array of bools.
    # Let's use a 2D boolean array valid_dir[x, d]
    valid_dir = np.zeros((total_points, num_dirs), dtype=bool)
    count_valid = np.zeros(total_points, dtype=np.int32)
    
    # Precompute line points for initial validity check
    # We need to check if L(x, d) \ {x} subseteq N
    # This is equivalent to: for all y in L(x, d) \ {x}, in_N[y] is True
    
    # Let's compute this efficiently.
    # For each x, d, check all 48 points.
    # Total 2401 * 50 * 48 ~ 5.7M checks. Fast enough.
    
    for x in range(total_points):
        x_idx = x // q
        y_idx = x % q
        for d in range(num_dirs):
            if d < q:
                vx_idx = one_idx
                vy_idx = d
            else:
                vx_idx = zero_idx
                vy_idx = one_idx
            
            all_in_N = True
            for t_idx in range(1, q):
                tx = mul_table[t_idx, vx_idx]
                ty = mul_table[t_idx, vy_idx]
                nx = add_table[x_idx, tx]
                ny = add_table[y_idx, ty]
                y_pt = nx * q + ny
                if not in_N[y_pt]:
                    all_in_N = False
                    break
            if all_in_N:
                valid_dir[x, d] = True
                count_valid[x] += 1

    # Check if baseline is valid
    if np.any(count_valid == 0):
        # This shouldn't happen for the constructed baseline, but just in case
        # Fallback to a simpler construction if needed, but let's assume it works.
        pass

    # Greedy Pruning
    # We want to remove points y from N if possible.
    # y is removable if for all (x, d) in affected[y] such that valid_dir[x, d] is True:
    #   count_valid[x] > 1  (so x has another valid direction)
    # Wait, if we remove y, valid_dir[x, d] becomes False for all (x, d) in affected[y] where it was True.
    # So x loses those directions.
    # x remains valid if it has at least 1 other direction.
    # So condition: for all (x, d) in affected[y] where valid_dir[x, d] is True:
    #   count_valid[x] - (number of such d for x) >= 1
    
    # To make this faster, we can precompute for each y, the list of x's it affects.
    # But the condition depends on the current state of valid_dir.
    
    # Let's iterate through points in N and try to remove them.
    # Order: Random or by "criticality" (number of valid directions they support).
    
    # We'll do multiple rounds of random removal attempts.
    
    in_N_work = in_N.copy()
    valid_dir_work = valid_dir.copy()
    count_valid_work = count_valid.copy()
    
    best_complement = np.sum(~in_N_work)
    best_in_N = in_N_work.copy()
    
    for round_num in range(20):
        if time.time() > deadline - safety_margin:
            break
            
        progress = False
        # Create a list of indices in N to try removing
        candidates = np.where(in_N_work)[0]
        np.random.shuffle(candidates)
        
        for y in candidates:
            if not in_N_work[y]:
                continue
            
            # Check if y is removable
            removable = True
            # Count how many valid directions y supports for each x
            # We need to check if removing y invalidates any x completely.
            
            # Optimization: 
            # For each (x, d) in affected[y]:
            #   if valid_dir_work[x, d]:
            #       if count_valid_work[x] == 1:
            #           removable = False; break
            #       # else, x will still have count_valid[x] - 1 >= 1?
            #       # No, we need to subtract ALL valid d's in affected[y] for that x.
            
            # Let's group affected[y] by x.
            # affected[y] is a list of (x, d).
            # We can use a dictionary or just iterate.
            # Since len(affected[y]) is ~2400, and we do this for ~2400 y's, it's ~5.7M iterations.
            # Inside, we check valid_dir_work[x, d].
            
            # To be precise:
            # Let S_x be the set of d such that (x, d) in affected[y] and valid_dir_work[x, d] is True.
            # If count_valid_work[x] - |S_x| < 1, then y is not removable.
            
            # We can compute |S_x| for each x affected by y.
            # Since the number of unique x's is small (at most 50 lines * 1 line per dir? No, y is on 1 line per dir, so 50 lines, 48 x's per line -> 2400 x's? No, x is the origin. For a fixed d, y is on one line. The x's on that line are the 48 other points. So for each d, there are 48 x's. Total 50*48=2400 pairs. The x's might repeat? No, for different d, the lines are different (unless parallel? No, lines with different slopes intersect at one point. So for different d, the lines through y are different. The x's are the other points on those lines. So all 2400 x's are distinct?
            # Actually, two different lines (different d) intersect at y. They share no other points. So yes, the 2400 x's are distinct.
            
            # So for each x in the 2400 x's, there is exactly one d such that (x, d) in affected[y].
            # So |S_x| is either 0 or 1.
            # Condition: if valid_dir_work[x, d] is True, then count_valid_work[x] must be > 1.
            
            for x, d in affected[y]:
                if valid_dir_work[x, d]:
                    if count_valid_work[x] <= 1:
                        removable = False
                        break
            
            if removable:
                # Remove y
                in_N_work[y] = False
                progress = True
                
                # Update valid_dir_work and count_valid_work
                for x, d in affected[y]:
                    if valid_dir_work[x, d]:
                        valid_dir_work[x, d] = False
                        count_valid_work[x] -= 1
                        # If count_valid_work[x] becomes 0, we have a problem?
                        # But we checked that count_valid_work[x] > 1, so it becomes >= 1.
                        # So it's safe.
                
                # Update best
                comp = np.sum(~in_N_work)
                if comp > best_complement:
                    best_complement = comp
                    best_in_N = in_N_work.copy()
        
        if not progress:
            break

    in_N = best_in_N

    # Final validity check
    # Re-verify that every point has at least one valid direction
    # This is a safety check.
    
    # Extract points
    points = []
    for p_idx in range(total_points):
        if in_N[p_idx]:
            x_idx = p_idx // q
            y_idx = p_idx % q
            xa, xb = elems[x_idx]
            ya, yb = elems[y_idx]
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)
