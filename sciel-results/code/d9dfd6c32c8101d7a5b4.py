import numpy as np
import time
import random
import heapq

def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2.
    Returns an array of points in N, where N is the Nikodym set.
    Uses baseline construction followed by aggressive greedy pruning.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 10.0  # Keep a larger safety margin to ensure we return

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
    # We want to maximize complement, so start with all True (in N) and remove some.
    # Actually, the definition of Nikodym set N requires that for every x, there is a line L_x containing x such that L_x \ {x} subset N.
    # A standard construction for F_q^2 is to take a set of lines and include all points on them? No, that's large.
    # The provided code uses: in_N[p_idx] = False if (trace_x + norm_y) == 0 and (ya, yb) in y_special.
    # This means in_N is True everywhere EXCEPT for those specific points.
    # So N is almost the entire space.
    
    in_N = np.ones(total_points, dtype=bool)
    
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
            
    # Vectorize baseline construction
    # x_idx ranges 0..48, y_idx ranges 0..48
    # p_idx = x_idx * q + y_idx
    # Condition: (Trace(x) + Norm(y)) % p == 0 AND (ya, yb) in y_special
    
    # Precompute trace and norm for all elements
    trace_arr = np.zeros(q, dtype=np.int64)
    norm_arr = np.zeros(q, dtype=np.int64)
    for i in range(q):
        a, b = elems[i]
        trace_arr[i] = (2 * a) % p
        norm_arr[i] = (a * a - b * b * alpha_sq) % p
    
    # Create boolean mask for y_special
    y_special_mask = np.zeros(q, dtype=bool)
    for ys in y_special:
        idx = elem_to_idx(ys[0], ys[1])
        y_special_mask[idx] = True
        
    # Broadcast to check condition for all pairs
    # trace_arr[:, None] + norm_arr[None, :]
    sum_arr = (trace_arr[:, None] + norm_arr[None, :]) % p
    condition = (sum_arr == 0) & y_special_mask[None, :]
    
    # Flatten to 1D
    condition_flat = condition.flatten()
    in_N[condition_flat] = False

    # Precompute line points
    # line_points[pt, d, t_idx] = index of point at distance t_idx along direction d from pt
    # We only need t_idx 1..48 for the "other than x" part
    line_points = np.zeros((total_points, num_dirs, q - 1), dtype=np.int64)
    
    one_idx = elem_to_idx(1, 0)  # element 1
    zero_idx = elem_to_idx(0, 0)  # element 0
    
    # Directions
    # d < q: v = (1, d)
    # d == q: v = (0, 1)
    
    # We can compute this faster by iterating directions and using vectorized additions
    # For a fixed direction d, the line through pt is pt + t*v_d.
    # Since we need line_points for ALL pt, we can compute offsets.
    
    # Let's stick to the loop but optimize the inner part if possible.
    # Actually, 2401 * 50 * 48 is ~5.7 million operations. This is fine in Python if not heavily nested with logic.
    # But we can precompute the "offset" indices for each direction.
    
    # offset[d, t] = index of element t * v_d? No, v_d is a pair of field elements.
    # Let's just use the loop, it's fast enough.
    
    for d_idx in range(num_dirs):
        if d_idx < q:
            vx_idx = one_idx
            vy_idx = d_idx
        else:
            vx_idx = zero_idx
            vy_idx = one_idx  # element (0,1) = 1
            
        # Precompute t*v for all t in 1..48
        txs = np.array([mul_table[t_idx, vx_idx] for t_idx in range(1, q)])
        tys = np.array([mul_table[t_idx, vy_idx] for t_idx in range(1, q)])
        
        # For each point pt, line_points[pt, d_idx, :] = pt + t*v
        # pt is (x, y). x_idx = pt // q, y_idx = pt % q
        # We can vectorize over pt?
        # x_indices = np.arange(total_points) // q
        # y_indices = np.arange(total_points) % q
        # new_x = add_table[x_indices[:, None], txs[None, :]] -> shape (2401, 48)
        # new_y = add_table[y_indices[:, None], tys[None, :]] -> shape (2401, 48)
        # line_points[:, d_idx, :] = new_x * q + new_y
        
        x_indices = np.arange(total_points) // q
        y_indices = np.arange(total_points) % q
        
        # add_table is (49, 49). x_indices is (2401,), txs is (48,)
        # We need add_table[x_indices[:, None], txs[None, :]]
        # This creates a (2401, 48) array.
        
        # To save memory/time, we can do this in chunks or just let numpy handle it.
        # 2401 * 48 * 8 bytes * 2 (x and y) ~ 18 MB. Acceptable.
        
        nx = add_table[x_indices[:, None], txs[None, :]]
        ny = add_table[y_indices[:, None], tys[None, :]]
        line_points[:, d_idx, :] = nx * q + ny

    # Initialize validity tracking
    # is_valid_dir[pt, d] = True if all points on line (excluding pt) are in N
    # valid_count[pt] = number of valid directions for pt
    
    is_valid_dir = np.zeros((total_points, num_dirs), dtype=bool)
    valid_count = np.zeros(total_points, dtype=np.int32)
    
    for d in range(num_dirs):
        mask = in_N[line_points[:, d, :]]
        valid_pts = np.all(mask, axis=1)
        is_valid_dir[:, d] = valid_pts
        valid_count += valid_pts.astype(np.int32)

    # covers[y, d] is the set of x such that y is on line x-d.
    # As derived, this is exactly line_points[y, d, :].
    covers = line_points

    # State for the main greedy loop
    in_N = in_N.copy()
    is_valid_dir = is_valid_dir.copy()
    valid_count = valid_count.copy()
    
    # Priority queue for removable points
    # (priority, y)
    # Lower priority is better.
    # Priority can be the number of "critical" constraints y satisfies.
    # If y is removed, it breaks constraints for x where y is on line x-d and x has no other valid line.
    # Actually, we want to remove y if it doesn't break any constraints.
    # So priority 0 means "safe to remove".
    
    # Let's define a function to compute the "cost" of removing y.
    # Cost = number of pairs (x, d) such that y is on line x-d AND is_valid_dir[x, d] is True AND valid_count[x] == 1.
    # If cost > 0, removing y will make some x invalid (valid_count becomes 0).
    # So we only remove if cost == 0.
    
    # To use a priority queue, we can just push all points with cost 0.
    # When we remove a point, costs of other points might change (increase? no, removing a point from N only decreases valid_counts, so it can only make points LESS safe to remove? 
    # Wait. If I remove y, valid_count[x] decreases for some x. 
    # This means valid_count[x] might drop from 2 to 1. 
    # This makes x MORE critical. 
    # So if z was on line x-d, z's cost increases (from 0 to 1 if x was the only other support? No).
    # Cost of z: # of x such that z on line x-d, valid[x,d]=True, valid_count[x]==1.
    # If I remove y, valid_count[x] might drop. 
    # If valid_count[x] drops to 1, and z is on line x-d, then z's cost increases.
    # So costs are monotonically non-decreasing? 
    # No, if valid_count[x] drops to 0, x is invalid. But we stop if any x becomes invalid.
    # So we must ensure we never create an invalid state.
    # Therefore, we can only remove y if cost(y) == 0.
    # After removing y, some x might have valid_count drop from 2 to 1.
    # This increases the cost of points z on lines through x.
    # So we need to update costs.
    
    # However, a simpler approach: 
    # Just maintain a list of "safe" points (cost 0).
    # Iterate through them. Remove one.
    # Update valid_counts.
    # Recompute costs for affected points?
    
    # Let's try a simpler structural change first:
    # Instead of static orders, use a dynamic priority queue where we always pick the point with the lowest "impact".
    # Impact = number of (x, d) pairs where y is on line x-d and is_valid_dir[x, d] is True.
    # This is the number of constraints y supports.
    # Removing a point with low support is "cheap" in terms of breaking valid lines, but we MUST ensure we don't break the last line for any x.
    
    # Let's stick to the "Cost" definition: 
    # Cost(y) = # of x such that y is on line x-d, is_valid_dir[x, d] == True, and valid_count[x] == 1.
    # If Cost(y) == 0, y is removable.
    
    # We will maintain a heap of (cost, y).
    # Initially compute cost for all y in N.
    
    def compute_cost(y):
        cost = 0
        for d in range(num_dirs):
            xs = covers[y, d, :]
            valid_mask = is_valid_dir[xs, d]
            if np.any(valid_mask):
                counts = valid_count[xs]
                cost += np.sum(counts[valid_mask] == 1)
        return cost

    # Initialize heap
    heap = []
    for y in range(total_points):
        if in_N[y]:
            c = compute_cost(y)
            if c == 0:
                heapq.heappush(heap, (c, y))

    # Greedy loop
    while heap:
        if time.time() > deadline - safety_margin:
            break
        
        cost, y = heapq.heappop(heap)
        
        # Check if y is still in N and still has cost 0
        # (Cost might have increased since it was pushed)
        if not in_N[y]:
            continue
        
        current_cost = compute_cost(y)
        if current_cost != 0:
            # It became critical. Push it back? Or just skip?
            # If we skip, it might become safe again? No, costs only increase (valid_counts decrease).
            # So if it was 0 and now > 0, it will never be 0 again.
            # So we can just discard it.
            continue
        
        # Remove y
        in_N[y] = False
        
        # Update valid_count and is_valid_dir for affected x
        affected_x = set()
        for d in range(num_dirs):
            xs = covers[y, d, :]
            valid_mask = is_valid_dir[xs, d]
            if np.any(valid_mask):
                # These x's will lose this valid direction
                # valid_count[x] -= 1
                # is_valid_dir[x, d] = False
                
                # We need to update the heap for points z that are on lines through these x's?
                # Actually, we need to update the cost of points z that might be affected by the change in valid_count[x].
                
                # Which z's cost depends on x?
                # z's cost depends on x if z is on line x-d'.
                # So for each x in xs[valid_mask], and each d', we need to check z in covers[x, d', :].
                # This is getting expensive.
                
                # Alternative:
                # Just re-compute cost for all points in N? No, too slow.
                
                # Let's optimize the update.
                # When we remove y:
                # 1. For each d, for each x in covers[y, d] where is_valid_dir[x, d] was True:
                #    - is_valid_dir[x, d] becomes False.
                #    - valid_count[x] decreases by 1.
                #    - If valid_count[x] becomes 1, then x is now "critical".
                #      Any point z on a line through x (with any direction d' where is_valid_dir[x, d'] is True) 
                #      now has its cost increased by 1 (if it wasn't already counted? No, cost counts x if valid_count[x]==1).
                #      Wait, cost(z) = # of x such that z on line x-d, valid[x,d], count[x]==1.
                #      So if x becomes critical (count 1), then for all z on lines through x (where valid[x,d'] is true), cost(z) increases by 1.
                
                # So we need to:
                # 1. Identify x's that become critical (count drops to 1).
                # 2. For each such x, identify all z on lines through x (where valid[x,d'] is true).
                # 3. Increase cost(z) by 1.
                # 4. If cost(z) was 0, it becomes 1, so it's no longer safe.
                #    We don't need to remove it from the heap, just check cost when popping.
                
                # This seems manageable.
                
                # Let's implement this update logic.
                
                # First, find x's that become critical.
                # We need to know which x's had count 2 and now have count 1.
                # Or just check after update.
                
                # Let's collect x's to update.
                # For each d:
                #   xs_d = covers[y, d, :]
                #   valid_mask = is_valid_dir[xs_d, d]
                #   critical_xs = xs_d[valid_mask & (valid_count[xs_d] == 2)] # These will become 1
                #   # Also, if count was 1, they become 0 -> Invalid! 
                #   # But we checked cost(y)==0, so no x with count 1 has y on its valid line.
                #   # So valid_count[xs_d[valid_mask]] >= 2.
                
                # Let's do the update.
                
                for x in xs[valid_mask]:
                    # x loses direction d
                    # Check if x becomes critical
                    old_count = valid_count[x]
                    valid_count[x] = old_count - 1
                    is_valid_dir[x, d] = False
                    
                    if old_count == 2:
                        # x becomes critical (count 1)
                        # Find all z on lines through x where is_valid_dir[x, d'] is True
                        # For each such z, cost(z) increases by 1.
                        for d_prime in range(num_dirs):
                            if is_valid_dir[x, d_prime]:
                                zs = covers[x, d_prime, :]
                                # For each z in zs, if in_N[z], cost[z] += 1
                                # We don't store cost in an array, we compute it on the fly?
                                # No, we need to update the heap or at least know when to skip.
                                # If we don't store cost, we can't update it.
                                # So we MUST store cost in an array.
                                
                                # Let's add a cost array.
                                pass

    # This dynamic update is complex. Let's simplify.
    # Since the graph is small (2401 nodes), maybe we can just re-compute costs for all points after each removal?
    # 2401 * 50 * 48 ~ 5.7M ops per removal.
    # If we remove 1000 points, that's 5.7B ops. Too slow in Python.
    
    # How about we just use the static "criticality" order but with multiple random restarts and more rounds?
    # The current code does this.
    
    # Let's try a different structural change:
    # Use a "Maximal Independent Set" style approach on the conflict graph?
    # Conflict graph: Edge (y1, y2) if removing y1 makes y2 critical? No.
    
    # Let's try a simpler heuristic:
    # Randomly shuffle the order of removal, but check validity strictly.
    # The current code does this.
    
    # What if we prioritize points that are on FEWER lines?
    # Every point is on 50 lines.
    
    # What if we prioritize points that are "central"?
    
    # Let's try to improve the baseline construction.
    # The baseline is N = { (x,y) : Trace(x) + Norm(y) != 0 OR (ya,yb) not in Y_special }.
    # Actually, the code sets in_N[condition] = False.
    # Condition: (Trace(x) + Norm(y) == 0) AND (ya,yb) in Y_special.
    # So N is the complement of this set.
    
    # Can we find a smaller N?
    # The size of the removed set is |{(x,y) : T(x)+N(y)=0, y in Y_special}|.
    # Y_special has p*p = 49 points? 
    # y_special = {(u^2, v) : u,v in 0..6}. Size 49.
    # For each y in Y_special, how many x satisfy T(x) = -N(y)?
    # T(x) is a surjective map from F_49 to F_7.
    # So for each y, there are 49/7 = 7 x's.
    # Total removed = 49 * 7 = 343.
    # So |N| = 2401 - 343 = 2058.
    # Complement = 343.
    # Score = 343 / 2401? No, score is proportional to complement.
    # Seed score 1.0 corresponds to some complement.
    # Current score 1.3333.
    
    # If we can remove MORE points from N, the complement grows.
    # The pruning algorithm removes points from N.
    # So it increases the complement.
    
    # The bottleneck is the pruning efficiency.
    
    # Let's try to speed up the pruning by using a better data structure for "is y on line x-d".
    # We already have covers.
    
    # Let's try to implement the dynamic cost update correctly.
    
    # State:
    # in_N: bool[2401]
    # is_valid_dir: bool[2401, 50]
    # valid_count: int[2401]
    # cost: int[2401]
    # heap: list of (cost, y)
    
    # Init cost
    cost = np.zeros(total_points, dtype=np.int32)
    for y in range(total_points):
        if in_N[y]:
            cost[y] = compute_cost(y)
            if cost[y] == 0:
                heapq.heappush(heap, (0, y))

    # Loop
    while heap:
        if time.time() > deadline - safety_margin:
            break
            
        c, y = heapq.heappop(heap)
        
        if not in_N[y]:
            continue
            
        if cost[y] != 0:
            continue
            
        # Remove y
        in_N[y] = False
        
        # Update
        # For each d:
        #   xs = covers[y, d, :]
        #   valid_mask = is_valid_dir[xs, d]
        #   For each x in xs[valid_mask]:
        #      is_valid_dir[x, d] = False
        #      valid_count[x] -= 1
        #      If valid_count[x] became 1 (was 2):
        #          For each d':
        #             If is_valid_dir[x, d']:
        #                zs = covers[x, d', :]
        #                For each z in zs:
        #                   If in_N[z]:
        #                      cost[z] += 1
        #                      If cost[z] was 0, it's now 1. (No heap update needed, just check on pop)
        
        for d in range(num_dirs):
            xs = covers[y, d, :]
            valid_mask = is_valid_dir[xs, d]
            if not np.any(valid_mask):
                continue
                
            # Get indices of x that are valid for this direction
            x_indices = np.where(valid_mask)[0]
            
            for x_idx in x_indices:
                x = xs[x_idx]
                
                # Update x
                is_valid_dir[x, d] = False
                old_count = valid_count[x]
                valid_count[x] = old_count - 1
                
                if old_count == 2:
                    # x becomes critical
                    # Update costs of z on lines through x
                    for d_prime in range(num_dirs):
                        if is_valid_dir[x, d_prime]:
                            zs = covers[x, d_prime, :]
                            # For each z in zs, if in_N[z], cost[z] += 1
                            # Vectorize?
                            # mask_z = in_N[zs]
                            # cost[zs[mask_z]] += 1
                            mask_z = in_N[zs]
                            if np.any(mask_z):
                                cost[zs[mask_z]] += 1

    # After loop, extract points from in_N
    points = []
    for p_idx in range(total_points):
        if in_N[p_idx]:
            x_idx = p_idx // q
            y_idx = p_idx % q
            xa, xb = elems[x_idx]
            ya, yb = elems[y_idx]
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)
