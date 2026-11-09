import numpy as np
import time
import random
import heapq

def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2.
    Returns an array of points in N, where N is the Nikodym set.
    Uses baseline construction followed by priority-queue greedy pruning.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 15.0

    p = 7
    q = p * p  # 49
    alpha_sq = 3  # alpha^2 = 3 in F_7

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
    in_N = np.ones(total_points, dtype=bool)
    
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
            
    trace_arr = np.zeros(q, dtype=np.int64)
    norm_arr = np.zeros(q, dtype=np.int64)
    for i in range(q):
        a, b = elems[i]
        trace_arr[i] = (2 * a) % p
        norm_arr[i] = (a * a - b * b * alpha_sq) % p
    
    y_special_mask = np.zeros(q, dtype=bool)
    for ys in y_special:
        idx = elem_to_idx(ys[0], ys[1])
        y_special_mask[idx] = True
        
    sum_arr = (trace_arr[:, None] + norm_arr[None, :]) % p
    condition = (sum_arr == 0) & y_special_mask[None, :]
    condition_flat = condition.flatten()
    in_N[condition_flat] = False

    # Precompute line points
    line_points = np.zeros((total_points, num_dirs, q - 1), dtype=np.int64)
    
    one_idx = elem_to_idx(1, 0)
    zero_idx = elem_to_idx(0, 0)
    
    for d_idx in range(num_dirs):
        if d_idx < q:
            vx_idx = one_idx
            vy_idx = d_idx
        else:
            vx_idx = zero_idx
            vy_idx = one_idx
            
        txs = np.array([mul_table[t_idx, vx_idx] for t_idx in range(1, q)])
        tys = np.array([mul_table[t_idx, vy_idx] for t_idx in range(1, q)])
        
        x_indices = np.arange(total_points) // q
        y_indices = np.arange(total_points) % q
        
        nx = add_table[x_indices[:, None], txs[None, :]]
        ny = add_table[y_indices[:, None], tys[None, :]]
        line_points[:, d_idx, :] = nx * q + ny

    # covers[y, d] = line_points[y, d, :] (by symmetry)
    covers = line_points

    # Initialize validity
    is_valid_dir = np.zeros((total_points, num_dirs), dtype=bool)
    valid_count = np.zeros(total_points, dtype=np.int32)
    
    for d in range(num_dirs):
        mask = in_N[line_points[:, d, :]]
        valid_pts = np.all(mask, axis=1)
        is_valid_dir[:, d] = valid_pts
        valid_count += valid_pts.astype(np.int32)

    def compute_cost(y, work_in_N, work_is_valid_dir, work_valid_count):
        """Compute the cost of removing point y: number of (x,d) pairs where y is critical."""
        if not work_in_N[y]:
            return float('inf')
        
        cost = 0
        for d in range(num_dirs):
            xs = covers[y, d, :]
            valid_mask = work_is_valid_dir[xs, d]
            if np.any(valid_mask):
                counts = work_valid_count[xs]
                cost += np.sum(counts[valid_mask] == 1)
        return cost

    def priority_prune(init_in_N, init_is_valid_dir, init_valid_count, max_removals=500):
        """
        Priority-queue greedy: always remove the cheapest removable point.
        Returns (in_N, is_valid_dir, valid_count, complement_size).
        """
        work_in_N = init_in_N.copy()
        work_is_valid_dir = init_is_valid_dir.copy()
        work_valid_count = init_valid_count.copy()
        
        # Build initial priority queue
        pq = []
        for y in range(total_points):
            if work_in_N[y]:
                c = compute_cost(y, work_in_N, work_is_valid_dir, work_valid_count)
                if c < float('inf'):
                    heapq.heappush(pq, (c, y))
        
        removals = 0
        while pq and removals < max_removals:
            if time.time() > deadline - safety_margin:
                break
            
            cost, y = heapq.heappop(pq)
            
            # Check if y is still in N and cost is still valid
            if not work_in_N[y]:
                continue
            
            # Recompute cost to check if it's still accurate
            actual_cost = compute_cost(y, work_in_N, work_is_valid_dir, work_valid_count)
            if actual_cost != cost:
                # Cost changed, re-push with new cost
                if actual_cost < float('inf'):
                    heapq.heappush(pq, (actual_cost, y))
                continue
            
            if actual_cost > 0:
                # y is not removable (it's critical for some x)
                continue
            
            # Remove y
            work_in_N[y] = False
            removals += 1
            
            # Update validity for affected points
            for d in range(num_dirs):
                xs = covers[y, d, :]
                valid_mask = work_is_valid_dir[xs, d]
                if np.any(valid_mask):
                    work_valid_count[xs] -= valid_mask.astype(np.int32)
                    work_is_valid_dir[xs, d][valid_mask] = False
                    
                    # Points whose valid_count changed need their costs updated
                    # We'll let the lazy deletion handle stale entries
            
            # Push neighbors that might now be removable
            # For each d, for each x in covers[y, d], x's cost may have decreased
            # But we'll just let the lazy deletion handle it by rechecking when popped
        
        comp = np.sum(~work_in_N)
        return work_in_N, work_is_valid_dir, work_valid_count, comp

    # Best solution tracking
    best_complement = np.sum(~in_N)
    best_in_N = in_N.copy()
    best_is_valid_dir = is_valid_dir.copy()
    best_valid_count = valid_count.copy()

    # Strategy 1: Priority queue from baseline
    if time.time() < deadline - safety_margin:
        w_in_N, w_is_valid_dir, w_valid_count, comp = priority_prune(
            in_N, is_valid_dir, valid_count, max_removals=1000
        )
        if comp > best_complement:
            best_complement = comp
            best_in_N = w_in_N.copy()
            best_is_valid_dir = w_is_valid_dir.copy()
            best_valid_count = w_valid_count.copy()

    # Strategy 2: Sequential pruning then priority queue
    if time.time() < deadline - safety_margin:
        seq_in_N = in_N.copy()
        seq_is_valid_dir = is_valid_dir.copy()
        seq_valid_count = valid_count.copy()
        
        # Sequential pass: remove points in order if removable
        for y in range(total_points):
            if time.time() > deadline - safety_margin:
                break
            if not seq_in_N[y]:
                continue
            removable = True
            for d in range(num_dirs):
                xs = covers[y, d, :]
                valid_mask = seq_is_valid_dir[xs, d]
                if np.any(valid_mask):
                    counts = seq_valid_count[xs]
                    if np.any(counts[valid_mask] == 1):
                        removable = False
                        break
            if removable:
                seq_in_N[y] = False
                for d in range(num_dirs):
                    xs = covers[y, d, :]
                    valid_mask = seq_is_valid_dir[xs, d]
                    if np.any(valid_mask):
                        seq_valid_count[xs] -= valid_mask.astype(np.int32)
                        seq_is_valid_dir[xs, d][valid_mask] = False
        
        # Then apply priority queue on the result
        w_in_N, w_is_valid_dir, w_valid_count, comp = priority_prune(
            seq_in_N, seq_is_valid_dir, seq_valid_count, max_removals=500
        )
        if comp > best_complement:
            best_complement = comp
            best_in_N = w_in_N.copy()
            best_is_valid_dir = w_is_valid_dir.copy()
            best_valid_count = w_valid_count.copy()

    # Strategy 3: Multiple random restarts with priority queue
    if time.time() < deadline - safety_margin:
        for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]:
            if time.time() > deadline - safety_margin:
                break
            rng = random.Random(seed)
            
            # Random perturbation: start from best, randomly remove some points first
            pert_in_N = best_in_N.copy()
            pert_is_valid_dir = best_is_valid_dir.copy()
            pert_valid_count = best_valid_count.copy()
            
            # Try removing up to 50 random points that are removable
            candidates = [y for y in range(total_points) if pert_in_N[y]]
            rng.shuffle(candidates)
            for y in candidates[:50]:
                if time.time() > deadline - safety_margin:
                    break
                removable = True
                for d in range(num_dirs):
                    xs = covers[y, d, :]
                    valid_mask = pert_is_valid_dir[xs, d]
                    if np.any(valid_mask):
                        counts = pert_valid_count[xs]
                        if np.any(counts[valid_mask] == 1):
                            removable = False
                            break
                if removable:
                    pert_in_N[y] = False
                    for d in range(num_dirs):
                        xs = covers[y, d, :]
                        valid_mask = pert_is_valid_dir[xs, d]
                        if np.any(valid_mask):
                            pert_valid_count[xs] -= valid_mask.astype(np.int32)
                            pert_is_valid_dir[xs, d][valid_mask] = False
            
            # Apply priority queue
            w_in_N, w_is_valid_dir, w_valid_count, comp = priority_prune(
                pert_in_N, pert_is_valid_dir, pert_valid_count, max_removals=300
            )
            if comp > best_complement:
                best_complement = comp
                best_in_N = w_in_N.copy()
                best_is_valid_dir = w_is_valid_dir.copy()
                best_valid_count = w_valid_count.copy()

    # Final validity check
    if np.any(best_valid_count == 0):
        # Invalid, revert to baseline
        in_N = np.ones(total_points, dtype=bool)
        in_N[condition_flat] = False
        points = []
        for p_idx in range(total_points):
            if in_N[p_idx]:
                x_idx = p_idx // q
                y_idx = p_idx % q
                xa, xb = elems[x_idx]
                ya, yb = elems[y_idx]
                points.append([[xa, xb], [ya, yb]])
        return np.array(points, dtype=np.int64)

    # Extract points from best_in_N
    points = []
    for p_idx in range(total_points):
        if best_in_N[p_idx]:
            x_idx = p_idx // q
            y_idx = p_idx % q
            xa, xb = elems[x_idx]
            ya, yb = elems[y_idx]
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)
