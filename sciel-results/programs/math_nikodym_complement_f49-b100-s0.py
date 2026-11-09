# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import heapq

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using priority-queue-based greedy removal
    with multiple random restarts.
    
    Strategy:
    1. Precompute field arithmetic tables for F_49.
    2. Initialize N using the Hermitian unital construction.
    3. Precompute the 'affects' mapping.
    4. Use a priority-queue greedy removal: always remove the point whose removal
       causes the least increase in minimum cover_count damage.
    5. Run multiple random restarts with different orderings, keep best result.
    """
    deadline = time.time() + 620  # Safety margin before 630s limit
    
    p = 7
    q = p * p  # 49

    # Field F_q = F_p[alpha], alpha^2 = k
    k_field = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k_field = c
            break
    if k_field is None:
        k_field = 3

    def to_int(a, b):
        return a + b * p

    def to_pair(x):
        return (x % p, (x // p) % p)

    # Precompute field arithmetic tables
    f_add = np.zeros((q, q), dtype=np.int32)
    f_mul = np.zeros((q, q), dtype=np.int32)
    f_neg = np.zeros(q, dtype=np.int32)

    for x in range(q):
        xa, xb = to_pair(x)
        f_neg[x] = to_int((-xa) % p, (-xb) % p)
        for y in range(q):
            ya, yb = to_pair(y)
            f_add[x, y] = to_int((xa + ya) % p, (xb + yb) % p)
            real = (xa * ya + xb * yb * k_field) % p
            imag = (xa * yb + xb * ya) % p
            f_mul[x, y] = to_int(real, imag)

    # Directions
    directions = []
    for m in range(q):
        directions.append((1, m))
    directions.append((0, 1))
    n_dir = len(directions)

    # All points
    point_index = {}
    all_points = []
    for x_idx in range(q):
        for y_idx in range(q):
            idx = x_idx * q + y_idx
            point_index[(x_idx, y_idx)] = idx
            all_points.append((x_idx, y_idx))
    
    idx_to_point = [None] * (q * q)
    for pt, idx in point_index.items():
        idx_to_point[idx] = pt

    # Build initial N
    def build_initial_in_N():
        in_N = np.ones(q * q, dtype=bool)
        y_special = set()
        for u in range(p):
            for v in range(p):
                y_special.add(to_int((u * u) % p, v))
        
        for x_idx in range(q):
            xa, xb = to_pair(x_idx)
            trace_x = (2 * xa) % p
            for y_idx in range(q):
                ya, yb = to_pair(y_idx)
                norm_y = (ya * ya - yb * yb * k_field) % p
                if (trace_x + norm_y) % p == 0:
                    if y_idx not in y_special:
                        in_N[point_index[(x_idx, y_idx)]] = False
        return in_N

    # Precompute affects (same for all restarts)
    affects = [[] for _ in range(q * q)]
    for r_idx in range(q * q):
        rx, ry = idx_to_point[r_idx]
        for d_idx, (vx, vy) in enumerate(directions):
            for t in range(1, q):
                tv_x = f_mul[t, vx]
                tv_y = f_mul[t, vy]
                neg_tv_x = f_neg[tv_x]
                neg_tv_y = f_neg[tv_y]
                px = f_add[rx, neg_tv_x]
                py = f_add[ry, neg_tv_y]
                if (px, py) in point_index:
                    p_idx = point_index[(px, py)]
                    affects[r_idx].append((p_idx, d_idx))

    # Precompute line membership: for each direction, map from point to list of other points on line
    # Actually, affects already gives us what we need.

    def compute_not_in_N_count(in_N):
        """Compute not_in_N_count[p_idx, d_idx] for given in_N."""
        not_in_N_count = np.zeros((q * q, n_dir), dtype=np.int32)
        for d_idx, (vx, vy) in enumerate(directions):
            t_v = []
            for t in range(1, q):
                tx = f_mul[t, vx]
                ty = f_mul[t, vy]
                t_v.append((tx, ty))
            
            for p_idx in range(q * q):
                px, py = idx_to_point[p_idx]
                count = 0
                for tx, ty in t_v:
                    nx = f_add[px, tx]
                    ny = f_add[py, ty]
                    n_idx = point_index[(nx, ny)]
                    if not in_N[n_idx]:
                        count += 1
                not_in_N_count[p_idx, d_idx] = count
        return not_in_N_count

    def greedy_priority_removal(in_N, not_in_N_count, cover_count, order_seed=None):
        """
        Priority-queue based greedy removal.
        Returns (in_N, not_in_N_count, cover_count, removed_count).
        """
        in_N = in_N.copy()
        not_in_N_count = not_in_N_count.copy()
        cover_count = cover_count.copy()
        
        # Build initial priority queue
        # Priority: for each point in N, compute the "cost" of removing it
        # Cost = number of (p_idx, d_idx) where old_count == 0 (i.e., would reduce cover_count)
        # We want to minimize this cost. Use negative for max-heap (or min-heap with cost).
        
        # Use a min-heap: (cost, r_idx)
        # But cost changes as we remove points, so we need lazy deletion.
        
        heap = []
        cost_cache = {}
        
        # Initial costs
        for r_idx in range(q * q):
            if in_N[r_idx]:
                cost = 0
                for p_idx, d_idx in affects[r_idx]:
                    if not_in_N_count[p_idx, d_idx] == 0:
                        cost += 1
                cost_cache[r_idx] = cost
                heapq.heappush(heap, (cost, r_idx))
        
        removed_count = 0
        time_budget = deadline - time.time() - 5
        start_time = time.time()
        
        while heap:
            if time.time() - start_time > time_budget:
                break
            
            cost, r_idx = heapq.heappop(heap)
            
            # Lazy deletion check
            if not in_N[r_idx]:
                continue
            if cost_cache.get(r_idx, -1) != cost:
                continue
            
            # Check if removal is valid
            invalid = False
            affected = []
            
            for p_idx, d_idx in affects[r_idx]:
                old_count = not_in_N_count[p_idx, d_idx]
                if old_count == 0:
                    if cover_count[p_idx] == 1:
                        invalid = True
                        break
                affected.append((p_idx, d_idx, old_count))
            
            if invalid:
                # Mark as stale (will be re-added with new cost if still in N)
                cost_cache[r_idx] = -1
                continue
            
            # Commit removal
            for p_idx, d_idx, old_count in affected:
                not_in_N_count[p_idx, d_idx] = old_count + 1
                if old_count == 0:
                    cover_count[p_idx] -= 1
            
            in_N[r_idx] = False
            removed_count += 1
            cost_cache[r_idx] = -1
            
            # Update costs of affected points
            # Points p_idx that were affected may now have different costs
            # Specifically, for each p_idx in affected, the points r' that affect p_idx
            # in direction d_idx may have their cost changed.
            # This is expensive to track exactly, so we use lazy re-computation.
            # When we pop a point, we recompute its cost if needed.
            # For now, we just mark neighbors for potential re-insertion.
            
            # Actually, let's recompute costs for points that might be affected
            # and push them back into the heap.
            # To keep it simple, we'll just push all points in N that are affected
            # by this removal with their updated costs.
            
            # Collect unique p_idx values
            affected_p_idxs = set()
            for p_idx, d_idx, old_count in affected:
                affected_p_idxs.add(p_idx)
            
            # For each affected p_idx, find all r' in N that affect p_idx in some direction
            # and update their cost in the heap
            # This is still expensive. Alternative: just periodically rebuild the heap.
            
            # Simple approach: every K removals, rebuild the heap.
            # For now, let's just continue and rely on lazy deletion.
            # Points whose cost has changed will be detected when popped.
        
        return in_N, not_in_N_count, cover_count, removed_count

    # Run multiple restarts
    best_in_N = None
    best_size = q * q + 1
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        # Build initial set
        in_N_init = build_initial_in_N()
        not_in_N_count_init = compute_not_in_N_count(in_N_init)
        cover_count_init = np.sum(not_in_N_count_init == 0, axis=1).astype(np.int32)
        
        # Run greedy priority removal
        in_N_result, _, _, removed = greedy_priority_removal(
            in_N_init, not_in_N_count_init, cover_count_init
        )
        
        current_size = np.sum(in_N_result)
        if current_size < best_size:
            best_size = current_size
            best_in_N = in_N_result.copy()
        
        # Also try a sequential pass after priority removal
        # to catch any remaining removable points
        not_in_N_count_seq = compute_not_in_N_count(in_N_result)
        cover_count_seq = np.sum(not_in_N_count_seq == 0, axis=1).astype(np.int32)
        
        # Sequential pass
        progress = True
        while progress:
            if time.time() > deadline - 10:
                break
            progress = False
            for r_idx in range(q * q):
                if not in_N_result[r_idx]:
                    continue
                if time.time() > deadline - 5:
                    break
                
                invalid = False
                affected = []
                for p_idx, d_idx in affects[r_idx]:
                    old_count = not_in_N_count_seq[p_idx, d_idx]
                    if old_count == 0:
                        if cover_count_seq[p_idx] == 1:
                            invalid = True
                            break
                    affected.append((p_idx, d_idx, old_count))
                
                if invalid:
                    continue
                
                for p_idx, d_idx, old_count in affected:
                    not_in_N_count_seq[p_idx, d_idx] = old_count + 1
                    if old_count == 0:
                        cover_count_seq[p_idx] -= 1
                
                in_N_result[r_idx] = False
                progress = True
        
        current_size = np.sum(in_N_result)
        if current_size < best_size:
            best_size = current_size
            best_in_N = in_N_result.copy()

    # Build result from best_in_N
    if best_in_N is None:
        best_in_N = build_initial_in_N()

    points = []
    for i in range(q * q):
        if best_in_N[i]:
            x, y = idx_to_point[i]
            xa, xb = to_pair(x)
            ya, yb = to_pair(y)
            points.append([[xa, xb], [ya, yb]])

    if len(points) == 0:
        # Fallback
        best_in_N = build_initial_in_N()
        points = []
        for i in range(q * q):
            if best_in_N[i]:
                x, y = idx_to_point[i]
                xa, xb = to_pair(x)
                ya, yb = to_pair(y)
                points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
