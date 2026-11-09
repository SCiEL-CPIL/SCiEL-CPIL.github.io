import numpy as np
import time
import heapq

def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2.
    Uses a MODEL-AND-OPTIMIZE approach with a priority-queue based greedy pruning.
    The priority of a point is the number of valid constraints it supports.
    Removing the point with the lowest priority minimizes the damage to the set's validity.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin before 600s
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

    # Precompute line points: for each point and direction, the other q-1 points on the line
    line_points = np.zeros((total_points, num_dirs, q - 1), dtype=np.int64)

    one_idx = elem_to_idx(1, 0)  # element 1
    zero_idx = elem_to_idx(0, 0)  # element 0

    for d_idx in range(num_dirs):
        if d_idx < q:
            vx_idx = one_idx
            vy_idx = d_idx
        else:
            vx_idx = zero_idx
            vy_idx = one_idx  # element (0,1) = 1

        for pt in range(total_points):
            x_idx = pt // q
            y_idx = pt % q
            for t_idx in range(1, q):
                tx = mul_table[t_idx, vx_idx]
                ty = mul_table[t_idx, vy_idx]
                nx = add_table[x_idx, tx]
                ny = add_table[y_idx, ty]
                line_points[pt, d_idx, t_idx - 1] = nx * q + ny

    # Precompute membership: mem[y, d, x] = True if y is on line through x in direction d
    # This is symmetric: y is on line(x,d) iff x is on line(y,d)
    # We can use line_points directly for this check, but a precomputed boolean matrix is faster.
    # However, storing mem (2401 x 50 x 2401) is ~288 MB, which is feasible but large.
    # Instead, we can check membership by looking up line_points.
    # For speed in the inner loop, let's precompute a list of neighbors for each point-direction.
    # neighbors[pt, d] is an array of points on the line through pt in direction d (including pt? No, line_points excludes pt).
    # Actually, the condition is: x has a valid direction d if all points on line(x,d) are in N.
    # Removing y breaks validity for x,d if y is on line(x,d) and x,d was valid.
    # y is on line(x,d) iff x is on line(y,d).
    # So, if we remove y, we must check all x such that y is on line(x,d).
    # This is equivalent to: for each d, x is in line_points[y, d] or x == y.
    # Wait, line_points[y, d] contains the q-1 other points.
    # So the set of x affected by removing y is: {y} union line_points[y, d].
    # Let's precompute 'affected_by[y, d]' as a boolean mask or list of x's.
    # Since q=49, the list is small. We can just use line_points[y, d] and check y separately.

    # Initialize validity tracking
    valid_count = np.zeros(total_points, dtype=np.int32)
    is_valid_dir = np.zeros((total_points, num_dirs), dtype=bool)

    for pt in range(total_points):
        for d in range(num_dirs):
            if np.all(in_N[line_points[pt, d, :]]):
                is_valid_dir[pt, d] = True
                valid_count[pt] += 1

    # Check initial validity
    if np.any(valid_count == 0):
        # Baseline might be invalid? Unlikely, but let's be safe.
        # If invalid, we return the baseline anyway, but we expect it to be valid.
        pass

    best_complement = np.sum(~in_N)
    best_in_N = in_N.copy()
    best_valid_count = valid_count.copy()
    best_is_valid_dir = is_valid_dir.copy()

    def try_prune_pq(max_rounds=5):
        nonlocal in_N, valid_count, is_valid_dir
        nonlocal best_complement, best_in_N, best_valid_count, best_is_valid_dir

        work = in_N.copy()
        wc = valid_count.copy()
        ird = is_valid_dir.copy()
        
        # Priority: number of valid constraints this point supports.
        # A point y supports constraint (x, d) if y is on line(x, d) and ird[x, d] is True.
        # Let support[y] = sum over d of count of x such that ird[x, d] and y is on line(x, d).
        # Note: y is on line(x, d) iff x is on line(y, d).
        # So support[y] = sum over d of count of x in {y} U line_points[y, d] such that ird[x, d] is True.
        
        def calc_support(work, wc, ird):
            support = np.zeros(total_points, dtype=np.int32)
            for y in range(total_points):
                if not work[y]:
                    continue
                s = 0
                for d in range(num_dirs):
                    # Check y itself
                    if ird[y, d]:
                        s += 1
                    # Check neighbors
                    for x in line_points[y, d, :]:
                        if ird[x, d]:
                            s += 1
                support[y] = s
            return support

        # Initial priority queue
        # We want to remove points with LOWEST support.
        # Heap in Python is min-heap.
        # We push (support, point_index).
        
        # We need to handle dynamic updates to support.
        # When we remove a point y, the support of other points might change.
        # Specifically, if we remove y, it no longer supports any constraints.
        # But also, removing y might invalidate some (x, d), which reduces the support of other points on those lines.
        # This is complex to update incrementally in a heap without stale entries.
        # Strategy: Lazy deletion.
        # Keep a version counter or just re-check validity when popping.
        
        # Let's use a simpler approach: 
        # 1. Calculate support for all active points.
        # 2. Push all active points into heap with their support.
        # 3. Pop the min support point.
        # 4. Check if it's still active and if its support is still the same (or just check if it's removable).
        # 5. If removable, remove it, update ird/wc, and recalculate support for affected points?
        #    Recalculating support for ALL points is O(N^2 * D). N=2400, D=50. 2400^2 * 50 = 288M ops. 
        #    Doing this for every removal (up to 2400 removals) is 2400 * 288M = 691 Billion ops. Too slow.
        
        # Better: 
        # The "support" of y is the number of valid lines passing through y.
        # When we remove y, we invalidate some lines.
        # The support of other points z on those invalidated lines decreases.
        # We can update support locally.
        
        # Let's try a different heuristic for priority that is cheaper to update.
        # Priority = valid_count of the point itself? No, that's not right.
        # Priority = number of points x for which y is on their UNIQUE valid line?
        # If y is on the unique valid line of x, removing y makes x invalid. This is "bad".
        # So we want to remove y that is on the unique valid line of FEWEST points.
        
        # Let's define criticality[y] = number of points x such that y is on the unique valid line of x.
        # If criticality[y] == 0, y is safe to remove (it doesn't break any unique validity).
        # If criticality[y] > 0, removing y breaks validity for those x.
        
        # How to compute criticality?
        # x has a unique valid line d if wc[x] == 1 and ird[x, d] is True.
        # y is on that line if y is in {x} U line_points[x, d].
        # So criticality[y] = count of x such that wc[x]==1 and y is on line(x, ird[x, :]).
        
        # We can maintain this incrementally.
        # Initial calculation:
        # For each x with wc[x]==1:
        #   find d such that ird[x, d]
        #   for y in {x} U line_points[x, d]:
        #       criticality[y] += 1
        
        # When we remove y:
        # 1. We set work[y] = False.
        # 2. We check which (x, d) become invalid.
        #    A (x, d) becomes invalid if y was on line(x, d) and ird[x, d] was True.
        #    Wait, ird[x, d] is True only if ALL points on line(x, d) are in N.
        #    So if we remove y, and y was on line(x, d), then (x, d) becomes invalid ONLY IF y was the ONLY missing point? No.
        #    ird[x, d] was True. This means all points on line(x, d) were in N.
        #    We are removing y from N. So y is no longer in N.
        #    Therefore, (x, d) becomes invalid.
        #    So, for every (x, d) such that ird[x, d] is True and y is on line(x, d), we set ird[x, d] = False and wc[x] -= 1.
        
        # 3. Now, the set of "unique valid lines" changes.
        #    - Some (x, d) that were unique (wc[x]==1) are now invalid (wc[x] becomes 0).
        #      This DECREASES criticality for points on that line.
        #    - Some (x, d) that were NOT unique (wc[x]>1) might become unique (wc[x] becomes 1) if they were not invalid?
        #      No, we only invalidate lines. We never validate them.
        #      So wc[x] only decreases.
        #      If wc[x] was 2 and becomes 1, then x now has a unique valid line.
        #      This INCREASES criticality for points on that line.
        
        # This incremental update is feasible.
        
        # Let's implement this.
        
        # Data structures:
        # criticality: np.array of int32
        # heap: list of (criticality, point_index)
        # active: boolean array (work)
        
        # Precompute line membership for faster "is y on line(x, d)" check?
        # We have line_points[x, d].
        # To check if y is on line(x, d), we can check if y == x or y in line_points[x, d].
        # Since line_points is a numpy array, `y in line_points[x, d]` is O(q).
        # Doing this for every update might be slow if we do it often.
        # But the number of updates is bounded by the number of invalidations.
        
        # Let's optimize the "find x such that y is on line(x, d)" step.
        # We precomputed that y is on line(x, d) iff x is on line(y, d).
        # So the set of x affected by removing y is exactly {y} U line_points[y, d] for each d.
        # We already have this list! It's just line_points[y, d].
        
        # Algorithm:
        # 1. Init criticality.
        # 2. Init heap with all active points.
        # 3. Loop:
        #    Pop (crit, y) from heap.
        #    If not work[y], continue (stale).
        #    If crit > 0, we might still want to remove it if no other option? 
        #    Actually, if crit > 0, removing y breaks validity for some x.
        #    We should only remove y if crit == 0.
        #    If crit == 0, remove y.
        #    Update ird, wc, criticality.
        #    Push updated criticality for affected points.
        #    If no points with crit == 0, break? Or try to remove min crit point anyway?
        #    The previous "greedy" removed any point that didn't make the set invalid.
        #    Removing a point with crit > 0 makes the set invalid for those x.
        #    So we MUST only remove points with crit == 0 to maintain validity.
        #    Wait, the condition for a valid Nikodym set is that EVERY point has at least one valid line.
        #    If we remove y, and this breaks the unique line of x, then x has 0 valid lines. Invalid.
        #    So yes, we can ONLY remove y if it is not on the unique valid line of any x.
        #    i.e., criticality[y] == 0.
        
        # If we reach a state where no point has criticality 0, we are stuck.
        # We can try to "repair" by adding points back? No, we are minimizing N.
        # We can try a different starting point or different heuristic.
        
        # Let's implement this "Criticality Greedy".
        
        # Init criticality
        criticality = np.zeros(total_points, dtype=np.int32)
        # For each x, if wc[x] == 1:
        for x in range(total_points):
            if wc[x] == 1:
                # Find the valid direction
                d = np.where(ird[x, :])[0][0]
                # Increment criticality for y in {x} U line_points[x, d]
                criticality[x] += 1
                for y in line_points[x, d, :]:
                    criticality[y] += 1
        
        # Init heap
        heap = []
        for y in range(total_points):
            if work[y]:
                heapq.heappush(heap, (criticality[y], y))
        
        removed_count = 0
        
        for _ in range(total_points):
            if time.time() > deadline - safety_margin:
                break
            
            if not heap:
                break
            
            crit, y = heapq.heappop(heap)
            
            if not work[y]:
                continue
            
            # If criticality is not 0, we cannot remove it safely.
            # But maybe the heap entry is stale?
            # We should check if criticality[y] is still crit.
            # Actually, if criticality[y] > 0, we can't remove it.
            # We can skip it and not push it back? No, it might become 0 later.
            # So we push it back? That's inefficient.
            # Better: only push points with criticality 0 into the heap?
            # But criticality changes.
            
            # Let's stick to: Pop min. If crit == 0, remove. Else, push back with updated crit?
            # If we push back, we might loop.
            # Alternative: Maintain a separate list of "safe" points (crit==0).
            # But criticality changes dynamically.
            
            # Let's try: 
            # If crit == 0:
            #   Remove y.
            #   Update neighbors.
            # Else:
            #   Push back (criticality[y], y) and break? No, maybe another point has crit 0.
            #   We should scan the heap for any point with crit 0?
            #   Since heap is sorted by crit, if the min is > 0, ALL are > 0.
            #   So if min crit > 0, we are stuck. Break.
            
            if crit > 0:
                # No safe points. Break.
                break
            
            # Remove y
            work[y] = False
            removed_count += 1
            
            # Update ird and wc
            # For each d, for each x in {y} U line_points[y, d]:
            #   If ird[x, d] is True:
            #       ird[x, d] = False
            #       wc[x] -= 1
            #       Update criticality for points on line(x, d)
            
            affected_x = set()
            for d in range(num_dirs):
                # x = y
                if ird[y, d]:
                    affected_x.add((y, d))
                # x in line_points[y, d]
                for x in line_points[y, d, :]:
                    if ird[x, d]:
                        affected_x.add((x, d))
            
            # Apply invalidations
            for x, d in affected_x:
                # This line is now broken
                ird[x, d] = False
                old_wc = wc[x]
                wc[x] -= 1
                new_wc = wc[x]
                
                # Update criticality
                # The line (x, d) was unique if old_wc == 1.
                # If old_wc == 1, it was contributing to criticality of points on line(x, d).
                # Now it's invalid, so it no longer contributes.
                # So we DECREASE criticality for points on line(x, d).
                
                if old_wc == 1:
                    # Decrement criticality for y' in {x} U line_points[x, d]
                    criticality[x] -= 1
                    for y_prime in line_points[x, d, :]:
                        criticality[y_prime] -= 1
                        if work[y_prime] and criticality[y_prime] == 0:
                            heapq.heappush(heap, (0, y_prime))
                
                # If new_wc == 1, it is now unique.
                # It now contributes to criticality of points on line(x, d).
                # So we INCREASE criticality for points on line(x, d).
                
                if new_wc == 1:
                    # Find the new valid direction
                    # There is exactly one d2 such that ird[x, d2] is True.
                    d2 = np.where(ird[x, :])[0][0]
                    # Increment criticality for y' in {x} U line_points[x, d2]
                    criticality[x] += 1
                    for y_prime in line_points[x, d2, :]:
                        criticality[y_prime] += 1
                        # If it becomes 0? No, it increases.
                        # But if it was 0 and becomes 1, it's no longer safe.
                        # We don't need to push it.
        
        comp = np.sum(~work)
        if comp > best_complement:
            best_complement = comp
            best_in_N = work.copy()
            best_valid_count = wc.copy()
            best_is_valid_dir = ird.copy()
        
        # Revert to best for next iteration
        in_N = best_in_N.copy()
        valid_count = best_valid_count.copy()
        is_valid_dir = best_is_valid_dir.copy()

    # Run the priority queue greedy
    try_prune_pq(max_rounds=1)
    
    # If we didn't improve, try some random restarts with the same logic?
    # The baseline is deterministic.
    # Maybe we can try to remove points in a different order initially?
    # No, the criticality logic is robust.
    
    # Final validity check
    actual_valid = np.zeros(total_points, dtype=np.int32)
    for pt in range(total_points):
        for d in range(num_dirs):
            if np.all(best_in_N[line_points[pt, d, :]]):
                actual_valid[pt] += 1
    
    if np.any(actual_valid == 0):
        # Invalid, revert to baseline
        in_N = np.ones(total_points, dtype=bool)
        for x_idx in range(q):
            xa, xb = elems[x_idx]
            trace_x = (2 * xa) % p
            for y_idx in range(q):
                ya, yb = elems[y_idx]
                norm_y = (ya * ya - yb * yb * alpha_sq) % p
                p_idx = x_idx * q + y_idx
                if (trace_x + norm_y) % p == 0 and (ya, yb) in y_special:
                    in_N[p_idx] = False
        best_in_N = in_N.copy()
        best_complement = np.sum(~in_N)

    # Extract points
    points = []
    for p_idx in range(total_points):
        if best_in_N[p_idx]:
            x_idx = p_idx // q
            y_idx = p_idx % q
            xa, xb = elems[x_idx]
            ya, yb = elems[y_idx]
            points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)
