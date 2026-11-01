import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k):
    """
    Greedy initialization using a boolean array for speed.
    """
    rows = []
    used = [False] * 1500
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            valid = True
            new_diffs = []
            for a in row:
                d = x - a
                if d >= len(used):
                    used.extend([False] * (d - len(used) + 100))
                if used[d]:
                    valid = False
                    break
                new_diffs.append(d)
            if valid:
                row.append(x)
                for d in new_diffs:
                    used[d] = True
        rows.append(row)
    return rows

def _row_diffs_list(row):
    """Compute the list of all positive differences in a row."""
    diffs = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return diffs

def _row_diffs_set(row):
    """Compute the set of all positive differences in a row."""
    return set(_row_diffs_list(row))

def _compute_all_used_set(rows):
    """Compute the set of all differences across all rows."""
    all_used = set()
    for row in rows:
        all_used |= _row_diffs_set(row)
    return all_used

def _global_max(rows):
    """Compute the maximum entry across all rows."""
    return max(max(row) for row in rows)

def _check_conflicts(new_diffs, old_diffs, all_used):
    """
    Check if new_diffs conflict with all_used, excluding those in old_diffs.
    Returns True if there is a conflict.
    """
    for d in new_diffs:
        if d in all_used and d not in old_diffs:
            return True
    return False

def _row_diffs_list_fast(row):
    """Compute the list of all positive differences in a row, optimized for small k."""
    diffs = []
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.append(row[j] - ri)
    return diffs

def _try_solve_csp(n_rows, k, max_val, time_limit):
    """
    Attempt to solve the CSP with max entry <= max_val using backtracking with forward checking.
    Returns a list of rows if successful, None otherwise.
    """
    start_time = time.time()
    
    # Total differences needed
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # If max_val is too small to hold this many distinct differences, fail immediately
    # Differences range from 1 to max_val
    if max_val < total_diffs:
        return None

    # Sort rows to construct? The problem is symmetric, so we just construct n_rows identical constraints.
    # We will construct them one by one.
    
    # To make it harder, we can try to construct rows in a way that leaves the most "room" for others?
    # Or just greedy order.
    
    rows = []
    all_used = set()
    
    # Heuristic: try to construct rows with smaller max values first?
    # Or just standard backtracking.
    
    # Since rows are indistinguishable, we can just fill them sequentially.
    # However, to avoid symmetry breaking issues, we might want to enforce some ordering on the rows themselves?
    # E.g., row 0 max <= row 1 max <= ...
    # This reduces the search space significantly.
    
    # Let's try a recursive backtracking that enforces row_max[0] <= row_max[1] <= ...
    
    # State:
    # rows: list of completed rows
    # current_row: list of values in the row currently being built
    # pos: index in current_row
    # all_used: set of differences used so far
    # min_start: minimum allowed value for the next element in current_row (to enforce ordering if needed, but here it's just increasing)
    # max_allowed_for_this_row: to enforce row_max ordering?
    
    # Actually, enforcing row_max ordering is tricky because we don't know the max until the row is complete.
    # But we know the max of the previous row. So the current row's max must be >= prev_max?
    # No, if we sort rows by max, then row i max >= row i-1 max.
    # This means we are searching for a multiset of rows.
    
    # Let's stick to a simpler approach: just fill rows sequentially without ordering constraint,
    # but use a good heuristic for choosing the next value.
    # To reduce symmetry, we can require that the first non-zero element of row i is >= first non-zero element of row i-1?
    # No, that's not a valid symmetry breaking for the whole set.
    
    # Let's try a different CSP formulation:
    # Variables: all entries of all rows.
    # This is huge.
    
    # Alternative: Randomized Restart Backtracking.
    
    def backtrack(row_idx, current_row, pos, all_used, min_val_so_far):
        if time.time() - start_time > time_limit:
            return None
        
        if pos == k + 1:
            # Row complete
            if row_idx == n_rows - 1:
                return current_row[:]
            # Start next row
            return backtrack(row_idx + 1, [0], 1, all_used, 0)
        
        if pos == 0:
            # Always start with 0
            return backtrack(row_idx, current_row, 1, all_used, 0)
        
        # Determine candidate values for current_row[pos]
        # Must be > current_row[pos-1]
        # Must be <= max_val
        
        # Heuristic: try values close to current_row[pos-1] first? Or spread out?
        # Spreading out might be better to use up large differences?
        # Actually, we want to minimize max, so we should try to keep values small.
        
        # But we also need to leave room for future elements.
        # If we pick a value too small, we might run out of room for the remaining k-pos elements.
        
        # Min value: current_row[pos-1] + 1
        # Max value: max_val
        
        # Also, we need to ensure that we can fit the remaining elements.
        # If we pick v, the remaining elements must be > v.
        # So we need at least (k - pos) values > v, up to max_val.
        # So v <= max_val - (k - pos)
        
        min_v = current_row[pos - 1] + 1
        max_v = min(max_val, max_val - (k - pos))
        
        if min_v > max_v:
            return None
        
        # Generate candidates
        # To break symmetry and speed up, we can try a few random candidates or a range.
        # Let's try a mix of greedy (small values) and random.
        
        # Actually, let's just iterate through candidates.
        # To make it faster, we can shuffle the candidates or use a step.
        
        # Let's try candidates in increasing order, but skip some if the search is deep.
        
        # A better heuristic: pick values that don't create "wasted" small differences?
        # No, all differences must be distinct.
        
        # Let's try a randomized approach for candidates to escape local minima.
        
        candidates = list(range(min_v, max_v + 1))
        
        # If too many candidates, sample them
        if len(candidates) > 50:
            random.shuffle(candidates)
            candidates = candidates[:50]
        
        # Sort them? Maybe try smaller ones first to keep max low?
        # But smaller ones might lead to dead ends later.
        # Let's try a mix.
        
        # Actually, let's just iterate.
        
        for v in candidates:
            # Check if v is valid with current_row
            new_diffs = []
            valid = True
            for i in range(pos):
                d = v - current_row[i]
                if d in all_used:
                    valid = False
                    break
                # Check internal duplicates in current_row + v
                # We already check against all_used, but we need to check against new_diffs
                if d in new_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            # Tentatively add
            current_row.append(v)
            for d in new_diffs:
                all_used.add(d)
            
            result = backtrack(row_idx, current_row, pos + 1, all_used, v)
            
            # Undo
            current_row.pop()
            for d in new_diffs:
                all_used.remove(d)
            
            if result is not None:
                return result
        
        return None

    # Start backtracking
    # We need to handle the fact that rows are indistinguishable.
    # To reduce search, we can enforce that the rows are sorted by their first non-zero element?
    # Or just let it search.
    
    # One issue: the search space is huge.
    # Backtracking might be too slow for n_rows=7, k=6.
    
    # Let's add a timeout and return None if it takes too long.
    
    # Initial call
    # We need to be careful with the recursion depth and state copying.
    # The above backtrack function modifies current_row and all_used in place, which is correct.
    
    # However, the symmetry of rows means that many solutions are equivalent.
    # We can enforce that row 0 <= row 1 <= ... in lexicographic order?
    # This is hard to implement in the current structure.
    
    # Let's try a simpler heuristic:
    # When starting a new row, ensure its first non-zero element is >= the first non-zero element of the previous row?
    # This is not a complete symmetry breaking, but it helps.
    
    # Actually, let's just rely on the timeout. If it finds a solution, great. If not, we fall back to SA.
    
    # One optimization: if we are building row i, and we have already built i-1 rows,
    # we can check if the remaining number of differences is possible.
    
    # Let's just run it with a short time limit.
    
    return backtrack(0, [0], 1, set(), 0)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Initialization
    rows = _greedy_init(n_rows, k)
    
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    
    # 2. Try CSP solver with decreasing max_val
    # Start from best_val - 1 and go down
    # But CSP is slow, so we only try a few values.
    
    # Let's try CSP for best_val - 1, best_val - 2, etc.
    # But only if we have time.
    
    # Actually, let's integrate CSP as a restart mechanism in SA?
    # No, let's just try it once or twice.
    
    # Try CSP with a tight time limit
    if time.time() < deadline - 30:
        for target in range(best_val - 1, best_val - 10, -1):
            if time.time() > deadline - 30:
                break
            sol = _try_solve_csp(n_rows, k, target, 2.0)
            if sol is not None:
                # Verify solution
                all_diffs = set()
                valid = True
                for row in sol:
                    if row[0] != 0:
                        valid = False
                        break
                    for i in range(len(row)):
                        for j in range(i+1, len(row)):
                            d = row[j] - row[i]
                            if d in all_diffs:
                                valid = False
                                break
                            all_diffs.add(d)
                    if not valid:
                        break
                if valid:
                    best_rows = sol
                    best_val = _global_max(best_rows)
                    # If we found a solution, we can try to improve it further with SA
                    break
    
    # 3. Simulated Annealing
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    row_maxes = [max(r) for r in current_rows]
    
    all_used = _compute_all_used_set(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    
    T = 400.0
    T_min = 0.3
    cool = 0.99997
    
    no_improve = 0
    max_no_improve = 20000
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a row and position
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        # Generate candidates
        candidates = []
        
        # Decreasing candidates
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(50, old_val - min_allowed)
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)
            
        # Increasing candidates
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 30
        max_increase = min(20, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            candidates.append(old_val + d)
            
        # Random jump
        if random.random() < 0.2:
            low = current_rows[ri][pi - 1] + 1
            high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 20
            if low <= high:
                candidates.append(random.randint(low, high))
        
        if not candidates:
            no_improve += 1
            T = max(T_min, T * cool)
            continue
            
        random.shuffle(candidates)
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            
            # Compute new diffs
            new_diffs_list = _row_diffs_list_fast(cand_row)
            
            # Check internal validity
            if len(set(new_diffs_list)) != len(new_diffs_list):
                continue
                
            new_diffs = set(new_diffs_list)
            old_diffs = row_diffs[ri]
            
            # Check conflicts
            if _check_conflicts(new_diffs, old_diffs, all_used):
                continue
                
            # Move is valid
            cand_row_max = max(cand_row)
            
            # Calculate delta
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.4:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.1))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                # Update state
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri] = cand_row
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = max(row_maxes)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
                    
                moved = True
                break
        
        if not moved:
            no_improve += 1
            
        T = max(T_min, T * cool)
        
        # Restart if stuck
        if no_improve > 10000:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = _compute_all_used_set(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 400.0
            no_improve = 0
    
    # 4. Polishing Phase
    polish_deadline = min(deadline, start_time + 148.0)
    
    current_best = best_val
    for target_offset in range(1, 6):
        target_val = current_best - target_offset
        if target_val < 100:
            break
            
        if time.time() > polish_deadline - 2:
            break
            
        current_rows = [r[:] for r in best_rows]
        current_val = best_val
        all_used = _compute_all_used_set(current_rows)
        row_diffs = [_row_diffs_set(r) for r in current_rows]
        row_maxes = [max(r) for r in current_rows]
        
        T = 50.0
        T_min = 0.1
        cool = 0.9999
        no_improve = 0
        max_steps = 3000
        
        for _ in range(max_steps):
            if time.time() > polish_deadline - 2:
                break
                
            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            old_val = current_rows[ri][pi]
            
            min_allowed = current_rows[ri][pi - 1] + 1
            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else target_val
            
            if min_allowed > max_allowed:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
                
            new_val = random.randint(min_allowed, max_allowed)
            if new_val == old_val:
                continue
                
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            
            new_diffs_list = _row_diffs_list_fast(cand_row)
            if len(set(new_diffs_list)) != len(new_diffs_list):
                continue
                
            new_diffs = set(new_diffs_list)
            old_diffs = row_diffs[ri]
            
            if _check_conflicts(new_diffs, old_diffs, all_used):
                continue
                
            cand_row_max = max(cand_row)
            if cand_row_max > target_val:
                continue
                
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.5:
                    accept = True
            else:
                if new_global_max <= target_val:
                    prob = math.exp(-delta / max(T, 0.1))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri] = cand_row
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = max(row_maxes)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                    if best_val <= target_val:
                        break
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
            T = max(T_min, T * cool)
            
            if no_improve > 500:
                break
    
    return best_rows

# EVOLVE-BLOCK-END
