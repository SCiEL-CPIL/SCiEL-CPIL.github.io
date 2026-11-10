import time
import random

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0, minimizing the
    largest entry, subject to all positive differences within rows being distinct
    across all rows.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Randomized Greedy to find a good initial upper bound
    def greedy_random(seed):
        rng = random.Random(seed)
        used_mask = 0
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            last_val = 0
            while len(row) < k + 1:
                # Find candidates for next value
                # We look for x > last_val such that all diffs x-a are unused
                # To speed up, we don't scan from 1, we scan from last_val + 1
                candidates = []
                x = last_val + 1
                # Heuristic: don't search too far if we find enough candidates
                # The max possible x is bounded by current best_scope if we have one, 
                # but for initial greedy, we just find the first few valid ones.
                while len(candidates) < 5:
                    # Calculate new differences mask
                    new_diffs_mask = 0
                    valid = True
                    for a in row:
                        d = x - a
                        bit = 1 << d
                        if new_diffs_mask & bit:
                            # Internal collision in this row (shouldn't happen if x > last_val and row is sorted? 
                            # Wait, x-a are distinct if a are distinct. 
                            # x-a1 != x-a2 iff a1 != a2. So internal collisions are impossible.
                            pass
                        if used_mask & bit:
                            valid = False
                            break
                        new_diffs_mask |= bit
                    if valid:
                        candidates.append((x, new_diffs_mask))
                    x += 1
                    # Safety break if x gets too large (shouldn't happen often in greedy)
                    if x > 200: 
                        break
                
                if not candidates:
                    return None, float('inf')
                
                # Pick randomly from candidates
                chosen_x, chosen_mask = rng.choice(candidates)
                row.append(chosen_x)
                last_val = chosen_x
                used_mask |= chosen_mask
            
            rows.append(row)
        
        scope = max(max(row) for row in rows)
        return rows, scope

    # Run greedy for a short duration
    greedy_time_limit = start_time + 20.0
    seed = 0
    while time.time() < greedy_time_limit:
        rows, scope = greedy_random(seed)
        if rows is not None and scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]
        seed += 1
        # If we found a very good solution (close to theoretical min), we might want to stop greedy sooner
        # Theoretical min for (7,5) is 105.
        if best_scope <= 105:
            break

    if best_rows is None:
        # Fallback to simple greedy if random failed or time ran out immediately
        rows, scope = greedy_random(0)
        if rows:
            best_rows = rows
            best_scope = scope
        else:
            # Absolute fallback: deterministic greedy
            used_mask = 0
            best_rows = []
            for _ in range(n_rows):
                row = [0]
                last_val = 0
                while len(row) < k + 1:
                    x = last_val + 1
                    while True:
                        new_diffs_mask = 0
                        valid = True
                        for a in row:
                            d = x - a
                            bit = 1 << d
                            if used_mask & bit:
                                valid = False
                                break
                            new_diffs_mask |= bit
                        if valid:
                            break
                        x += 1
                    row.append(x)
                    last_val = x
                    used_mask |= new_diffs_mask
                best_rows.append(row)
            best_scope = max(max(r) for r in best_rows)

    # Phase 2: Backtracking to try to lower the scope
    # We try to find a solution with scope M_target = best_scope - 1, then -2, etc.
    # We use a backtracking solver that enforces an ordering on the rows' last elements
    # to break symmetry: Last(row_i) <= Last(row_j) for i < j.
    
    def backtrack_solve(target_scope):
        """
        Attempts to find a valid set of rows with max entry <= target_scope.
        Returns list of rows or None.
        """
        if target_scope < n_rows * (k + 1) * k // 2: # Rough lower bound check
            return None
            
        used_mask = 0
        rows = []
        # To break symmetry, we enforce that the last element of row i is <= last element of row i+1
        # We will store the last element of the previous row to constrain the current row's max.
        # Actually, it's easier to just enforce the max of the current row is <= target_scope
        # and that the sequence of row maxima is non-decreasing.
        
        def dfs(row_idx, current_row, last_val, prev_row_max):
            if time.time() > deadline:
                return None
            
            if len(current_row) == k + 1:
                # Row complete
                row_max = current_row[-1]
                if row_max > prev_row_max:
                    # Violates symmetry breaking (if we enforce non-decreasing maxes)
                    # Note: This is valid because rows are interchangeable.
                    # We can always sort the final rows by their last element.
                    # So we can assume WLOG that we build them in order of increasing last element.
                    pass 
                # Actually, if we don't enforce it, we waste search space.
                # Let's enforce: current row's max must be >= prev_row_max.
                # Wait, if we enforce max_i <= max_{i+1}, we cover all permutations.
                # So:
                if row_max < prev_row_max:
                    return None
                
                rows.append(current_row[:])
                
                if row_idx == n_rows - 1:
                    return rows[:]
                
                # Start next row
                return dfs(row_idx + 1, [0], 0, row_max)
            
            else:
                # Find next element for current_row
                # Candidates must be > last_val and <= target_scope
                # Also, if this is the last element (pos k), it must be >= prev_row_max (if we are enforcing order)
                # But we check row_max at the end.
                
                # Heuristic: If we are at the last position, we might want to try values
                # that are just large enough to satisfy the order constraint, or smaller to save space?
                # Smaller is better for scope minimization, but we are fixed to target_scope.
                
                start_val = last_val + 1
                # Optimization: If we need to place the last element, and we are enforcing order,
                # the minimum value for the last element is max(start_val, prev_row_max).
                # However, intermediate elements don't have this constraint relative to prev_row_max,
                # only the final row max does.
                
                for x in range(start_val, target_scope + 1):
                    # Check differences
                    new_diffs_mask = 0
                    valid = True
                    for a in current_row:
                        d = x - a
                        bit = 1 << d
                        if new_diffs_mask & bit:
                             # Internal duplicate? Impossible as argued before.
                             pass
                        if used_mask & bit:
                            valid = False
                            break
                        new_diffs_mask |= bit
                    
                    if not valid:
                        continue
                    
                    # Additional pruning:
                    # If this is the last element (index k), check if x >= prev_row_max
                    if len(current_row) == k:
                        if x < prev_row_max:
                            continue
                    
                    # Recurse
                    current_row.append(x)
                    used_mask_new = used_mask | new_diffs_mask
                    # Pass updated used_mask. Since Python integers are immutable, we pass by value effectively.
                    res = dfs(row_idx, current_row, x, prev_row_max)
                    if res is not None:
                        return res
                    current_row.pop()
                
                return None

        # Start DFS
        # We start with row 0, prev_row_max = 0
        return dfs(0, [0], 0, 0)

    # Try to improve
    current_target = best_scope - 1
    # Lower bound: We need total_diffs = n_rows * (k+1)*k//2 distinct differences.
    # The largest difference is at most M.
    # The number of available differences is M.
    # So M >= total_diffs.
    min_possible = n_rows * (k + 1) * k // 2
    
    while current_target >= min_possible and time.time() < deadline - 5.0:
        # Try to solve for current_target
        # To avoid getting stuck in a very hard instance for a long time,
        # we can limit the time per target or use randomized restarts.
        # For now, we try once per target.
        
        # We can add randomization to the DFS by shuffling the candidate order?
        # The current DFS tries x in increasing order.
        # To diversify, we could try a few random restarts of the DFS for the same target.
        
        found = False
        # Try a few random restarts of the backtracking for this target
        # We modify the DFS to accept a seed for shuffling candidates?
        # That's complex. Instead, we can just run the deterministic DFS.
        # If it fails quickly, we move to next target.
        # If it takes long, it might find a solution.
        
        # Let's add a time check inside the loop of targets
        if time.time() > deadline - 2.0:
            break
            
        res = backtrack_solve(current_target)
        if res is not None:
            best_rows = res
            best_scope = current_target
            # Found a solution for this target. Can we go lower?
            current_target -= 1
            # If we found it very fast, we might try to go lower immediately.
        else:
            # No solution for this target.
            current_target -= 1
            # If it took a long time to fail, we might not have time for more.
            # But backtrack_solve has a global deadline check.

    return best_rows

# EVOLVE-BLOCK-END
