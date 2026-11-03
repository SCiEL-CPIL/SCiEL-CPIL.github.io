import time
import random
import math

def _scope(rows):
    if not rows:
        return 0
    return max(a for r in rows for a in r[1:])

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _diffs_for_row(row):
    diffs = []
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.append(row[j] - row[i])
    return diffs

def _construct_rows_backtrack(n_rows, k, deadline, rng, target_scope):
    """
    Construct n_rows valid rows using backtracking with forward checking.
    Returns a list of rows if successful, None otherwise.
    """
    used_diffs = set()
    rows = []
    
    for row_idx in range(n_rows):
        if time.time() > deadline:
            return None
        
        # Backtrack to find a valid row
        current_row = [0]
        current_diffs = set()
        
        # We use a recursive helper inside this function for clarity, 
        # but implement it iteratively or with a local recursive function.
        
        def backtrack(pos, current_row, current_diffs, used_diffs, k, max_val, deadline, rng):
            if time.time() > deadline:
                return None
            
            n = k + 1
            if pos == n:
                return list(current_row)
            
            prev = current_row[-1]
            min_next = prev + 1
            
            # Forward checking: 
            # We need to place (n - pos - 1) more numbers.
            # The smallest they can be is min_next, min_next+1, ...
            # The largest one will be min_next + (n - pos - 2) + 1 = min_next + (n - pos - 1)
            # This must be <= max_val
            if min_next + (n - pos - 1) > max_val:
                return None
            
            # Generate candidates
            # To encourage tight packing, we try smaller values first, but shuffle slightly to escape local traps
            candidates = list(range(min_next, max_val + 1))
            
            # Optimization: If the range is large, we might want to bias towards smaller values 
            # but allow some randomness. However, for DTS, smaller values are generally better for fitting in scope.
            # A simple heuristic: try a few small ones, then randomize the rest.
            if len(candidates) > 10:
                # Keep the first 5 in order (smallest), shuffle the rest
                head = candidates[:5]
                tail = candidates[5:]
                rng.shuffle(tail)
                candidates = head + tail
            
            for val in candidates:
                if time.time() > deadline:
                    return None
                
                # Check conflicts
                conflict = False
                new_diffs = []
                for idx in range(pos):
                    d = val - current_row[idx]
                    if d in current_diffs or d in used_diffs:
                        conflict = True
                        break
                    new_diffs.append(d)
                
                if conflict:
                    continue
                
                # Commit
                current_row.append(val)
                current_diffs.update(new_diffs)
                
                result = backtrack(pos + 1, current_row, current_diffs, used_diffs, k, max_val, deadline, rng)
                
                # Uncommit
                current_row.pop()
                for d in new_diffs:
                    current_diffs.discard(d)
                
                if result is not None:
                    return result
            
            return None

        row = backtrack(1, current_row, current_diffs, used_diffs, k, target_scope, deadline, rng)
        
        if row is None:
            return None
        
        rows.append(row)
        for d in _diffs_for_row(row):
            used_diffs.add(d)
    
    return rows

def _greedy_construct(n_rows, k, rng):
    """
    Fallback greedy construction if backtracking fails or as a starting point.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            conflict = False
            for a in row:
                if (x - a) in used:
                    conflict = True
                    break
            if not conflict:
                row.append(x)
                for a in row[:-1]:
                    used.add(x - a)
        rows.append(row)
    return rows

def construct(n_rows, k):
    deadline = time.time() + 145.0  # Safety margin
    rng = random.Random(42)
    
    best_result = None
    best_scope = float('inf')
    
    # Estimate a reasonable starting scope
    # Counting bound: m * k(k+1)/2 <= m^2 => m >= sqrt(k(k+1)/2 * m_rows) is not quite right.
    # Total diffs = n_rows * k(k+1)/2. Max diff is scope.
    # Scope must be at least sqrt(2 * total_diffs) roughly? No, diffs are distinct integers 1..scope.
    # So scope >= total_diffs.
    # total_diffs = 7 * 15 = 105. So scope >= 105.
    # Known best is 112.
    
    min_possible_scope = n_rows * k * (k + 1) // 2
    # Start slightly above the counting bound, increasing if necessary
    # We try a range of target scopes
    
    start_scope = min_possible_scope
    # Add a small buffer to avoid immediate failure due to structural constraints
    current_target = start_scope + 2 
    
    # Try to construct with increasing scope limits
    # We limit the number of restarts/attempts per scope to avoid wasting time
    max_attempts_per_scope = 500000  # Iterations of backtracking
    
    while time.time() < deadline - 10:
        if current_target > 200: # Hard cap for this specific problem instance (7,5)
            break
            
        # Try to construct with this target scope
        # We use a deadline for the backtracking process itself
        bt_deadline = time.time() + 5.0 # Allow 5 seconds per scope attempt
        
        # We might need multiple attempts with different random seeds/orders if the first fails
        # However, randomized backtracking usually finds a solution quickly if one exists
        # If it fails, it's likely the scope is too small.
        
        rows = _construct_rows_backtrack(n_rows, k, bt_deadline, rng, current_target)
        
        if rows is not None:
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_result = rows
            # If we found a solution, try to see if a smaller scope works?
            # Since we started from the bottom and went up, the first success is the best for this target.
            # But maybe a smaller target failed due to bad luck?
            # We can try a few more times with the same target or slightly smaller.
            # For now, if we found one, we can try to optimize it or accept it.
            # Given the time constraints, accepting the first valid solution found with the smallest feasible scope is a good strategy.
            break
        
        # If failed, increase target scope
        current_target += 1
        
    # If backtracking failed to find anything (or time ran out), use greedy
    if best_result is None:
        best_result = _greedy_construct(n_rows, k, rng)
        best_scope = _scope(best_result)
        
    # Optional: Local search to improve scope?
    # Backtracking with forward checking is usually quite tight.
    # A simple local search (SA) might help squeeze out 1-2 units.
    
    if time.time() < deadline - 5:
        # Implement a simple SA to try to reduce scope
        current_rows = [list(r) for r in best_result]
        current_scope = best_scope
        best_rows = [list(r) for r in current_rows]
        
        # Build diff counter for SA
        diff_counter = {}
        for r in current_rows:
            for d in _diffs_for_row(r):
                diff_counter[d] = diff_counter.get(d, 0) + 1
        
        def get_collisions(counter):
            return sum(max(0, v - 1) for v in counter.values())
        
        collisions = get_collisions(diff_counter)
        
        T = 10.0
        cooling = 0.99
        min_T = 0.01
        
        sa_deadline = time.time() + (deadline - time.time()) * 0.8
        
        while time.time() < sa_deadline:
            # Pick random row and position
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(1, k)
            
            old_val = current_rows[i][j]
            prev_val = current_rows[i][j-1]
            next_val = current_rows[i][j+1] if j < k else old_val + 10000
            
            # Generate new value
            # Try to reduce if possible
            if rng.random() < 0.5:
                upper = old_val - 1
                if upper <= prev_val:
                    continue
                new_val = rng.randint(prev_val + 1, upper)
            else:
                new_val = old_val + rng.randint(-5, 5)
                
            if new_val <= prev_val or (j < k and new_val >= next_val) or new_val == old_val:
                continue
                
            # Check validity of new row
            new_row = current_rows[i][:]
            new_row[j] = new_val
            new_diffs = _diffs_for_row(new_row)
            if len(new_diffs) != len(set(new_diffs)):
                continue
            
            # Update diff counter
            old_diffs = _diffs_for_row(current_rows[i])
            for d in old_diffs:
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            for d in new_diffs:
                diff_counter[d] = diff_counter.get(d, 0) + 1
                
            new_collisions = get_collisions(diff_counter)
            new_scope = _scope(current_rows)
            
            # Cost function
            old_cost = current_scope + collisions * 1000
            new_cost = new_scope + new_collisions * 1000
            
            if new_cost <= old_cost or rng.random() < math.exp(-(new_cost - old_cost) / max(T, min_T)):
                current_rows[i] = new_row
                current_scope = new_scope
                collisions = new_collisions
                
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in current_rows]
            else:
                # Revert
                for d in new_diffs:
                    diff_counter[d] -= 1
                    if diff_counter[d] == 0:
                        del diff_counter[d]
                for d in old_diffs:
                    diff_counter[d] = diff_counter.get(d, 0) + 1
            
            T = max(T * cooling, min_T)
            
        if best_scope < _scope(best_result):
            best_result = best_rows

    return best_result
