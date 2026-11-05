import random
import time
import sys

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    time_limit = 118.0  # Leave buffer for 120s limit

    def time_up():
        return time.time() - start_time > time_limit

    total_diffs = n_rows * k * (k + 1) // 2
    min_scope = total_diffs

    best_result = None
    best_scope = float('inf')

    # Greedy initialization
    def greedy_construct():
        nonlocal best_result, best_scope
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = set(x - a for a in row)
                    if not new_diffs & used:
                        break
                    x += 1
                row.append(x)
                used |= new_diffs
            rows.append(row)
        
        scope = max(r[-1] for r in rows)
        if scope < best_scope:
            best_scope = scope
            best_result = [r[:] for r in rows]
        return rows, used

    # Run greedy first
    _, _ = greedy_construct()

    # Backtracking solver with bitmask for used differences
    def solve_with_scope(target_scope, rng_seed):
        nonlocal best_result, best_scope
        
        rng = random.Random(rng_seed)
        # Use a set for used differences (simpler, and S is small ~200)
        used_diffs = set()
        rows = []
        
        # Precompute all valid difference sets for a single row within scope
        # A row is defined by its k+1 elements starting at 0
        # We'll use backtracking to build rows
        
        def backtrack(row_idx, current_row):
            nonlocal used_diffs, rows, best_scope, best_result
            
            if time_up():
                return False
                
            if row_idx == n_rows:
                # All rows constructed
                actual_scope = max(r[-1] for r in rows)
                if actual_scope < best_scope:
                    best_scope = actual_scope
                    best_result = [r[:] for r in rows]
                return True
            
            if len(current_row) == k + 1:
                # Row complete
                rows.append(current_row)
                if backtrack(row_idx + 1, [0]):
                    return True
                rows.pop()
                return False
            
            last_val = current_row[-1]
            upper_bound = target_scope - 1
            
            if last_val + 1 > upper_bound:
                return False
            
            # Generate candidates with heuristic ordering
            candidates = []
            
            for x in range(last_val + 1, upper_bound + 1):
                new_diffs = set(x - a for a in current_row)
                if new_diffs & used_diffs:
                    continue
                
                # Heuristic: prefer larger differences (less scarce)
                cost = sum(1.0 / d for d in new_diffs)
                noise = rng.random() * 0.01
                candidates.append((cost + noise, x, new_diffs))
                
                if len(candidates) > 20:
                    break
            
            if not candidates:
                return False
            
            candidates.sort(key=lambda item: item[0])
            
            # Try candidates in order
            for cost, x, new_diffs in candidates:
                old_used = used_diffs.copy()
                used_diffs = used_diffs | new_diffs
                
                if backtrack(row_idx, current_row + [x]):
                    return True
                
                used_diffs = old_used
            
            return False
        
        return backtrack(0, [0])

    # Main search loop with restarts
    seed_counter = 0
    consecutive_failures = 0
    
    while time.time() - start_time < time_limit:
        target = best_scope
        found = solve_with_scope(target, seed_counter)
        
        if found:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
        
        seed_counter += 1
    
    if best_result is None:
        # Fallback to greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    return best_result
