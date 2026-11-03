"""Difference triangle set (4,7): Optimized CSP solver using bitsets for difference tracking."""
import random
import sys
import time
sys.setrecursionlimit(100000)

def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows, each of length k+1.
    Uses a bitset-based backtracking solver with strong pruning and forward checking.
    """
    # Improved greedy baseline with multiple restarts
    def greedy():
        best = None
        best_scope = float('inf')
        for attempt in range(20):
            rng = random.Random(attempt * 7919)
            used = 0
            rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    valid = True
                    for a in row:
                        d = x - a
                        if (used >> d) & 1:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for a in row[:-1]:
                            d = x - a
                            used |= (1 << d)
                rows.append(row)
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best = rows
        return best

    best_rows = greedy()
    best_scope = max(max(r) for r in best_rows)
    
    def solve_scope(scope, seed, max_nodes=5000000):
        rng = random.Random(seed)
        used_bits = 0
        rows = []
        nodes = [0]
        
        def build_row(idx):
            nonlocal used_bits, nodes
            if nodes[0] > max_nodes:
                return False
            nodes[0] += 1
            
            if idx == n_rows:
                return True
            
            row = [0]
            row_diff_bits = 0
            
            # Symmetry breaking: rows sorted by second element
            if idx == 0:
                min_start = 1
            else:
                min_start = rows[idx - 1][1] + 1
            
            def rec(row, row_diff_bits, min_x):
                nonlocal used_bits, nodes
                if nodes[0] > max_nodes:
                    return False
                nodes[0] += 1
                
                if len(row) == k + 1:
                    if used_bits & row_diff_bits:
                        return False
                    
                    used_bits |= row_diff_bits
                    rows.append(row)
                    if build_row(idx + 1):
                        return True
                    else:
                        rows.pop()
                        used_bits ^= row_diff_bits
                        return False

                remaining = (k + 1) - len(row)
                max_x = scope - (remaining - 1)
                
                if max_x < min_x:
                    return False

                blocked = used_bits | row_diff_bits
                candidates = []
                
                for x in range(min_x, max_x + 1):
                    ok = True
                    for a in row:
                        d = x - a
                        if (blocked >> d) & 1:
                            ok = False
                            break
                    if ok:
                        candidates.append(x)
                
                if not candidates:
                    return False
                
                # Forward checking: verify remaining positions can be filled
                # For each candidate, check if enough future values exist
                feasible_candidates = []
                for x in candidates:
                    new_blocked = blocked | 0
                    new_diffs = 0
                    for a in row:
                        d = x - a
                        new_diffs |= (1 << d)
                    new_blocked = blocked | new_diffs
                    
                    # Check if we can still fill remaining - 1 positions
                    rem = remaining - 1
                    # Need rem more values from x+1 to scope
                    # Each must have distinct differences with all row + [x]
                    # Simple check: ensure there are enough candidate values
                    count = 0
                    for future_x in range(x + 1, scope + 1):
                        ok2 = True
                        for b in row + [x]:
                            d = future_x - b
                            if (new_blocked >> d) & 1:
                                ok2 = False
                                break
                        if ok2:
                            count += 1
                            if count >= rem:
                                break
                    if count >= rem:
                        feasible_candidates.append(x)
                
                if not feasible_candidates:
                    return False
                
                # Value ordering: prefer values that leave more flexibility
                # Sort by number of future candidates (ascending = fail first)
                def flexibility(x):
                    new_diffs = 0
                    for a in row:
                        d = x - a
                        new_diffs |= (1 << d)
                    new_blocked = blocked | new_diffs
                    count = 0
                    for future_x in range(x + 1, scope + 1):
                        ok2 = True
                        for b in row + [x]:
                            d = future_x - b
                            if (new_blocked >> d) & 1:
                                ok2 = False
                                break
                        if ok2:
                            count += 1
                    return count
                
                # For speed, just use a simple heuristic
                if len(feasible_candidates) <= 8:
                    x = feasible_candidates[0]
                else:
                    # Pick from a diverse subset
                    step = max(1, len(feasible_candidates) // 10)
                    subset = feasible_candidates[::step][:10]
                    if seed % 2 == 0:
                        x = subset[0]
                    else:
                        x = rng.choice(subset)
                
                new_diffs = 0
                for a in row:
                    d = x - a
                    new_diffs |= (1 << d)
                
                if rec(row + [x], row_diff_bits | new_diffs, x + 1):
                    return True
                return False
            
            return rec(row, row_diff_bits, min_start)
        
        success = build_row(0)
        if success:
            return rows
        return None

    t0 = time.time()
    time_limit = 115
    best_found = best_rows
    best_scope_found = best_scope
    
    # Compute a better lower bound
    # Counting bound: total differences = n_rows * k*(k+1)/2
    # Each difference in [1, scope] used at most once
    # So scope >= n_rows * k*(k+1)/2
    total_diffs = n_rows * k * (k + 1) // 2
    lower_bound = total_diffs
    
    # For DTS(4,7): 4 * 7*8/2 = 112
    # Start from min(best_scope - 1, lower_bound + 25) to avoid too many scopes
    start_scope = min(best_scope - 1, lower_bound + 30)
    scopes_to_check = list(range(start_scope, lower_bound - 1, -1))
    
    # Adaptive time allocation
    total_scopes = len(scopes_to_check)
    
    for i, scope in enumerate(scopes_to_check):
        if time.time() - t0 > time_limit:
            break
        
        # Allocate more time to higher scopes (more likely to have solutions)
        # and less to lower ones
        remaining_scopes = total_scopes - i
        elapsed = time.time() - t0
        remaining_time = time_limit - elapsed
        # Weight: give proportionally more time to earlier (higher) scopes
        weight = (total_scopes - i) / (sum(range(1, total_scopes + 1)))
        slice_time = max(0.5, remaining_time * weight * 1.5)
        slice_time = min(slice_time, 8.0)
        
        if slice_time < 0.3:
            break
            
        found = None
        seed = 0
        scope_start = time.time()
        
        while time.time() - scope_start < slice_time:
            budget = 8000000
            res = solve_scope(scope, seed, max_nodes=budget)
            if res:
                found = res
                break
            seed += 1
            # After 5 failures, increase budget slightly
            if seed % 5 == 0:
                budget = int(budget * 1.2)
        
        if found:
            best_found = found
            best_scope_found = scope
        # Continue searching for smaller scopes

    return best_found
