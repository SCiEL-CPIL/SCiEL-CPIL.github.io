import random
import time
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    time_limit = 115.0  # Leave buffer for 120s limit
    
    MAX_SCOPE_ESTIMATE = 256

    def time_up():
        return time.time() - start_time > time_limit

    best_result = None
    best_scope = float('inf')
    
    def greedy_initialization():
        used_diffs = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            last_val = 0
            while len(row) < k + 1:
                found = False
                for x in range(last_val + 1, MAX_SCOPE_ESTIMATE):
                    conflict = False
                    new_mask = 0
                    for a in row:
                        d = x - a
                        bit = 1 << d
                        if used_diffs & bit:
                            conflict = True
                            break
                        new_mask |= bit
                    if not conflict:
                        row.append(x)
                        used_diffs |= new_mask
                        last_val = x
                        found = True
                        break
                if not found:
                    return None, float('inf')
            rows.append(row)
        
        scope = max(r[-1] for r in rows)
        return rows, scope

    init_rows, init_scope = greedy_initialization()
    if init_rows:
        best_result = [r[:] for r in init_rows]
        best_scope = init_scope

    def solve_attempt(target_scope, rng_seed):
        nonlocal best_result, best_scope
        
        rng = random.Random(rng_seed)
        used_diffs = 0
        rows = []
        
        def _backtrack(row_idx, current_row, current_max_val):
            nonlocal used_diffs, rows, best_scope, best_result
            
            if time_up():
                return False
                
            if row_idx == n_rows:
                # All rows constructed
                # Check if we found a better solution
                # Since we enforce x < target_scope, current_max_val < target_scope
                # But we want the actual max.
                # Actually, if we complete all rows, the scope is max(r[-1] for r in rows)
                # But we are building row by row.
                # The current_max_val is the max of all elements seen so far.
                # Since rows are built sequentially and we don't backtrack previous rows' maxes easily in this structure,
                # let's just calculate it.
                actual_scope = max(r[-1] for r in rows)
                if actual_scope < best_scope:
                    best_scope = actual_scope
                    best_result = [r[:] for r in rows]
                return True
            
            if len(current_row) == k + 1:
                # Row complete
                rows.append(current_row)
                if _backtrack(row_idx + 1, [0], current_max_val):
                    return True
                rows.pop()
                return False
            
            last_val = current_row[-1]
            
            # Upper bound for next element x
            # x must be > last_val
            # x must be < target_scope (strictly, because if x == target_scope, scope is at least target_scope,
            # but we are looking for scope < target_scope? 
            # The task says "largest entry as small as possible".
            # If we set target_scope = S, we are looking for a solution with max entry < S.
            # So x < S.
            upper_bound = target_scope - 1
            
            # Additional pruning: if we need to place (k+1 - len(current_row)) more elements,
            # and the minimum possible next element is last_val + 1, then the maximum possible
            # final element is at least last_val + (k + 1 - len(current_row))
            remaining_elements = k + 1 - len(current_row)
            if last_val + remaining_elements >= target_scope:
                return False
            
            candidates = []
            
            # Heuristic:
            # We want to prefer candidates that use "large" differences (low scarcity).
            # Cost = sum(1.0 / d) for d in new_diffs.
            # We also want to keep x small.
            
            for x in range(last_val + 1, upper_bound + 1):
                # Check if new differences conflict with used_diffs
                # new_diffs = {x - a for a in current_row}
                # We need to check if any (x - a) is in used_diffs.
                # Also, we need to check if (x - a) are distinct among themselves?
                # Yes, because a are distinct, x - a are distinct.
                
                # Bitmask check
                conflict = False
                new_mask = 0
                for a in current_row:
                    d = x - a
                    if d >= MAX_SCOPE_ESTIMATE:
                        # Should not happen if target_scope is small
                        continue
                    bit = 1 << d
                    if used_diffs & bit:
                        conflict = True
                        break
                    new_mask |= bit
                
                if conflict:
                    continue
                
                # Calculate cost
                # We can approximate cost by summing 1/d
                # To make it faster, we can just use the smallest d as a proxy?
                # Or sum of 1/d is fine for small k.
                cost = 0.0
                for a in current_row:
                    d = x - a
                    cost += 1.0 / d
                
                # Add small random noise to break ties and allow exploration
                noise = rng.random() * 0.01
                
                candidates.append((cost + noise, x, new_mask))
                
                # Limit candidates to avoid explosion
                # If we have too many, we might want to pick the best few.
                # But for now, we break after a certain number to keep it fast.
                if len(candidates) > 30:
                    break
            
            if not candidates:
                return False
            
            # Sort candidates
            candidates.sort(key=lambda item: item[0])
            
            # Try candidates
            # Randomize the order of the top candidates to explore different branches
            # We can shuffle the top 5 or so.
            top_n = min(8, len(candidates))
            top_candidates = candidates[:top_n]
            rng.shuffle(top_candidates)
            remaining_candidates = candidates[top_n:]
            
            for cost, x, new_mask in top_candidates + remaining_candidates:
                # Recurse
                old_used = used_diffs
                used_diffs = used_diffs | new_mask
                
                if _backtrack(row_idx, current_row + [x], max(current_max_val, x)):
                    return True
                
                # Restore state
                used_diffs = old_used
            
            return False
        
        return _backtrack(0, [0], 0)

    seed_counter = 0
    current_target = best_scope if best_result else 200
    
    while time.time() - start_time < time_limit:
        if best_result is not None:
            target = best_scope
        else:
            target = current_target
            
        found = solve_attempt(target, seed_counter)
        seed_counter += 1

    if best_result is None:
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
