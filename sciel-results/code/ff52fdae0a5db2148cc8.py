import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Theoretical lower bound: total number of differences
    num_diffs = n_rows * k * (k + 1) // 2
    
    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)
    
    if best_scope <= num_diffs:
        return best_solution

    # 2. Iterative deepening / target scope search
    # We try to find solutions with scope M, M-1, M-2 ...
    # We start from best_scope - 1 down to num_diffs
    
    # To avoid spending too much time on a single target, we limit attempts per target
    # and reduce attempts as we go lower if time is tight.
    
    target_scope = best_scope - 1
    
    # We will try to reduce the scope.
    # A key insight: if we can't find a solution for M, we might not be able to find M-1 easily.
    # But sometimes M-1 is easier due to different structure? Unlikely.
    # However, random restarts help.
    
    while target_scope >= num_diffs and time.time() < deadline - 5.0:
        found = False
        # Try multiple random restarts for this target scope
        # The number of tries can be adaptive.
        # If we are close to the bound, try more.
        max_tries = 50
        if time.time() > deadline - 30.0:
            max_tries = 10
        if time.time() > deadline - 10.0:
            max_tries = 3
            
        for attempt in range(max_tries):
            if time.time() > deadline - 5.0:
                break
            
            # Random seed based on target and attempt
            seed = hash((target_scope, attempt, n_rows, k)) % (2**32)
            solution = _backtracking_search(n_rows, k, target_scope, deadline, seed)
            
            if solution:
                new_scope = max(max(row) for row in solution)
                if new_scope <= target_scope:
                    best_scope = new_scope
                    best_solution = solution
                    found = True
                    # If we found a solution, we can try to reduce further immediately
                    # But the outer loop will decrement target_scope anyway.
                    break
            
        if not found:
            # If we failed to find a solution for this target, 
            # it's likely we can't find lower ones either, 
            # but it's not strictly true. 
            # However, to save time, we might break if we've failed for a while.
            # For now, we just decrement and try again, but with fewer tries if time is low.
            pass
        
        target_scope -= 1
        
        # If we hit the lower bound, stop
        if target_scope < num_diffs:
            break

    return best_solution

def greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _backtracking_search(n_rows, k, max_scope, deadline, seed):
    """
    Backtracking search to build n_rows of k+1 increasing integers starting at 0,
    with all pairwise differences distinct across all rows, and max element <= max_scope.
    Uses bitmask for fast difference checking.
    """
    rng = random.Random(seed)
    rows = []
    k_plus_1 = k + 1
    
    # Bitmask for used differences. 
    # max_scope is around 120-150, so an integer bitmask is efficient.
    used_mask = 0
    
    def build_row(pos, current_mark, row_so_far):
        nonlocal used_mask, deadline
        
        if time.time() > deadline:
            return None
            
        if pos == k_plus_1:
            return list(row_so_far)
        
        # Prune upper bound: we need at least (k_plus_1 - pos - 1) more elements,
        # each at least 1 greater than previous.
        # So max x is max_scope - (k_plus_1 - pos - 1)
        max_x = max_scope - (k_plus_1 - pos - 1)
        
        if max_x <= current_mark:
            return None
            
        # Generate candidates
        # We want to try smaller values first to keep the scope tight, 
        # but randomize to escape local minima.
        candidates = list(range(current_mark + 1, max_x + 1))
        
        if not candidates:
            return None
        
        # Shuffle candidates but bias towards smaller ones?
        # A common heuristic: try the first few smallest, then some random ones.
        # Or just shuffle and try.
        # Let's try a mix: take the first 5 smallest, and 5 random from the rest, shuffle them.
        # This ensures we don't get stuck in a "too big" path immediately.
        
        if len(candidates) > 10:
            smallest = candidates[:5]
            rest = candidates[5:]
            random_part = rng.sample(rest, min(5, len(rest)))
            candidates_to_try = smallest + random_part
            rng.shuffle(candidates_to_try)
        else:
            candidates_to_try = candidates
            rng.shuffle(candidates_to_try)
        
        for x in candidates_to_try:
            # Check conflicts
            conflict = False
            # Check against global used_mask
            # And against internal row differences (which are not in used_mask yet, 
            # but we must ensure they don't conflict with each other or global)
            # Actually, we can just check against used_mask and then add to a local set?
            # No, used_mask is global for ALL rows.
            # Internal differences in the current row must also be distinct from each other.
            # But since the row is increasing, x - a_i are distinct for different a_i.
            # So we only need to check if (x - a_i) is in used_mask.
            
            new_diff_bits = 0
            for a in row_so_far:
                d = x - a
                if d > max_scope: # Should not happen due to max_x bound
                    conflict = True
                    break
                # Check if bit d is set in used_mask
                if (used_mask >> d) & 1:
                    conflict = True
                    break
                new_diff_bits |= (1 << d)
            
            if conflict:
                continue
            
            # Also need to check if new_diff_bits conflict with each other?
            # No, d = x - a_i. Since a_i are distinct, d are distinct.
            
            # Tentatively accept
            used_mask |= new_diff_bits
            
            res = build_row(pos + 1, x, row_so_far + [x])
            if res is not None:
                return res
            
            # Backtrack
            used_mask ^= new_diff_bits
        
        return None
    
    for i in range(n_rows):
        # Randomize the start of the row? No, must start at 0.
        # But we can randomize the order of rows? 
        # The problem is symmetric with respect to rows.
        # However, constructing rows sequentially with a fixed order is standard.
        # To add randomness, we could shuffle the row indices, but the content matters.
        
        row = build_row(1, 0, [0])
        if row is None:
            return None
        rows.append(row)
        
        # Mark all differences in this row as used in the global mask
        for j in range(k_plus_1):
            for i2 in range(j + 1, k_plus_1):
                d = row[i2] - row[j]
                used_mask |= (1 << d)

    return rows
