import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # 1. Baseline Greedy Solution to get an initial upper bound
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    
    best_scope = max(max(row) for row in best_solution)
    
    # Calculate theoretical lower bound
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs
    
    if best_scope <= lower_bound:
        return best_solution

    # 2. Bitset-accelerated backtracking to reduce scope
    current_target = best_scope - 1
    
    while time.time() < deadline and current_target >= lower_bound:
        # Try to find a solution with max entry <= current_target
        found = False
        attempts = 0
        max_attempts = 50 # Increase attempts for better chance of finding a solution in tight scope
        
        while time.time() < deadline and not found and attempts < max_attempts:
            attempts += 1
            solution = search_with_scope_randomized(n_rows, k, current_target, deadline)
            if solution:
                new_scope = max(max(row) for row in solution)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_solution = solution
                    found = True
                    if best_scope <= lower_bound:
                        break
        
        if found:
            current_target = best_scope - 1
        else:
            if time.time() > deadline - 10:
                break
            current_target -= 1

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

def search_with_scope_randomized(n_rows, k, max_scope, deadline):
    """
    Attempts to construct n_rows rows of k+1 elements each,
    such that all elements are <= max_scope and all differences are unique.
    Uses randomization to explore different structures.
    """
    used = bytearray(max_scope + 1)
    rows = []
    
    def backtrack(row_idx, pos, current_mark, row_so_far):
        if time.time() > deadline:
            return None
            
        if pos == k + 1:
            return list(row_so_far)
        
        valid_candidates = []
        
        # Calculate max possible x to allow remaining elements to fit
        remaining = (k + 1) - pos
        max_x = max_scope - (remaining - 1)
        
        for x in range(current_mark + 1, max_x + 1):
            conflict = False
            for a in row_so_far:
                d = x - a
                if used[d]:
                    conflict = True
                    break
            if not conflict:
                valid_candidates.append(x)
                
        if not valid_candidates:
            return None
        
        # Randomize the order of candidates to explore different branches
        # Bias towards smaller candidates but allow some randomness
        candidates_to_try = list(valid_candidates)
        if len(candidates_to_try) > 10:
            first_10 = candidates_to_try[:10]
            random.shuffle(first_10)
            candidates_to_try = first_10 + candidates_to_try[10:]
        else:
            random.shuffle(candidates_to_try)
            
        for x in candidates_to_try:
            new_diffs = []
            for a in row_so_far:
                d = x - a
                used[d] = 1
                new_diffs.append(d)
                
            res = backtrack(row_idx, pos + 1, x, row_so_far + [x])
            if res is not None:
                return res
                
            for d in new_diffs:
                used[d] = 0
                
        return None

    for i in range(n_rows):
        row = backtrack(i, 1, 0, [0])
        if row is None:
            return None
        rows.append(row)
        
    return rows
