import time
import random

def construct(n_rows, k):
    start_time = time.time()
    # 150s limit, use 110s for safety
    deadline = start_time + 110.0
    
    # 1. Baseline Greedy Solution
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    
    best_scope = max(max(row) for row in best_solution)
    
    # Calculate theoretical lower bound
    # Total number of differences is n_rows * k * (k+1) / 2
    # These must be distinct integers in [1, M]
    # So M >= n_rows * k * (k+1) / 2
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs
    
    if best_scope <= lower_bound:
        return best_solution

    # 2. Randomized Search to Improve Scope
    # We try to find a solution with scope < best_scope
    # We iteratively decrease the target scope
    
    # To allow for randomization, we set a seed or just use random
    # We run multiple attempts
    
    while time.time() < deadline:
        target_scope = best_scope - 1
        if target_scope < lower_bound:
            break
        
        # Try to find a solution with max element <= target_scope
        # We do this with randomized backtracking
        # We might fail, so we retry
        
        # To avoid getting stuck in one specific failure mode,
        # we can just call the search function which randomizes internally.
        
        solution = randomized_search(n_rows, k, target_scope, deadline)
        
        if solution:
            new_scope = max(max(row) for row in solution)
            if new_scope < best_scope:
                best_scope = new_scope
                best_solution = solution
                
                # If we reached the lower bound, we can stop
                if best_scope <= lower_bound:
                    break
            else:
                # If solution found but scope didn't improve (shouldn't happen if target_scope < best_scope)
                # Actually, if we find a solution with max <= target_scope, and target_scope < best_scope,
                # then new_scope <= target_scope < best_scope.
                # So it always improves.
                pass

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

def randomized_search(n_rows, k, max_scope, deadline):
    used_diffs = set()
    rows = []
    
    # Recursive helper
    def build_row(row_idx, pos, current_mark, row_so_far):
        if time.time() > deadline:
            return None
            
        if pos == k + 1:
            return row_so_far
        
        # Find valid candidates for the next mark
        # x must be > current_mark and <= max_scope
        # Diffs {x - a for a in row_so_far} must be disjoint from used_diffs
        
        valid_candidates = []
        # We can break early if we find enough candidates?
        # No, we want to sample from the smallest ones.
        # So we iterate from current_mark + 1 upwards.
        
        for x in range(current_mark + 1, max_scope + 1):
            # Quick check: if x is too large, we might not be able to finish?
            # Not a simple check.
            
            new_diffs = {x - a for a in row_so_far}
            if new_diffs & used_diffs:
                continue
            
            valid_candidates.append(x)
            
            # Optimization: If we have many candidates, we don't need to check all
            # if we are going to sample from the smallest ones.
            # But we don't know the "validity" of future steps.
            # For now, collect up to a limit?
            # If we collect all, it's O(M). M is small.
            
        if not valid_candidates:
            return None
        
        # Heuristic: Sample from the smallest valid candidates
        # This encourages smaller scopes
        # We take the first 20 (smallest) and sample from them
        sample_pool = valid_candidates[:20]
        sample_size = min(5, len(sample_pool))
        
        # Randomly pick a subset of candidates to try
        # This introduces randomness to escape local minima
        candidates_to_try = random.sample(sample_pool, sample_size)
        
        # Shuffle them to randomize the order of attempts
        random.shuffle(candidates_to_try)
        
        for x in candidates_to_try:
            new_diffs = {x - a for a in row_so_far}
            
            # Add to used
            used_diffs.update(new_diffs)
            
            # Recurse
            res = build_row(row_idx, pos + 1, x, row_so_far + [x])
            if res is not None:
                return res
            
            # Undo
            used_diffs.difference_update(new_diffs)
            
        return None

    for i in range(n_rows):
        row = build_row(i, 1, 0, [0])
        if row is None:
            return None
        rows.append(row)
        
    return rows
