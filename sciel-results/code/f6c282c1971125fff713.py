import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0
    
    # 1. Baseline Greedy Solution
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    
    best_scope = max(max(row) for row in best_solution)
    
    # Calculate theoretical lower bound
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs
    
    if best_scope <= lower_bound:
        return best_solution

    # 2. Try to improve by reducing scope
    # Start from best_scope - 1 and try to find solutions with smaller scopes
    current_target = best_scope - 1
    
    while time.time() < deadline and current_target >= lower_bound:
        # Try multiple random restarts at this target scope
        success = False
        attempts = 0
        max_attempts_per_target = 50
        
        while time.time() < deadline and not success and attempts < max_attempts_per_target:
            attempts += 1
            solution = randomized_search(n_rows, k, current_target, deadline)
            if solution:
                new_scope = max(max(row) for row in solution)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_solution = solution
                    success = True
                    if best_scope <= lower_bound:
                        break
        
        if success:
            current_target = best_scope - 1
        else:
            # Failed to find solution at this target, try slightly larger target
            # or give up if we've spent too much time
            if time.time() > deadline - 30:
                break
            # Try a target that's a bit less aggressive
            current_target = max(lower_bound, current_target + 1)
            if current_target >= best_scope:
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

def randomized_search(n_rows, k, max_scope, deadline):
    used_diffs = set()
    rows = []
    
    def build_row(row_idx, pos, current_mark, row_so_far):
        if time.time() > deadline:
            return None
            
        if pos == k + 1:
            return row_so_far
        
        # Find valid candidates for the next mark
        valid_candidates = []
        
        for x in range(current_mark + 1, max_scope + 1):
            new_diffs = {x - a for a in row_so_far}
            if new_diffs & used_diffs:
                continue
            
            valid_candidates.append(x)
            
            # Early termination: if we have enough candidates, stop
            if len(valid_candidates) >= 30:
                break
        
        if not valid_candidates:
            return None
        
        # Heuristic: prefer smaller candidates but with some randomness
        # Use a weighted sampling that favors smaller values
        sample_pool = valid_candidates[:min(20, len(valid_candidates))]
        
        # Weighted random selection: earlier candidates have higher probability
        weights = [1.0 / (i + 1) for i in range(len(sample_pool))]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Pick 3-5 candidates to try
        num_to_try = min(4, len(sample_pool))
        indices = list(range(len(sample_pool)))
        random.shuffle(indices)
        
        # Prioritize smaller candidates by sorting selected indices
        selected_indices = sorted(indices[:num_to_try])
        
        for idx in selected_indices:
            x = sample_pool[idx]
            new_diffs = {x - a for a in row_so_far}
            
            used_diffs.update(new_diffs)
            
            res = build_row(row_idx, pos + 1, x, row_so_far + [x])
            if res is not None:
                return res
            
            used_diffs.difference_update(new_diffs)
            
        return None

    for i in range(n_rows):
        row = build_row(i, 1, 0, [0])
        if row is None:
            return None
        rows.append(row)
        
    return rows
