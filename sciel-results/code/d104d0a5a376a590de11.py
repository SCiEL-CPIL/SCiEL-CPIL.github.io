import time
import random
import sys
from functools import lru_cache
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Theoretical lower bound
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs
    
    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)
    
    if best_scope <= lower_bound:
        return best_solution
    
    # 2. Local improvement to reduce scope
    best_solution = _improve_scope(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)
    
    if best_scope <= lower_bound:
        return best_solution
    
    # 3. Try to find better solutions with randomized backtracking
    # for target scopes below current best
    attempt = 0
    while time.time() < deadline - 10.0:
        target_scope = best_scope - 1
        if target_scope < lower_bound:
            break
        
        # Try multiple random restarts for this target scope
        found = False
        for _ in range(30):
            if time.time() > deadline - 10.0:
                break
            
            solution = _backtracking_search(n_rows, k, target_scope, deadline, attempt)
            if solution:
                new_scope = max(max(row) for row in solution)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_solution = solution
                    found = True
                    # Try to improve further
                    improved = _improve_scope(best_solution, n_rows, k, deadline)
                    new_scope = max(max(row) for row in improved)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = improved
                    if best_scope <= lower_bound:
                        return best_solution
                    break
            attempt += 1
        
        if not found and target_scope <= lower_bound + 5:
            # If we can't find anything near the lower bound, 
            # try a slightly larger target to make progress
            target_scope = best_scope - 1
            if target_scope < lower_bound:
                break
    
    # 4. Final polish pass - try to squeeze out last bit of improvement
    if time.time() < deadline - 2.0:
        polished = _polish_scope(best_solution, n_rows, k, deadline)
        if polished:
            new_scope = max(max(row) for row in polished)
            if new_scope < best_scope:
                best_scope = new_scope
                best_solution = polished
    
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

def _backtracking_search(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    Backtracking search to build n_rows of k+1 increasing integers starting at 0,
    with all pairwise differences distinct across all rows, and max element <= max_scope.
    Uses symmetry breaking: row[i][1] > row[i-1][1] for i > 0.
    """
    rng = random.Random(seed_offset * 12345 + 678)
    used_diffs = set()
    rows = []
    k_plus_1 = k + 1
    
    def build_row(pos, current_mark, row_so_far, min_first_elem=1):
        if time.time() > deadline:
            return None
            
        if pos == k_plus_1:
            return list(row_so_far)
        
        # Find valid candidates
        valid_candidates = []
        # Prune upper bound: we need at least (k_plus_1 - pos - 1) more elements,
        # each at least 1 greater than previous, so max possible x is:
        max_x = max_scope - (k_plus_1 - pos - 1)
        if max_x <= current_mark:
            return None
            
        # Symmetry breaking: enforce increasing first elements
        start_x = max(current_mark + 1, min_first_elem)
        for x in range(start_x, max_x + 1):
            new_diffs = {x - a for a in row_so_far}
            if new_diffs & used_diffs:
                continue
            valid_candidates.append(x)
        
        if not valid_candidates:
            return None
        
        # Randomize the order of candidates to escape local minima
        # But bias towards smaller values to keep scope small
        n_smallest = min(15, len(valid_candidates))
        smallest = valid_candidates[:n_smallest]
        
        # With some probability, also consider larger candidates
        if len(valid_candidates) > n_smallest and rng.random() < 0.4:
            extra = rng.sample(valid_candidates[n_smallest:], min(8, len(valid_candidates) - n_smallest))
            candidates_to_try = smallest + extra
        else:
            candidates_to_try = smallest
        
        rng.shuffle(candidates_to_try)
        
        for x in candidates_to_try:
            new_diffs = {x - a for a in row_so_far}
            
            used_diffs.update(new_diffs)
            # For next position, enforce symmetry breaking on first element
            if pos == 1:
                next_min = x + 1
            else:
                next_min = 1
            res = build_row(pos + 1, x, row_so_far + [x], next_min)
            if res is not None:
                return res
            used_diffs.difference_update(new_diffs)
        
        return None
    
    for i in range(n_rows):
        # Symmetry breaking: first element of row i must be > first element of row i-1
        if i == 0:
            min_first = 1
        else:
            min_first = rows[i-1][1] + 1
        
        row = build_row(1, 0, [0], min_first)
        if row is None:
            return None
        rows.append(row)
        used_diffs.update({row[j] - row[i] for i in range(k_plus_1) for j in range(i + 1, k_plus_1)})
    
    return rows

def _improve_scope(solution, n_rows, k, deadline):
    """
    Try to reduce the scope of a valid solution by local search.
    We try to shift elements down to reduce the maximum value.
    """
    if not solution:
        return solution
    
    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1
    
    # Collect all used differences
    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])
    
    # Try to reduce scope by shifting elements
    # Strategy: for each row, try to shift elements down
    improved = True
    pass_count = 0
    while improved and time.time() < deadline and pass_count < 50:
        improved = False
        pass_count += 1
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos in range(k_plus_1 - 1, 0, -1):  # from end to start
                # Try to shift this element down
                original = row[pos]
                # Try values from original-1 down to row[pos-1]+1
                for new_val in range(original - 1, row[pos - 1], -1):
                    # Check if new value creates conflicts
                    new_diffs = {new_val - row[i] for i in range(pos)}
                    # Also need to check diffs with elements after pos
                    for j in range(pos + 1, k_plus_1):
                        new_diffs.add(row[j] - new_val)
                    
                    # Remove old diffs
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(original - row[i])
                    for j in range(pos + 1, k_plus_1):
                        old_diffs.add(row[j] - original)
                    
                    # Check if new_diffs conflict with (used_diffs - old_diffs)
                    if new_diffs & (used_diffs - old_diffs):
                        continue
                    
                    # Check if new_diffs are all positive
                    if any(d <= 0 for d in new_diffs):
                        continue
                    
                    # Apply the change
                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    row[pos] = new_val
                    improved = True
                    break
    
    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        # Revert if no improvement
        return [list(row) for row in solution]

def _polish_scope(solution, n_rows, k, deadline):
    """
    Final high-precision polish pass: try multiple small perturbations
    and accept if they lead to a valid solution with smaller scope.
    """
    if not solution:
        return solution
    
    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1
    
    # Collect all used differences
    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])
    
    rng = random.Random(42)
    
    # Try random perturbations: pick a random element and try shifting it
    for _ in range(200):
        if time.time() > deadline:
            break
        
        # Pick a random row and position (not the first element)
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, k_plus_1 - 1)
        row = best[row_idx]
        original = row[pos]
        
        # Try shifting down by 1 or 2
        for delta in [1, 2]:
            new_val = original - delta
            if new_val <= row[pos - 1]:
                continue
            
            # Check if new value creates conflicts
            new_diffs = {new_val - row[i] for i in range(pos)}
            for j in range(pos + 1, k_plus_1):
                new_diffs.add(row[j] - new_val)
            
            # Remove old diffs
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(original - row[i])
            for j in range(pos + 1, k_plus_1):
                old_diffs.add(row[j] - original)
            
            # Check if new_diffs conflict with (used_diffs - old_diffs)
            if new_diffs & (used_diffs - old_diffs):
                continue
            
            # Check if new_diffs are all positive
            if any(d <= 0 for d in new_diffs):
                continue
            
            # Apply the change
            used_diffs.difference_update(old_diffs)
            used_diffs.update(new_diffs)
            row[pos] = new_val
            break
    
    # Run a quick improvement pass after perturbations
    result = _improve_scope(best, n_rows, k, deadline)
    new_scope = max(max(row) for row in result)
    if new_scope < best_scope:
        return result
    else:
        return [list(row) for row in solution]
