import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0

    # Theoretical lower bound: we need n_rows * k*(k+1)/2 distinct positive differences
    # Each difference is at least 1, at most scope, so scope >= n_rows * k*(k+1)/2
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
    attempt_counter = 0
    while time.time() < deadline - 5.0:
        target_scope = best_scope - 1
        if target_scope < lower_bound:
            break

        # Try multiple random restarts for this target scope
        found = False
        for attempt in range(30):
            if time.time() > deadline - 5.0:
                break

            solution = _backtracking_search(n_rows, k, target_scope, deadline, attempt_counter)
            attempt_counter += 1
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

        if not found:
            # If we can't find anything at target_scope, try a slightly larger target
            # to make progress, then keep reducing
            if target_scope <= lower_bound + 5:
                # We're near the lower bound but can't find a solution; give up
                break
            else:
                # Try target_scope + 1 to make some progress
                target_scope = min(best_scope - 1, target_scope + 2)
                if target_scope >= best_scope:
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

def _backtracking_search(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    Backtracking search to build n_rows of k+1 increasing integers starting at 0,
    with all pairwise differences distinct across all rows, and max element <= max_scope.
    """
    rng = random.Random(seed_offset * 7919 + 12345)
    used_diffs = set()
    rows = []
    
    # Precompute some useful values
    total_positions = k + 1
    
    def build_row(row_idx, pos, current_mark, row_so_far):
        if time.time() > deadline:
            return None
            
        if pos == total_positions:
            return list(row_so_far)
        
        # Find valid candidates
        valid_candidates = []
        for x in range(current_mark + 1, max_scope + 1):
            new_diffs = {x - a for a in row_so_far}
            if new_diffs & used_diffs:
                continue
            valid_candidates.append(x)
        
        if not valid_candidates:
            return None
        
        # Heuristic: prefer smaller values to keep scope tight
        # But also consider some larger values for diversity
        n_smallest = min(8, len(valid_candidates))
        smallest = valid_candidates[:n_smallest]
        
        # With some probability, also consider larger candidates
        if len(valid_candidates) > n_smallest and rng.random() < 0.4:
            extra = rng.sample(valid_candidates[n_smallest:], min(4, len(valid_candidates) - n_smallest))
            candidates_to_try = smallest + extra
        else:
            candidates_to_try = smallest
        
        rng.shuffle(candidates_to_try)
        
        for x in candidates_to_try:
            new_diffs = {x - a for a in row_so_far}
            
            # Check if we can actually complete the row with remaining positions
            # Each subsequent element needs to be at least 1 more than the previous
            remaining = total_positions - pos - 1
            if x + remaining > max_scope:
                continue
            
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
        used_diffs.update({b - a for a in row for b in row if b > a})
    
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
    
    # Collect all used differences
    used_diffs = set()
    for row in best:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used_diffs.add(row[j] - row[i])
    
    # Try to reduce scope by shifting elements
    # Strategy: for each row, try to shift elements down
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos in range(len(row) - 1, 0, -1):  # from end to start
                # Try to shift this element down
                original = row[pos]
                for new_val in range(row[pos - 1] + 1, original):
                    # Check if new value creates conflicts
                    new_diffs = {new_val - row[i] for i in range(pos)}
                    # Also need to check diffs with elements after pos
                    for j in range(pos + 1, len(row)):
                        new_diffs.add(row[j] - new_val)
                    
                    # Remove old diffs
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(original - row[i])
                    for j in range(pos + 1, len(row)):
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
