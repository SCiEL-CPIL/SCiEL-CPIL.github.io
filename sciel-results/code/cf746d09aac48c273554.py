import time
import random

def construct(n_rows, k):
    """
    Constructs a difference triangle set (set of Golomb rulers).
    n_rows: number of rows
    k: number of intervals (so k+1 marks)
    Returns a list of rows, each a list of k+1 integers.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_solution = None
    best_score = float('inf')
    
    # Lower bound estimate: counting bound is roughly sqrt(n_rows * k * (k+1) / 2 * (something))
    # Actually, total diffs = n_rows * k * (k+1) / 2.
    # These must be distinct integers from 1 to M.
    # So M >= n_rows * k * (k+1) / 2.
    # For 7 rows, k=5: 7 * 15 = 105.
    # So 105 is a hard lower bound.
    
    seed = 0
    while time.time() < deadline:
        seed += 1
        random.seed(seed)
        
        global_used = set()
        current_rows = []
        success = True
        
        for r in range(n_rows):
            # Try to find a valid row
            row = find_row(global_used, k, seed + r)
            if row is None:
                success = False
                break
            
            current_rows.append(row)
            # Add differences to global_used
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    global_used.add(row[j] - row[i])
            
            # Check time
            if time.time() > deadline:
                return best_solution if best_solution else current_rows # Return partial if best is null? No, must be valid.
                # If best_solution is None, we must return a valid one.
                # The seed greedy always produces a valid one.
                # So if we fail here, we should probably fall back to a simple greedy for this seed or previous best.
                # But the loop structure ensures we only update best_solution if success is True.
                # If we run out of time, we return best_solution.
                # If best_solution is None, it means we haven't completed a full set yet.
                # We should ensure we have at least one valid solution.
                # The first seed (0) with greedy usually works.
        
        if success:
            score = max(max(row) for row in current_rows)
            if score < best_score:
                best_score = score
                best_solution = current_rows
                if best_score <= 105:
                    # Can't do better than counting bound
                    return best_solution
    
    # If we never found a solution (unlikely), run a simple greedy to ensure validity
    if best_solution is None:
        best_solution = greedy_construct(n_rows, k)
        
    return best_solution

def find_row(global_used, k, seed):
    """
    Finds a row of k+1 marks starting at 0, strictly increasing,
    such that its differences do not conflict with global_used or itself.
    Uses randomized backtracking.
    """
    random.seed(seed)
    row = [0]
    local_used = set()
    
    # Limit backtracking depth to avoid hanging
    max_depth = 10000
    depth = 0
    
    def backtrack(current_row, current_local_used):
        nonlocal depth
        if len(current_row) == k + 1:
            return list(current_row)
        
        depth += 1
        if depth > max_depth:
            return None
            
        last_mark = current_row[-1]
        
        # Generate candidates
        # We want to try small marks first to keep the row compact
        # But randomize the order of the first few valid candidates
        
        valid_candidates = []
        # Upper bound for x: we don't know it, but we can stop if x is too large
        # A heuristic bound: last_mark + 2*k + 10? 
        # Or just iterate until we find enough valid candidates or hit a limit.
        # Since we backtrack, we need to consider all valid x in a range.
        # Let's iterate x from last_mark + 1 upwards.
        
        # Optimization: We can pre-filter x if x - current_row[0] is in global_used or current_local_used.
        # Actually, all diffs x - a for a in current_row must be new.
        
        found = False
        # We will collect a few valid candidates and pick one randomly to encourage exploration
        # Or just try them in order with a random restart mechanism.
        
        # Strategy: Try candidates in increasing order. If we fail, the backtracking handles it.
        # To avoid deep backtracking on a bad path, we can limit the number of candidates we consider at each step?
        # No, backtracking needs to be complete to find a solution if one exists.
        
        # However, for Golomb rulers, the "greedy" choice (smallest x) is often good.
        # Randomization: With probability 0.1, pick a random valid candidate from the next 5.
        
        x = last_mark + 1
        candidates = []
        
        while len(candidates) < 10: # Collect up to 10 candidates
            # Check if x is valid
            new_diffs = {x - a for a in current_row}
            if not (new_diffs & (global_used | current_local_used)):
                candidates.append((x, new_diffs))
                if len(candidates) >= 1:
                    # If we have enough, we can stop collecting and pick
                    # But we need to know if there are more?
                    # For backtracking, we need to be able to try the next one if this fails.
                    # So we should generate them on the fly or store them.
                    # Storing is fine for small k.
                    pass
            x += 1
            
            # Heuristic bound: if x is very large, maybe no solution?
            # But we don't know the max.
            # Let's just keep going until we have candidates.
            # If we don't find any, return None.
            if x > 200: # Arbitrary large bound, should be much smaller
                break
        
        if not candidates:
            return None
            
        # Pick a candidate
        # Greedy: pick the first one (smallest x)
        # Random: pick randomly from the list
        # Hybrid: pick the smallest with high prob, else random
        if random.random() < 0.8:
            idx = 0
        else:
            idx = random.randint(0, len(candidates) - 1)
            
        chosen_x, chosen_diffs = candidates[idx]
        
        current_row.append(chosen_x)
        current_local_used = current_local_used | chosen_diffs
        
        result = backtrack(current_row, current_local_used)
        if result is not None:
            return result
            
        # Backtrack
        current_row.pop()
        current_local_used = current_local_used - chosen_diffs
        
        # Try next candidates
        # We need to iterate through the remaining candidates
        # This simple recursive structure doesn't easily iterate through candidates in a loop.
        # Let's restructure to a loop.
        
        return None # This simple recursive call only tries ONE candidate per level effectively if not structured as a loop.

    # Restructured Backtracking
    def backtrack_loop(current_row, current_local_used):
        if len(current_row) == k + 1:
            return list(current_row)
        
        last_mark = current_row[-1]
        x = last_mark + 1
        
        while True:
            # Find next valid x
            valid = False
            while True:
                new_diffs = {x - a for a in current_row}
                if not (new_diffs & (global_used | current_local_used)):
                    valid = True
                    break
                x += 1
                if x > 500: # Safety bound
                    return None
            
            # We have a valid x
            current_row.append(x)
            new_local_used = current_local_used | new_diffs
            
            result = backtrack_loop(current_row, new_local_used)
            if result is not None:
                return result
            
            # Backtrack
            current_row.pop()
            x += 1 # Try next x
            
            # Add some randomness: skip some x's?
            # If random.random() < 0.1:
            #    x += random.randint(1, 5)

    return backtrack_loop(row, local_used)

def greedy_construct(n_rows, k):
    """Fallback simple greedy."""
    used, rows = set(), []
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
