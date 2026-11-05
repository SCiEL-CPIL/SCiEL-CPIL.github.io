import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Generate Pool of Candidates
    # We need rows of length k+1 (k intervals)
    # Each row is a Golomb Ruler
    # We want a diverse pool with small max values
    
    pool = []
    rng = random.Random(42)
    
    def generate_candidate(max_limit=200):
        # Randomized greedy generation of a Golomb Ruler
        row = [0]
        used_diffs = set()
        
        for _ in range(k):
            last = row[-1]
            # Find valid next values
            # We want to keep values small, so start from last+1
            # But to ensure diversity, we might skip some
            
            valid_nexts = []
            # Limit search range to avoid infinite loops or huge numbers
            # If we can't find a small one, we might fail or return a large one
            # We want candidates that are "small"
            
            # Heuristic: Search up to last + 20 initially, expand if needed
            # But for a pool, we just want *some* valid rulers
            
            # Let's use a simple backtracking with randomization for robustness
            # Or just greedy with random jumps
            
            # Greedy with randomization:
            # 1. Find smallest valid x
            # 2. With prob 0.5, use it.
            # 3. Else, find a random valid x in [smallest, smallest + 10]
            
            smallest_valid = None
            # Search range: last+1 to last+50 (arbitrary, but should cover most)
            # If k is small, this is fine.
            
            for x in range(last + 1, last + 50):
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs:
                    if smallest_valid is None:
                        smallest_valid = x
                    valid_nexts.append(x)
                    if len(valid_nexts) > 10:
                        break # Enough candidates
            
            if not valid_nexts:
                # Expand search? Or fail?
                # For now, return None if we can't find one quickly
                return None
            
            if rng.random() < 0.7:
                next_x = smallest_valid
            else:
                next_x = rng.choice(valid_nexts)
            
            row.append(next_x)
            used_diffs |= {next_x - a for a in row[:-1]}
            
        max_val = row[-1]
        # Only keep if max_val is reasonably small
        # The counting bound is 105, so we hope for < 120
        if max_val < 150: 
            return (row, used_diffs, max_val)
        return None

    # Generate pool
    target_pool_size = 2000
    attempts = 0
    max_attempts = 10000
    
    while len(pool) < target_pool_size and attempts < max_attempts and time.time() < deadline - 20.0:
        cand = generate_candidate()
        if cand:
            pool.append(cand)
        attempts += 1
        
    if not pool:
        # Fallback: Simple greedy
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

    # Sort pool by max_val
    pool.sort(key=lambda x: x[2])
    
    # 2. Greedy Selection
    def greedy_select(pool, n_rows):
        selected = []
        used_diffs = set()
        
        for row, diffs, max_val in pool:
            if not diffs & used_diffs:
                selected.append((row, diffs, max_val))
                used_diffs |= diffs
                if len(selected) == n_rows:
                    break
        return selected

    current_solution = greedy_select(pool, n_rows)
    
    if len(current_solution) < n_rows:
        # If greedy failed, try to relax or fallback
        # This shouldn't happen if pool is large enough
        # Fallback to simple greedy
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

    current_max = max(item[2] for item in current_solution)
    best_solution = current_solution
    best_max = current_max

    # 3. Local Search / Iterated Local Search
    # Try to improve by swapping rows
    
    # Pre-compute compatibility?
    # Compatibility of candidate C with solution S (excluding row i)
    # means C.diffs & (union of diffs of S \ {i}) == empty
    
    # To speed up, we can just check on the fly.
    
    while time.time() < deadline:
        # Pick a random row index to replace
        i = rng.randint(0, n_rows - 1)
        
        # Compute union of diffs of all other rows
        other_diffs = set()
        for j, (_, d, _) in enumerate(current_solution):
            if j != i:
                other_diffs |= d
        
        # Find a candidate that is compatible with other_diffs
        # and has a smaller max_val than current_solution[i]
        # Or just any candidate that improves the global max
        
        current_row_max = current_solution[i][2]
        
        # Search pool for a better candidate
        # Since pool is sorted, we can iterate
        # But we need one that is compatible
        
        # Optimization: Only consider candidates with max_val < current_max
        # And potentially < current_row_max if we want to reduce that specific row
        
        # Let's just iterate through the pool (which is sorted)
        # and find the first compatible one with max_val < current_max
        
        # To avoid always picking the same one, we can randomize the start or selection
        
        # Simple approach: Iterate through pool
        replacement_found = False
        for cand in pool:
            if cand[2] >= current_max:
                break # Since sorted, no better max available
            
            # Check compatibility
            if not cand[1] & other_diffs:
                # Check if this candidate is better than the current row i
                # If cand[2] < current_row_max, it's definitely an improvement for that row
                # But does it improve the global max?
                # If current_max is determined by another row, replacing i with a smaller one
                # doesn't reduce global max unless i WAS the max.
                
                # Let's just accept if it reduces the max of the replaced row
                # and is compatible.
                
                if cand[2] < current_row_max:
                    # Replace
                    current_solution[i] = cand
                    new_max = max(item[2] for item in current_solution)
                    
                    if new_max < best_max:
                        best_max = new_max
                        best_solution = [item[0] for item in current_solution]
                        current_solution = list(current_solution) # Copy? No, we modify in place
                    
                    replacement_found = True
                    break # Found one, restart loop to pick another row
        
        if not replacement_found:
            # Try to swap with a candidate that has same max but different diffs?
            # Or just continue
            
            # If we can't find a replacement for any row, we might be stuck.
            # We can try to "perturb" by swapping two rows?
            # Or just break if no progress for a while?
            
            # For now, just continue.
            
            # To avoid infinite loop of doing nothing, we can add a counter
            # But the time limit will stop it.
            
            pass

    return best_solution
# EVOLVE-BLOCK-END
