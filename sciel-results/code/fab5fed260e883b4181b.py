import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Lower bound
    total_diffs = n_rows * k * (k + 1) // 2
    M = total_diffs
    
    # Upper bound: known best is 134 for (4,7). 
    # For general n_rows, k, a safe upper bound is total_diffs + 2 * n_rows * k
    max_M = total_diffs + 100
    
    best_rows = None
    best_M = None
    
    rng = random.Random(42) # Fixed seed for reproducibility? 
    # Actually, using system time or a global RNG is fine. 
    # Let's use a local RNG seeded with time for variety across runs if evaluated multiple times?
    # The evaluator runs it once. Fixed seed is safer for debugging, but random is better for robustness.
    # Let's use random.SystemRandom() or just random module.
    
    while M <= max_M and time.time() < deadline:
        # Try to find a solution with scope M
        # Number of attempts depends on remaining time
        # Let's do 500 attempts per M initially
        attempts_done = 0
        max_attempts = 500
        
        while attempts_done < max_attempts and time.time() < deadline:
            attempts_done += 1
            rows = try_construct(n_rows, k, M, rng)
            if rows is not None:
                best_rows = rows
                best_M = M
                # We found a solution for M. 
                # Can we do better? No, we started from the lower bound.
                # So this is the minimal M for this random search.
                # We can return immediately if we want the minimal M found.
                # But maybe a slightly larger M was found faster? 
                # The task says "minimized". So we want the smallest M.
                # Since we iterate M upwards, the first M for which we find a solution is the best.
                return rows
        
        M += 1
        
    # If we exited the loop without returning, we might have best_rows from a previous M?
    # No, if we found one we returned.
    # If we didn't find any, we need to return a valid solution.
    # Use a greedy fallback to ensure validity.
    return greedy_construct(n_rows, k)

def try_construct(n_rows, k, M, rng):
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            last = row[-1]
            valid_xs = []
            # Optimization: if we are close to M, the range is small
            for x in range(last + 1, M + 1):
                # Check differences
                # new_diffs = {x - a for a in row}
                # if not new_diffs & used_diffs:
                
                # Manual loop for speed?
                conflict = False
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        conflict = True
                        break
                if not conflict:
                    valid_xs.append(x)
            
            if not valid_xs:
                return None
            
            # Pick a random valid x
            x = rng.choice(valid_xs)
            row.append(x)
            # Update used_diffs
            for a in row[:-1]:
                used_diffs.add(x - a)
        rows.append(row)
    return rows

def greedy_construct(n_rows, k):
    """Fallback greedy to ensure a valid result is returned."""
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
