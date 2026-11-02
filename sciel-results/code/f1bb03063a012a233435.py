"""
Difference triangle set (7,5): construct 7 rows of 6 increasing integers starting at 0
so that all positive differences within rows are distinct across all rows.
Minimize the largest entry.

Approach: Iterated Local Search (ILS) with row-level neighborhood.
1. Initial solution via randomized greedy.
2. ILS: repeatedly replace one row with a new valid row found by randomized backtracking
   within the current global max bound, accepting if it reduces the global max.
3. Periodic restarts with tighter max bounds to drive down the global maximum.
"""

import time
import random
import sys
import math

def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _all_diffs(rows):
    """Return the set of all positive differences across all rows."""
    used = set()
    for row in rows:
        used |= _row_diffs(row)
    return used

def _global_max(rows):
    return max(row[-1] for row in rows)

def _find_valid_row(used_others, k, max_val, rng, max_attempts=20000):
    """
    Find a valid row of k+1 integers starting at 0, max <= max_val,
    whose differences don't conflict with used_others.
    Uses randomized backtracking.
    """
    n = k + 1  # number of elements in row
    
    def backtrack(pos, current_row, current_diffs, attempts_left):
        if pos == n:
            return current_row[:]
        if attempts_left <= 0:
            return None
        
        # Determine range for next element
        if pos == 0:
            candidates = [0]
        else:
            lo = current_row[-1] + 1
            hi = max_val
            if lo > hi:
                return None
            # Generate candidates in random order for diversity
            # But bias toward smaller values to keep max small
            candidates = list(range(lo, hi + 1))
            rng.shuffle(candidates)
        
        for val in candidates:
            # Compute new differences
            new_diffs = set()
            for i in range(pos):
                d = val - current_row[i]
                if d in used_others or d in current_diffs:
                    break
                new_diffs.add(d)
            else:
                # All new diffs are valid
                current_row.append(val)
                current_diffs |= new_diffs
                result = backtrack(pos + 1, current_row, current_diffs, attempts_left - 1)
                if result is not None:
                    return result
                current_row.pop()
                current_diffs -= new_diffs
        
        return None
    
    # Try multiple times with different randomizations
    for attempt in range(50):
        result = backtrack(0, [], set(), max_attempts // 50)
        if result is not None:
            return result
    
    return None

def _find_valid_row_greedy(used_others, k, max_val, rng, bias=0.7):
    """
    Find a valid row using randomized greedy (faster but less thorough).
    """
    n = k + 1
    row = [0]
    diffs = set()
    x = 1
    attempts = 0
    max_attempts = max_val * 4
    
    while len(row) < n:
        if x > max_val or attempts > max_attempts:
            return None
        new_diffs = {x - a for a in row}
        if not (new_diffs & used_others) and not (new_diffs & diffs):
            if rng.random() < bias:
                row.append(x)
                diffs |= new_diffs
                x += 1
            else:
                x += 1
        else:
            x += 1
        attempts += 1
    
    return row

def _construct_initial(n_rows, k, max_val, rng):
    """
    Construct an initial set of rows using randomized greedy.
    """
    used = set()
    rows = []
    for i in range(n_rows):
        # Try to find a row within max_val
        row = None
        for attempt in range(20):
            row = _find_valid_row_greedy(used, k, max_val, rng, bias=0.8)
            if row is not None:
                break
        if row is None:
            # Fallback: use a wider range
            row = _find_valid_row_greedy(used, k, max_val + 30, rng, bias=0.9)
            if row is None:
                # Last resort: deterministic greedy
                row = [0]
                diffs = set()
                x = 1
                while len(row) < k + 1:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used) and not (new_diffs & diffs):
                        row.append(x)
                        diffs |= new_diffs
                        x += 1
                    else:
                        x += 1
        rows.append(row)
        used |= _row_diffs(row)
    return rows

def construct(n_rows, k):
    """
    Main entry point. Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences are distinct across all rows.
    Minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    rng = random.Random(42)
    
    # Phase 1: Initial construction
    # Start with a reasonable max_val estimate
    # Counting bound: need 105 distinct diffs, so max_val >= 105
    # But we need them to form valid rows, so start higher
    initial_max = 120
    
    rows = _construct_initial(n_rows, k, initial_max, rng)
    best_rows = [r[:] for r in rows]
    best_max = _global_max(best_rows)
    
    current_rows = [r[:] for r in rows]
    current_max = _global_max(current_rows)
    
    # Phase 2: ILS - iteratively try to reduce the global max
    # Strategy: try to find a solution with max_val = current_max - 1
    # If successful, update and repeat. If not, perturb and retry.
    
    stall_count = 0
    max_stall = 50
    
    while time.time() < deadline - 10:
        target_max = current_max - 1
        if target_max < 105:  # below counting bound
            break
        
        found_better = False
        attempts = 0
        max_attempts = 30
        
        while attempts < max_attempts and not found_better and time.time() < deadline - 10:
            attempts += 1
            
            # Try to construct a solution with max <= target_max
            # Use randomized approach: shuffle row order, use different random seeds
            row_order = list(range(n_rows))
            rng.shuffle(row_order)
            
            used = set()
            new_rows = [None] * n_rows
            
            success = True
            for idx in row_order:
                # Get used diffs from already-placed rows
                used_so_far = set()
                for j in range(n_rows):
                    if new_rows[j] is not None:
                        used_so_far |= _row_diffs(new_rows[j])
                
                # Try to find a row for position idx
                row = None
                for try_count in range(10):
                    row = _find_valid_row_greedy(used_so_far, k, target_max, rng, bias=0.7)
                    if row is not None:
                        break
                if row is None:
                    # Try backtracking for this row
                    row = _find_valid_row(used_so_far, k, target_max, rng, max_attempts=5000)
                
                if row is None:
                    success = False
                    break
                new_rows[idx] = row
            
            if success:
                new_max = _global_max(new_rows)
                if new_max <= target_max:
                    current_rows = new_rows
                    current_max = new_max
                    if new_max < best_max:
                        best_rows = [r[:] for r in current_rows]
                        best_max = new_max
                    found_better = True
                    stall_count = 0
                else:
                    stall_count += 1
            else:
                stall_count += 1
        
        if not found_better:
            # Try a different strategy: perturb current solution and retry
            if stall_count >= max_stall:
                # Restart with a fresh construction at current_max
                rows = _construct_initial(n_rows, k, current_max, rng)
                current_rows = [r[:] for r in rows]
                current_max = _global_max(current_rows)
                stall_count = 0
            else:
                # Small perturbation: replace one random row
                row_idx = rng.randrange(n_rows)
                used_others = set()
                for i in range(n_rows):
                    if i != row_idx:
                        used_others |= _row_diffs(current_rows[i])
                
                new_row = _find_valid_row(used_others, k, current_max, rng, max_attempts=10000)
                if new_row is not None:
                    current_rows[row_idx] = new_row
                    new_max = _global_max(current_rows)
                    if new_max < best_max:
                        best_rows = [r[:] for r in current_rows]
                        best_max = new_max
        
        # Check if we've hit the theoretical minimum
        if best_max <= 105:
            break
    
    # Phase 3: Final attempt to tighten - try to reduce each individual row
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        improved = False
        for row_idx in range(n_rows):
            row = best_rows[row_idx]
            current_row_max = row[-1]
            
            # Try to find a smaller row for this position
            used_others = set()
            for r_idx in range(n_rows):
                if r_idx != row_idx:
                    used_others |= _row_diffs(best_rows[r_idx])
            
            for trial_max in range(current_row_max - 1, 10, -1):
                new_row = _find_valid_row_greedy(used_others, k, trial_max, rng, bias=0.9)
                if new_row is not None:
                    best_rows[row_idx] = new_row
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_rows
