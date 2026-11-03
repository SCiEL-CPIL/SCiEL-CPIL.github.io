import time
import random
import math
from collections import Counter
from itertools import combinations

def _compute_diffs(row):
    """Compute all pairwise differences for a single row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _get_scope(rows):
    """Return the largest entry in any row."""
    if not rows:
        return 0
    return max(max(row) for row in rows)

def _row_to_diffs(row):
    """Convert a row to its list of consecutive differences."""
    return [row[i+1] - row[i] for i in range(len(row)-1)]

def _diffs_to_row(diffs):
    """Convert consecutive differences back to a row starting at 0."""
    row = [0]
    for d in diffs:
        row.append(row[-1] + d)
    return row

def _is_valid(rows, k):
    """Check if all differences are distinct across all rows."""
    seen = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i-1]:
                return False
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _calculate_fitness(rows, k, max_scope_hint=200):
    """
    Calculate fitness: scope + penalty for collisions.
    Returns (score, diff_counts, used_diffs) to allow fast updates if needed, 
    but here we just return the score for simplicity, optimized by early exit if possible.
    """
    all_diffs = []
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                all_diffs.append(row[j] - row[i])
    
    counts = Counter(all_diffs)
    collisions = sum(max(0, c - 1) for c in counts.values())
    scope = _get_scope(rows)
    
    # Heavy penalty for collisions to prioritize validity
    return scope + 10000 * collisions

def _greedy_init(n_rows, k, rng):
    """
    Greedy initialization: build each row by choosing the smallest available value.
    This ensures a valid starting point with a reasonable scope.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _local_search(rows, n_rows, k, deadline, rng):
    """
    Optimized Local Search with Simulated Annealing.
    Moves:
    1. Change a single element in a row (maintaining strict increase).
    2. Swap two elements (if it maintains order? No, just re-sorting isn't allowed as values are fixed positions? 
       Actually, rows are sets. We can permute them? No, "increasing integers". 
       So we just change values.
    """
    current_rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_fit = _calculate_fitness(current_rows, k)
    current_fit = best_fit
    
    T = 10.0
    cooling = 0.9995
    min_T = 0.01
    
    # Pre-calculate current diffs for each row to speed up delta calculation?
    # For simplicity and robustness against complex delta logic, we recalc fitness 
    # but optimize the loop.
    
    while T > min_T and time.time() < deadline:
        # Pick a random row and position
        r_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, k) # Don't touch 0
        
        old_val = current_rows[r_idx][pos]
        
        # Generate new value
        # Bias towards smaller values to reduce scope
        if rng.random() < 0.6:
            # Try to decrease
            lower_bound = current_rows[r_idx][pos-1] + 1
            upper_bound = old_val - 1
            if lower_bound <= upper_bound:
                new_val = rng.randint(lower_bound, upper_bound)
            else:
                continue
        else:
            # Try to increase or randomize
            lower_bound = current_rows[r_idx][pos-1] + 1
            # Cap increase to not blow up scope too much
            max_val = max(old_val + 5, current_rows[r_idx][-1] + 5)
            if lower_bound < max_val:
                new_val = rng.randint(lower_bound, max_val)
            else:
                continue
        
        if new_val == old_val:
            continue
            
        # Check bounds with next element
        if pos < k:
            if new_val >= current_rows[r_idx][pos+1]:
                continue
        
        # Apply move
        current_rows[r_idx][pos] = new_val
        
        # Calculate new fitness
        new_fit = _calculate_fitness(current_rows, k)
        
        # Acceptance criterion
        if new_fit < current_fit or (T > 0 and rng.random() < math.exp(-(new_fit - current_fit) / T)):
            current_fit = new_fit
            if new_fit < best_fit:
                best_fit = new_fit
                best_rows = [list(r) for r in current_rows]
        else:
            # Revert
            current_rows[r_idx][pos] = old_val
            
        T *= cooling
        
        # Periodic check to see if we can compress
        # This is a heuristic: if we have been stuck, try a "compression" move
        # For now, standard SA is sufficient if fast enough.
        
    return best_rows

def construct(n_rows, k):
    # Strict deadline guard: 110s to be safe against 150s kill
    deadline = time.time() + 110.0
    rng = random.Random(42)
    
    # Initial greedy solution
    rows = _greedy_init(n_rows, k, rng)
    
    best_rows = [list(r) for r in rows]
    best_fit = _calculate_fitness(rows, k)
    
    # Run local search
    # Split time: 80s for main search, 20s for final polish/restart if needed
    main_deadline = time.time() + 80.0
    
    if time.time() < main_deadline:
        rows = _local_search(rows, n_rows, k, main_deadline, rng)
        fit = _calculate_fitness(rows, k)
        if fit < best_fit:
            best_fit = fit
            best_rows = [list(r) for r in rows]
            
    # If we have time, try a few restarts or a second pass with different seed
    if time.time() < deadline - 5:
        # Try a different random seed for restart
        rng2 = random.Random(123)
        rows2 = _greedy_init(n_rows, k, rng2)
        # Quick local search on restart
        restart_deadline = deadline - 5
        rows2 = _local_search(rows2, n_rows, k, restart_deadline, rng2)
        fit2 = _calculate_fitness(rows2, k)
        if fit2 < best_fit:
            best_fit = fit2
            best_rows = [list(r) for r in rows2]
            
    # Verify validity
    if not _is_valid(best_rows, k):
        # Fallback to greedy if somehow invalid (shouldn't happen with valid init and valid moves)
        return _greedy_init(n_rows, k, rng)
    
    return best_rows
