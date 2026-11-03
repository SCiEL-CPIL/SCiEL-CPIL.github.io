"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import sys
import time
import random

import math
import time

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimize the largest entry (the scope).
    
    Uses Simulated Annealing with a penalty for conflicts to escape local minima.
    """
    if n_rows <= 0:
        return []
    if k == 0:
        return [[0] for _ in range(n_rows)]
    
    # Initialize with a valid greedy solution
    rows = _greedy_construct(n_rows, k)
    
    # Simulated Annealing
    current_rows = [row[:] for row in rows]
    current_cost = _cost(current_rows)
    
    best_rows = [row[:] for row in current_rows]
    best_cost = current_cost
    
    start_time = time.time()
    timeout = 110  # Leave 10s for overhead
    
    # Parameters
    T = 500.0
    T_min = 1.0
    cooling_rate = 0.9995
    
    while time.time() - start_time < timeout:
        # Generate a neighbor
        new_rows = _get_neighbor(current_rows, k)
        new_cost = _cost(new_rows)
        
        # Acceptance criterion
        if new_cost < current_cost or random.random() < math.exp(-(new_cost - current_cost) / T):
            current_rows = new_rows
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_rows = [row[:] for row in current_rows]
        
        # Cool down
        T = max(T_min, T * cooling_rate)
        
        # If we have a valid solution (0 conflicts) and we've cooled down, 
        # we might want to restart to find a better valid solution
        if current_cost < 10000 and T < 10: # 10000 is penalty for 1 conflict
             # Restart with a different random seed or perturb best
             if random.random() < 0.05:
                 current_rows = _perturb(best_rows, k)
                 current_cost = _cost(current_rows)
                 T = 100.0

    # Ensure the returned solution is valid
    if _cost(best_rows) >= 10000:
        # Fallback to greedy if SA failed to find a valid solution
        return _greedy_construct(n_rows, k)
        
    return best_rows


def _cost(rows):
    """
    Calculate cost: number of conflicts * 10000 + max value.
    Conflicts are counted as the number of duplicate differences.
    Also penalizes non-increasing rows.
    """
    used_diffs = {}
    conflicts = 0
    max_val = 0
    inc_violations = 0
    
    for row in rows:
        for i in range(len(row)):
            if row[i] > max_val:
                max_val = row[i]
            if i > 0 and row[i] <= row[i-1]:
                inc_violations += 1
            for j in range(i):
                d = row[i] - row[j]
                if d in used_diffs:
                    conflicts += 1
                else:
                    used_diffs[d] = 1
                    
    return (conflicts + inc_violations) * 10000 + max_val


def _get_neighbor(rows, k):
    """
    Generate a neighbor by randomly modifying one element in one row.
    """
    new_rows = [row[:] for row in rows]
    n_rows = len(new_rows)
    
    # Pick a random row
    r = random.randint(0, n_rows - 1)
    
    # Pick a random index to change (1 to k, index 0 is fixed at 0)
    i = random.randint(1, k)
    
    # Determine bounds for the new value
    # We allow a wide range to explore, but penalize violations in cost
    current_val = new_rows[r][i]
    # Pick a new value in a range around the current value
    # Or a random value in [0, 200]
    new_val = random.randint(0, 250)
    
    new_rows[r][i] = new_val
    
    return new_rows


def _perturb(rows, k):
    """
    Create a perturbed version of a valid solution to restart SA.
    """
    new_rows = [row[:] for row in rows]
    # Randomly swap some elements or shift them
    for _ in range(10):
        r = random.randint(0, len(new_rows) - 1)
        i = random.randint(1, k)
        # Try to move element
        new_val = random.randint(0, 250)
        new_rows[r][i] = new_val
                 
    return new_rows


def _greedy_construct(n_rows, k):
    """Greedy fallback: same as the original seed program."""
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

# EVOLVE-BLOCK-END
