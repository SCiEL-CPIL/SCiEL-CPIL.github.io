"""Improved program using Simulated Annealing with Row Reconstruction."""
# EVOLVE-BLOCK-START
import random
import math
import time

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry using Simulated Annealing.
    """
    # 1. Greedy Initialization
    rows, used = _greedy_init(n_rows, k)
    
    # 2. Simulated Annealing
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    # Initial temperature
    T = 10.0
    cooling_rate = 0.999
    
    current_max = max(max(row) for row in rows)
    best_rows = [row[:] for row in rows]
    best_max = current_max
    
    rng = random.Random(42)
    
    while T > 0.01 and (time.time() - start_time) < time_limit:
        # Pick a random row to reconstruct
        row_idx = rng.randint(0, n_rows - 1)
        old_row = rows[row_idx]
        old_row_max = max(old_row)
        
        # Remove old row's differences from used
        old_diffs = _get_diffs(old_row)
        for d in old_diffs:
            used.discard(d)
        
        # Reconstruct row
        new_row = _reconstruct_row(used, k, rng, target_max=current_max)
        
        if new_row is None:
            # Failed to reconstruct, restore old row
            used.update(old_diffs)
            continue
            
        new_row_max = max(new_row)
        
        # Calculate new global max
        # We can optimize this by keeping a sorted list of row maxes, 
        # but for small n_rows, recomputing is fine.
        # However, we only change one row.
        # Let's just compute the max of all other rows and compare.
        # Actually, we can just check if new_row_max < current_max or if it's the max.
        
        # To be safe and simple, let's compute the potential new global max.
        # This is O(n_rows * k).
        other_maxes = [max(rows[i]) for i in range(n_rows) if i != row_idx]
        if other_maxes:
            max_of_others = max(other_maxes)
        else:
            max_of_others = 0
            
        potential_new_global_max = max(max_of_others, new_row_max)
        
        # Acceptance criterion
        if potential_new_global_max < current_max:
            # Descent, always accept
            accept = True
        else:
            # Uphill, accept with probability
            delta = potential_new_global_max - current_max
            if delta == 0:
                accept = True
            else:
                accept_prob = math.exp(-delta / T)
                accept = rng.random() < accept_prob
        
        if accept:
            # Update state
            new_diffs = _get_diffs(new_row)
            used.update(new_diffs)
            rows[row_idx] = new_row
            current_max = potential_new_global_max
            
            if current_max < best_max:
                best_max = current_max
                best_rows = [row[:] for row in rows]
        else:
            # Reject, restore old row
            used.update(old_diffs)
            
        # Cool down
        T *= cooling_rate
        
    return best_rows

def _greedy_init(n_rows, k):
    """
    Initialize rows using a greedy strategy.
    """
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
                used.update(new_diffs)
        rows.append(row)
    return rows, used

def _reconstruct_row(used, k, rng, target_max=None):
    """
    Reconstruct a row of length k+1 starting at 0, avoiding 'used' differences.
    Uses backtracking with a bias towards smaller values.
    """
    # We want to minimize the max element.
    # We can try to fit the row within a certain bound.
    # Let's use a greedy approach with limited backtracking.
    
    def backtrack(row, depth):
        if len(row) == k + 1:
            return True
        
        last = row[-1]
        # Try small gaps first to keep values low
        # We need to ensure we don't go too high if we have a target
        # But for now, just try gaps 1..20 in random order biased towards small?
        # Actually, sorting by gap is good for minimization.
        # But we need diversity for SA.
        
        # Let's try gaps from 1 to 30.
        # To make it stochastic, shuffle the range but keep small gaps more likely?
        # Or just shuffle 1..30.
        gaps = list(range(1, 31))
        rng.shuffle(gaps)
        
        for gap in gaps:
            candidate = last + gap
            # Optional: if target_max is set and candidate > target_max, skip?
            # No, we might need to go higher to find a valid configuration.
            # But if we are trying to improve, we should prefer lower.
            
            new_diffs = {candidate - a for a in row}
            if new_diffs & used:
                continue
            
            row.append(candidate)
            used.update(new_diffs)
            
            if backtrack(row, depth + 1):
                return True
            
            used.difference_update(new_diffs)
            row.pop()
            
        return False

    row = [0]
    if backtrack(row, 0):
        return row
    return None

def _get_diffs(row):
    """
    Get all positive differences in a row.
    """
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs
# EVOLVE-BLOCK-END
