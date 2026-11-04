"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _diffs_of_row(row):
    """Return set of positive differences for a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _total_scope(rows):
    """Return the maximum entry across all rows."""
    return max(max(r) for r in rows)


def _all_diffs_used(rows):
    """Return the set of all differences used across all rows."""
    used = set()
    for r in rows:
        used |= _diffs_of_row(r)
    return used


def _greedy_row(used, k, start_val=1):
    """
    Greedily construct a row of k+1 integers starting at 0.
    Each new element is the smallest integer > last that doesn't create
    a difference already in 'used'.
    Returns (row, new_diffs) or None if stuck.
    """
    row = [0]
    new_diffs = set()
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not (new & used) and not (new & new_diffs):
            row.append(x)
            new_diffs |= new
    return row, new_diffs


def _greedy_construct(n_rows, k):
    """Pure greedy construction."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row, new_diffs = _greedy_row(used, k)
        if row is None:
            return None
        rows.append(row)
        used |= new_diffs
    return rows


def _randomized_greedy_row(used, k, rng, pool_size=5):
    """
    Randomized greedy: at each step, pick from the top 'pool_size' smallest
    valid candidates with some randomness.
    """
    row = [0]
    new_diffs = set()
    x = 0
    while len(row) < k + 1:
        # Collect valid candidates
        candidates = []
        x += 1
        while len(candidates) < pool_size:
            new = {x - a for a in row}
            if not (new & used) and not (new & new_diffs):
                candidates.append(x)
            x += 1
        if not candidates:
            return None, None
        # Pick randomly from candidates, weighted toward smaller
        # Use a simple approach: pick the first candidate with prob 0.6,
        # otherwise pick uniformly from all candidates
        if rng.random() < 0.6:
            choice = candidates[0]
        else:
            choice = rng.choice(candidates)
        # Add all differences up to choice
        row.append(choice)
        for a in row[:-1]:
            new_diffs.add(choice - a)
        x = choice
    return row, new_diffs


def _randomized_greedy(n_rows, k, rng, pool_size=5):
    """Randomized greedy construction with multiple restarts."""
    best_rows = None
    best_scope = float('inf')
    
    max_restarts = 50
    deadline = time.time() + 100  # Leave margin for safety
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        used = set()
        rows = []
        success = True
        for _ in range(n_rows):
            row, new_diffs = _randomized_greedy_row(used, k, rng, pool_size)
            if row is None:
                success = False
                break
            rows.append(row)
            used |= new_diffs
        if success:
            scope = _total_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
    
    return best_rows if best_rows is not None else _greedy_construct(n_rows, k)


def _is_valid(rows):
    """Check if all positive differences across all rows are distinct."""
    used = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                d = r[j] - r[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _simulated_annealing(rows, k, rng, start_temp=100.0, cool_rate=0.95, min_temp=0.1, deadline=None):
    """
    Simulated Annealing to minimize the scope.
    Moves: Select a random row, a random index (1..k), and shift it by +/- delta.
    Objective: Max entry (scope).
    """
    current_rows = [r[:] for r in rows]
    current_scope = _total_scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    temp = start_temp
    i = 0
    while temp > min_temp:
        if deadline and time.time() > deadline:
            break
        
        # Select a random row and a random element (index 1 to k)
        row_idx = rng.randint(0, len(current_rows) - 1)
        row = current_rows[row_idx]
        elem_idx = rng.randint(1, len(row) - 1)
        
        # Try to move the element
        old_val = row[elem_idx]
        # Random step size, biased towards small steps
        step = rng.randint(-5, 5)
        if step == 0:
            continue
        new_val = old_val + step
        
        # Check bounds: must be strictly increasing
        if new_val <= row[elem_idx - 1] or (elem_idx < len(row) - 1 and new_val >= row[elem_idx + 1]):
            continue
            
        # Apply move
        row[elem_idx] = new_val
        new_scope = _total_scope(current_rows)
        
        # Check validity
        if not _is_valid(current_rows):
            # Revert move
            row[elem_idx] = old_val
            continue
        
        # Acceptance probability
        delta_scope = new_scope - current_scope
        if delta_scope < 0 or rng.random() < pow(2.71828, -delta_scope / temp):
            # Accept
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in current_rows]
        else:
            # Revert move
            row[elem_idx] = old_val
            
        temp *= cool_rate
        i += 1
        
    return best_rows


def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    """
    rng = random.Random(42)
    
    # Start with greedy
    rows = _greedy_construct(n_rows, k)
    if rows is None:
        # Fallback: very simple construction
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    # Try randomized greedy with multiple restarts
    deadline = time.time() + 110  # Leave 10s margin
    best_rows = rows
    best_scope = _total_scope(rows)
    
    # Phase 1: Randomized greedy restarts
    for restart in range(30):
        if time.time() > deadline:
            break
        rand_rows = _randomized_greedy(n_rows, k, rng, pool_size=rng.choice([3, 5, 8]))
        if rand_rows is not None:
            scope = _total_scope(rand_rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rand_rows]
    
    # Phase 2: Simulated Annealing on best solution
    if time.time() < deadline - 5:
        # Run SA for a few iterations/restarts
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            sa_rows = _simulated_annealing(best_rows, k, rng, start_temp=50.0, cool_rate=0.99, min_temp=1.0, deadline=deadline - 2)
            scope = _total_scope(sa_rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in sa_rows]
    
    return best_rows

# EVOLVE-BLOCK-END
