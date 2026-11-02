"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _compute_used_diffs(rows):
    """Compute set of all positive differences used across all rows."""
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])
    return used

def _row_diffs(row):
    """Get all positive differences in a single row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _is_valid(rows):
    """Check if all differences are distinct across all rows."""
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _scope(rows):
    """Return the maximum entry (scope) of the rows."""
    return max(max(row) for row in rows)

def _greedy_construct(n_rows, k, rng):
    """Greedy construction with randomization for multi-start."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            # Try candidates in a slightly randomized order near the smallest
            candidates = []
            for trial in range(10):
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    candidates.append((x, new))
                    if len(candidates) >= 3:
                        break
            if not candidates:
                # Fallback: keep searching
                while True:
                    x += 1
                    new = {x - a for a in row}
                    if not new & used:
                        candidates.append((x, new))
                        break
            # Pick randomly among early candidates for diversity
            x, new = rng.choice(candidates)
            row.append(x)
            used |= new
        rows.append(row)
    return rows

def _local_compact(rows, n_rows, k, rng, time_budget=5.0):
    """Aggressively compact rows by trying to reduce entries."""
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    start = time.time()
    
    # Try to reduce each row's entries
    for iteration in range(200):
        if time.time() - start > time_budget:
            break
        for r_idx in range(n_rows):
            row = list(best_rows[r_idx])
            # Try to reduce the last entry
            if len(row) > 1:
                old_last = row[-1]
                # Try smaller values for last position
                for new_last in range(old_last - 1, 0, -1):
                    if new_last <= row[-2]:
                        break
                    test_row = row[:-1] + [new_last]
                    # Check if valid with other rows
                    test_rows = [list(r) for r in best_rows]
                    test_rows[r_idx] = test_row
                    if _is_valid(test_rows):
                        new_scope = _scope(test_rows)
                        if new_scope < best_scope:
                            best_rows = test_rows
                            best_scope = new_scope
                            break
    
    return best_rows, best_scope

def _simulated_annealing(rows, n_rows, k, rng, time_budget=30.0):
    """Simulated annealing to minimize scope."""
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    start = time.time()
    
    T = 10.0
    cooling = 0.995
    
    while T > 0.01 and (time.time() - start) < time_budget:
        # Pick a random row and a random position to perturb
        r_idx = rng.randint(0, n_rows - 1)
        row = list(current_rows[r_idx])
        pos = rng.randint(1, len(row) - 1)
        
        # Perturb: move this position by a small random amount
        delta = rng.choice([-3, -2, -1, 1, 2, 3, 5, 8])
        old_val = row[pos]
        new_val = old_val + delta
        
        # Ensure increasing order
        if pos > 0 and new_val <= row[pos - 1]:
            new_val = row[pos - 1] + 1
        if pos < len(row) - 1 and new_val >= row[pos + 1]:
            new_val = row[pos + 1] - 1
        if new_val <= 0:
            continue
            
        row[pos] = new_val
        test_rows = [list(r) for r in current_rows]
        test_rows[r_idx] = row
        
        if _is_valid(test_rows):
            new_scope = _scope(test_rows)
            delta_scope = new_scope - current_scope
            if delta_scope <= 0 or rng.random() < np.exp(-delta_scope / T):
                current_rows = test_rows
                current_scope = new_scope
                if new_scope < best_scope:
                    best_rows = [list(r) for r in current_rows]
                    best_scope = new_scope
        T *= cooling
    
    return best_rows, best_scope

def construct(n_rows, k):
    """Construct difference triangle set with minimal scope."""
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Multi-start greedy constructions
    n_starts = 10
    for i in range(n_starts):
        if time.time() > deadline - 30:
            break
        rows = _greedy_construct(n_rows, k, rng)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback to simple greedy
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
    
    # Phase 2: Local compaction on best solution
    best_rows, best_scope = _local_compact(best_rows, n_rows, k, rng, time_budget=10.0)
    
    # Phase 3: Simulated annealing
    remaining_time = max(0, deadline - time.time() - 5)
    if remaining_time > 5:
        best_rows, best_scope = _simulated_annealing(
            best_rows, n_rows, k, rng, time_budget=remaining_time
        )
    
    return best_rows

# EVOLVE-BLOCK-END
