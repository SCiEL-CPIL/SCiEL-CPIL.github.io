"""Difference triangle set (4,7) — simulated annealing with local compaction."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return set of positive differences for a single row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _compute_used_diffs(rows):
    """Return the union of all differences across all rows."""
    used = set()
    for row in rows:
        used |= _row_diffs(row)
    return used


def _is_valid(rows):
    """Check if all differences across rows are distinct."""
    used = set()
    for row in rows:
        diffs = _row_diffs(row)
        if used & diffs:
            return False
        used |= diffs
    return True


def _scope(rows):
    """Largest entry across all rows."""
    return max(max(r) for r in rows)


def _greedy_construct(n_rows, k, rng=None):
    """Deterministic greedy construction with optional randomization."""
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


def _local_compact(rows, rng, max_iters=50):
    """Try to reduce the scope by moving individual elements left."""
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    for _ in range(max_iters):
        improved = False
        # Try each row
        for ri in range(len(best_rows)):
            row = best_rows[ri]
            # Try to reduce each element (except first which is 0)
            for ei in range(len(row) - 1, 0, -1):
                old_val = row[ei]
                # Try to move this element left
                for new_val in range(old_val - 1, max(row[ei-1] + 1, 0) - 1, -1):
                    if new_val <= row[ei-1]:
                        break
                    # Check if this move is valid
                    test_row = row[:ei] + [new_val] + row[ei+1:]
                    # Ensure strictly increasing
                    if not all(test_row[i] < test_row[i+1] for i in range(len(test_row)-1)):
                        continue
                    # Check differences
                    test_diffs = _row_diffs(test_row)
                    # Get all other diffs
                    other_diffs = set()
                    for rj in range(len(best_rows)):
                        if rj != ri:
                            other_diffs |= _row_diffs(best_rows[rj])
                    if not (test_diffs & other_diffs):
                        # Valid move
                        row[ei] = new_val
                        best_rows[ri] = row
                        new_scope = _scope(best_rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                        improved = True
                        break
        if not improved:
            break
    return best_rows, best_scope


def _simulated_annealing(n_rows, k, rng, time_limit):
    """Simulated annealing to find better solutions."""
    start_time = time.time()
    
    # Start with greedy
    rows = _greedy_construct(n_rows, k, rng)
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    # Initial local compaction
    rows, current_scope = _local_compact(rows, rng, max_iters=100)
    if current_scope < best_scope:
        best_scope = current_scope
        best_rows = [list(r) for r in rows]
    
    temperature = max(1.0, best_scope * 0.1)
    cooling_rate = 0.995
    
    while time.time() - start_time < time_limit and temperature > 0.1:
        # Generate a perturbation
        trial_rows = [list(r) for r in rows]
        
        # Randomly modify one or two elements
        n_mods = rng.choice([1, 1, 1, 2])
        for _ in range(n_mods):
            ri = rng.randrange(len(trial_rows))
            ei = rng.randrange(1, len(trial_rows[ri]))  # Don't modify first element
            row = trial_rows[ri]
            # Try to move element by a small amount
            delta = rng.choice([-2, -1, 1, 2])
            new_val = row[ei] + delta
            # Check bounds
            lower = row[ei-1] + 1
            upper = row[ei+1] - 1 if ei + 1 < len(row) else best_scope + 20
            if new_val < lower or new_val > upper:
                continue
            # Check validity of this single change
            test_row = row[:ei] + [new_val] + row[ei+1:]
            if not all(test_row[i] < test_row[i+1] for i in range(len(test_row)-1)):
                continue
            test_diffs = _row_diffs(test_row)
            other_diffs = set()
            for rj in range(len(trial_rows)):
                if rj != ri:
                    other_diffs |= _row_diffs(trial_rows[rj])
            if test_diffs & other_diffs:
                continue
            row[ei] = new_val
            trial_rows[ri] = row
        
        # Check if trial is valid
        if not _is_valid(trial_rows):
            temperature *= cooling_rate
            continue
        
        trial_scope = _scope(trial_rows)
        delta_scope = trial_scope - current_scope
        
        # Accept if better or with probability exp(-delta/T)
        if delta_scope < 0:
            rows = [list(r) for r in trial_rows]
            current_scope = trial_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
                # Do local compaction after improvement
                rows, current_scope = _local_compact(rows, rng, max_iters=50)
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in rows]
        else:
            import math
            prob = math.exp(-delta_scope / max(temperature, 0.1))
            if rng.random() < prob:
                rows = [list(r) for r in trial_rows]
                current_scope = trial_scope
        
        temperature *= cooling_rate
    
    return best_rows, best_scope


def construct(n_rows, k):
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # Start with deterministic greedy
    best = _greedy_construct(n_rows, k)
    best_scope = _scope(best)
    
    # Local compaction
    best, best_scope = _local_compact(best, random.Random(0), max_iters=200)
    
    # Simulated annealing with multiple restarts
    rng = random.Random(42)
    
    # First SA run with longer time
    time_for_sa1 = min(40.0, deadline - time.time())
    if time_for_sa1 > 5:
        sa_rows, sa_scope = _simulated_annealing(n_rows, k, rng, time_for_sa1)
        if sa_scope < best_scope:
            best_scope = sa_scope
            best = sa_rows
    
    # Additional restarts if time remains
    while time.time() < deadline - 5.0:
        # New random restart
        trial_rng = random.Random(rng.randrange(10**6))
        trial = _greedy_construct(n_rows, k, trial_rng)
        trial, trial_scope = _local_compact(trial, trial_rng, max_iters=100)
        
        if trial_scope < best_scope:
            best_scope = trial_scope
            best = trial
        
        # Short SA from this starting point
        remaining = deadline - time.time() - 5.0
        if remaining > 10:
            sa_rows, sa_scope = _simulated_annealing(n_rows, k, trial_rng, min(20.0, remaining))
            if sa_scope < best_scope:
                best_scope = sa_scope
                best = sa_rows
    
    return best

# EVOLVE-BLOCK-END
