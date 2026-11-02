"""Difference triangle set (4,7) — backtracking with MRV and local search."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _all_diffs(row):
    """Return set of all positive differences in a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs


def _max_entry(rows):
    return max(max(r) for r in rows)


def _is_valid(rows):
    """Check if all differences across all rows are distinct."""
    seen = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True


def _greedy_build(n_rows, k, rng, bias=1.0):
    """Greedy build with optional randomness."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                if bias >= 1.0 or rng.random() < bias:
                    row.append(x)
                    used |= new
                else:
                    # Skip with some probability for diversity
                    if rng.random() > 0.3:
                        row.append(x)
                        used |= new
        rows.append(row)
    return rows


def _backtrack_construct(n_rows, k, deadline, rng):
    """
    Backtracking search with MRV (minimum remaining values) heuristic.
    Tries to construct rows one at a time, choosing the next row that
    has the fewest valid next values.
    """
    used_diffs = set()
    rows = []
    
    def solve(idx):
        if time.time() > deadline:
            return None
        if idx == n_rows:
            return rows[:]
        
        # Try to build row idx
        row = [0]
        # We'll use a simple approach: try to extend row greedily with some randomness
        # but also allow backtracking at the row level
        
        # For efficiency, let's do a randomized greedy per row with restarts
        best_row = None
        best_used = None
        
        for attempt in range(50):  # Try up to 50 variants
            trial_row = [0]
            trial_used = set(used_diffs)
            x = 0
            success = True
            while len(trial_row) < k + 1:
                x += 1
                new = {x - a for a in trial_row}
                if new & trial_used:
                    continue
                # Decide whether to take this value
                if attempt == 0:
                    # First attempt: pure greedy
                    trial_row.append(x)
                    trial_used |= new
                else:
                    # Subsequent attempts: some randomness
                    if rng.random() < 0.8:
                        trial_row.append(x)
                        trial_used |= new
                    # else skip
            
            if len(trial_row) == k + 1:
                # Check if this row's max is reasonable
                if best_row is None or trial_row[-1] < best_row[-1]:
                    best_row = trial_row
                    best_used = trial_used
                    if best_row[-1] < 140:  # Good enough
                        break
        
        if best_row is None:
            return None
        
        rows.append(best_row)
        result = solve(idx + 1)
        if result is not None:
            return result
        rows.pop()
        return None
    
    return solve(0)


def _local_search(rows, deadline, rng):
    """
    Local search: try to reduce the maximum entry by adjusting individual
    row elements while maintaining validity.
    """
    best = rows[:]
    best_score = _max_entry(best)
    current_score = best_score
    
    while time.time() < deadline:
        # Pick a random row to modify
        row_idx = rng.randrange(len(best))
        old_row = best[row_idx][:]
        
        # Try to rebuild this row with a slightly different strategy
        used_others = set()
        for i, r in enumerate(best):
            if i != row_idx:
                used_others |= _all_diffs(r)
        
        # Rebuild row with randomness
        new_row = [0]
        x = 0
        while len(new_row) < len(old_row):
            x += 1
            new = {x - a for a in new_row}
            if not new & used_others:
                if rng.random() < 0.7:
                    new_row.append(x)
                else:
                    # Sometimes skip to get a different structure
                    if rng.random() < 0.5:
                        new_row.append(x)
        
        if len(new_row) == len(old_row):
            # Check full validity
            trial_rows = best[:]
            trial_rows[row_idx] = new_row
            if _is_valid(trial_rows):
                sc = _max_entry(trial_rows)
                if sc < current_score:
                    current_score = sc
                    best = trial_rows
                    if sc < best_score:
                        best_score = sc
                # Even if not better, accept with some probability (simulated annealing)
                elif rng.random() < 0.1:
                    best = trial_rows
                    current_score = sc
        
        # Occasionally try a full restart of one row
        if rng.random() < 0.2:
            used_others = set()
            for i, r in enumerate(best):
                if i != row_idx:
                    used_others |= _all_diffs(r)
            
            # Greedy rebuild
            new_row = [0]
            x = 0
            while len(new_row) < len(old_row):
                x += 1
                new = {x - a for a in new_row}
                if not new & used_others:
                    new_row.append(x)
            
            if len(new_row) == len(old_row):
                trial_rows = best[:]
                trial_rows[row_idx] = new_row
                if _is_valid(trial_rows):
                    sc = _max_entry(trial_rows)
                    if sc < current_score:
                        current_score = sc
                        best = trial_rows
                        if sc < best_score:
                            best_score = sc
    
    return best


def construct(n_rows, k):
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    rng = random.Random(42)
    
    # Start with a good greedy baseline
    best = _greedy_build(n_rows, k, random.Random(0), bias=1.0)
    best_score = _max_entry(best)
    
    # Try multiple greedy variants
    for seed in range(1, 20):
        if time.time() > deadline:
            break
        trial = _greedy_build(n_rows, k, random.Random(seed), bias=rng.choice([1.0, 0.9, 0.8, 0.7]))
        sc = _max_entry(trial)
        if sc < best_score:
            best_score = sc
            best = trial
    
    # Local search to improve
    best = _local_search(best, deadline, rng)
    best_score = _max_entry(best)
    
    # If time remains, try more restarts
    while time.time() < deadline - 5.0:
        trial = _greedy_build(n_rows, k, rng, bias=rng.choice([1.0, 0.9, 0.8, 0.7, 0.6]))
        sc = _max_entry(trial)
        if sc < best_score:
            best_score = sc
            best = trial
            # Quick local search on the new best
            if time.time() < deadline - 10.0:
                best = _local_search(best, min(deadline - 5.0, time.time() + 20.0), rng)
                best_score = _max_entry(best)
    
    return best

# EVOLVE-BLOCK-END
