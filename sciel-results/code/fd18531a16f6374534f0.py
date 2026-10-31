"""Difference triangle set: minimize largest entry for n_rows rows of k+1 increasing ints from 0."""
import time
import random

_START_TIME = None

def _time_left(limit=145.0):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return limit - (time.time() - _START_TIME)

def _greedy_sequential(n_rows, k):
    used = set()
    rows = []
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

def _greedy_round_robin(n_rows, k, max_iter=100000):
    """Round-robin: at each step, pick the row that is most constrained and extend it."""
    used = set()
    rows = [[0] for _ in range(n_rows)]
    total_needed = n_rows * (k + 1)
    placed = n_rows  # all rows have 0
    for _ in range(max_iter):
        if placed == total_needed:
            break
        if time.time() - _START_TIME > 145.0:
            break
        # Find the row with fewest valid next candidates (most constrained)
        best_row = -1
        best_count = float('inf')
        best_candidate = None
        for r in range(n_rows):
            if len(rows[r]) >= k + 1:
                continue
            last = rows[r][-1]
            # Count valid next candidates in a reasonable range
            # Just try to find the smallest valid next
            count = 0
            candidate = None
            for x in range(last + 1, last + 200):  # look ahead
                new = {x - a for a in rows[r]}
                if not (new & used):
                    count += 1
                    if candidate is None:
                        candidate = x
            if count == 0:
                # No candidates in this range; try larger
                x = last + 200
                while True:
                    new = {x - a for a in rows[r]}
                    if not (new & used):
                        count = 1
                        candidate = x
                        break
                    x += 1
            if count < best_count:
                best_count = count
                best_row = r
                best_candidate = candidate
                if count == 1:
                    break
        if best_row == -1:
            break
        x = best_candidate
        rows[best_row].append(x)
        for a in rows[best_row]:
            used.add(x - a)
        placed += 1
    # Check completeness
    for r in range(n_rows):
        if len(rows[r]) < k + 1:
            return None
    return rows

def _try_multiple_seeds(n_rows, k, n_trials=50, deadline=145.0):
    """Try round-robin greedy with different random perturbations."""
    best_rows = None
    best_max = float('inf')
    for trial in range(n_trials):
        if time.time() - _START_TIME > deadline:
            break
        rows = _greedy_round_robin(n_rows, k)
        if rows is None:
            continue
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = rows
            if best_max == n_rows * k * (k + 1) // 2:
                break  # hit lower bound
    return best_rows

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    deadline = 145.0
    
    # Sequential greedy baseline
    seq_rows = _greedy_sequential(n_rows, k)
    seq_max = max(max(r) for r in seq_rows)
    
    # Lower bound
    lower_bound = n_rows * k * (k + 1) // 2
    
    best_rows = seq_rows
    best_max = seq_max
    
    # Try round-robin greedy multiple times
    if (seq_max - lower_bound) > 0 and n_rows * (k + 1) <= 60:
        rr_rows = _try_multiple_seeds(n_rows, k, n_trials=100, deadline=deadline)
        if rr_rows is not None:
            rr_max = max(max(r) for r in rr_rows)
            if rr_max < best_max:
                best_rows = rr_rows
                best_max = rr_max
    
    return best_rows
