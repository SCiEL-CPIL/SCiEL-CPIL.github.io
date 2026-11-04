"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry such that all positive differences
    within rows are distinct across all rows.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 115s safety margin before 150s kill

    # Best solution found so far
    best_rows = None
    best_max = float('inf')

    # First, get a baseline using greedy
    baseline_rows = _greedy_randomized(n_rows, k, seed=42)
    if baseline_rows is not None:
        best_rows = baseline_rows
        best_max = max(max(row) for row in baseline_rows)

    # Try local search to improve
    random.seed(42)
    improved = _local_search(n_rows, k, best_rows, best_max, start_time, deadline)
    if improved is not None:
        imp_max = max(max(row) for row in improved)
        if imp_max < best_max:
            best_max = imp_max
            best_rows = improved

    # Try a few more greedy restarts with different seeds
    for seed in range(43, 103):
        if time.time() > deadline:
            break
        rows = _greedy_randomized(n_rows, k, seed=seed)
        if rows is not None:
            row_max = max(max(row) for row in rows)
            if row_max < best_max:
                best_max = row_max
                best_rows = rows
                # Try local search on this new best
                improved = _local_search(n_rows, k, best_rows, best_max, time.time(), deadline)
                if improved is not None:
                    imp_max = max(max(row) for row in improved)
                    if imp_max < best_max:
                        best_max = imp_max
                        best_rows = improved

    if best_rows is None:
        best_rows = _greedy_randomized(n_rows, k, seed=0)
    
    return best_rows


def _greedy_randomized(n_rows, k, seed=42):
    """
    Randomized greedy: each row starts at 0 and picks the next mark from
    a small window of candidates, choosing randomly among valid ones.
    """
    rng = random.Random(seed)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            # Look ahead in a window
            window = 10
            candidates = []
            for dx in range(1, window + 1):
                x_candidate = x + dx
                new_diffs = {x_candidate - a for a in row}
                if not (new_diffs & used):
                    candidates.append(x_candidate)
            
            if not candidates:
                # Expand window if no candidates found
                window = x + 50
                for dx in range(1, window + 1):
                    x_candidate = x + dx
                    new_diffs = {x_candidate - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x_candidate)
                if not candidates:
                    return None  # Failed
            
            # Pick randomly from candidates
            x = rng.choice(candidates)
            row.append(x)
            new_diffs = {x - a for a in row[:-1]}
            used |= new_diffs
        rows.append(row)
    return rows


def _local_search(n_rows, k, initial_rows, initial_max, start_time, deadline):
    """
    Local search: try to reduce the maximum entry by modifying rows.
    Uses simulated annealing-like approach.
    """
    rows = [list(row) for row in initial_rows]
    current_max = initial_max
    best_rows = [list(row) for row in rows]
    best_max = current_max
    
    # Compute used differences
    def get_used_diffs(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used
    
    used = get_used_diffs(rows)
    
    # Try to reduce the maximum by perturbing rows
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        # Pick a random row to modify
        row_idx = random.randint(0, n_rows - 1)
        old_row = list(rows[row_idx])
        
        # Try to create a new row with smaller maximum
        # Remove the largest element and try to replace it
        if len(rows[row_idx]) <= 1:
            continue
        
        # Try perturbing: remove last element and try different values
        new_row = list(rows[row_idx])
        
        # Remove the last element
        last_val = new_row.pop()
        
        # Compute diffs from remaining elements
        remaining_diffs = set()
        for i in range(len(new_row)):
            for j in range(i + 1, len(new_row)):
                remaining_diffs.add(new_row[j] - new_row[i])
        
        # Find global used diffs excluding this row's remaining diffs
        global_used = set(used)
        for d in remaining_diffs:
            global_used.discard(d)
        
        # Try to find a new last value
        found = False
        for new_last in range(1, last_val + 1):
            if new_last <= new_row[-1]:
                continue
            new_diffs = {new_last - a for a in new_row}
            # Check against global used AND remaining_diffs (prefix internal diffs)
            if not (new_diffs & global_used) and not (new_diffs & remaining_diffs):
                new_row.append(new_last)
                found = True
                break
        
        if not found:
            # Try a completely new row
            new_row = _try_build_row(k, global_used, random, deadline)
            if new_row is None:
                continue
        
        # Check if this is better
        new_row_max = max(new_row)
        if new_row_max < max(old_row):
            # Accept this change
            rows[row_idx] = new_row
            # Update used set
            used = get_used_diffs(rows)
            current_max = max(max(row) for row in rows)
            
            if current_max < best_max:
                best_max = current_max
                best_rows = [list(row) for row in rows]
    
    if best_max < initial_max:
        return best_rows
    return None


def _try_build_row(k, global_used, rng, deadline):
    """Try to build a row of k+1 elements starting at 0 avoiding global_used diffs."""
    row = [0]
    x = 0
    used_in_row = set()
    while len(row) < k + 1:
        if time.time() > deadline:
            return None
        # Try candidates
        for dx in range(1, 20):
            x_candidate = x + dx
            new_diffs = {x_candidate - a for a in row}
            # Check against global used AND diffs already in this row
            if not (new_diffs & global_used) and not (new_diffs & used_in_row):
                row.append(x_candidate)
                x = x_candidate
                used_in_row |= new_diffs
                break
        else:
            return None
    return row
# EVOLVE-BLOCK-END
