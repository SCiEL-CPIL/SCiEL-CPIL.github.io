"""Improved construct for difference triangle set (4,7) using randomized greedy + local search."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _valid_rows(rows):
    """Check that all positive differences within each row are distinct across all rows."""
    used = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    return max(max(row) for row in rows)

def _greedy_row(used, k, randomize=False, seed=None):
    """Build a single row of k+1 increasing integers starting at 0.
    Greedily picks the smallest (or randomly perturbed) next value whose
    differences to earlier entries in the row are all unused."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
    
    row = [0]
    x = 0
    while len(row) < k + 1:
        # Try candidates in order (greedy) or with some randomization
        if randomize:
            # Look ahead a bit and pick from a small random window
            candidates = []
            x += 1
            for _ in range(3):  # consider a few candidates
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    candidates.append(x)
                x += 1
            if not candidates:
                # Fallback: keep incrementing until we find one
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x)
                        break
                    x += 1
            x = candidates[0]  # pick the first valid (smallest in window)
        else:
            x += 1
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                continue
        row.append(x)
        used.update(x - a for a in row)
    return row

def _local_search(rows, k, used, max_iters=5000, rng=None):
    """Try to reduce the maximum entry by shifting/removing/adding entries.
    Returns improved rows or the original if no improvement."""
    if rng is None:
        rng = random
    
    best_max = _max_entry(rows)
    rows = [list(r) for r in rows]
    
    for _ in range(max_iters):
        # Pick a random row and a random position to try to reduce
        ri = rng.randrange(len(rows))
        row = rows[ri]
        if len(row) <= 1:
            continue
        pi = rng.randrange(1, len(row))  # position to try to reduce (not 0)
        
        old_val = row[pi]
        # Try to decrease row[pi] by some amount
        delta = rng.randint(1, min(5, old_val - row[pi-1]))
        if delta <= 0:
            continue
        new_val = old_val - delta
        if new_val <= row[pi-1]:
            continue
        if new_val >= row[pi+1] if pi+1 < len(row) else False:
            continue
        
        # Check if this change is valid: remove old diffs, add new diffs
        # Old diffs involving position pi:
        old_diffs = set()
        for j in range(pi):
            old_diffs.add(row[pi] - row[j])
        for j in range(pi+1, len(row)):
            old_diffs.add(row[j] - row[pi])
        
        # New diffs:
        new_diffs = set()
        for j in range(pi):
            new_diffs.add(new_val - row[j])
        for j in range(pi+1, len(row)):
            new_diffs.add(row[j] - new_val)
        
        # Check: new_diffs should not conflict with (used - old_diffs)
        available = used - old_diffs
        if new_diffs & available:
            continue
        
        # Apply
        row[pi] = new_val
        used = (used - old_diffs) | new_diffs
        new_max = _max_entry(rows)
        if new_max < best_max:
            best_max = new_max
            rows = [list(r) for r in rows]
    
    return rows, best_max

def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across rows.
    Minimize the largest entry."""
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s kill
    
    best_rows = None
    best_max = float('inf')
    
    # Multiple restarts with different random seeds
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        rng = random.Random(attempt * 7919 + 13)
        
        # Try a few strategies per attempt
        for strategy in range(3):
            if time.time() > deadline:
                break
            
            used = set()
            rows = []
            
            # Greedy construction with some randomization
            for r in range(n_rows):
                if strategy == 0:
                    # Pure greedy
                    row = _greedy_row(used, k, randomize=False, seed=None)
                elif strategy == 1:
                    # Slightly randomized greedy
                    row = _greedy_row(used, k, randomize=True, seed=rng)
                else:
                    # Greedy but with a random offset in candidate selection
                    row = _greedy_row(used, k, randomize=True, seed=rng)
                rows.append(row)
            
            current_max = _max_entry(rows)
            if current_max < best_max:
                best_max = current_max
                best_rows = [list(r) for r in rows]
            
            # Local search to try to improve
            if time.time() < deadline - 10:
                improved_rows, improved_max = _local_search(
                    rows, k, used, max_iters=2000, rng=rng
                )
                if improved_max < best_max:
                    best_max = improved_max
                    best_rows = improved_rows
    
    if best_rows is None:
        # Fallback: pure greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            rows.append(_greedy_row(used, k, randomize=False, seed=None))
        best_rows = rows
    
    return best_rows

# EVOLVE-BLOCK-END
