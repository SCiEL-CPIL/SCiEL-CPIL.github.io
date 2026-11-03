"""Improved program for the difference triangle set task."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _validate(rows, k):
    """Check that all differences are distinct across rows."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d <= 0:
                    return False
                if d in used:
                    return False
                used.add(d)
    return True

def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(row) for row in rows)

def _greedy_row(used, k, start_x=1):
    """Build a single row greedily starting from 0."""
    row = [0]
    x = start_x
    while len(row) < k + 1:
        new_diffs = {x - a for a in row}
        if not (new_diffs & used):
            row.append(x)
            used |= new_diffs
        x += 1
    return row

def _greedy_all(n_rows, k):
    """Build all rows greedily."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        rows.append(row)
    return rows

def _local_search(rows, k, max_iters=5000, rng=None):
    """Local search: try to reduce the maximum entry by swapping/adjusting elements."""
    if rng is None:
        rng = random.Random(42)
    
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    used = set()
    for row in best_rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        # Try to reduce the last element of each row
        for ri in range(len(best_rows)):
            row = best_rows[ri]
            last = row[-1]
            # Try to find a smaller value for the last element
            for new_last in range(last - 1, row[-2], -1):
                # Check if replacing last with new_last keeps all diffs distinct
                # Remove old diffs involving last
                old_diffs = {last - a for a in row[:-1]}
                # Add new diffs
                new_diffs = {new_last - a for a in row[:-1]}
                # Check new_diffs don't conflict with (used - old_diffs)
                remaining_used = used - old_diffs
                if not (new_diffs & remaining_used):
                    # Accept
                    used = remaining_used | new_diffs
                    row[-1] = new_last
                    new_scope = _scope(best_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                    improved = True
                    break
    return best_rows, best_scope

def _randomized_greedy(n_rows, k, rng, max_attempts=20):
    """Try multiple randomized greedy constructions and keep the best."""
    best_rows = None
    best_scope = float('inf')
    
    for attempt in range(max_attempts):
        used = set()
        rows = []
        success = True
        for _ in range(n_rows):
            row = [0]
            x = 1
            # Add some randomization in ordering of candidate positions
            while len(row) < k + 1:
                # Try a few random candidates near the minimum
                found = False
                base_x = x
                # Try sequential first, then some random jumps
                for _ in range(5):
                    candidate = base_x + rng.randint(0, 3)
                    new_diffs = {candidate - a for a in row}
                    if not (new_diffs & used):
                        row.append(candidate)
                        used |= new_diffs
                        x = candidate + 1
                        found = True
                        break
                if not found:
                    # Fallback to sequential
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            row.append(x)
                            used |= new_diffs
                            x += 1
                            break
                        x += 1
            rows.append(row)
        
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
    
    return best_rows, best_scope

def construct(n_rows, k):
    """Construct a difference triangle set with n_rows rows of k+1 integers."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with greedy
    rows = _greedy_all(n_rows, k)
    best_scope = _scope(rows)
    
    # Local search to improve
    rows, best_scope = _local_search(rows, k, max_iters=10000)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        return rows
    
    # Try randomized greedy with local search
    rng = random.Random(12345)
    for seed in range(50):
        if time.time() > deadline:
            break
        rng.seed(seed)
        r_rows, r_scope = _randomized_greedy(n_rows, k, rng, max_attempts=5)
        if r_rows is not None:
            r_rows, r_scope = _local_search(r_rows, k, max_iters=3000, rng=rng)
            if r_scope < best_scope:
                best_scope = r_scope
                rows = r_rows
    
    return rows

# EVOLVE-BLOCK-END
