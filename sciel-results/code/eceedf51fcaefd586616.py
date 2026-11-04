"""Difference triangle set (4,7): minimize the largest entry (scope)."""
import time
import random
import math

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within a row are distinct across all rows.
    Return the list of rows with the smallest possible scope found within the time limit.
    """
    t_start = time.time()
    DEADLINE = 140  # safety margin before 150s kill
    
    # Precompute the number of differences per row: k*(k+1)/2
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound on scope from counting: need at least total_diffs distinct positive diffs
    # max diff = scope, so scope*(scope+1)/2 >= total_diffs
    lb_scope = 0
    while lb_scope * (lb_scope + 1) // 2 < total_diffs:
        lb_scope += 1
    
    best_rows = None
    best_scope = float('inf')
    
    # Strategy 1: Pure greedy (deterministic)
    rows = _pure_greedy(n_rows, k)
    scope = max(max(r) for r in rows)
    if scope < best_scope:
        best_scope = scope
        best_rows = rows
    
    if time.time() - t_start > DEADLINE:
        return best_rows
    
    # Strategy 2: Randomized greedy with restarts
    # Use multiple random seeds and different orderings
    rng = random.Random(42)
    attempt = 0
    while time.time() - t_start < DEADLINE:
        attempt += 1
        rows = _randomized_greedy(n_rows, k, rng)
        if rows is not None:
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
                # If we hit the lower bound, we can't do better
                if best_scope == lb_scope:
                    break
        # Occasionally try a different strategy
        if attempt % 5 == 0 and time.time() - t_start < DEADLINE:
            rows = _greedy_with_ordering(n_rows, k, rng)
            if rows is not None:
                scope = max(max(r) for r in rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = rows
                    if best_scope == lb_scope:
                        break
    
    # Strategy 3: Local search / improvement on the best found
    if time.time() - t_start < DEADLINE and best_rows is not None:
        improved = _local_search_improve(best_rows, n_rows, k, rng)
        if improved is not None:
            scope = max(max(r) for r in improved)
            if scope < best_scope:
                best_scope = scope
                best_rows = improved
    
    return best_rows


def _pure_greedy(n_rows, k):
    """Standard greedy: each row takes smallest available mark."""
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


def _randomized_greedy(n_rows, k, rng):
    """
    Randomized greedy: when multiple candidates are available, pick randomly.
    Also randomize the order in which we consider next values.
    """
    used = set()
    rows = []
    for r_idx in range(n_rows):
        row = [0]
        x = 0
        # Track the smallest x that could work; we'll collect a few options
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                # Found a valid x; with some probability accept immediately,
                # otherwise look ahead a bit for alternatives
                if rng.random() < 0.3:
                    # Look ahead up to 5 more values for alternatives
                    candidates = [x]
                    for dx in range(1, 6):
                        xx = x + dx
                        nd = {xx - a for a in row}
                        if not nd & used:
                            candidates.append(xx)
                    chosen = rng.choice(candidates)
                    row.append(chosen)
                    used |= {chosen - a for a in row[:-1]}
                    x = chosen - 1  # continue from here
                    break
                else:
                    row.append(x)
                    used |= new_diffs
                    break
        rows.append(row)
    return rows


def _greedy_with_ordering(n_rows, k, rng):
    """
    Greedy but with randomized row ordering and mark selection bias.
    """
    used = set()
    rows = [None] * n_rows
    # Process rows in random order
    order = list(range(n_rows))
    rng.shuffle(order)
    
    for r_idx in order:
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                # Randomized choice: accept with probability, else skip a bit
                if rng.random() < 0.7:
                    row.append(x)
                    used |= new_diffs
                else:
                    # Skip this one, try a few more
                    found = False
                    for dx in range(1, 4):
                        xx = x + dx
                        nd = {xx - a for a in row}
                        if not nd & used:
                            row.append(xx)
                            used |= nd
                            x = xx - 1
                            found = True
                            break
                    if not found:
                        row.append(x)
                        used |= new_diffs
        rows[r_idx] = row
    
    return rows


def _local_search_improve(rows, n_rows, k, rng):
    """
    Try to reduce the scope by adjusting the last element of each row
    or shifting elements slightly.
    """
    best_scope = max(max(r) for r in rows)
    current_rows = [list(r) for r in rows]
    
    for _ in range(50):  # limited iterations
        if time.time() > 140:
            break
        # Try to reduce the maximum element
        max_idx = max(range(n_rows), key=lambda i: max(current_rows[i]))
        row = current_rows[max_idx]
        current_max = max(row)
        
        # Try replacing the last element with a smaller one
        improved = False
        for new_last in range(current_max - 1, row[-2], -1):
            candidate = row[:-1] + [new_last]
            if _is_valid_set(candidate, current_rows, max_idx):
                current_rows[max_idx] = candidate
                best_scope = min(best_scope, max(max(r) for r in current_rows))
                improved = True
                break
        
        if not improved:
            # Try adjusting an internal element
            pos = rng.randint(1, len(row) - 2)
            old_val = row[pos]
            for delta in [-1, 1, -2, 2]:
                new_val = old_val + delta
                if new_val <= row[pos-1] or new_val >= row[pos+1]:
                    continue
                candidate = row[:pos] + [new_val] + row[pos+1:]
                if _is_valid_set(candidate, current_rows, max_idx):
                    current_rows[max_idx] = candidate
                    best_scope = min(best_scope, max(max(r) for r in current_rows))
                    improved = True
                    break
        
        if not improved:
            break
    
    if best_scope < max(max(r) for r in rows):
        return current_rows
    return None


def _is_valid_set(candidate_row, all_rows, replace_idx):
    """Check if candidate_row (with replace_idx in all_rows) creates valid distinct diffs."""
    # Get all diffs from other rows
    used = set()
    for i, r in enumerate(all_rows):
        if i == replace_idx:
            continue
        for a in range(len(r)):
            for b in range(a+1, len(r)):
                used.add(r[b] - r[a])
    
    # Check candidate row diffs
    for a in range(len(candidate_row)):
        for b in range(a+1, len(candidate_row)):
            d = candidate_row[b] - candidate_row[a]
            if d in used:
                return False
    return True

# EVOLVE-BLOCK-END
