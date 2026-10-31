"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _row_diffs(row):
    """Return set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _count_conflicts(rows):
    """Count total duplicate differences across all rows."""
    from collections import Counter
    c = Counter()
    for row in rows:
        for d in _row_diffs(row):
            c[d] += 1
    return sum(v - 1 for v in c.values() if v > 1)

def _greedy_row(k, used, conflict_weight=0.0):
    """
    Greedily build a row of k+1 increasing ints starting at 0.
    'used' is a set of differences already taken by other rows.
    conflict_weight: if >0, prefer marks that create fewer conflicts.
    Returns the row.
    """
    row = [0]
    x = 1
    while len(row) < k + 1:
        best_x = None
        best_score = None
        # Look ahead a window
        for candidate in range(x, x + 50):
            new_diffs = {candidate - a for a in row}
            conflicts = len(new_diffs & used)
            # Prefer fewer conflicts, then smaller value
            score = (conflicts, candidate)
            if best_score is None or score < best_score:
                best_score = score
                best_x = candidate
            # Early stop if no conflicts and small enough
            if conflicts == 0 and candidate - row[-1] <= 2:
                break
        row.append(best_x)
        used |= {best_x - a for a in row[:-1]}
        x = best_x + 1
    return row

def _repair_row(idx, rows, k, used_others):
    """
    Rebuild row idx using greedy that avoids used_others (diffs from all other rows).
    Returns new row.
    """
    # Collect used diffs from all OTHER rows
    used = set()
    for i, row in enumerate(rows):
        if i == idx:
            continue
        used |= _row_diffs(row)
    # Greedy build
    row = [0]
    x = 1
    while len(row) < k + 1:
        best_x = None
        best_score = None
        for candidate in range(x, x + 80):
            new_diffs = {candidate - a for a in row}
            conflicts = len(new_diffs & used)
            score = (conflicts, candidate)
            if best_score is None or score < best_score:
                best_score = score
                best_x = candidate
            if conflicts == 0:
                break
        row.append(best_x)
        used |= {best_x - a for a in row[:-1]}
        x = best_x + 1
    rows[idx] = row
    return row

def _scope(rows):
    return max(max(r) for r in rows)

def _try_shrink(rows, k, deadline):
    """
    Try to reduce scope by shrinking the last element of each row.
    For each row, try to find a smaller last mark that keeps all diffs distinct.
    """
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(rows)):
            row = list(rows[idx])
            old_last = row[-1]
            # Try to reduce the last element
            for new_last in range(old_last - 1, row[-2], -1):
                test_row = row[:-1] + [new_last]
                # Check validity
                test_diffs = _row_diffs(test_row)
                # Check against all other rows
                ok = True
                for j, other in enumerate(rows):
                    if j == idx:
                        continue
                    other_diffs = _row_diffs(other)
                    if test_diffs & other_diffs:
                        ok = False
                        break
                if ok:
                    rows[idx] = test_row
                    improved = True
                    break
    return rows

def _kick(rows, k, rng):
    """Perturb a random row by randomizing its tail."""
    idx = rng.randint(0, len(rows) - 1)
    row = list(rows[idx])
    # Randomize the last few elements
    tail_start = max(1, len(row) - 3)
    base = row[tail_start - 1] if tail_start > 0 else 0
    for i in range(tail_start, len(row)):
        row[i] = base + rng.randint(1, 50)
        base = row[i]
    # Ensure strictly increasing
    for i in range(1, len(row)):
        if row[i] <= row[i-1]:
            row[i] = row[i-1] + 1
    rows[idx] = row

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    rng = random.Random(42)
    
    # Phase 1: Initial construction using conflict-aware greedy
    rows = []
    used = set()
    for i in range(n_rows):
        row = _greedy_row(k, used)
        rows.append(row)
        used |= _row_diffs(row)
    
    # Phase 2: Repair conflicts via iterative local search
    for iteration in range(2000):
        if time.time() > deadline:
            break
        # Count conflicts per row
        from collections import Counter
        diff_counter = Counter()
        for row in rows:
            for d in _row_diffs(row):
                diff_counter[d] += 1
        total_conflicts = sum(v - 1 for v in diff_counter.values() if v > 1)
        
        if total_conflicts == 0:
            break
        
        # Find most conflicted row
        row_conflicts = [0] * n_rows
        for i, row in enumerate(rows):
            for d in _row_diffs(row):
                if diff_counter[d] > 1:
                    row_conflicts[i] += diff_counter[d] - 1
        worst = max(range(n_rows), key=lambda i: row_conflicts[i])
        _repair_row(worst, rows, k, None)
    
    # Phase 3: Optimization - shrink scope
    rows = _try_shrink(rows, k, deadline)
    
    # Phase 4: Iterated kicks for further improvement
    best_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    
    for it in range(500):
        if time.time() > deadline:
            break
        _kick(rows, k, rng)
        # Quick repair
        for _ in range(50):
            if time.time() > deadline:
                break
            from collections import Counter
            diff_counter = Counter()
            for row in rows:
                for d in _row_diffs(row):
                    diff_counter[d] += 1
            total_conflicts = sum(v - 1 for v in diff_counter.values() if v > 1)
            if total_conflicts == 0:
                break
            row_conflicts = [0] * n_rows
            for i, row in enumerate(rows):
                for d in _row_diffs(row):
                    if diff_counter[d] > 1:
                        row_conflicts[i] += diff_counter[d] - 1
            worst = max(range(n_rows), key=lambda i: row_conflicts[i])
            _repair_row(worst, rows, k, None)
        
        # Check validity
        all_diffs = []
        for row in rows:
            all_diffs.extend(_row_diffs(row))
        if len(all_diffs) == len(set(all_diffs)):
            sc = _scope(rows)
            if sc < best_scope:
                best_scope = sc
                best_rows = [list(r) for r in rows]
                # Try to shrink further
                best_rows = _try_shrink(best_rows, k, deadline)
                best_scope = _scope(best_rows)
    
    return best_rows
# EVOLVE-BLOCK-END
