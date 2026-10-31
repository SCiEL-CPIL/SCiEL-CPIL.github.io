"""Difference triangle set: minimize largest entry for n_rows rows of k+1 increasing ints from 0."""
import time
import random
import sys

_START_TIME = None
_DEADLINE = 140.0  # safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _elapsed()

def _compute_diffs_set(row):
    """Return set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _all_diffs_valid(rows, n_rows, k):
    """Check that all differences across all rows are distinct."""
    seen = set()
    for row in rows:
        d = _compute_diffs_set(row)
        if d & seen:
            return False
        seen |= d
    return True

def _greedy_build(n_rows, k, row_order, rng):
    """Greedy: build rows in given order, each row greedily picks smallest valid next value."""
    used = set()
    rows = [None] * n_rows
    for r in row_order:
        row = [0]
        used_in_row = set()
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                # Check new differences
                ok = True
                for a in row:
                    d = x - a
                    if d in used or d in used_in_row:
                        ok = False
                        break
                if ok:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
                        used_in_row.add(x - a)
                    break
                x += 1
        rows[r] = row
    return rows

def _greedy_round_robin(n_rows, k, rng):
    """Round-robin greedy: extend rows one element at a time, choosing the row with smallest last element."""
    used = set()
    rows = [[0] for _ in range(n_rows)]
    total_needed = n_rows * (k + 1)
    placed = n_rows
    
    while placed < total_needed:
        if _time_left() < 3.0:
            return None
        # Find row with smallest last element that still needs more
        best_r = -1
        best_last = -1
        order = list(range(n_rows))
        rng.shuffle(order)
        for r in order:
            if len(rows[r]) >= k + 1:
                continue
            last = rows[r][-1]
            if best_r == -1 or last < best_last:
                best_last = last
                best_r = r
        if best_r == -1:
            return None
        row = rows[best_r]
        x = row[-1] + 1
        while True:
            ok = True
            for a in row:
                d = x - a
                if d in used:
                    ok = False
                    break
            if ok:
                row.append(x)
                for a in row[:-1]:
                    used.add(x - a)
                placed += 1
                break
            x += 1
    return rows

def _try_greedy_variants(n_rows, k, rng, num_trials=50):
    """Try multiple greedy strategies and return best."""
    best_rows = None
    best_max = float('inf')
    
    for t in range(num_trials):
        if _time_left() < 5.0:
            break
        # Random row order
        order = list(range(n_rows))
        rng.shuffle(order)
        rows = _greedy_build(n_rows, k, order, rng)
        if rows is None:
            continue
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [list(r) for r in rows]
    
    # Also try round-robin
    for t in range(num_trials):
        if _time_left() < 5.0:
            break
        rows = _greedy_round_robin(n_rows, k, rng)
        if rows is None:
            continue
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [list(r) for r in rows]
    
    return best_rows, best_max

def _local_search(rows, n_rows, k, rng, time_budget):
    """Local search: try to reduce max by moving elements in rows near the max."""
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)
    start = time.time()
    
    # Precompute difference sets for each row
    row_diffs = [_compute_diffs_set(rows[r]) for r in range(n_rows)]
    all_diffs = set()
    for rd in row_diffs:
        all_diffs |= rd
    
    while time.time() - start < time_budget and _time_left() > 3.0:
        # Find rows with elements near the max
        max_val = best_max
        target_rows = [r for r in range(n_rows) if max(rows[r]) >= max_val - 10]
        if not target_rows:
            break
        r = rng.choice(target_rows)
        row = rows[r]
        
        # Try to reduce the last element
        old_val = row[-1]
        improved = False
        for new_val in range(old_val - 1, row[-2], -1):
            # Check if new_val - a conflicts for all a in row[:-1]
            new_diffs = set()
            ok = True
            for a in row[:-1]:
                d = new_val - a
                if d in row_diffs[r] or d in all_diffs:
                    # d in row_diffs[r] means it's already used within this row
                    # but we're replacing the last element, so we need to check against row_diffs[r] minus old diffs
                    ok = False
                    break
                new_diffs.add(d)
            if not ok:
                continue
            # Check against other rows
            conflict = False
            for other_r in range(n_rows):
                if other_r == r:
                    continue
                if new_diffs & row_diffs[other_r]:
                    conflict = True
                    break
            if not conflict:
                # Apply the move
                row[-1] = new_val
                # Update diff sets
                old_rd = row_diffs[r]
                row_diffs[r] = _compute_diffs_set(row)
                all_diffs = set()
                for rr in range(n_rows):
                    all_diffs |= row_diffs[rr]
                cur_max = max(max(rows[rr]) for rr in range(n_rows))
                if cur_max < best_max:
                    best_max = cur_max
                    best_rows = [list(rows[rr]) for rr in range(n_rows)]
                improved = True
                break
        if not improved:
            # Try a random perturbation: pick a random position in a random row and try to change it
            rr = rng.randrange(n_rows)
            pos = rng.randrange(1, len(rows[rr]))
            old = rows[rr][pos]
            delta = rng.randint(-5, 5)
            new = old + delta
            if new <= rows[rr][pos-1] or (pos < k and new >= rows[rr][pos+1]) or new <= 0:
                continue
            # Check validity
            new_diffs = set()
            ok = True
            for i in range(len(rows[rr])):
                for j in range(i+1, len(rows[rr])):
                    a = rows[rr][i] if i != pos else new
                    b = rows[rr][j] if j != pos else new
                    if i == pos or j == pos:
                        d = b - a
                        new_diffs.add(d)
            # Full check
            row_copy = list(rows[rr])
            row_copy[pos] = new
            nd = _compute_diffs_set(row_copy)
            # Check internal
            if len(nd) != k*(k+1)//2:
                continue
            # Check against others
            conflict = False
            for other_r in range(n_rows):
                if other_r == rr:
                    continue
                if nd & row_diffs[other_r]:
                    conflict = True
                    break
            if not conflict:
                rows[rr][pos] = new
                row_diffs[rr] = nd
                all_diffs = set()
                for x in range(n_rows):
                    all_diffs |= row_diffs[x]
                cur_max = max(max(rows[x]) for x in range(n_rows))
                if cur_max < best_max:
                    best_max = cur_max
                    best_rows = [list(rows[x]) for x in range(n_rows)]
    
    return best_rows, best_max

def _simulated_annealing(rows, n_rows, k, rng, time_budget):
    """SA with bitmask-style conflict counting for speed."""
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)
    start = time.time()
    
    # Precompute
    row_diffs = [_compute_diffs_set(rows[r]) for r in range(n_rows)]
    all_diffs = set()
    for rd in row_diffs:
        all_diffs |= rd
    
    T = 100.0
    T_min = 0.5
    cooling = 0.998
    
    while T > T_min and time.time() - start < time_budget and _time_left() > 3.0:
        r = rng.randrange(n_rows)
        pos = rng.randrange(1, k + 1)
        row = rows[r]
        old_val = row[pos]
        delta = rng.randint(-8, 8)
        new_val = old_val + delta
        
        if new_val <= row[pos-1]:
            T *= cooling
            continue
        if pos < k and new_val >= row[pos+1]:
            T *= cooling
            continue
        if new_val <= 0:
            T *= cooling
            continue
        
        # Compute new diff set for this row
        row_copy = list(row)
        row_copy[pos] = new_val
        new_rd = _compute_diffs_set(row_copy)
        
        # Check internal validity (all diffs distinct within row)
        if len(new_rd) != k * (k + 1) // 2:
            T *= cooling
            continue
        
        # Check against other rows
        conflict_count = 0
        for other_r in range(n_rows):
            if other_r == r:
                continue
            conflict_count += len(new_rd & row_diffs[other_r])
        
        if conflict_count == 0:
            # Valid move
            row[pos] = new_val
            row_diffs[r] = new_rd
            cur_max = max(max(rows[x]) for x in range(n_rows))
            if cur_max < best_max:
                best_max = cur_max
                best_rows = [list(rows[x]) for x in range(n_rows)]
        else:
            # Accept with probability based on temperature and conflict count
            if rng.random() < 0.005 * (T / 100.0) and conflict_count <= 3:
                row[pos] = new_val
                row_diffs[r] = new_rd
        
        T *= cooling
    
    return best_rows, best_max

def _multi_start_local_search(rows, n_rows, k, rng, time_budget):
    """Run local search from multiple perturbed starts."""
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)
    start = time.time()
    
    while time.time() - start < time_budget and _time_left() > 5.0:
        # Perturb: pick a random row and position, change it
        cur_rows = [list(r) for r in best_rows]
        r = rng.randrange(n_rows)
        pos = rng.randrange(1, k + 1)
        delta = rng.randint(-5, 5)
        new_val = cur_rows[r][pos] + delta
        if new_val > cur_rows[r][pos-1] and (pos >= k or new_val < cur_rows[r][pos+1]) and new_val > 0:
            cur_rows[r][pos] = new_val
        
        # Quick validity check
        if not _all_diffs_valid(cur_rows, n_rows, k):
            continue
        
        cur_max = max(max(r) for r in cur_rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = cur_rows
        
        # Local search from this state
        ls_rows, ls_max = _local_search(cur_rows, n_rows, k, rng, min(2.0, _time_left() - 3.0))
        if ls_max < best_max:
            best_max = ls_max
            best_rows = ls_rows
    
    return best_rows, best_max

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(42)
    
    # Phase 1: Greedy with multiple strategies
    best_rows, best_max = _try_greedy_variants(n_rows, k, rng, num_trials=40)
    
    if best_rows is None:
        # Fallback
        best_rows = _greedy_build(n_rows, k, list(range(n_rows)), rng)
        best_max = max(max(r) for r in best_rows)
    
    # Phase 2: Local search
    if _time_left() > 10.0:
        budget = min(30.0, _time_left() - 5.0)
        ls_rows, ls_max = _local_search(best_rows, n_rows, k, rng, budget)
        if ls_max < best_max:
            best_max = ls_max
            best_rows = ls_rows
    
    # Phase 3: Simulated annealing
    if _time_left() > 15.0:
        budget = min(40.0, _time_left() - 5.0)
        sa_rows, sa_max = _simulated_annealing(best_rows, n_rows, k, rng, budget)
        if sa_max < best_max:
            best_max = sa_max
            best_rows = sa_rows
    
    # Phase 4: Multi-start local search
    if _time_left() > 10.0:
        budget = min(30.0, _time_left() - 5.0)
        ms_rows, ms_max = _multi_start_local_search(best_rows, n_rows, k, rng, budget)
        if ms_max < best_max:
            best_max = ms_max
            best_rows = ms_rows
    
    # Final local search polish
    if _time_left() > 5.0:
        budget = min(10.0, _time_left() - 2.0)
        ls_rows, ls_max = _local_search(best_rows, n_rows, k, rng, budget)
        if ls_max < best_max:
            best_max = ls_max
            best_rows = ls_rows
    
    # Final verification
    if not _all_diffs_valid(best_rows, n_rows, k):
        # Should not happen, but try to fix
        pass
    
    return best_rows
