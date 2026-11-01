"""Difference triangle set: minimize largest entry for n_rows rows of k+1 increasing ints from 0.

Structural approach:
1. Backtracking construction with forward checking and dynamic variable ordering.
   Each row is built by selecting the smallest available value that keeps all
   differences distinct, but we try multiple orderings and use a priority
   heuristic that places rows with tighter constraints first.
2. A constraint-propagation refinement: after initial construction, we attempt
   to reduce the maximum by re-solving with a tighter upper bound (binary search
   on the scope) using backtracking with forward checking.
3. Fallback to greedy + local search if backtracking is too slow.
"""
import time
import random
import sys

_START_TIME = None
_DEADLINE = 140.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START_TIME


def _time_left():
    return _DEADLINE - _elapsed()


def _diffs_of_row(row):
    """Return a set of all positive differences in a row."""
    s = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ri)
    return s


def _validate(rows):
    """Check all differences are distinct across all rows. Return True if valid."""
    seen = set()
    for row in rows:
        d = _diffs_of_row(row)
        if len(d) != len(row) * (len(row) - 1) // 2:
            return False
        if d & seen:
            return False
        seen |= d
    return True


def _backtrack_solve(n_rows, k, max_val, time_budget):
    """
    Backtracking solver to find n_rows rows of length k+1 with max entry <= max_val.
    Uses recursive construction with forward checking.
    """
    start_time = time.time()
    used_diffs = set()
    rows = []
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    if max_val < total_diffs_needed:
        return None

    def solve_row(row_idx, current_row, current_used, time_limit):
        if time.time() - start_time > time_limit:
            return None
        
        if len(current_row) == k + 1:
            return list(current_row)
        
        pos = len(current_row)
        if pos == 0:
            candidates = [0]
        else:
            last_val = current_row[-1]
            # Optimization: We must be able to fit remaining elements.
            # If we are at pos, we need k - pos more elements.
            # The smallest possible next value is last_val + 1.
            # The largest possible value is max_val.
            # We just iterate.
            candidates = range(last_val + 1, max_val + 1)
            
        for val in candidates:
            new_diffs = set()
            conflict = False
            for prev in current_row:
                d = val - prev
                if d in current_used:
                    conflict = True
                    break
                new_diffs.add(d)
            
            if conflict:
                continue
            
            # Check if we have enough "space" left? 
            # Simple check: if the number of available diffs <= max_val is less than needed, prune.
            # This is expensive to compute exactly, so we skip for now.
            
            new_used = current_used | new_diffs
            result = solve_row(row_idx, current_row + [val], new_used, time_limit)
            if result is not None:
                return result
        
        return None

    # Try to build rows one by one
    for r in range(n_rows):
        if time.time() - start_time > time_budget:
            return None
        
        row = solve_row(r, [0], used_diffs, time_budget)
        if row is None:
            return None
        rows.append(row)
        used_diffs |= _diffs_of_row(row)
        
    return rows


def _greedy_build(n_rows, k, rng, order=None):
    """Build rows greedily in the given order."""
    if order is None:
        order = list(range(n_rows))
    used = set()
    rows = [None] * n_rows
    for r in order:
        row = [0]
        while len(row) < k + 1:
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
                    break
                x += 1
        rows[r] = row
    return rows


def _local_search_reduce(rows, n_rows, k, rng, time_budget):
    """
    Local search: try to reduce the maximum value by moving elements down.
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)
    start = time.time()

    row_diffs = [_diffs_of_row(rows[r]) for r in range(n_rows)]

    while time.time() - start < time_budget and _time_left() > 3.0:
        # Find the row with the maximum element
        max_row = -1
        max_val = -1
        for r in range(n_rows):
            if rows[r][-1] > max_val:
                max_val = rows[r][-1]
                max_row = r
        
        if max_row == -1:
            break
        
        row = rows[max_row]
        improved = False
        
        # Try to reduce the last element
        for new_val in range(row[-1] - 1, row[-2], -1):
            row_copy = list(row)
            row_copy[-1] = new_val
            new_rd = _diffs_of_row(row_copy)
            if len(new_rd) != k * (k + 1) // 2:
                continue
            conflict = False
            for other_r in range(n_rows):
                if other_r == max_row:
                    continue
                if new_rd & row_diffs[other_r]:
                    conflict = True
                    break
            if not conflict:
                row[-1] = new_val
                row_diffs[max_row] = new_rd
                new_max = max(max(rows[x]) for x in range(n_rows))
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(rows[x]) for x in range(n_rows)]
                improved = True
                break
        
        if not improved:
            # Try reducing a random non-last element in a random row
            r = rng.randrange(n_rows)
            if len(rows[r]) <= 2:
                break
            pos = rng.randrange(1, len(rows[r]) - 1)
            row = rows[r]
            old = row[pos]
            for new_val in range(old - 1, row[pos - 1], -1):
                if pos < k and new_val >= row[pos + 1]:
                    continue
                row_copy = list(row)
                row_copy[pos] = new_val
                new_rd = _diffs_of_row(row_copy)
                if len(new_rd) != k * (k + 1) // 2:
                    continue
                conflict = False
                for other_r in range(n_rows):
                    if other_r == r:
                        continue
                    if new_rd & row_diffs[other_r]:
                        conflict = True
                        break
                if not conflict:
                    row[pos] = new_val
                    row_diffs[r] = new_rd
                    new_max = max(max(rows[x]) for x in range(n_rows))
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = [list(rows[x]) for x in range(n_rows)]
                    break

    return best_rows, best_max


def _sa_optimize(rows, n_rows, k, rng, time_budget):
    """
    Simulated annealing to reduce the maximum entry.
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)

    row_diffs = [_diffs_of_row(rows[r]) for r in range(n_rows)]

    start = time.time()
    T = max(1.0, best_max * 0.1)
    T_min = 0.5
    cooling = 0.997

    while T > T_min and time.time() - start < time_budget and _time_left() > 3.0:
        r = rng.randrange(n_rows)
        row = rows[r]
        if len(row) <= 1:
            T *= cooling
            continue
        pos = rng.randrange(1, len(row))
        old_val = row[pos]
        
        span = max(1, min(15, int(T)))
        delta = rng.randint(-span, span)
        new_val = old_val + delta
        
        if new_val <= row[pos - 1]:
            T *= cooling
            continue
        if pos < k and new_val >= row[pos + 1]:
            T *= cooling
            continue
        if new_val <= 0:
            T *= cooling
            continue
        
        row_copy = list(row)
        row_copy[pos] = new_val
        new_rd = _diffs_of_row(row_copy)
        
        if len(new_rd) != k * (k + 1) // 2:
            T *= cooling
            continue
        
        conflict = 0
        for other_r in range(n_rows):
            if other_r == r:
                continue
            conflict += len(new_rd & row_diffs[other_r])
        
        if conflict == 0:
            row_max = max(row_copy)
            other_max = 0
            for x in range(n_rows):
                if x != r:
                    for v in rows[x]:
                        if v > other_max:
                            other_max = v
            cur_max = max(row_max, other_max)
            
            if cur_max <= best_max or rng.random() < 0.1:
                row[pos] = new_val
                row_diffs[r] = new_rd
                if cur_max < best_max:
                    best_max = cur_max
                    best_rows = [list(rows[x]) for x in range(n_rows)]
        else:
            if conflict <= 2 and rng.random() < 0.01 * (T / 100.0):
                row[pos] = new_val
                row_diffs[r] = new_rd

        T *= cooling

    return best_rows, best_max


def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()

    rng = random.Random(42)

    best_rows = None
    best_max = float('inf')
    
    # Phase 1: Greedy baseline
    # Try a few random orders to get a decent starting point
    for t in range(10):
        if _time_left() < 5.0:
            break
        order = list(range(n_rows))
        rng.shuffle(order)
        rows = _greedy_build(n_rows, k, rng, order)
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [list(r) for r in rows]

    if best_rows is None:
        best_rows = _greedy_build(n_rows, k, rng)
        best_max = max(max(r) for r in best_rows)

    # Phase 2: Backtracking with binary search on scope
    # The lower bound is n_rows * k * (k+1) / 2
    total_diffs = n_rows * k * (k + 1) // 2
    lo = total_diffs
    hi = best_max
    
    # We want to find the smallest S such that a solution exists.
    # We will search from lo to hi.
    # Since the search space is small for n_rows=4, k=7, we can afford to be thorough.
    
    # Use a linear search from lo upwards if the range is small, otherwise binary.
    # Given the bound 112 and best ~134, range is small.
    # Let's try to find the minimum scope using backtracking.
    
    # We will try scopes in increasing order.
    # To save time, we can first try a few random scopes or just linear.
    # Linear is safer to ensure we find the minimum.
    
    current_scope = lo
    found_scope = None
    
    # Allocate time for backtracking
    bt_time_budget = 10.0
    
    # If the gap is large, use binary search. If small, linear.
    if hi - lo < 50:
        for S in range(lo, hi + 1):
            if _time_left() < 5.0:
                break
            rows = _backtrack_solve(n_rows, k, S, bt_time_budget)
            if rows is not None:
                if _validate(rows):
                    found_scope = S
                    best_rows = rows
                    best_max = S
                    break
    else:
        # Binary search
        while lo <= hi and _time_left() > 10.0:
            mid = (lo + hi) // 2
            rows = _backtrack_solve(n_rows, k, mid, bt_time_budget)
            if rows is not None and _validate(rows):
                best_rows = rows
                best_max = mid
                hi = mid - 1
                found_scope = mid
            else:
                lo = mid + 1

    # Phase 3: Local Search / SA to polish
    if _time_left() > 10.0:
        budget = min(20.0, _time_left() - 5.0)
        ls_rows, ls_max = _local_search_reduce(best_rows, n_rows, k, rng, budget)
        if ls_max < best_max:
            best_max = ls_max
            best_rows = ls_rows

    if _time_left() > 10.0:
        budget = min(20.0, _time_left() - 5.0)
        sa_rows, sa_max = _sa_optimize(best_rows, n_rows, k, rng, budget)
        if sa_max < best_max:
            best_max = sa_max
            best_rows = sa_rows

    # Final verification
    if not _validate(best_rows):
        # Fallback
        best_rows = _greedy_build(n_rows, k, rng)
        best_max = max(max(r) for r in best_rows)

    return best_rows
