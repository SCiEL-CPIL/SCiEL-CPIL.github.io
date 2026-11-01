"""Difference triangle set: minimize largest entry for n_rows rows of k+1 increasing ints from 0."""
import time
import random
from itertools import combinations

_START_TIME = None
_DEADLINE = 140.0

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _elapsed()

def _all_diffs(row):
    """Return set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _validate(rows, n_rows, k):
    seen = set()
    for row in rows:
        if len(row) != k + 1 or row[0] != 0:
            return False
        for i in range(k):
            if row[i] >= row[i + 1]:
                return False
        d = _all_diffs(row)
        if len(d) != k * (k + 1) // 2:
            return False
        if d & seen:
            return False
        seen |= d
    return True

def _greedy_row(used_diffs, k, start=0):
    """Build one row greedily given a set of already-used differences."""
    row = [0]
    while len(row) < k + 1:
        x = row[-1] + 1
        while True:
            valid = True
            for a in row:
                if (x - a) in used_diffs:
                    valid = False
                    break
            if valid:
                row.append(x)
                for a in row[:-1]:
                    used_diffs.add(x - a)
                break
            x += 1
    return row

def _greedy_build(n_rows, k, order, rng):
    used = set()
    rows = [None] * n_rows
    for r in order:
        rows[r] = _greedy_row(used, k)
    return rows

def _lower_bound(n_rows, k, max_val):
    """Counting lower bound: need n_rows * k*(k+1)/2 distinct diffs in [1, max_val]."""
    needed = n_rows * k * (k + 1) // 2
    available = max_val * (max_val + 1) // 2
    return needed <= available

def _backtracking(n_rows, k, upper_bound, rng, time_limit):
    """
    Backtracking search: construct rows one by one, choosing each element
    with pruning based on remaining difference capacity.
    """
    start_time = time.time()
    best = None
    best_max = upper_bound
    num_diffs_per_row = k * (k + 1) // 2

    def search(rows, used_diffs, row_idx, max_so_far):
        nonlocal best, best_max

        if time.time() - start_time > time_limit:
            return

        if row_idx == n_rows:
            if max_so_far < best_max:
                best_max = max_so_far
                best = [list(r) for r in rows]
            return

        # Prune: if even with remaining rows we can't beat best, stop
        if max_so_far >= best_max:
            return

        # Counting bound: remaining rows need (n_rows - row_idx) * num_diffs_per_row
        # distinct diffs, all must be <= some value. If max_so_far is already
        # too small to accommodate, prune.
        remaining_rows = n_rows - row_idx
        needed_diffs = remaining_rows * num_diffs_per_row
        # All diffs must be in [1, best_max - 1] (since we need strict improvement)
        # Actually diffs can be up to the max entry which is at most best_max - 1
        # But conservatively, if we're building a row with max entry M, diffs go up to M
        # Let's use: available diffs in [1, best_max-1] must be >= needed
        avail = (best_max - 1) * best_max // 2
        if avail < needed_diffs:
            return

        row = [0]
        row_diffs = set()

        def extend_row(pos, current_max):
            nonlocal best, best_max

            if time.time() - start_time > time_limit:
                return

            if pos == k + 1:
                # Row complete
                rows.append(row)
                all_diffs = used_diffs | row_diffs
                search(rows, all_diffs, row_idx + 1, max(current_max, best_max))
                rows.pop()
                return

            if current_max >= best_max:
                return

            # Determine range of valid next values
            last = row[pos - 1] if pos > 0 else 0
            # Upper bound: we need the row's max to be < best_max
            # Also, the new value must create diffs not in used_diffs
            upper = min(best_max - 1, 200)  # cap for practicality

            for x in range(last + 1, upper + 1):
                if current_max < x:
                    if x >= best_max:
                        break
                new_diffs = []
                valid = True
                for i in range(pos):
                    d = x - row[i]
                    if d in used_diffs or d in row_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                if not valid:
                    continue

                row.append(x)
                row_diffs.update(new_diffs)
                extend_row(pos + 1, max(current_max, x))
                row.pop()
                for d in new_diffs:
                    row_diffs.discard(d)

            # If we haven't found a solution with improvement, try relaxing
            # but only if we haven't exhausted the range

        extend_row(1, 0)

    # Try multiple random row orderings
    for attempt in range(10):
        if time.time() - start_time > time_limit:
            break
        order = list(range(n_rows))
        rng.shuffle(order)
        # Re-index: we build rows in 'order' sequence
        # Simplification: just build rows 0..n_rows-1 in shuffled priority
        search([], set(), 0, 0)
        if best is not None:
            break

    return best, best_max

def _local_search(rows, n_rows, k, rng, time_budget):
    """
    Local search: try to reduce max entry by replacing elements in rows
    that contain the maximum, with smaller valid values.
    """
    start = time.time()
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)

    while time.time() - start < time_budget and _time_left() > 3.0:
        max_val = max(max(r) for r in rows)
        # Find rows containing values near max
        target_rows = [r for r in range(n_rows) if max(rows[r]) >= max_val - 10]
        if not target_rows:
            target_rows = list(range(n_rows))
        r = rng.choice(target_rows)
        row = rows[r]

        # Try to reduce each element from the end
        improved = False
        for pos in range(k, 0, -1):
            old = row[pos]
            lower = row[pos - 1] + 1
            upper = (row[pos + 1] - 1) if pos < k else old - 1
            if upper < lower:
                continue

            # Try smaller values first (to reduce max)
            for new in range(upper, lower - 1, -1):
                row[pos] = new
                if not _check_row_valid(rows, r, n_rows, k):
                    row[pos] = old
                    continue
                cur_max = max(max(row) for row in rows)
                if cur_max < best_max:
                    best_max = cur_max
                    best_rows = [list(row) for row in rows]
                improved = True
                break
            if improved:
                break

        if not improved:
            # Random perturbation to escape local minimum
            rr = rng.randrange(n_rows)
            pos = rng.randrange(1, k + 1)
            old = rows[rr][pos]
            delta = rng.randint(-5, 5)
            new = old + delta
            if new <= rows[rr][pos - 1] or (pos < k and new >= rows[rr][pos + 1]) or new <= 0:
                continue
            rows[rr][pos] = new
            if not _check_row_valid(rows, rr, n_rows, k):
                rows[rr][pos] = old

    return best_rows, best_max

def _check_row_valid(rows, changed_row, n_rows, k):
    """Check if the solution is still valid after modifying one row."""
    seen = set()
    for r in range(n_rows):
        row = rows[r]
        if len(row) != k + 1 or row[0] != 0:
            return False
        for i in range(k):
            if row[i] >= row[i + 1]:
                return False
        d = _all_diffs(row)
        if len(d) != k * (k + 1) // 2:
            return False
        if d & seen:
            return False
        seen |= d
    return True

def _iterated_local_search(rows, n_rows, k, rng, time_budget):
    """ILS: repeated local search with random kicks."""
    start = time.time()
    current_rows = [list(r) for r in rows]
    current_max = max(max(r) for r in current_rows)
    best_rows = [list(r) for r in current_rows]
    best_max = current_max

    while time.time() - start < time_budget and _time_left() > 5.0:
        # Kick: perturb 1-3 random elements
        perturbed = [list(r) for r in current_rows]
        num_kicks = rng.randint(1, 3)
        for _ in range(num_kicks):
            r = rng.randrange(n_rows)
            pos = rng.randrange(1, k + 1)
            old = perturbed[r][pos]
            delta = rng.randint(-4, 4)
            new = old + delta
            if new <= perturbed[r][pos - 1] or (pos < k and new >= perturbed[r][pos + 1]) or new <= 0:
                continue
            perturbed[r][pos] = new

        if not _validate(perturbed, n_rows, k):
            continue

        # Local search
        budget = min(8.0, time_budget - (time.time() - start), _time_left() - 5.0)
        if budget < 1.0:
            break
        ls_rows, ls_max = _local_search(perturbed, n_rows, k, rng, budget)

        if ls_max < best_max:
            best_max = ls_max
            best_rows = ls_rows
            current_rows = ls_rows
            current_max = ls_max
        elif ls_max == best_max and rng.random() < 0.3:
            # Accept sideways move with some probability
            current_rows = ls_rows
            current_max = ls_max

    return best_rows, best_max

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    rng = random.Random(12345)

    # Phase 1: Greedy baseline with multiple orderings
    best_rows = None
    best_max = float('inf')

    for t in range(30):
        if _time_left() < 10.0:
            break
        order = list(range(n_rows))
        rng.shuffle(order)
        rows = _greedy_build(n_rows, k, order, rng)
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = rows

    # Phase 2: Backtracking to find a better solution
    if _time_left() > 30.0 and best_max > 0:
        bt_rows, bt_max = _backtracking(n_rows, k, best_max - 1, rng, time_limit=25.0)
        if bt_rows is not None and bt_max < best_max:
            best_max = bt_max
            best_rows = bt_rows

    # Phase 3: Iterated local search refinement
    if _time_left() > 15.0:
        ils_budget = min(50.0, _time_left() - 10.0)
        ils_rows, ils_max = _iterated_local_search(best_rows, n_rows, k, rng, ils_budget)
        if ils_max < best_max:
            best_max = ils_max
            best_rows = ils_rows

    # Final verification
    if not _validate(best_rows, n_rows, k):
        best_rows = _greedy_build(n_rows, k, list(range(n_rows)), rng)
        best_max = max(max(r) for r in best_rows)

    return best_rows
