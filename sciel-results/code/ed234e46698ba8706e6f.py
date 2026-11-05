"""Difference triangle set (7,5): construct rows of k+1 increasing ints starting at 0
with all pairwise differences distinct across all rows. Minimize the maximum entry."""
import time
import random
from itertools import combinations

# Deadline: keep a safety margin well under 150s wall-clock.
DEADLINE_SECONDS = 135.0


def _validate(rows):
    """Return True if rows are valid: each row starts at 0, strictly increasing,
    and all positive differences across all rows are distinct."""
    used = set()
    for row in rows:
        if not row or row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _max_entry(rows):
    return max(max(row) for row in rows)


def _greedy_row(used, k):
    """Build one row greedily: start at 0, pick smallest next x such that
    all new differences x-a for a in row are unused."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new_diffs = {x - a for a in row}
        if not (new_diffs & used):
            row.append(x)
            used |= new_diffs
    return row


def _greedy_full(n_rows, k):
    """Run the original greedy to get a baseline set of rows."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        rows.append(row)
    return rows


def _row_diffs(row):
    """Return set of all positive differences within a row."""
    return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}


def _all_diffs(rows):
    used = set()
    for row in rows:
        used |= _row_diffs(row)
    return used


def _local_search(rows, k, time_budget=20.0):
    """Try to reduce the max entry by mutating rows (remove a mark, re-greedy the tail)
    and accepting if valid and max_entry doesn't increase; occasionally accept equal
    to escape plateaus. Uses simulated-annealing-like acceptance."""
    best_rows = [list(r) for r in rows]
    best_max = _max_entry(best_rows)
    start = time.time()
    rng = random.Random(12345)

    n = len(rows)
    current_max = best_max

    # We'll try to improve by:
    # 1. Picking a random row
    # 2. Picking a random position to "truncate" (keep prefix, discard rest)
    # 3. Re-greedy the tail with the current global used set (minus this row's diffs)
    # 4. Check validity; if valid and max_entry < current_max, accept
    #    If valid and max_entry == current_max, accept with some probability

    for _ in range(500000):
        if time.time() - start > time_budget:
            break

        idx = rng.randrange(n)
        row = rows[idx]
        # Choose a truncation point: keep at least 1 element (the 0)
        # and at most len(row)-1 so we re-add at least one
        if len(row) <= 2:
            continue
        trunc = rng.randrange(1, len(row))  # keep row[:trunc]
        prefix = row[:trunc]

        # Compute used set without this row's contributions
        other_used = set()
        for i, r in enumerate(rows):
            if i != idx:
                other_used |= _row_diffs(r)

        # Re-greedy from prefix
        new_row = list(prefix)
        x = prefix[-1]
        while len(new_row) < k + 1:
            x += 1
            new_diffs = {x - a for a in new_row}
            if not (new_diffs & other_used):
                new_row.append(x)
                other_used |= new_diffs

        # Check full validity with this new row
        all_rows = list(rows)
        all_rows[idx] = new_row
        if not _validate(all_rows):
            continue

        new_max = _max_entry(all_rows)
        if new_max < current_max:
            rows = all_rows
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_rows = [list(r) for r in rows]
        elif new_max == current_max and rng.random() < 0.1:
            # Accept to explore
            rows = all_rows

    return best_rows, best_max


def _random_restart_greedy(n_rows, k, rng, max_scope_hint):
    """Try a randomized greedy: at each step, pick among several candidate x values
    weighted to prefer smaller ones, but with some randomness to escape bad choices."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            # Try a window of candidates
            found = False
            for attempt in range(10):
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
                    found = True
                    break
            if not found:
                return None  # failed
        rows.append(row)
    return rows


def _hill_climb_single_row(rows, k, time_budget=10.0):
    """For each row, try replacing it with a different valid row that has a smaller
    max entry, while keeping other rows fixed. Iterate until no improvement."""
    start = time.time()
    improved = True
    while improved and time.time() - start < time_budget:
        improved = False
        for idx in range(len(rows)):
            row = rows[idx]
            old_max = max(row)
            # Compute used set of other rows
            other_used = set()
            for i, r in enumerate(rows):
                if i != idx:
                    other_used |= _row_diffs(r)

            # Try to build a new row with max < old_max
            target_max = old_max - 1
            if target_max < k:  # need at least k+1 distinct values including 0
                continue

            # Greedy build with constraint that all entries <= target_max
            new_row = [0]
            x = 0
            success = True
            while len(new_row) < k + 1:
                found = False
                while x < target_max:
                    x += 1
                    new_diffs = {x - a for a in new_row}
                    if not (new_diffs & other_used):
                        new_row.append(x)
                        other_used |= new_diffs
                        found = True
                        break
                if not found:
                    success = False
                    break
            if success and max(new_row) < old_max:
                # Full validation
                all_rows = list(rows)
                all_rows[idx] = new_row
                if _validate(all_rows):
                    rows = all_rows
                    improved = True

    return rows


def construct(n_rows, k):
    start_time = time.time()

    # Phase 1: Greedy baseline
    rows = _greedy_full(n_rows, k)
    best_rows = [list(r) for r in rows]
    best_max = _max_entry(best_rows)

    # Phase 2: Hill-climb single rows to reduce max
    remaining = DEADLINE_SECONDS - (time.time() - start_time) - 30.0
    if remaining > 5.0:
        rows = _hill_climb_single_row(rows, k, time_budget=min(remaining * 0.3, 20.0))
        cur_max = _max_entry(rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [list(r) for r in rows]

    # Phase 3: Local search with truncation + re-greedy
    remaining = DEADLINE_SECONDS - (time.time() - start_time) - 15.0
    if remaining > 5.0:
        rows2, max2 = _local_search(rows, k, time_budget=min(remaining * 0.4, 30.0))
        if max2 < best_max:
            best_max = max2
            best_rows = rows2

    # Phase 4: Multiple random restarts with greedy, keep best
    remaining = DEADLINE_SECONDS - (time.time() - start_time) - 10.0
    if remaining > 5.0:
        rng = random.Random(42)
        restart_budget = remaining * 0.5
        restart_start = time.time()
        for _ in range(200):
            if time.time() - restart_start > restart_budget:
                break
            rows_r = _random_restart_greedy(n_rows, k, rng, best_max)
            if rows_r is None:
                continue
            rmax = _max_entry(rows_r)
            if rmax < best_max:
                best_max = rmax
                best_rows = [list(r) for r in rows_r]
                # Quick hill-climb on the new best
                remaining2 = DEADLINE_SECONDS - (time.time() - start_time) - 10.0
                if remaining2 > 3.0:
                    improved = _hill_climb_single_row(best_rows, k, time_budget=min(remaining2 * 0.2, 8.0))
                    imax = _max_entry(improved)
                    if imax < best_max:
                        best_max = imax
                        best_rows = [list(r) for r in improved]

    # Final safety check
    if not _validate(best_rows):
        # Fallback to pure greedy
        best_rows = _greedy_full(n_rows, k)

    return best_rows
