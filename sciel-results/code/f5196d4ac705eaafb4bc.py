"""Rewritten program for difference triangle set (4,7) task."""
import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0.
    All positive differences within rows must be distinct across all rows.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin before 150s limit

    # First, get a baseline greedy solution
    baseline = greedy_construct(n_rows, k)
    baseline_scope = max(max(row) for row in baseline)

    best_rows = baseline
    best_scope = baseline_scope

    # Try randomized local search / backtracking with time limit
    # We'll use a randomized greedy approach with multiple restarts
    # and local improvement

    random.seed(42)

    attempts = 0
    while time.time() < deadline and attempts < 500:
        attempts += 1
        rows = randomized_greedy(n_rows, k, random_state=random)
        scope = max(max(row) for row in rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = rows
            # Try to improve further with local search
            if time.time() < deadline:
                improved = local_improve(best_rows, n_rows, k, time.time(), deadline)
                if improved is not None:
                    imp_scope = max(max(row) for row in improved)
                    if imp_scope < best_scope:
                        best_scope = imp_scope
                        best_rows = improved

    return best_rows


def greedy_construct(n_rows, k):
    """Standard greedy: pick smallest next mark whose differences are unused."""
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


def randomized_greedy(n_rows, k, random_state=None):
    """
    Randomized greedy: at each step, consider candidate positions in a range,
    randomly permute them, and pick the first valid one.
    """
    if random_state is None:
        random_state = random

    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        # Upper bound for search: start from a reasonable estimate
        # The greedy gives some scope; use that as upper bound
        while len(row) < k + 1:
            # Look ahead a window of candidates
            window_start = x + 1
            window_size = max(1, int(5 + 3 * len(row)))
            candidates = list(range(window_start, window_start + window_size))
            random_state.shuffle(candidates)
            found = False
            for cand in candidates:
                new_diffs = {cand - a for a in row}
                if not new_diffs & used:
                    row.append(cand)
                    used |= new_diffs
                    x = cand
                    found = True
                    break
            if not found:
                # Extend window
                x = window_start + window_size - 1
    return rows


def local_improve(rows, n_rows, k, start_time, deadline):
    """
    Try to reduce the scope by swapping or adjusting the largest element(s).
    Simple approach: try to replace the last element of each row with a smaller
    valid one, iterating.
    """
    if time.time() > deadline:
        return None

    rows = [list(r) for r in rows]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_rows):
            # Try to reduce the last element
            old_last = rows[i][-1]
            # Try smaller values
            for new_last in range(old_last - 1, 0, -1):
                # Check if replacing last element keeps all diffs unique
                # Remove old diffs involving old_last
                old_diffs = {old_last - a for a in rows[i][:-1]}
                # Compute all other diffs
                other_diffs = set()
                for j in range(n_rows):
                    row_j = rows[j]
                    for a in range(len(row_j)):
                        for b in range(a + 1, len(row_j)):
                            d = row_j[b] - row_j[a]
                            if not (j == i and b == len(row_j) - 1):
                                other_diffs.add(d)
                # New diffs with new_last
                new_diffs = {new_last - a for a in rows[i][:-1]}
                if new_diffs & other_diffs:
                    continue
                # Also need new_last > rows[i][-2]
                if new_last <= rows[i][-2]:
                    continue
                # Valid replacement
                rows[i][-1] = new_last
                improved = True
                break
    return rows
