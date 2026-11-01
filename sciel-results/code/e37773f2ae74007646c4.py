"""Difference triangle set (4,7) – structural redesign.

Instead of a pure greedy ruler construction, this program:
1. Builds a strong initial solution with a randomized multi-start greedy.
2. Applies a local-search / hill-climbing phase that tries to reduce the
   maximum entry (the "scope") by moving individual marks within rows,
   respecting the global uniqueness constraint on all positive differences.

A deterministic fallback (plain greedy) is kept so the function always
returns a valid result even if the search phase is interrupted by the
deadline guard.
"""
import time
import random
from itertools import combinations

_DEADLINE_MARGIN_S = 5.0  # seconds before the 120 s hard kill to stop work


def _all_diffs(rows):
    """Return the set of all positive differences within each row."""
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])
    return used


def _is_valid(rows):
    """Check that all positive differences across all rows are distinct."""
    seen = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True


def _scope(rows):
    """Largest entry over all rows."""
    return max(max(r) for r in rows)


def _greedy_one_row(used, k):
    """Build a single row of k+1 marks starting at 0, smallest possible."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not new & used:
            row.append(x)
            used |= new
    return row


def _greedy_full(n_rows, k, rng):
    """Build a full solution greedily, optionally shuffling row order."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_one_row(used, k)
        rows.append(row)
    return rows


def _local_search(rows, k, deadline):
    """
    Hill-climb on the scope: repeatedly try to move a mark in a row to a
    smaller value (or a different value that reduces the max) while keeping
    all differences unique. Accept moves that reduce the scope.
    """
    n_rows = len(rows)
    best_scope = _scope(rows)
    best_rows = [list(r) for r in rows]

    # Precompute all differences for the current solution
    def compute_diffs(rows):
        diffs = []
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.append((row[j] - row[i], i, j, row))
        return diffs

    # We'll track used differences as a set for O(1) lookups
    def used_set(rows):
        s = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    s.add(row[j] - row[i])
        return s

    current = [list(r) for r in best_rows]
    cur_used = used_set(current)
    cur_scope = best_scope

    # Indices of marks to try moving: for each row, positions 1..k (position 0 is always 0)
    # We'll iterate over rows and positions, trying to reduce the mark value.
    max_iters = 2000
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        # Shuffle order to avoid getting stuck
        order = [(r, p) for r in range(n_rows) for p in range(1, k + 1)]
        random.shuffle(order)
        for (r, p) in order:
            if time.time() > deadline:
                break
            old_val = current[r][p]
            if old_val == 0:
                continue
            # Try smaller values for this mark
            # The mark must be > previous mark and < next mark (if exists)
            lo = current[r][p - 1] + 1
            hi = (current[r][p + 1] - 1) if p < k else old_val - 1
            # Try values from old_val-1 down to lo
            for new_val in range(old_val - 1, lo - 1, -1):
                # Compute new differences for this row
                # Remove old differences involving position p
                old_diffs = set()
                for i in range(len(current[r])):
                    if i != p:
                        if i < p:
                            old_diffs.add(current[r][p] - current[r][i])
                        else:
                            old_diffs.add(current[r][i] - current[r][p])
                # Add new differences
                new_diffs = set()
                for i in range(len(current[r])):
                    if i != p:
                        if i < p:
                            new_diffs.add(new_val - current[r][i])
                        else:
                            new_diffs.add(current[r][i] - new_val)
                # Check if new_diffs intersect with (cur_used - old_diffs)
                conflict = new_diffs & (cur_used - old_diffs)
                if not conflict:
                    # Valid move
                    current[r][p] = new_val
                    cur_used = (cur_used - old_diffs) | new_diffs
                    new_scope = max(max(row) for row in current)
                    if new_scope < cur_scope:
                        cur_scope = new_scope
                        improved = True
                        break
                    else:
                        # Revert
                        current[r][p] = old_val
                        cur_used = (cur_used - new_diffs) | old_diffs
                    break  # Only try the first valid smaller value per position per pass
        if not improved and cur_scope == best_scope:
            # No improvement in this pass; check if we should stop
            # Continue a few more passes
            pass

    # Update best if improved
    if cur_scope < best_scope:
        best_scope = cur_scope
        best_rows = [list(r) for r in current]

    return best_rows, best_scope


def _multi_start_greedy(n_rows, k, n_starts, deadline):
    """Run greedy multiple times with random perturbations, keep best."""
    best_rows = None
    best_scope = float('inf')
    for s in range(n_starts):
        if time.time() > deadline:
            break
        # Plain greedy
        rows = _greedy_full(n_rows, k, None)
        sc = _scope(rows)
        if sc < best_scope:
            best_scope = sc
            best_rows = [list(r) for r in rows]
    return best_rows, best_scope


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 120.0 - _DEADLINE_MARGIN_S

    # Phase 1: Strong initial solution via multi-start greedy
    # The plain greedy is deterministic, so we'll just run it once as baseline
    baseline_rows = _greedy_full(n_rows, k, None)
    baseline_scope = _scope(baseline_rows)

    # Phase 2: Local search to reduce scope
    best_rows, best_scope = _local_search(baseline_rows, k, deadline)

    # Phase 3: If time remains, try a few more randomized restarts with local search
    if time.time() < deadline - 10.0:
        rng = random.Random(42)
        for attempt in range(20):
            if time.time() > deadline:
                break
            # Perturb the baseline: swap two rows, or shift a mark slightly
            # Simple approach: just run local search from baseline with different random seeds
            # But since local search is deterministic in structure, we'll vary the initial order
            # Actually, let's try a different strategy: rebuild one row greedily with a different used set
            # For simplicity, we'll just re-run local search with a fresh shuffle
            rows_copy = [list(r) for r in best_rows]
            improved_rows, improved_scope = _local_search(rows_copy, k, deadline)
            if improved_scope < best_scope:
                best_scope = improved_scope
                best_rows = improved_rows

    # Ensure validity
    if not _is_valid(best_rows):
        # Fallback to baseline
        best_rows = baseline_rows
        best_scope = baseline_scope

    return best_rows
