"""Difference triangle set: minimize the largest entry (scope)."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences within rows are globally distinct.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit

    # Total number of differences per row: k*(k+1)/2
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Lower bound on scope: need at least total_diffs distinct positive differences,
    # each <= scope, so scope >= total_diffs. Also scope >= k (trivial).
    # A better lower bound: the sum of the smallest total_diffs positive ints is
    # total_diffs*(total_diffs+1)/2, but that's not directly helpful for max.
    # Actually, we just need total_diffs distinct values in [1, scope], so scope >= total_diffs.
    lower_bound = total_diffs

    # Start search from a good initial upper bound
    # Use greedy to get an initial solution
    best_rows = _greedy_construct(n_rows, k)
    best_scope = max(max(row) for row in best_rows)

    # Try to improve via randomized local search / backtracking
    # Strategy: for a target scope S, try to find a valid assignment.
    # We'll try decreasing S from best_scope down to lower_bound.

    # First, let's try a smarter greedy that's aware of a target scope
    # Then use local search to polish

    # Try local search with scope reduction
    current_rows = [list(r) for r in best_rows]
    current_scope = best_scope

    # Local search: try to reduce scope by swapping/moving entries
    while time.time() < deadline:
        improved = False
        for _ in range(200):  # iterations per scope attempt
            if time.time() > deadline:
                break
            # Try to find a smaller valid configuration
            new_rows, new_scope = _local_search_step(current_rows, k, current_scope)
            if new_rows is not None and new_scope < current_scope:
                current_rows = new_rows
                current_scope = new_scope
                improved = True
                if current_scope <= lower_bound:
                    return current_rows
                break
        if not improved:
            # Try a randomized restart
            if time.time() > deadline:
                break
            # Randomize and try greedy with target scope
            target = current_scope - 1
            if target < lower_bound:
                break
            candidate = _greedy_with_target(n_rows, k, target)
            if candidate is not None:
                cs = max(max(row) for row in candidate)
                if cs < current_scope:
                    current_rows = candidate
                    current_scope = cs
                    improved = True
                    if current_scope <= lower_bound:
                        return current_rows
            if not improved and time.time() < deadline:
                # Try backtracking for a specific row
                candidate = _backtrack_improve(current_rows, k, current_scope - 1)
                if candidate is not None:
                    cs = max(max(row) for row in candidate)
                    if cs < current_scope:
                        current_rows = candidate
                        current_scope = cs
                        if current_scope <= lower_bound:
                            return current_rows

    return current_rows


def _greedy_construct(n_rows, k):
    """Standard greedy: each row takes smallest next mark with unused diffs."""
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


def _greedy_with_target(n_rows, k, target_scope):
    """Greedy construction constrained to stay within target_scope."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if x > target_scope:
                return None
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _local_search_step(rows, k, current_scope):
    """
    Try to reduce scope by one via a local move.
    Returns (new_rows, new_scope) or (None, None).
    """
    n_rows = len(rows)
    # Collect all used differences
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])

    # Try to find a row and position where we can reduce the max element
    # Strategy: for each row, try to replace the last element with a smaller one
    # that still gives valid diffs, then re-greedy the rest if needed.
    
    # Simpler: try swapping the largest element in each row with a smaller candidate
    for r_idx in range(n_rows):
        row = rows[r_idx]
        if len(row) < 2:
            continue
        last_val = row[-1]
        # Try smaller values for the last position
        for new_val in range(last_val - 1, 0, -1):
            if new_val <= row[-2]:
                break
            # Check if replacing row[-1] with new_val is valid
            test_row = row[:-1] + [new_val]
            # Compute diffs for this row
            row_diffs = set()
            valid = True
            for i in range(len(test_row)):
                for j in range(i + 1, len(test_row)):
                    d = test_row[j] - test_row[i]
                    if d in row_diffs:
                        valid = False
                        break
                    row_diffs.add(d)
                if not valid:
                    break
            if not valid:
                continue
            # Check against other rows' diffs
            other_diffs = set()
            for o_idx in range(n_rows):
                if o_idx == r_idx:
                    continue
                orow = rows[o_idx]
                for i in range(len(orow)):
                    for j in range(i + 1, len(orow)):
                        other_diffs.add(orow[j] - orow[i])
            if not row_diffs & other_diffs:
                # Valid! Create new rows
                new_rows = [list(r) for r in rows]
                new_rows[r_idx] = test_row
                new_scope = max(max(r) for r in new_rows)
                if new_scope < current_scope:
                    return new_rows, new_scope
    return None, None


def _backtrack_improve(rows, k, target_scope):
    """
    Try to reconstruct one row with a smaller max, keeping others fixed.
    Returns new rows or None.
    """
    n_rows = len(rows)
    # Collect diffs from all other rows
    other_diffs = set()
    for r_idx in range(n_rows):
        row = rows[r_idx]
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                other_diffs.add(row[j] - row[i])

    # For each row, try to find an alternative with max <= target_scope
    for r_idx in range(n_rows):
        old_row = rows[r_idx]
        old_max = max(old_row)
        if old_max <= target_scope:
            continue
        # Try to find a new row with max <= target_scope
        new_row = _backtrack_row(k, target_scope, other_diffs)
        if new_row is not None:
            new_rows = [list(r) for r in rows]
            new_rows[r_idx] = new_row
            return new_rows
    return None


def _backtrack_row(k, max_val, forbidden_diffs):
    """
    Find a row of k+1 increasing ints starting at 0, max <= max_val,
    with no diff in forbidden_diffs and no internal duplicate diffs.
    """
    row = [0]
    used_diffs = set()

    def backtrack(pos, last_val):
        if pos == k + 1:
            return True
        for v in range(last_val + 1, max_val + 1):
            new_diffs = {v - a for a in row}
            if new_diffs & used_diffs or new_diffs & forbidden_diffs:
                continue
            row.append(v)
            used_diffs |= new_diffs
            if backtrack(pos + 1, v):
                return True
            row.pop()
            used_diffs -= new_diffs
        return False

    if backtrack(1, 0):
        return list(row)
    return None

# EVOLVE-BLOCK-END
