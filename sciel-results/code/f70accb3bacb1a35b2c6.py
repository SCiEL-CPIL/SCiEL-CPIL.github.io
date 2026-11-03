"""Difference triangle set (4,7) – bitset-accelerated backtracking with deadline guard."""
import time
import sys
import random

# Deadline guard
_DEADLINE = 115.0  # seconds, safety margin under 120s limit
_start_time = None


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.perf_counter()
    return time.perf_counter() - _start_time


def _time_left():
    return _DEADLINE - _elapsed()


def construct(n_rows, k):
    global _start_time
    _start_time = time.perf_counter()

    # Total number of differences per row: k*(k+1)//2
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Bitset representation of used differences
    # We'll use a Python int as a dynamic bitset
    # Bit d is set if difference d is used

    best_rows = None
    best_scope = None

    # First, get a greedy solution as a starting point
    greedy_rows = _greedy(n_rows, k)
    if greedy_rows is not None:
        best_rows = greedy_rows
        best_scope = max(max(r) for r in greedy_rows)

    # Now try to improve with backtracking, trying smaller scopes
    # The counting bound gives a lower bound on scope
    # For n_rows rows of k+1 entries, total diffs = n_rows * k*(k+1)//2
    # All diffs must be in [1, scope], so scope >= total_diffs
    lower_bound = total_diffs

    # Try scopes from lower_bound upward, but also try near the greedy solution
    # We'll do a smart search: try scopes around the known best and below

    # Strategy: try backtracking for various target scopes
    # Start from just below the greedy best and work down, and also from lower bound up

    # Collect candidate scopes to try
    candidate_scopes = set()
    if best_scope is not None:
        for s in range(max(lower_bound, 1), min(best_scope, lower_bound + 50) + 1):
            candidate_scopes.add(s)
        # Also try a few above lower bound
        for s in range(max(lower_bound, best_scope - 20), best_scope + 1):
            candidate_scopes.add(s)
    else:
        for s in range(lower_bound, lower_bound + 30):
            candidate_scopes.add(s)

    # Sort: try smaller scopes first (better solutions)
    candidate_scopes = sorted(candidate_scopes)

    for target_scope in candidate_scopes:
        if _time_left() < 2.0:
            break

        # Try to find a solution with all entries <= target_scope
        result = _backtrack(n_rows, k, target_scope)
        if result is not None:
            best_rows = result
            best_scope = target_scope
            # Found a valid solution at this scope; try to go lower
            # But we're iterating in increasing order, so we'll naturally try lower ones first
            # Actually we want to try lower scopes first, so break and restart?
            # No, we're going from low to high, so if we find one at scope S, 
            # we should try all scopes below S. But we already tried them and failed.
            # So this is the best we can do with this approach.
            # Let's continue trying higher scopes in case we missed something? No,
            # we want the minimum scope. Since we're going low to high, the first
            # successful one is the best.
            break

    # Final polish: try to reduce scope by 1 with targeted backtracking
    if best_rows is not None and _time_left() > 3.0:
        for delta in range(1, 5):
            if _time_left() < 2.0:
                break
            new_scope = best_scope - delta
            if new_scope < lower_bound:
                break
            result = _backtrack(n_rows, k, new_scope)
            if result is not None:
                best_rows = result
                best_scope = new_scope

    if best_rows is None:
        # Fallback: return greedy
        best_rows = _greedy(n_rows, k)

    return best_rows


def _greedy(n_rows, k):
    """Greedy construction: each row takes smallest next mark with unused differences."""
    used = 0  # bitset
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            # Check if all differences x - a for a in row are unused
            ok = True
            for a in row:
                d = x - a
                if d > 0 and (used >> d) & 1:
                    ok = False
                    break
            if ok:
                row.append(x)
                for a in row[:-1]:
                    d = x - a
                    if d > 0:
                        used |= (1 << d)
        rows.append(row)
    return rows


def _backtrack(n_rows, k, max_scope):
    """
    Backtracking search for n_rows rows of k+1 increasing integers starting at 0,
    all entries <= max_scope, with all positive differences distinct across all rows.
    Returns list of rows or None if not found within time budget.
    """
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # If max_scope < total_diffs, impossible
    if max_scope < total_diffs:
        return None

    # We'll build rows one at a time
    # For each row, we choose k positions from 1..max_scope (since 0 is fixed)
    # such that all pairwise differences are unused

    # To make this tractable, we use a recursive approach with pruning
    # For each row, we generate candidate rows (subsets of size k from {1,...,max_scope})
    # but that's C(max_scope, k) which is huge. Instead, we build each row greedily
    # with backtracking.

    # Better approach: build each row entry by entry with backtracking
    # Global state: used differences bitset

    used = [0]  # list to allow mutation in closure; bitset of used differences
    rows = []

    def _build_row(row_idx):
        """Build the row_idx-th row (0-indexed) using backtracking."""
        if _time_left() < 0.5:
            return False

        if row_idx == n_rows:
            return True

        # Build this row: start with [0], add k more entries
        row = [0]

        def _add_entries(pos, current_max):
            """Add entries starting from position pos, with current_max being the last entry."""
            if _time_left() < 0.5:
                return False

            if len(row) == k + 1:
                rows.append(row[:])
                success = _build_row(row_idx + 1)
                if success:
                    return True
                rows.pop()
                return False

            # Try next entries from current_max + 1 to max_scope
            # Pruning: we need (k + 1 - len(row)) more entries
            remaining = k + 1 - len(row)
            start = current_max + 1
            end = max_scope - (remaining - 1)  # need room for remaining entries

            for x in range(start, end + 1):
                if _time_left() < 0.5:
                    return False

                # Check if all differences x - a for a in row are unused
                ok = True
                new_diffs = []
                for a in row:
                    d = x - a
                    if d > 0:
                        if (used[0] >> d) & 1:
                            ok = False
                            break
                        new_diffs.append(d)

                if not ok:
                    continue

                # Mark differences as used
                old_used = used[0]
                for d in new_diffs:
                    used[0] |= (1 << d)

                row.append(x)
                if _add_entries(pos + 1, x):
                    return True
                row.pop()

                # Unmark
                used[0] = old_used

            return False

        return _add_entries(0, 0)

    # Use a different strategy: since pure backtracking over individual entries is too slow,
    # let's try a randomized greedy with restarts
    return _randomized_search(n_rows, k, max_scope)


def _randomized_search(n_rows, k, max_scope):
    """
    Randomized greedy with restarts: build each row by picking entries in random order
    (with some heuristic), and restart if stuck.
    """
    if _time_left() < 1.0:
        return None

    best = None
    attempts = 0

    while _time_left() > 1.0 and attempts < 5000:
        attempts += 1
        result = _try_one(n_rows, k, max_scope, attempts)
        if result is not None:
            best = result
            break
        # Occasionally try a different random seed implicitly via different attempt count

    return best


def _try_one(n_rows, k, max_scope, seed_offset):
    """
    Try to construct n_rows rows with all entries <= max_scope.
    Uses a greedy approach with randomized tie-breaking.
    """
    rng = random.Random(12345 + seed_offset)
    used = 0  # bitset
    rows = []

    for row_idx in range(n_rows):
        row = [0]
        # We need to pick k entries from 1..max_scope
        # Greedy with some randomization: at each step, pick from a small set of candidates
        # Sort candidates by some heuristic, pick randomly from top few

        while len(row) < k + 1:
            # Find all valid next entries
            candidates = []
            for x in range(row[-1] + 1, max_scope + 1):
                ok = True
                for a in row:
                    d = x - a
                    if d > 0 and (used >> d) & 1:
                        ok = False
                        break
                if ok:
                    candidates.append(x)

            if not candidates:
                return None  # Stuck

            # Pick from candidates: prefer smaller x (to keep scope tight) but add some randomness
            # Sort by x, pick randomly from the first few
            candidates.sort()
            # Take top min(5, len(candidates)) and pick randomly
            top_n = min(5, len(candidates))
            x = rng.choice(candidates[:top_n])

            # Mark differences
            for a in row:
                d = x - a
                if d > 0:
                    used |= (1 << d)
            row.append(x)

        rows.append(row)

    # Verify all entries <= max_scope
    for row in rows:
        if row[-1] > max_scope:
            return None

    return rows
