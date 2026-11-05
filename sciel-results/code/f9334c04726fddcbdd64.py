"""
Construct difference triangle sets (n_rows, k) minimizing the largest entry.
Combines greedy baseline, bitset-accelerated CSP with constraint propagation,
and local-search refinement.
"""
import time
import random
from collections import defaultdict

_TIME_LIMIT = 148.0  # leave 2s safety margin before 150s kill
_DEADLINE = None
_DEADLINE_SET = False


def _time_left():
    global _DEADLINE, _DEADLINE_SET
    if not _DEADLINE_SET:
        _DEADLINE = time.time() + _TIME_LIMIT
        _DEADLINE_SET = True
    return max(0.0, _DEADLINE - time.time())


def _validate(rows):
    """Check that all positive differences within each row are distinct across all rows."""
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _scope(rows):
    return max(max(r) for r in rows) if rows else 0


def _greedy_construct(n_rows, k, used, rng=None):
    """Build one row greedily, optionally with small randomization."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not new & used:
            row.append(x)
            used |= new
    return row


def _csp_solve(n_rows, k, max_scope, timeout, seed=0):
    """
    CSP solver using bitset difference tracking and randomized backtracking.
    Returns list of rows or None.
    """
    deadline = time.time() + timeout
    used_bits = 0  # bitset of used differences
    rows = []
    rng = random.Random(seed)

    def build_row(current_row, used_bits, deadline):
        if time.time() > deadline:
            return False

        if len(current_row) == k + 1:
            return True

        last = current_row[-1]
        remaining = k + 1 - len(current_row)

        # We need 'remaining' more elements, each at least 1 apart
        # So the last element must be at most max_scope
        # And we need room for 'remaining-1' more elements after x
        min_x = last + 1
        max_x = max_scope - (remaining - 1)

        if min_x > max_x:
            return False

        candidates = []
        for x in range(min_x, max_x + 1):
            # Compute new differences as bitset
            new_diff_bits = 0
            conflict = False
            for a in current_row:
                d = x - a
                if (used_bits >> d) & 1:
                    conflict = True
                    break
                new_diff_bits |= (1 << d)

            if conflict:
                continue

            # Check internal differences don't conflict
            internal_ok = True
            for i in range(len(current_row)):
                for j in range(i + 1, len(current_row)):
                    d = current_row[j] - current_row[i]
                    if (new_diff_bits >> d) & 1:
                        internal_ok = False
                        break
                if not internal_ok:
                    break
            if not internal_ok:
                continue

            candidates.append((x, new_diff_bits))

        if not candidates:
            return False

        # Shuffle to add randomness
        rng.shuffle(candidates)

        for x, new_diff_bits in candidates:
            if time.time() > deadline:
                return False

            current_row.append(x)
            new_used = used_bits | new_diff_bits

            if build_row(current_row, new_used, deadline):
                return True

            current_row.pop()

        return False

    for row_idx in range(n_rows):
        if time.time() > deadline:
            return None

        current_row = [0]
        if build_row(current_row, used_bits, deadline):
            rows.append(current_row)
            # Update used_bits
            for i in range(len(current_row)):
                for j in range(i + 1, len(current_row)):
                    d = current_row[j] - current_row[i]
                    used_bits |= (1 << d)
        else:
            return None

    return rows


def _local_search(rows, n_rows, k, max_iters=50000):
    """
    Local search: try to reduce the largest entry by moving marks.
    We focus on the row with the maximum entry and try to shrink it.
    """
    best_scope = _scope(rows)
    rows = [r[:] for r in rows]
    iters = 0

    # Precompute all differences from other rows for quick lookup
    def get_other_diffs(rows, row_idx):
        other_diffs = set()
        for r2 in rows:
            if id(r2) == id(rows[row_idx]):
                continue
            for i in range(len(r2)):
                for j in range(i + 1, len(r2)):
                    other_diffs.add(r2[j] - r2[i])
        return other_diffs

    while iters < max_iters and _time_left() > 5.0:
        iters += 1
        # Find the row with the maximum entry
        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        row = rows[max_idx]
        current_max = row[-1]

        if current_max <= best_scope:
            break

        # Try to reduce the last element
        improved = False
        other_diffs = get_other_diffs(rows, max_idx)

        # Try moving the last mark down
        for new_last in range(current_max - 1, row[-2], -1):
            if new_last <= row[-2]:
                break
            new_row = row[:-1] + [new_last]
            # Check internal differences
            internal_ok = True
            internal_diffs = set()
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    d = new_row[j] - new_row[i]
                    if d in internal_diffs:
                        internal_ok = False
                        break
                    internal_diffs.add(d)
            if not internal_ok:
                continue
            # Check against other rows
            if internal_diffs & other_diffs:
                continue
            # Valid! Update
            rows[max_idx] = new_row
            new_scope = _scope(rows)
            if new_scope < best_scope:
                best_scope = new_scope
            improved = True
            break

        if not improved:
            # Try reducing other marks in the max row
            found_move = False
            other_diffs = get_other_diffs(rows, max_idx)
            for pos in range(len(row) - 2, 0, -1):
                old_val = row[pos]
                # Try reducing this mark
                lower_bound = row[pos - 1] + 1
                for new_val in range(old_val - 1, lower_bound - 1, -1):
                    if new_val < lower_bound:
                        break
                    new_row = row[:pos] + [new_val] + row[pos + 1:]
                    # Check strictly increasing
                    if any(new_row[i] >= new_row[i + 1] for i in range(len(new_row) - 1)):
                        continue
                    internal_diffs = set()
                    internal_ok = True
                    for i in range(len(new_row)):
                        for j in range(i + 1, len(new_row)):
                            d = new_row[j] - new_row[i]
                            if d in internal_diffs:
                                internal_ok = False
                                break
                            internal_diffs.add(d)
                    if not internal_ok:
                        continue
                    if internal_diffs & other_diffs:
                        continue
                    # Valid
                    rows[max_idx] = new_row
                    new_scope = _scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                    found_move = True
                    break
                if found_move:
                    break
            if not found_move:
                # Try random restart of the max row
                if random.random() < 0.1 and _time_left() > 20.0:
                    # Try rebuilding the max row with a slightly different strategy
                    used_other = get_other_diffs(rows, max_idx)
                    # Try greedy with randomization
                    new_row = [0]
                    x = 0
                    while len(new_row) < k + 1:
                        x += 1
                        if x > best_scope:
                            break
                        new_diffs = {x - a for a in new_row}
                        if not (new_diffs & used_other):
                            # Check internal
                            internal = set()
                            ok = True
                            for i in range(len(new_row)):
                                for j in range(i + 1, len(new_row)):
                                    d = new_row[j] - new_row[i]
                                    if d in internal:
                                        ok = False
                                        break
                                    internal.add(d)
                            if ok and not (new_diffs & internal):
                                new_row.append(x)
                    if len(new_row) == k + 1 and new_row[-1] <= best_scope:
                        rows[max_idx] = new_row
                        new_scope = _scope(rows)
                        if new_scope < best_scope:
                            best_scope = new_scope

    return rows, best_scope


def construct(n_rows, k):
    global _DEADLINE, _DEADLINE_SET
    _DEADLINE = time.time() + _TIME_LIMIT
    _DEADLINE_SET = True

    rng = random.Random(42)

    # Phase 1: Greedy construction
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_construct(n_rows, k, used, rng)
        rows.append(row)

    if not _validate(rows):
        # Fallback: simple greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)

    best_scope = _scope(rows)
    best_rows = [r[:] for r in rows]

    # Phase 2: Try CSP with decreasing max_scope
    if _time_left() > 20.0:
        # Try to find a better solution using CSP
        # Start from best_scope - 1 and go down
        for target_scope in range(best_scope - 1, max(0, best_scope - 30), -1):
            if _time_left() < 5.0:
                break
            # Give more time to larger targets
            timeout = min(10.0, _time_left() - 3.0)
            if timeout < 1.0:
                break
            # Try a few seeds
            for seed in range(3):
                if _time_left() < 3.0:
                    break
                result = _csp_solve(n_rows, k, target_scope, timeout / 3.0, seed=42 + seed)
                if result is not None:
                    if _validate(result):
                        new_scope = _scope(result)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [r[:] for r in result]
                        break

    # Phase 3: Local search
    if _time_left() > 10.0:
        rows_ls, scope_ls = _local_search(best_rows, n_rows, k)
        if scope_ls < best_scope:
            best_scope = scope_ls
            best_rows = rows_ls

    # Phase 4: Additional local search passes
    if _time_left() > 8.0:
        rows_ls2, scope_ls2 = _local_search(best_rows, n_rows, k, max_iters=20000)
        if scope_ls2 < best_scope:
            best_scope = scope_ls2
            best_rows = rows_ls2

    # Final validation
    if not _validate(best_rows):
        # Return the greedy result
        return rows

    return best_rows
