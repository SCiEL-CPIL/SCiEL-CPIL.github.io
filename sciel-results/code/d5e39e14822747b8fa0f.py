"""Difference triangle set (4,7): construct rows with distinct global differences.

New approach: precompute all difference-sets (sets of 36 distinct differences that
arise from 9 increasing values starting at 0), then pack 4 disjoint such sets.
"""
import time
import random
import sys


def construct(n_rows, k):
    deadline = time.time() + 145.0  # safety margin before 150s kill

    # Total differences per row
    diffs_per_row = (k + 1) * k // 2  # 36 for k=7
    total_diffs = n_rows * diffs_per_row  # 144

    # Lower bound on scope
    lb = total_diffs

    # Greedy fallback
    def greedy_construct():
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

    best_rows = greedy_construct()
    best_scope = max(max(r) for r in best_rows)

    # Precompute all candidate difference-sets up to a limit
    # A difference-set is a set of 36 distinct positive integers
    # that equals {a_j - a_i : i < j} for some increasing 0 <= a_0 < ... < a_8
    # We'll enumerate by trying to build the row values directly and collect the diff set.

    def enumerate_diff_sets(max_val):
        """Enumerate all difference-sets from rows of k+1 values in [0, max_val]."""
        seen = set()
        sets_list = []
        # We'll use a recursive builder that tracks the diff set
        def rec(row, diff_set, idx):
            if idx == k + 1:
                key = frozenset(diff_set)
                if key not in seen:
                    seen.add(key)
                    sets_list.append((list(diff_set), list(row)))
                return
            # Pruning: the diff set must have exactly diffs_per_row elements
            if len(diff_set) > diffs_per_row:
                return
            # Minimum next value
            min_next = (row[-1] if row else 0) + 1
            # Maximum next value: must leave room for remaining diffs
            # Each new value adds (len(row)) new diffs
            remaining_vals = k + 1 - idx
            remaining_diffs = diffs_per_row - len(diff_set)
            # Each of the remaining_vals new values adds at least 1 new diff
            # but the last value adds len(row) diffs
            # Simple bound: max_val
            for v in range(min_next, max_val + 1):
                new_diffs = {v - a for a in row}
                if len(new_diffs) != len(row):
                    # Duplicate difference within this row
                    continue
                if new_diffs & diff_set:
                    continue
                new_diff_set = diff_set | new_diffs
                # Check we can still reach diffs_per_row
                # Remaining positions: remaining_vals - 1 more values
                # Each adds at least 1 diff, so we need remaining_diffs >= remaining_vals - 1
                if diffs_per_row - len(new_diff_set) < remaining_vals - 1:
                    continue
                row.append(v)
                rec(row, new_diff_set, idx + 1)
                row.pop()

        rec([], set(), 0)
        return sets_list

    # Try to enumerate with a reasonable bound
    # The scope is at most best_scope, so values are at most best_scope
    # But enumerating all rows up to best_scope is too large.
    # Instead, we'll use a smarter approach: build rows incrementally and pack.

    # Alternative: use backtracking to build 4 rows with disjoint diff sets.
    # We'll build one row at a time, choosing values that don't conflict.

    def backtrack_pack():
        """Build n_rows rows with disjoint difference sets, minimizing max value."""
        used = set()
        rows = []

        def rec(row_idx, row, diff_set):
            if row_idx == n_rows:
                scope = max(max(r) for r in rows)
                return scope
            if len(row) == k + 1:
                rows.append(row)
                scope = rec(row_idx + 1, [], set())
                rows.pop()
                return scope
            # Choose next value
            min_next = (row[-1] if row else 0) + 1
            # Try values in increasing order
            for v in range(min_next, best_scope + 1):
                new_diffs = {v - a for a in row}
                if len(new_diffs) != len(row):
                    continue
                if new_diffs & used or new_diffs & diff_set:
                    continue
                # Pruning: check remaining capacity
                remaining_vals = k + 1 - (len(row) + 1)
                remaining_diffs = diffs_per_row - (len(diff_set) + len(new_diffs))
                if remaining_diffs < remaining_vals:
                    continue
                row.append(v)
                new_diff_set = diff_set | new_diffs
                scope = rec(row_idx, row, new_diff_set)
                if scope < best_scope:
                    return scope
                row.pop()
            return float('inf')

        # This is too slow. Let's use a different strategy.
        return float('inf')

    # Use a randomized local search: start from greedy, then try to reduce max value
    # by moving individual entries.

    def local_search(rows, max_iters=50000):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])

        current_scope = max(max(r) for r in rows)
        best_scope_ls = current_scope
        best_rows_ls = [list(r) for r in rows]

        for it in range(max_iters):
            if time.time() > deadline:
                break
            # Pick a random row and a random position to try moving
            ridx = random.randrange(n_rows)
            pos = random.randrange(1, k + 1)  # positions 1..k (not 0)
            old_val = rows[ridx][pos]

            # Try to move this value to a smaller value
            # The value must be > previous value and < next value
            min_val = rows[ridx][pos - 1] + 1
            max_val = rows[ridx][pos + 1] - 1 if pos < k else current_scope

            if max_val < min_val:
                continue

            # Try values in a random order, biased towards smaller
            candidates = list(range(min_val, max_val + 1))
            random.shuffle(candidates)
            candidates.sort(key=lambda x: random.random() ** 0.3 * (x - min_val))

            # Remove old diffs involving this position
            old_diffs = set()
            for j in range(len(rows[ridx])):
                if j != pos:
                    d = abs(rows[ridx][pos] - rows[ridx][j])
                    old_diffs.add(d)

            # Remove from used
            for d in old_diffs:
                used.discard(d)

            placed = False
            for cand in candidates:
                if cand == old_val:
                    continue
                new_diffs = set()
                for j in range(len(rows[ridx])):
                    if j != pos:
                        d = abs(cand - rows[ridx][j])
                        new_diffs.add(d)
                if new_diffs & used:
                    continue
                # Check no duplicate within row
                if len(new_diffs) != len(rows[ridx]) - 1:
                    continue
                # Accept
                rows[ridx][pos] = cand
                used |= new_diffs
                new_scope = max(max(r) for r in rows)
                if new_scope < best_scope_ls:
                    best_scope_ls = new_scope
                    best_rows_ls = [list(r) for r in rows]
                placed = True
                break

            if not placed:
                # Restore
                rows[ridx][pos] = old_val
                used |= old_diffs

        return best_rows_ls, best_scope_ls

    # Run local search from greedy
    ls_rows, ls_scope = local_search(best_rows, max_iters=100000)
    if ls_scope < best_scope:
        best_scope = ls_scope
        best_rows = ls_rows

    # Try multiple random restarts with different greedy orders
    for restart in range(50):
        if time.time() > deadline:
            break
        random.seed(restart * 12345 + 67890)
        # Randomized greedy
        used = set()
        rows = []
        success = True
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                if x > best_scope + 50:
                    success = False
                    break
            if not success:
                break
            rows.append(row)
        if success:
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows

            # Local search
            ls_rows, ls_scope = local_search(best_rows, max_iters=20000)
            if ls_scope < best_scope:
                best_scope = ls_scope
                best_rows = ls_rows

    return best_rows


if __name__ == "__main__":
    rows = construct(4, 7)
    for r in rows:
        print(r)
    print("Scope:", max(max(r) for r in rows))
