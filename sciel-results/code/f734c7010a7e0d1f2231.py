"""Rewritten program: column-wise list scheduling + local search for difference triangle sets."""
import time
import random
import math

def _deadline_guard(start, margin=15.0, limit=150.0):
    """Return True if we should stop (elapsed + margin >= limit)."""
    return (time.time() - start) + margin >= limit

def _score_of(rows, k):
    """Largest entry across all rows."""
    return max(max(r) for r in rows) if rows else 0

def _is_valid(rows, k):
    """Check that all positive differences within rows are distinct across rows."""
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def construct(n_rows, k):
    start = time.time()
    deadline = start + 145.0  # safety margin before 150s kill

    # ---- Phase 1: Column-wise list scheduling ----
    # Instead of building row by row, we build column by column.
    # For each column j (from 1 to k), assign a value to each row such that
    # the new differences (val - existing values in that row) don't conflict
    # with any previously used difference globally.
    #
    # We process rows in an order that tries to keep the running max small.
    # For each column, we sort rows by their current last element (ascending)
    # and greedily assign the smallest feasible value.

    # used_diffs: set of all differences used so far
    # rows: list of lists, each starting with [0]
    rows = [[0] for _ in range(n_rows)]
    used_diffs = set()

    # Precompute: for each column, we need to pick a value for each row.
    # The value must be > rows[r][-1] and all (val - rows[r][i]) for i in range(len(rows[r]))
    # must not be in used_diffs.

    for col in range(k):  # we're adding column index col+1, so col goes 0..k-1
        if _deadline_guard(start, 10.0, 150.0):
            break

        # Sort row indices by their current last element (ascending) to process
        # "smaller" rows first, which tends to keep values tight.
        row_order = sorted(range(n_rows), key=lambda r: rows[r][-1])

        for r in row_order:
            # Find smallest x > rows[r][-1] such that all (x - rows[r][i]) not in used_diffs
            x = rows[r][-1] + 1
            while True:
                conflict = False
                for i in range(len(rows[r])):
                    d = x - rows[r][i]
                    if d in used_diffs:
                        conflict = True
                        break
                if not conflict:
                    break
                x += 1

            # Commit
            for i in range(len(rows[r])):
                used_diffs.add(x - rows[r][i])
            rows[r].append(x)

    best_score = _score_of(rows, k)
    best_rows = [list(r) for r in rows]

    if _deadline_guard(start, 8.0, 150.0):
        return best_rows

    # ---- Phase 2: Local search to reduce the maximum entry ----
    # We try to "compress" rows by shifting or swapping elements.
    # Strategy: pick a random row and a random position, try to reduce the value
    # by replacing it with a smaller feasible value, or try swapping two rows'
    # assignments to see if it helps.

    # A more effective approach: try to find a row that has a large gap and
    # "pull" its later elements down. We do this by rebuilding that row with
    # a greedy approach that tries to keep values as small as possible given
    # the current used_diffs (minus this row's diffs).

    random.seed(42)
    improved = True
    while improved and not _deadline_guard(start, 8.0, 150.0):
        improved = False
        # Try each row in random order
        for r in random.sample(range(n_rows), n_rows):
            if _deadline_guard(start, 8.0, 150.0):
                break

            # Remove this row's differences from used_diffs
            row_diffs = set()
            for i in range(len(best_rows[r])):
                for j in range(i + 1, len(best_rows[r])):
                    row_diffs.add(best_rows[r][j] - best_rows[r][i])
            for d in row_diffs:
                used_diffs.discard(d)

            # Rebuild this row greedily, trying to minimize the max
            # Start with 0, then greedily pick smallest feasible x
            new_row = [0]
            # We want to keep the row's max as small as possible.
            # Greedy: pick smallest x > last such that all new diffs are free.
            for _ in range(k):
                x = new_row[-1] + 1
                while True:
                    ok = True
                    for a in new_row:
                        d = x - a
                        if d in used_diffs:
                            ok = False
                            break
                    if ok:
                        break
                    x += 1
                for a in new_row:
                    used_diffs.add(x - a)
                new_row.append(x)

            new_max = new_row[-1]
            old_max = best_rows[r][-1]

            if new_max < old_max:
                best_rows[r] = new_row
                best_score = _score_of(best_rows, k)
                improved = True
            else:
                # Put the original row back
                for d in row_diffs:
                    used_diffs.add(d)
                # Rebuild used_diffs properly: we need to re-add this row's diffs
                # Actually, we removed them and didn't add the new ones (since we didn't accept).
                # Let's just re-add the original diffs.
                # Wait, we already discarded them. We need to re-add.
                for i in range(len(best_rows[r])):
                    for j in range(i + 1, len(best_rows[r])):
                        used_diffs.add(best_rows[r][j] - best_rows[r][i])

    # ---- Phase 3: Try to improve further with a different row ordering in Phase 1 ----
    # If we still have time, try a few different random orderings for the column-wise build.
    if not _deadline_guard(start, 5.0, 150.0):
        for trial in range(20):
            if _deadline_guard(start, 5.0, 150.0):
                break

            rows2 = [[0] for _ in range(n_rows)]
            used2 = set()

            for col in range(k):
                if _deadline_guard(start, 5.0, 150.0):
                    break
                # Random permutation of row order for this column
                row_order = list(range(n_rows))
                random.shuffle(row_order)

                for r in row_order:
                    x = rows2[r][-1] + 1
                    while True:
                        conflict = False
                        for i in range(len(rows2[r])):
                            d = x - rows2[r][i]
                            if d in used2:
                                conflict = True
                                break
                        if not conflict:
                            break
                        x += 1
                    for i in range(len(rows2[r])):
                        used2.add(x - rows2[r][i])
                    rows2[r].append(x)

            trial_score = _score_of(rows2, k)
            if trial_score < best_score:
                best_score = trial_score
                best_rows = [list(r) for r in rows2]

    return best_rows
