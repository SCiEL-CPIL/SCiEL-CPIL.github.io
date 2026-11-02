"""
Row-level stochastic reconstruction + compaction for difference-triangle sets.

Strategy (different family from seed):
- Build each row by sampling increasing integers from a candidate pool,
  explicitly rejecting any candidate whose differences with already-placed
  elements collide with the global used-difference set.
- After an initial full construction, iteratively pick the row with the
  largest max entry, freeze the other rows, and rebuild the chosen row from
  scratch using only differences not used by the frozen rows.
- Apply aggressive right-to-left compaction after every rebuild.
- Multiple restarts with different RNG seeds to escape local minima.
"""

import time
import random
import sys


def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    deadline = time.time() + 118.0
    m = k + 1  # entries per row
    total_diffs = n_rows * k * (k + 1) // 2

    # Lower bound from counting
    lb = n_rows * k * (k + 1) // 2

    def compute_all_diffs(rows):
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        for row in rows:
            if len(row) != m or row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in diffs:
                        return False
                    diffs.add(d)
        return True

    def greedy_row(used_diffs, max_val):
        """Build a single row greedily, avoiding used_diffs."""
        row = [0]
        for _ in range(k):
            placed = False
            for x in range(row[-1] + 1, max_val + 1):
                ok = True
                for a in row:
                    if (x - a) in used_diffs:
                        ok = False
                        break
                if ok:
                    for a in row:
                        used_diffs.add(x - a)
                    row.append(x)
                    placed = True
                    break
            if not placed:
                return None
        return row

    def greedy_full(max_val):
        """Build all rows greedily."""
        used = set()
        rows = []
        for _ in range(n_rows):
            r = greedy_row(used, max_val)
            if r is None:
                return None
            rows.append(r)
        return rows

    def rand_row(used_diffs, max_val, rng, attempts=200):
        """Randomized row construction: pick candidates from a pool."""
        for _ in range(attempts):
            row = [0]
            for _ in range(k):
                # Candidate range: from row[-1]+1 to max_val
                lo = row[-1] + 1
                hi = max_val
                if lo > hi:
                    return None
                # Try a few random candidates
                found = False
                for _ in range(50):
                    x = rng.randint(lo, hi)
                    ok = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            ok = False
                            break
                        new_diffs.append(d)
                    if ok:
                        for d in new_diffs:
                            used_diffs.add(d)
                        row.append(x)
                        found = True
                        break
                if not found:
                    return None
            return row
        return None

    def rand_full(max_val, rng):
        """Build all rows with randomized construction."""
        used = set()
        rows = []
        for _ in range(n_rows):
            r = rand_row(used, max_val, rng)
            if r is None:
                return None
            rows.append(r)
        return rows

    def compact_rows(rows):
        """Right-to-left compaction: reduce each element to minimum feasible."""
        all_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    all_diffs.add(row[j] - row[i])

        improved = True
        while improved:
            improved = False
            for r_idx in range(len(rows)):
                row = rows[r_idx]
                for pos in range(len(row) - 1, 0, -1):
                    current_val = row[pos]
                    for new_val in range(row[pos - 1] + 1, current_val):
                        old_diffs = set()
                        for i in range(pos):
                            old_diffs.add(current_val - row[i])
                        for j in range(pos + 1, len(row)):
                            old_diffs.add(row[j] - current_val)

                        new_diffs = set()
                        for i in range(pos):
                            new_diffs.add(new_val - row[i])
                        for j in range(pos + 1, len(row)):
                            new_diffs.add(row[j] - new_val)

                        conflict = False
                        for d in new_diffs:
                            if d in old_diffs:
                                continue
                            if d in all_diffs:
                                conflict = True
                                break

                        if not conflict:
                            for d in old_diffs:
                                all_diffs.discard(d)
                            for d in new_diffs:
                                all_diffs.add(d)
                            row[pos] = new_val
                            improved = True
                            break
        return rows

    def rebuild_row(rows, r_idx, max_val, rng, attempts=30):
        """Rebuild row r_idx from scratch, keeping other rows frozen."""
        # Compute used diffs from all OTHER rows
        frozen_diffs = set()
        for i, row in enumerate(rows):
            if i == r_idx:
                continue
            for a in range(len(row)):
                for b in range(a + 1, len(row)):
                    frozen_diffs.add(row[b] - row[a])

        # Try to build a new row avoiding frozen_diffs
        best_row = None
        best_max = float('inf')
        for _ in range(attempts):
            row = [0]
            success = True
            for _ in range(k):
                lo = row[-1] + 1
                hi = max_val
                if lo > hi:
                    success = False
                    break
                found = False
                # Try candidates in increasing order with some randomness
                candidates = list(range(lo, hi + 1))
                rng.shuffle(candidates)
                # But prefer smaller values: sort by value with some random perturbation
                candidates.sort(key=lambda x: x + rng.random() * 5)
                for x in candidates:
                    ok = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in frozen_diffs:
                            ok = False
                            break
                        new_diffs.append(d)
                    if ok:
                        # Check for internal conflicts
                        for d in new_diffs:
                            if d in set(nd for nd in new_diffs if nd != d):
                                # This is a set, so no duplicates possible
                                pass
                        for a in row:
                            frozen_diffs.add(x - a)
                        row.append(x)
                        found = True
                        break
                if not found:
                    success = False
                    break
            if success:
                row_max = row[-1]
                if row_max < best_max:
                    best_max = row_max
                    best_row = row
                # Reset frozen_diffs for next attempt
                # Actually we modified frozen_diffs; need to restore
                # Let's redo this more carefully

        # The above modifies frozen_diffs incorrectly. Let's redo properly.
        frozen_diffs2 = set()
        for i, row in enumerate(rows):
            if i == r_idx:
                continue
            for a in range(len(row)):
                for b in range(a + 1, len(row)):
                    frozen_diffs2.add(row[b] - row[a])

        best_row = None
        best_max = float('inf')
        for _ in range(attempts):
            row = [0]
            success = True
            temp_used = set(frozen_diffs2)
            for _ in range(k):
                lo = row[-1] + 1
                hi = max_val
                if lo > hi:
                    success = False
                    break
                found = False
                candidates = list(range(lo, hi + 1))
                rng.shuffle(candidates)
                candidates.sort(key=lambda x: x + rng.random() * 3)
                for x in candidates:
                    ok = True
                    for a in row:
                        d = x - a
                        if d in temp_used:
                            ok = False
                            break
                    if ok:
                        for a in row:
                            temp_used.add(x - a)
                        row.append(x)
                        found = True
                        break
                if not found:
                    success = False
                    break
            if success:
                row_max = row[-1]
                if row_max < best_max:
                    best_max = row_max
                    best_row = list(row)

        if best_row is not None:
            rows[r_idx] = best_row
            return True
        return False

    # Initial construction
    best_rows = None
    best_scope = float('inf')

    # Try greedy first
    for start_scope in range(max(lb, 100), max(lb, 100) + 30):
        rows = greedy_full(start_scope)
        if rows is not None:
            best_rows = rows
            best_scope = max(max(r) for r in rows)
            break

    if best_rows is None:
        # Fallback: very greedy with no scope limit
        used = set()
        best_rows = []
        for _ in range(n_rows):
            r = greedy_row(used, 1000)
            if r is None:
                break
            best_rows.append(r)
        if len(best_rows) != n_rows:
            # Absolute fallback
            best_rows = []
            used = set()
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < m:
                    x += 1
                    ok = True
                    for a in row:
                        if (x - a) in used:
                            ok = False
                            break
                    if ok:
                        for a in row:
                            used.add(x - a)
                        row.append(x)
                best_rows.append(row)
        best_scope = max(max(r) for r in best_rows)

    # Compact initial solution
    best_rows = compact_rows(best_rows)
    best_scope = max(max(r) for r in best_rows)

    # Iterative improvement: rebuild the most expensive row
    rng = random.Random(12345)
    start_time = time.time()

    for iteration in range(200):
        if time.time() > deadline - 5:
            break

        # Find the row with the largest max entry
        max_entries = [max(r) for r in best_rows]
        target_idx = max(range(n_rows), key=lambda i: max_entries[i])
        current_max = max_entries[target_idx]

        # Try to rebuild with a slightly tighter scope
        target_scope = current_max - 1
        if target_scope < lb:
            target_scope = lb

        success = rebuild_row(best_rows, target_idx, target_scope, rng, attempts=20)

        if success:
            # Compact
            best_rows = compact_rows(best_rows)
            new_scope = max(max(r) for r in best_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                # Try another round
            else:
                # Revert: rebuild with original scope
                rebuild_row(best_rows, target_idx, current_max, rng, attempts=10)
                best_rows = compact_rows(best_rows)
                best_scope = max(max(r) for r in best_rows)
        else:
            # Try a full random restart with a tighter scope
            if iteration % 10 == 0:
                attempt_scope = best_scope - 1
                if attempt_scope >= lb:
                    rows = rand_full(attempt_scope, rng)
                    if rows is not None:
                        rows = compact_rows(rows)
                        new_scope = max(max(r) for r in rows)
                        if new_scope < best_scope:
                            best_rows = rows
                            best_scope = new_scope

    # Final compaction
    best_rows = compact_rows(best_rows)
    best_scope = max(max(r) for r in best_rows)

    # Validate
    if not is_valid(best_rows):
        # Fallback to greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < m:
                x += 1
                ok = True
                for a in row:
                    if (x - a) in used:
                        ok = False
                        break
                if ok:
                    for a in row:
                        used.add(x - a)
                    row.append(x)
            rows.append(row)
        return rows

    return best_rows


if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
