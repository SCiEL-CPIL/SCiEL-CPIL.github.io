"""Difference triangle set (7,5) – combined approach:
   (a) fast interleaved greedy initialisation,
   (b) constraint-propagated targeted reduction,
   (c) simulated-annealing local search with row-rebuild + entry-shift,
   (d) systematic downward compaction, and
   (e) backtracking at the current scope for tighter packing.
"""
import time
import random
import math
import sys

_START = time.time()
_DEADLINE = _START + 145.0


def _elapsed():
    return time.time() - _START


def _dead():
    return time.time() >= _DEADLINE


def _diffs_of_row(row):
    d = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            d.add(row[j] - ai)
    return d


def _scope(rows):
    return max(max(r) for r in rows)


def _valid(rows, n_rows, k):
    if len(rows) != n_rows:
        return False
    used = set()
    for r in rows:
        if len(r) != k + 1 or r[0] != 0:
            return False
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                if r[j] <= r[i]:
                    return False
        for d in _diffs_of_row(r):
            if d in used:
                return False
            used.add(d)
    return True


class DiffTracker:
    def __init__(self, max_diff):
        self.max_diff = max_diff
        self.used = bytearray(max_diff + 1)
        self.count = 0

    def add(self, diffs):
        for d in diffs:
            if d <= self.max_diff and not self.used[d]:
                self.used[d] = 1
                self.count += 1

    def remove(self, diffs):
        for d in diffs:
            if d <= self.max_diff and self.used[d]:
                self.used[d] = 0
                self.count -= 1

    def check(self, diffs):
        for d in diffs:
            if d <= self.max_diff and self.used[d]:
                return False
        return True


def _row_diffs_incremental(row, new_x):
    return {new_x - a for a in row}


def _interleaved_greedy(n_rows, k, max_mark, rnd=None):
    tracker = DiffTracker(max_mark + 1)
    rows = [[0] for _ in range(n_rows)]
    for _ in range(n_rows * k):
        candidates = [i for i in range(n_rows) if len(rows[i]) < k + 1]
        if not candidates:
            break
        i = min(candidates, key=lambda idx: rows[idx][-1])
        row = rows[i]
        x = row[-1] + 1
        found = False
        while x <= max_mark:
            new_diffs = _row_diffs_incremental(row, x)
            if tracker.check(new_diffs):
                row.append(x)
                tracker.add(new_diffs)
                found = True
                break
            x += 1
        if not found:
            return None, None
    if all(len(r) == k + 1 for r in rows):
        return rows, tracker
    return None, None


def _greedy_row(tracker, k, max_mark, rnd=None, noise=0.0):
    row = [0]
    for _ in range(k):
        x = row[-1] + 1
        found = False
        while x <= max_mark:
            if noise > 0 and rnd and rnd.random() < noise:
                x += 1
                continue
            new_diffs = _row_diffs_incremental(row, x)
            if tracker.check(new_diffs):
                row.append(x)
                tracker.add(new_diffs)
                found = True
                break
            x += 1
        if not found:
            return None
    return row


def _targeted_reduction(rows, n_rows, k, time_budget):
    best = [r[:] for r in rows]
    best_scope = _scope(best)
    deadline = time.time() + time_budget
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for ri in range(n_rows):
            row = best[ri]
            for pi in range(len(row) - 1, 0, -1):
                old_val = row[pi]
                other_diffs = set()
                for r2 in best:
                    if r2 is not row:
                        other_diffs |= _diffs_of_row(r2)
                row_diffs_other = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        if i != pi and j != pi:
                            row_diffs_other.add(row[j] - row[i])
                forbidden = other_diffs | row_diffs_other
                for new_val in range(old_val - 1, row[pi - 1], -1):
                    if pi < len(row) - 1 and new_val >= row[pi + 1]:
                        continue
                    new_diffs_pi = set()
                    for j in range(len(row)):
                        if j != pi:
                            if j < pi:
                                new_diffs_pi.add(new_val - row[j])
                            else:
                                new_diffs_pi.add(row[j] - new_val)
                    if new_diffs_pi & forbidden:
                        continue
                    row[pi] = new_val
                    new_scope = _scope(best)
                    if new_scope < best_scope:
                        best_scope = new_scope
                    improved = True
                    break
    return best, best_scope


def _replace_row(rows, ri, used, k, max_mark, rnd, randomize_prob=0.0):
    """Try to replace row ri with a smaller-scope alternative."""
    old_diffs = _diffs_of_row(rows[ri])
    used_other = used - old_diffs
    old_scope = rows[ri][-1]

    for attempt in range(8):
        row = [0]
        local = set()
        x = 0
        success = True
        while len(row) < k + 1:
            x += 1
            if x >= old_scope:
                success = False
                break
            if rnd.random() < randomize_prob and x < old_scope - 1:
                x += 1
                if x >= old_scope:
                    success = False
                    break
            bad = False
            for a in row:
                dv = x - a
                if dv in used_other or dv in local:
                    bad = True
                    break
            if not bad:
                row.append(x)
                local.update({x - a for a in row[:-1]})
        if success and len(row) == k + 1:
            new_diffs = _diffs_of_row(row)
            if not (new_diffs & used_other):
                return row, used_other | new_diffs
    return None, None


def _iterated_local_search(rows, n_rows, k, max_mark, time_budget, rnd):
    """Simulated annealing with row-rebuild and entry-shift moves."""
    best = [r[:] for r in rows]
    best_scope = _scope(best)
    tracker = DiffTracker(max_mark + 1)
    for r in best:
        tracker.add(_diffs_of_row(r))

    deadline = time.time() + time_budget
    T = 15.0
    cooling = 0.9995
    min_T = 0.05

    while time.time() < deadline and T > min_T:
        ri = rnd.randint(0, n_rows - 1)
        old_row = best[ri]
        old_diffs = _diffs_of_row(old_row)

        if rnd.random() < 0.65:
            tracker.remove(old_diffs)
            noise = 0.15 + 0.35 * (T / 15.0)
            new_row = _greedy_row(tracker, k, max_mark, rnd, noise=noise)
            if new_row is None:
                tracker.add(old_diffs)
                T *= cooling
                continue
            new_diffs = _diffs_of_row(new_row)
            new_scope = max(max(r2) for r2 in best)
            new_scope = max(new_scope, max(new_row))
            delta = new_scope - best_scope
            if delta <= 0 or rnd.random() < math.exp(-max(0, delta) / T):
                best[ri] = new_row
                tracker.add(new_diffs)
                if new_scope < best_scope:
                    best_scope = new_scope
                    T = 15.0
            else:
                tracker.add(old_diffs)
        else:
            pi = rnd.randint(1, len(old_row) - 1)
            old_val = old_row[pi]
            moved = False
            for offset in [-3, -2, -1, 1, 2, 3]:
                new_val = old_val + offset
                if new_val <= old_row[pi - 1]:
                    continue
                if pi < len(old_row) - 1 and new_val >= old_row[pi + 1]:
                    continue
                new_row = old_row[:]
                new_row[pi] = new_val
                new_diffs = _diffs_of_row(new_row)
                tracker.remove(old_diffs)
                conflict = False
                for d in new_diffs:
                    if d <= max_mark and tracker.used[d]:
                        conflict = True
                        break
                if not conflict:
                    new_scope = max(max(r2) for r2 in best)
                    new_scope = max(new_scope, max(new_row))
                    delta = new_scope - best_scope
                    if delta <= 0 or rnd.random() < math.exp(-max(0, delta) / T):
                        best[ri] = new_row
                        tracker.add(new_diffs)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            T = 15.0
                        moved = True
                        break
                    else:
                        tracker.add(old_diffs)
                        break
                else:
                    tracker.add(old_diffs)
                    break
            if not moved:
                T *= cooling
        T *= cooling

    return best, best_scope


def _backtrack_fill(n_rows, k, max_mark, rnd, time_budget=5.0):
    deadline = time.time() + time_budget
    rows = [None] * n_rows
    tracker = DiffTracker(max_mark + 1)

    def _fill_row(row_idx, row, tr, pos):
        if time.time() > deadline:
            return False
        if pos == k:
            rows[row_idx] = row[:]
            return _solve_next(row_idx + 1)
        start_x = row[-1] + 1
        for x in range(start_x, max_mark + 1):
            new_diffs = _row_diffs_incremental(row, x)
            if tr.check(new_diffs):
                row.append(x)
                tr.add(new_diffs)
                if _fill_row(row_idx, row, tr, pos + 1):
                    return True
                tr.remove(new_diffs)
                row.pop()
        return False

    def _solve_next(idx):
        if time.time() > deadline:
            return False
        if idx == n_rows:
            return True
        row = [0]
        return _fill_row(idx, row, tracker, 0)

    for attempt in range(300):
        if time.time() > deadline:
            break
        perm = list(range(n_rows))
        rnd.shuffle(perm)
        rows[:] = [None] * n_rows
        tracker = DiffTracker(max_mark + 1)

        def solve2(pos):
            if time.time() > deadline:
                return False
            if pos == n_rows:
                return True
            orig_idx = perm[pos]
            row = [0]
            if _fill_row2(orig_idx, row, tracker, 0, pos):
                return True
            return False

        def _fill_row2(row_idx, row, tr, pos, solve_pos):
            if time.time() > deadline:
                return False
            if pos == k:
                rows[row_idx] = row[:]
                return solve2(solve_pos + 1)
            start_x = row[-1] + 1
            for x in range(start_x, max_mark + 1):
                new_diffs = _row_diffs_incremental(row, x)
                if tr.check(new_diffs):
                    row.append(x)
                    tr.add(new_diffs)
                    if _fill_row2(row_idx, row, tr, pos + 1, solve_pos):
                        return True
                    tr.remove(new_diffs)
                    row.pop()
            return False

        if solve2(0):
            return rows
    return None


def construct(n_rows, k):
    rnd = random.Random(42)
    best_rows = None
    best_scope = float('inf')
    lb = n_rows * k * (k + 1) // 2

    # Phase 1: Interleaved greedy for a valid starting point
    rows, _ = _interleaved_greedy(n_rows, k, 200)
    if rows is not None:
        best_rows = rows
        best_scope = _scope(rows)

    if _dead():
        return best_rows

    # Phase 2: Targeted reduction
    if best_rows is not None:
        remaining = _DEADLINE - time.time() - 8.0
        if remaining > 2.0:
            best_rows, best_scope = _targeted_reduction(best_rows, n_rows, k, remaining)

    if _dead():
        return best_rows

    # Phase 3: ILS with row-rebuild + entry-shift
    if best_rows is not None:
        remaining = _DEADLINE - time.time() - 10.0
        if remaining > 3.0:
            best_rows, best_scope = _iterated_local_search(
                best_rows, n_rows, k, best_scope, remaining, rnd
            )

    if _dead():
        return best_rows

    # Phase 4: Another targeted reduction pass
    if best_rows is not None:
        remaining = _DEADLINE - time.time() - 5.0
        if remaining > 1.5:
            best_rows, best_scope = _targeted_reduction(best_rows, n_rows, k, remaining)

    if _dead():
        return best_rows

    # Phase 5: Backtracking at current scope
    if best_rows is not None and best_scope > lb:
        remaining = _DEADLINE - time.time() - 3.0
        if remaining > 3.0:
            result = _backtrack_fill(n_rows, k, best_scope, rnd, time_budget=min(remaining, 12.0))
            if result is not None:
                sc = _scope(result)
                if sc < best_scope:
                    best_scope = sc
                    best_rows = result

    if _dead():
        return best_rows

    # Phase 6: Final ILS polish
    if best_rows is not None:
        remaining = _DEADLINE - time.time() - 2.0
        if remaining > 1.0:
            best_rows, best_scope = _iterated_local_search(
                best_rows, n_rows, k, best_scope, remaining, rnd
            )

    if _dead():
        return best_rows

    # Final validation
    if best_rows is not None and not _valid(best_rows, n_rows, k):
        rows, _ = _interleaved_greedy(n_rows, k, 200)
        if rows is not None:
            best_rows = rows
            best_scope = _scope(rows)

    return best_rows
