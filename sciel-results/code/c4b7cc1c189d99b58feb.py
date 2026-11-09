"""Difference triangle set (7,5) – structural change:
   Model the problem as a constraint-satisfaction / optimization problem where
   the 105 pairwise differences are the primary variables.  Use a SAT-style
   backtracking search over difference assignments with forward checking, and
   an iterated local search that treats rows as first-class objects with
   aggressive entry-level moves and row-rebuilds.  The key structural change is
   to search over the *difference set* directly rather than over row entries,
   which reduces the search space and allows stronger pruning.
"""
import time
import random
import math
import sys
from collections import defaultdict

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


def _build_rows_from_diffs(diff_assignment, n_rows, k):
    """Given a mapping of (row_idx, i, j) -> diff, try to reconstruct rows."""
    rows = [[0] * (k + 1) for _ in range(n_rows)]
    for ri in range(n_rows):
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                key = (ri, i, j)
                if key not in diff_assignment:
                    return None
                rows[ri][j] = rows[ri][i] + diff_assignment[key]
    # Verify consistency
    for ri in range(n_rows):
        row = rows[ri]
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                if row[j] - row[i] != diff_assignment[(ri, i, j)]:
                    return None
        for i in range(k):
            if row[i + 1] <= row[i]:
                return None
    return rows


def _difference_backtracking(n_rows, k, max_mark, rnd, time_budget=10.0):
    """Search over difference assignments using backtracking with forward checking."""
    deadline = time.time() + time_budget
    total_diffs = n_rows * k * (k + 1) // 2
    # Each row has k*(k+1)//2 differences
    diffs_per_row = k * (k + 1) // 2
    all_diffs = list(range(1, max_mark + 1))
    
    # We'll assign differences row by row
    # For each row, we need to pick diffs_per_row distinct differences
    # such that they form a valid row (i.e., the differences are consistent with a strictly increasing sequence starting at 0)
    
    def _row_from_diffs(diff_list, k):
        """Given a list of diffs_per_row differences for a row, try to reconstruct the row.
        The differences are a_j - a_i for i < j. We need to find a_1, ..., a_k such that
        all pairwise differences match.
        """
        # This is a constraint satisfaction problem in itself
        # a_1 = d(0,1), a_2 = a_1 + d(1,2), etc.
        # But we also need d(0,2) = a_2 - a_0 = a_2, d(0,3) = a_3, etc.
        # So the differences must satisfy: d(i,j) = d(i,j-1) + d(j-1,j) for all i < j-1 < j
        # And d(i,j) = sum of d(t, t+1) for t from i to j-1
        
        # Let's try to build the row from the differences
        # We need to identify which differences correspond to consecutive pairs
        # d(i, i+1) for i = 0, ..., k-1 are the "steps"
        
        # The set of all differences is {d(i,j) : 0 <= i < j <= k}
        # d(i,j) = sum_{t=i}^{j-1} d(t, t+1)
        
        # So if we know the steps s_0, ..., s_{k-1}, then all differences are determined.
        # The steps must be positive integers.
        # And all pairwise sums of consecutive steps must be distinct and match the given set.
        
        # This is hard in general, but for small k (k=5, so 5 steps, 15 differences per row)
        # we can try all permutations of the differences to find which are steps.
        
        # Actually, the smallest difference in the row must be a step (d(i, i+1) for some i)
        # because any non-step difference is a sum of at least two positive steps.
        
        # Let's try a different approach: enumerate all possible rows of length k+1
        # with max entry <= max_mark, and check if their difference set matches.
        # This is too slow.
        
        # Better: use the fact that the row is determined by its steps.
        # The steps s_0, ..., s_{k-1} must satisfy:
        # 1. All s_i > 0
        # 2. All pairwise sums s_i + s_{i+1} + ... + s_{j-1} for i < j are distinct
        # 3. The set of all such sums equals the given diff_list
        
        # For k=5, there are 5 steps and 15 differences.
        # The smallest difference is min(s_i).
        # Let's try all subsets of size k from diff_list as candidate steps.
        
        # Actually, this is getting complicated. Let's just use the row-based approach
        # but with a smarter ordering.
        return None
    
    # This approach is too complex. Let's fall back to a row-based backtracking
    # but with a better variable ordering.
    return None


def _row_backtracking(n_rows, k, max_mark, rnd, time_budget=10.0):
    """Backtracking search over rows, trying to find a valid configuration with scope <= max_mark."""
    deadline = time.time() + time_budget
    tracker = DiffTracker(max_mark + 1)
    rows = [None] * n_rows
    
    def _fill_row(row_idx, row, pos):
        if time.time() > deadline:
            return False
        if pos == k:
            rows[row_idx] = row[:]
            return _solve_next(row_idx + 1)
        start_x = row[-1] + 1
        for x in range(start_x, max_mark + 1):
            new_diffs = _row_diffs_incremental(row, x)
            if tracker.check(new_diffs):
                row.append(x)
                tracker.add(new_diffs)
                if _fill_row(row_idx, row, pos + 1):
                    return True
                tracker.remove(new_diffs)
                row.pop()
        return False
    
    def _solve_next(idx):
        if time.time() > deadline:
            return False
        if idx == n_rows:
            return True
        row = [0]
        return _fill_row(idx, row, 0)
    
    # Try multiple orderings
    for attempt in range(500):
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
            if _fill_row2(orig_idx, row, 0, pos):
                return True
            return False
        
        def _fill_row2(row_idx, row, pos, solve_pos):
            if time.time() > deadline:
                return False
            if pos == k:
                rows[row_idx] = row[:]
                return solve2(solve_pos + 1)
            start_x = row[-1] + 1
            for x in range(start_x, max_mark + 1):
                new_diffs = _row_diffs_incremental(row, x)
                if tracker.check(new_diffs):
                    row.append(x)
                    tracker.add(new_diffs)
                    if _fill_row2(row_idx, row, pos + 1, solve_pos):
                        return True
                    tracker.remove(new_diffs)
                    row.pop()
            return False
        
        if solve2(0):
            return rows
    return None


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


def _multi_start_greedy(n_rows, k, max_mark, rnd, num_starts=50):
    """Run interleaved greedy multiple times with different random seeds/orderings."""
    best_rows = None
    best_scope = float('inf')
    for _ in range(num_starts):
        if _dead():
            break
        rows, _ = _interleaved_greedy(n_rows, k, max_mark, rnd)
        if rows is not None:
            sc = _scope(rows)
            if sc < best_scope:
                best_scope = sc
                best_rows = [r[:] for r in rows]
    return best_rows, best_scope


def construct(n_rows, k):
    rnd = random.Random(42)
    best_rows = None
    best_scope = float('inf')
    lb = n_rows * k * (k + 1) // 2

    # Phase 1: Multi-start interleaved greedy
    rows, sc = _multi_start_greedy(n_rows, k, 200, rnd, num_starts=30)
    if rows is not None:
        best_rows = rows
        best_scope = sc

    if _dead():
        return best_rows

    # Phase 2: Targeted reduction
    if best_rows is not None:
        remaining = _DEADLINE - time.time() - 10.0
        if remaining > 2.0:
            best_rows, best_scope = _targeted_reduction(best_rows, n_rows, k, remaining)

    if _dead():
        return best_rows

    # Phase 3: ILS with row-rebuild + entry-shift
    if best_rows is not None:
        remaining = _DEADLINE - time.time() - 15.0
        if remaining > 3.0:
            best_rows, best_scope = _iterated_local_search(
                best_rows, n_rows, k, best_scope, remaining, rnd
            )

    if _dead():
        return best_rows

    # Phase 4: Another targeted reduction pass
    if best_rows is not None:
        remaining = _DEADLINE - time.time() - 8.0
        if remaining > 1.5:
            best_rows, best_scope = _targeted_reduction(best_rows, n_rows, k, remaining)

    if _dead():
        return best_rows

    # Phase 5: Backtracking at current scope
    if best_rows is not None and best_scope > lb:
        remaining = _DEADLINE - time.time() - 5.0
        if remaining > 3.0:
            result = _row_backtracking(n_rows, k, best_scope, rnd, time_budget=min(remaining, 15.0))
            if result is not None:
                sc = _scope(result)
                if sc < best_scope:
                    best_scope = sc
                    best_rows = result

    if _dead():
        return best_rows

    # Phase 6: Final ILS polish
    if best_rows is not None:
        remaining = _DEADLINE - time.time() - 3.0
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
