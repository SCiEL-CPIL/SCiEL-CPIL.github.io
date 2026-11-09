"""Difference triangle set (7,5) – structural redesign:
   Model-and-optimize with (a) constraint-propagated construction,
   (b) iterated local search with row-rebuild + entry-shift moves,
   (c) systematic downward compaction, and (d) difference-assignment CSP.
"""
import time
import random
import sys
import math

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
    """Structural change: ILS with row-rebuild and entry-shift moves.
       Uses simulated-annealing acceptance to escape local minima."""
    best = [r[:] for r in rows]
    best_scope = _scope(best)
    tracker = DiffTracker(max_mark + 1)
    for r in best:
        tracker.add(_diffs_of_row(r))

    deadline = time.time() + time_budget
    T = 15.0
    cooling = 0.9995
    min_T = 0.05
    best_T = T

    while time.time() < deadline and T > min_T:
        ri = rnd.randint(0, n_rows - 1)
        old_row = best[ri]
        old_diffs = _diffs_of_row(old_row)

        if rnd.random() < 0.65:
            # Row rebuild with noise
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
                    best_T = T
                    T = 15.0  # restart temperature on improvement
            else:
                tracker.add(old_diffs)
        else:
            # Entry shift
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
                            best_T = T
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


def _difference_csp(n_rows, k, max_mark, rnd, time_budget=10.0):
    """
    Difference-assignment CSP:
    - Each row has k+1 entries: a_0=0, a_1, ..., a_k
    - All differences a_j - a_i (i<j) across all rows must be distinct
    - We model this as assigning distinct values to each difference
    - For each row, the differences must be consistent: a_j = a_i + d_ij
    - Use backtracking with constraint propagation on difference values
    """
    deadline = time.time() + time_budget
    total_diffs = n_rows * k * (k + 1) // 2  # 105 for 7x5
    
    # For each row, the differences are determined by the entries
    # We'll use a different approach: assign entries directly but with
    # stronger constraint propagation using the difference space
    
    # Precompute all possible difference positions
    # For row r, position j (1..k), the differences are a_j - a_i for i<j
    # These must all be distinct across all rows
    
    # Strategy: build rows one by one, but use a global difference bitmap
    # and try to minimize the maximum entry by searching from low to high
    
    # Use a smarter backtracking: for each row, try to construct it
    # such that it uses "small" differences that aren't taken yet
    
    used_diffs = bytearray(max_mark + 1)
    rows = [None] * n_rows
    
    def _count_used():
        return sum(used_diffs)
    
    def _build_row(row_idx, max_val, rnd, depth=0):
        if time.time() > deadline:
            return False
        if row_idx == n_rows:
            return True
        
        row = [0]
        if not _build_row_recursive(row_idx, row, used_diffs, max_val, rnd, 0):
            return False
        rows[row_idx] = row[:]
        return True
    
    def _build_row_recursive(row_idx, row, used, max_val, rnd, pos):
        if time.time() > deadline:
            return False
        if pos == k:
            return True
        
        start_x = row[-1] + 1
        # Try values in a randomized order but biased towards smaller values
        candidates = list(range(start_x, max_val + 1))
        # Sort by how "conflicting" they are - prefer values that create
        # differences that are less likely to conflict
        # For now, just shuffle with a bias
        rnd.shuffle(candidates)
        
        for x in candidates:
            new_diffs = _row_diffs_incremental(row, x)
            # Check if all new differences are available
            conflict = False
            for d in new_diffs:
                if d > max_mark or used[d]:
                    conflict = True
                    break
            if conflict:
                continue
            
            # Also check that new differences don't conflict with each other
            # (they shouldn't since row is strictly increasing, but just in case)
            if len(new_diffs) != len(set(new_diffs)):
                continue
            
            # Accept this value
            for d in new_diffs:
                used[d] = 1
            row.append(x)
            
            if _build_row_recursive(row_idx, row, used, max_val, rnd, pos + 1):
                return True
            
            # Backtrack
            row.pop()
            for d in new_diffs:
                used[d] = 0
        
        return False
    
    # Try multiple random orders and different max_val bounds
    for max_val in range(max_mark, max_mark - 10, -1):
        if time.time() > deadline:
            break
        for attempt in range(50):
            if time.time() > deadline:
                break
            used_diffs = bytearray(max_val + 1)
            rows = [None] * n_rows
            
            # Try different row orderings
            order = list(range(n_rows))
            rnd.shuffle(order)
            
            success = True
            for ri in order:
                row = [0]
                if not _build_row_recursive(ri, row, used_diffs, max_val, rnd, 0):
                    success = False
                    break
                rows[ri] = row[:]
            
            if success and all(rows[ri] is not None for ri in range(n_rows)):
                # Validate
                if _valid(rows, n_rows, k):
                    return rows
    
    return None


def _hybrid_csp_search(n_rows, k, max_mark, rnd, time_budget=15.0):
    """
    Hybrid CSP + local search:
    1. Use the difference-CSP to find a valid configuration at or near max_mark
    2. Then use local search to reduce the scope
    3. Alternate between CSP restarts and local search improvements
    """
    deadline = time.time() + time_budget
    best_rows = None
    best_scope = max_mark + 1
    
    # First, try to find any valid configuration
    rows = _difference_csp(n_rows, k, max_mark, rnd, time_budget=5.0)
    if rows is not None:
        best_rows = rows
        best_scope = _scope(rows)
    
    # Now alternate between CSP restarts at lower bounds and local search
    remaining_budget = deadline - time.time()
    if remaining_budget < 1.0:
        return best_rows, best_scope
    
    # Try to improve using local search
    if best_rows is not None:
        remaining = deadline - time.time() - 2.0
        if remaining > 1.0:
            best_rows, best_scope = _iterated_local_search(
                best_rows, n_rows, k, best_scope, remaining, rnd
            )
    
    # Try CSP at lower bounds
    for target_scope in range(best_scope - 1, max(best_scope - 15, 100), -1):
        if time.time() > deadline - 1.0:
            break
        remaining = deadline - time.time() - 1.0
        if remaining < 2.0:
            break
        rows = _difference_csp(n_rows, k, target_scope, rnd, time_budget=min(remaining, 5.0))
        if rows is not None:
            sc = _scope(rows)
            if sc < best_scope:
                best_scope = sc
                best_rows = rows
                # Polish with local search
                remaining2 = deadline - time.time() - 1.0
                if remaining2 > 1.0:
                    best_rows, best_scope = _iterated_local_search(
                        best_rows, n_rows, k, best_scope, remaining2, rnd
                    )
    
    return best_rows, best_scope


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

    # Phase 3: ILS with row-rebuild + entry-shift (structural core)
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

    # Phase 5: Hybrid CSP + local search (new structural approach)
    if best_rows is not None and best_scope > lb:
        remaining = _DEADLINE - time.time() - 3.0
        if remaining > 5.0:
            result_rows, result_scope = _hybrid_csp_search(
                n_rows, k, best_scope, rnd, time_budget=min(remaining, 15.0)
            )
            if result_rows is not None:
                if result_scope < best_scope:
                    best_scope = result_scope
                    best_rows = result_rows

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
