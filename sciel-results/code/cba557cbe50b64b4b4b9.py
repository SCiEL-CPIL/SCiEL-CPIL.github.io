"""Difference triangle set (7,5) -- combined structural approach.

Strategy:
  - Phase A: Interleaved greedy + randomized greedy to find an initial valid packing.
  - Phase B: Element-level local search that minimizes individual element values
    (structural change from row-replacement SA), with simulated-annealing acceptance.
  - Phase C: Backtracking repair to squeeze the final scope down.

The key structural change is moving from row-level replacement to element-level
moves, which allows finer-grained optimization and avoids the coarse granularity
of replacing entire rows.
"""

import time
import random
import sys
import math

try:
    import numpy as np
    _HAVE_NUMPY = True
except Exception:
    _HAVE_NUMPY = False

_START = time.time()
_DEADLINE = _START + 142.0


def _elapsed():
    return time.time() - _START


def _dead():
    return time.time() >= _DEADLINE


def _diffs_of_row(row):
    d = set()
    for i in range(len(row)):
        ai = row[i]
        for j in range(i + 1, len(row)):
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


def _greedy_row(used, k, max_mark, rnd):
    row = [0]
    local = set()
    x = 0
    while len(row) < k + 1:
        x += 1
        if x > max_mark:
            return None, local, False
        new = []
        bad = False
        for a in row:
            dv = x - a
            if dv in used or dv in local:
                bad = True
                break
            new.append(dv)
        if not bad:
            row.append(x)
            local.update(new)
    return row, local, True


def _greedy_fill(n_rows, k, max_mark, rnd):
    used = set()
    rows = []
    for _ in range(n_rows):
        row, local, ok = _greedy_row(used, k, max_mark, rnd)
        if not ok:
            return None, None
        rows.append(row)
        used |= local
    return rows, used


def _interleaved_greedy(n_rows, k, max_mark, rnd=None):
    """Build rows interleaved: at each step, extend the row with the smallest
    current maximum. This tends to keep all rows balanced and compact."""
    rows = [[0] for _ in range(n_rows)]
    used = set()

    for _ in range(n_rows * k):
        candidates = [i for i in range(n_rows) if len(rows[i]) < k + 1]
        if not candidates:
            break
        i = min(candidates, key=lambda idx: rows[idx][-1])
        row = rows[i]
        x = row[-1] + 1
        found = False
        while x <= max_mark:
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
                found = True
                break
            x += 1
        if not found:
            return None, None

    for r in rows:
        if len(r) != k + 1:
            return None, None
    return rows, used


def _randomized_greedy_fill(n_rows, k, max_mark, rnd, n_attempts=30):
    """Try multiple random orderings of rows to find a compact packing."""
    best = None
    best_scope = None
    for _ in range(n_attempts):
        order = list(range(n_rows))
        rnd.shuffle(order)
        used = set()
        rows = [None] * n_rows
        ok = True
        for ri in order:
            row, local, ok2 = _greedy_row(used, k, max_mark, rnd)
            if not ok2:
                ok = False
                break
            rows[ri] = row
            used |= local
        if ok and all(r is not None for r in rows):
            s = _scope(rows)
            if best_scope is None or s < best_scope:
                best_scope = s
                best = rows
    return best, best_scope


def _element_local_search(rows, n_rows, k, time_budget, rnd):
    """Element-level local search: try to decrease individual element values.
    
    Structural change: instead of replacing entire rows, we try to move
    individual elements to smaller valid positions. This gives much finer
    granularity and can find improvements that row-replacement misses.
    """
    if rows is None:
        return None, None

    current = [list(r) for r in rows]
    best = [list(r) for r in current]
    best_scope = _scope(best)
    current_scope = best_scope

    used = set()
    for r in current:
        used |= _diffs_of_row(r)

    deadline = time.time() + time_budget
    T = 20.0
    cooling = 0.998
    no_improve = 0
    max_no_improve = 5000

    while time.time() < deadline and T > 0.05:
        if _dead():
            break

        # Pick a random row and a random element (not the first, which is 0)
        ri = rnd.randint(0, n_rows - 1)
        ei = rnd.randint(1, k)  # elements 1..k (element 0 is fixed at 0)

        old_val = current[ri][ei]

        # Try to find a smaller valid value for this element
        # The element must be > previous element and < next element (if exists)
        lo = current[ri][ei - 1] + 1
        hi = old_val - 1  # try to decrease

        if hi < lo:
            T *= cooling
            continue

        # Try values in decreasing order from old_val-1 down to lo
        # but with some randomization to escape local optima
        found_new = False
        best_candidate = None

        # Try a window of candidate values
        search_range = min(hi, lo + int(T * 3) + 5)
        candidates = list(range(lo, search_range + 1))
        rnd.shuffle(candidates)
        # Also try the immediate smaller values
        for v in range(old_val - 1, max(lo, old_val - 15), -1):
            if v not in candidates:
                candidates.append(v)

        for v in candidates:
            if v <= current[ri][ei - 1]:
                continue
            if ei < k and v >= current[ri][ei + 1]:
                continue

            # Check if placing v at position ei keeps all diffs valid
            # Remove old element's contribution, add new
            row = current[ri]
            old_diffs = set()
            for j in range(k + 1):
                if j != ei:
                    if j < ei:
                        old_diffs.add(row[ei] - row[j])
                    else:
                        old_diffs.add(row[j] - row[ei])

            # New diffs with v
            new_diffs = set()
            for j in range(k + 1):
                if j != ei:
                    if j < ei:
                        new_diffs.add(v - row[j])
                    else:
                        new_diffs.add(row[j] - v)

            # Check: new_diffs must not conflict with used minus old_diffs
            # and new_diffs must not have internal conflicts (they can't since
            # they all involve v and distinct other elements)
            # Actually new_diffs could have duplicates if two other elements
            # are at the same distance from v -- but since the row is strictly
            # increasing, v - row[j] for j < ei are all distinct, and
            # row[j] - v for j > ei are all distinct. And v - row[j] > 0
            # for j < ei, row[j] - v > 0 for j > ei. Could v - row[j1] ==
            # row[j2] - v? That would mean row[j1] + row[j2] = 2v, which is
            # possible. So we need to check new_diffs for duplicates.
            if len(new_diffs) != len([x for x in new_diffs]):
                # Actually set handles this; check size
                pass

            # More careful check:
            all_new = list(new_diffs)
            if len(set(all_new)) != len(all_new):
                continue

            # Check against global used (minus old_diffs)
            conflict = False
            for d in new_diffs:
                if d in used and d not in old_diffs:
                    conflict = True
                    break
            if conflict:
                continue

            # Also need to check that new_diffs don't overlap with old_diffs
            # in a way that creates issues... actually no, we're replacing
            # old_diffs with new_diffs, so we just need new_diffs to be
            # disjoint from (used - old_diffs)
            # And new_diffs should be internally consistent (no duplicates)
            # which we checked above.

            # Compute new scope
            new_row = list(row)
            new_row[ei] = v
            new_scope = max(max(r) for i, r in enumerate(current) if i != ri)
            new_scope = max(new_scope, max(new_row))

            # Acceptance
            delta = new_scope - current_scope
            if delta <= 0 or rnd.random() < math.exp(-delta / max(T, 0.01)):
                # Accept: update state
                used -= old_diffs
                used |= new_diffs
                current[ri] = new_row
                current_scope = new_scope
                found_new = True

                if new_scope < best_scope:
                    best_scope = new_scope
                    best = [list(r) for r in current]
                    no_improve = 0
                else:
                    no_improve += 1
                break

        if not found_new:
            no_improve += 1

        if no_improve > max_no_improve:
            # Restart with a fresh random perturbation
            for _ in range(100):
                ri2 = rnd.randint(0, n_rows - 1)
                ei2 = rnd.randint(1, k)
                old_val2 = current[ri2][ei2]
                lo2 = current[ri2][ei2 - 1] + 1
                hi2 = (current[ri2][ei2 + 1] - 1) if ei2 < k else best_scope
                if hi2 > lo2:
                    new_val2 = rnd.randint(lo2, min(hi2, lo2 + int(T * 5)))
                    row2 = current[ri2]
                    old_diffs2 = set()
                    for j in range(k + 1):
                        if j != ei2:
                            if j < ei2:
                                old_diffs2.add(row2[ei2] - row2[j])
                            else:
                                old_diffs2.add(row2[j] - row2[ei2])
                    new_diffs2 = set()
                    for j in range(k + 1):
                        if j != ei2:
                            if j < ei2:
                                new_diffs2.add(new_val2 - row2[j])
                            else:
                                new_diffs2.add(row2[j] - new_val2)
                    # Check validity
                    conflict2 = False
                    for d in new_diffs2:
                        if d in used and d not in old_diffs2:
                            conflict2 = True
                            break
                    if not conflict2 and len(new_diffs2) == len(list(new_diffs2)):
                        used -= old_diffs2
                        used |= new_diffs2
                        new_row2 = list(row2)
                        new_row2[ei2] = new_val2
                        current[ri2] = new_row2
                        current_scope = max(max(r) for r in current)
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best = [list(r) for r in current]
                        no_improve = 0
                        break

        T *= cooling

    return best, best_scope


def _backtrack_repair(n_rows, k, target, rnd, time_budget=3.0):
    """Search for a packing into [0..target] via backtracking."""
    rows = [None] * n_rows
    used = set()

    def fill_row(ridx, row_id, marks):
        if time.time() > time_budget:
            return False
        if ridx == k + 1:
            rows[row_id] = [0] + list(marks)
            return True
        start = (marks[-1] + 1) if marks else 1
        cands = list(range(start, target + 1))
        rnd.shuffle(cands)
        for x in cands:
            prev = [0] + list(marks)
            newl = []
            bad = False
            for a in prev:
                dv = x - a
                if dv in used:
                    bad = True
                    break
                newl.append(dv)
            if bad:
                continue
            marks.append(x)
            used.update(newl)
            if fill_row(ridx + 1, row_id, marks):
                return True
            used.difference_update(newl)
            marks.pop()
        return False

    for _ in range(200):
        if time.time() > time_budget:
            break
        perm = list(range(n_rows))
        rnd.shuffle(perm)
        rows = [None] * n_rows
        used.clear()
        ok = True
        for ri in perm:
            marks = []
            if not fill_row(0, ri, marks):
                ok = False
                break
        if ok and all(rows[t] is not None for t in range(n_rows)):
            return rows
    return None


def construct(n_rows, k):
    rnd = random.Random(42)
    best_rows = None
    best_scope = None

    # ---- Phase A: find initial valid packing ----
    lb = n_rows * k * (k + 1) // 2

    # Try interleaved greedy first (often gives better initial packings)
    for m in range(lb, lb + 100):
        if _dead():
            break
        rows, used = _interleaved_greedy(n_rows, k, m, rnd)
        if rows is not None and _valid(rows, n_rows, k):
            s = _scope(rows)
            if best_scope is None or s < best_scope:
                best_scope = s
                best_rows = rows
            break

    # Try randomized greedy at various scopes
    if best_rows is None or _dead():
        for m in range(lb, lb + 50):
            if _dead():
                break
            rows, s = _randomized_greedy_fill(n_rows, k, m, rnd, n_attempts=20)
            if rows is not None and _valid(rows, n_rows, k):
                if best_scope is None or s < best_scope:
                    best_scope = s
                    best_rows = rows
                break

    # Fallback: plain greedy with no cap
    if best_rows is None:
        used = set()
        rows = []
        for _ in range(n_rows):
            row, local, ok = _greedy_row(used, k, 10 ** 9, rnd)
            if not ok:
                break
            rows.append(row)
            used |= local
        if len(rows) == n_rows:
            best_rows = rows
            best_scope = _scope(rows)

    if best_rows is None:
        # Last resort
        best_rows = []
        for i in range(n_rows):
            row = [0]
            base = (i + 1) * (k + 1) * (k + 1) * (k + 1)
            x = 0
            while len(row) < k + 1:
                x += 1
                row.append(base + x)
            best_rows.append(row)
        best_scope = _scope(best_rows)

    # ---- Phase B: element-level local search ----
    if not _dead():
        time_left = _DEADLINE - time.time() - 5.0
        if time_left > 2.0:
            improved, new_scope = _element_local_search(
                best_rows, n_rows, k, time_left, rnd
            )
            if improved is not None and _valid(improved, n_rows, k):
                if new_scope < best_scope:
                    best_rows = improved
                    best_scope = new_scope

    # ---- Phase C: backtracking repair ----
    if not _dead():
        time_left = _DEADLINE - time.time() - 2.0
        if time_left > 1.0:
            repaired = _backtrack_repair(
                n_rows, k, best_scope, rnd, time_budget=min(time_left, 5.0)
            )
            if repaired is not None and _valid(repaired, n_rows, k):
                s = _scope(repaired)
                if s < best_scope:
                    best_rows = repaired
                    best_scope = s

    # ---- Phase B2: one more round of element local search if time remains ----
    if not _dead():
        time_left = _DEADLINE - time.time() - 2.0
        if time_left > 3.0:
            improved, new_scope = _element_local_search(
                best_rows, n_rows, k, time_left, rnd
            )
            if improved is not None and _valid(improved, n_rows, k):
                if new_scope < best_scope:
                    best_rows = improved
                    best_scope = new_scope

    # Final sanity
    out = []
    for r in best_rows:
        out.append(list(r))
    return out
