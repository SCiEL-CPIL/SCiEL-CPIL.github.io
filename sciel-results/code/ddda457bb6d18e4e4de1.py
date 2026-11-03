import time
import random
import math
from collections import Counter

# Explicit wall-clock guard: evaluator kills at 150s; keep a 5s safety margin.
_START = time.time()
_DEADLINE = _START + 145.0

def _time_left():
    return _DEADLINE - time.time()


def _scope(rows):
    if not rows:
        return 0
    return max(a for r in rows for a in r[1:])


def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True


def _diffs_for_row(row):
    diffs = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return diffs


def _row_diffs_set(row):
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s


def _fast_greedy(n_rows, k, rng, scope_limit=None):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if scope_limit is not None and x > scope_limit:
                return None
            conflict = False
            for a in row:
                if (x - a) in used:
                    conflict = True
                    break
            if not conflict:
                row.append(x)
                for a in row[:-1]:
                    used.add(x - a)
        rows.append(row)
    return rows


def _csp_backtrack_row(pos, current_row, current_diffs, used_diffs, k, max_val, deadline, rng):
    if time.time() > deadline:
        return None
    n = k + 1
    if pos == n:
        return list(current_row)

    prev = current_row[-1]
    min_next = prev + 1

    if min_next + (n - pos - 1) > max_val:
        return None

    max_next = min(max_val, prev + (max_val - prev))
    candidates = list(range(min_next, max_next + 1))

    if len(candidates) > 4:
        mid = len(candidates) // 3
        rng.shuffle(candidates[mid:])

    for val in candidates:
        if time.time() > deadline:
            return None

        conflict = False
        new_diffs = []
        for idx in range(pos):
            d = val - current_row[idx]
            if d in current_diffs or d in used_diffs:
                conflict = True
                break
            new_diffs.append(d)

        if conflict:
            continue

        current_row.append(val)
        current_diffs.update(new_diffs)

        result = _csp_backtrack_row(pos + 1, current_row, current_diffs, used_diffs, k, max_val, deadline, rng)

        current_row.pop()
        for d in new_diffs:
            current_diffs.discard(d)

        if result is not None:
            return result

    return None


def _construct_rows_csp(n_rows, k, deadline, rng, target_scope):
    used_diffs = set()
    rows = []

    for row_idx in range(n_rows):
        if time.time() > deadline:
            return None

        best_row = None
        best_max_val = float('inf')

        for attempt in range(50):
            if time.time() > deadline:
                break

            current_row = [0]
            current_diffs = set()

            result = _csp_backtrack_row(1, current_row, current_diffs, used_diffs, k, target_scope, deadline, rng)

            if result is not None:
                row_max = result[-1]
                if row_max < best_max_val:
                    best_max_val = row_max
                    best_row = result
                if best_max_val <= target_scope:
                    break

        if best_row is None:
            return None

        rows.append(best_row)
        for d in _diffs_for_row(best_row):
            used_diffs.add(d)

    return rows


def _scope_squeeze_fast(rows, k, deadline, rng):
    rows = [list(r) for r in rows]
    n_rows = len(rows)
    current_scope = _scope(rows)

    row_diff_sets = [_row_diffs_set(r) for r in rows]

    for iteration in range(200):
        if time.time() > deadline:
            break

        max_rows = [i for i, r in enumerate(rows) if r[-1] == current_scope]
        if not max_rows:
            break

        progress_made = False
        for i in max_rows:
            if time.time() > deadline:
                break

            row = rows[i]
            old_max = row[-1]
            prev = row[k - 1]
            old_diffs = row_diff_sets[i]

            for new_val in range(old_max - 1, prev, -1):
                new_diffs = set()
                valid = True
                for idx in range(k):
                    d = new_val - row[idx]
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)

                if not valid:
                    continue

                conflict = False
                for j in range(n_rows):
                    if j == i:
                        continue
                    if new_diffs & row_diff_sets[j]:
                        conflict = True
                        break

                if not conflict:
                    row[k] = new_val
                    row_diff_sets[i] = new_diffs
                    current_scope = new_val
                    progress_made = True
                    break

            if progress_made:
                break

        if not progress_made:
            break

    return rows


def _sa_optimize(rows, k, rng, time_budget=60.0):
    """Simulated-annealing refinement over a conflict-counting representation.

    State: list of rows. We keep an incremental diff-count Counter so that a
    move's cost change (scope + 1000*conflicts) is computed in O(#diffs of one
    row) rather than by rescanning all rows. Because conflicts are penalised
    rather than forbidden, the search can traverse invalid intermediate states
    to escape local optima that the hard-constraint row-rebuild search cannot.
    """
    n_rows = len(rows)
    current = [list(r) for r in rows]
    best = [list(r) for r in current]

    # Build global diff counts and conflict tally incrementally.
    diff_counts = Counter()
    conflicts = 0

    def _add_diff(d):
        nonlocal conflicts
        c = diff_counts.get(d, 0)
        if c == 1:
            conflicts += 1
        diff_counts[d] = c + 1

    def _rem_diff(d):
        nonlocal conflicts
        c = diff_counts.get(d, 0)
        if c == 2:
            conflicts -= 1
        if c <= 1:
            diff_counts.pop(d, None)
        else:
            diff_counts[d] = c - 1

    for row in current:
        for d in _diffs_for_row(row):
            _add_diff(d)

    def _scope_of(state):
        return max(r[-1] for r in state)

    def _cost(state):
        return _scope_of(state) + 1000 * conflicts

    current_scope = _scope_of(current)
    best_scope = current_scope
    best_conflicts = conflicts
    best_cost = current_scope + 1000 * conflicts

    start = time.time()
    temp = 8.0
    cooling = 0.999
    min_temp = 0.3

    while time.time() - start < time_budget and _time_left() > 4.0:
        r = rng.random()

        if r < 0.62:
            # --- Move: rebuild one row greedily (may be invalid) ---
            ri = rng.randrange(n_rows)
            old_row = current[ri]
            for d in _diffs_for_row(old_row):
                _rem_diff(d)

            used = set(diff_counts.keys())
            max_limit = current_scope + rng.randint(0, 6)
            new_row = _fast_greedy(1, k, rng, scope_limit=max_limit)
            if new_row is None:
                # restore
                for d in _diffs_for_row(old_row):
                    _add_diff(d)
                continue

            for d in _diffs_for_row(new_row):
                _add_diff(d)

            current[ri] = new_row
            new_scope = _scope_of(current)
            new_cost = new_scope + 1000 * conflicts

        else:
            # --- Move: shift one interior/last entry of a random row ---
            ri = rng.randrange(n_rows)
            row = current[ri]
            idx = rng.randint(1, len(row) - 1)
            old_val = row[idx]

            if rng.random() < 0.6:
                new_val = old_val - rng.randint(1, 3)
            else:
                new_val = old_val + rng.randint(-3, 3)

            if new_val <= 0 or new_val == old_val:
                continue
            if idx > 1 and new_val <= row[idx - 1]:
                continue
            if idx < len(row) - 1 and new_val >= row[idx + 1]:
                continue

            for d in _diffs_for_row(row):
                _rem_diff(d)
            row[idx] = new_val
            for d in _diffs_for_row(row):
                _add_diff(d)

            new_scope = _scope_of(current)
            new_cost = new_scope + 1000 * conflicts

        # Acceptance
        delta = new_cost - (current_scope + 1000 * conflicts)
        if delta < 0 or (temp > min_temp and rng.random() < math.exp(-delta / temp)):
            current_scope = new_scope
            if current_scope + 1000 * conflicts < best_cost:
                best_cost = current_scope + 1000 * conflicts
                best_scope = current_scope
                best_conflicts = conflicts
                best = [list(r) for r in current]
        else:
            # reject: (state already reverted? no - we mutated in place) so revert
            # For rebuild move we must restore old row; for shift move restore old value.
            if r < 0.62:
                # revert rebuild
                new_row = current[ri]
                for d in _diffs_for_row(new_row):
                    _rem_diff(d)
                current[ri] = old_row
                for d in _diffs_for_row(old_row):
                    _add_diff(d)
            else:
                row = current[ri]
                for d in _diffs_for_row(row):
                    _rem_diff(d)
                row[idx] = old_val
                for d in _diffs_for_row(row):
                    _add_diff(d)

        temp = max(temp * cooling, min_temp)

    # Return the best (validity-preferred) state.
    return best, best_scope, best_conflicts


def construct(n_rows, k):
    rng = random.Random(12345)

    best_result = None
    best_score = float('inf')

    # Phase 1: CSP backtracking construction (structural backbone).
    for target_scope in [112, 114, 116, 120, 125, 130]:
        if _time_left() < 25.0:
            break
        rows = _construct_rows_csp(n_rows, k, _DEADLINE - 15.0, rng, target_scope)
        if rows is not None:
            s = _scope(rows)
            if s < best_score:
                best_score = s
                best_result = rows
            if s <= target_scope:
                break

    # Phase 2: greedy fallback.
    if best_result is None:
        rows = _fast_greedy(n_rows, k, rng)
        if rows is not None:
            best_result = rows
            best_score = _scope(rows)
        else:
            used = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    conflict = False
                    for a in row:
                        if (x - a) in used:
                            conflict = True
                            break
                    if not conflict:
                        row.append(x)
                        for a in row[:-1]:
                            used.add(x - a)
                rows.append(row)
            best_result = rows
            best_score = _scope(rows)

    # Phase 3: simulated-annealing refinement (new structural search family).
    if _time_left() > 18.0:
        budget = min(70.0, _time_left() - 12.0)
        sa_rows, sa_scope, sa_conf = _sa_optimize(best_result, k, rng, time_budget=budget)
        if sa_conf == 0 and _valid(sa_rows, k) and sa_scope < best_score:
            best_score = sa_scope
            best_result = sa_rows

    # Phase 4: fast scope squeeze.
    if _time_left() > 5.0:
        squeezed = _scope_squeeze_fast(best_result, k, _DEADLINE - 2.0, rng)
        if _valid(squeezed, k):
            s = _scope(squeezed)
            if s < best_score:
                best_score = s
                best_result = squeezed

    # Final validation + safety fallback.
    if best_result is None or not _valid(best_result, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                conflict = False
                for a in row:
                    if (x - a) in used:
                        conflict = True
                        break
                if not conflict:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
                rows.append(row)
        best_result = rows

    return best_result
