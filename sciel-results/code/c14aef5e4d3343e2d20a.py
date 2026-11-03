import time
import random
import math
import sys
from collections import Counter

_DEADLINE = 0.0
_BEST = None


def _scope(rows):
    m = 0
    for r in rows:
        if r:
            m = max(m, r[-1])
    return m


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


def _diffs_of(row):
    n = len(row)
    return [row[j] - row[i] for i in range(n) for j in range(i + 1, n)]


def _fast_greedy(k, n_rows, rng=None, spread=False):
    used = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = 0
            for a in row:
                new |= (1 << (x - a))
            if (used & new) == 0:
                row.append(x)
                used |= new
                if spread:
                    x += max(1, (k - len(row) + 1) * 2 - 2)
        rows.append(row)
    return rows


def _dfs_row(k, max_scope, used, rng, max_attempts=60):
    """Try to build a row via DFS/backtracking, avoiding used diffs."""
    for _ in range(max_attempts):
        row = [0]
        diffs = set()

        def dfs(idx, last):
            if idx == k + 1:
                return True
            # Prune: enough room left?
            remaining = k + 1 - idx
            if last + remaining > max_scope:
                return False
            start = last + 1
            # Randomize order sometimes
            candidates = list(range(start, max_scope + 1))
            if rng.random() < 0.5 and len(candidates) > 1:
                # shuffle a subset
                for i in range(len(candidates) - 1, 0, -1):
                    j = rng.randint(0, i)
                    candidates[i], candidates[j] = candidates[j], candidates[i]
            for t in candidates:
                new_diffs = set()
                ok = True
                for a in row:
                    d = t - a
                    if d in used or d in diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                if ok:
                    row.append(t)
                    diffs.update(new_diffs)
                    if dfs(idx + 1, t):
                        return True
                    row.pop()
                    diffs.difference_update(new_diffs)
            return False

        if dfs(1, 0):
            return row
    return None


def _build_diff_counts(rows):
    counts = Counter()
    for r in rows:
        for d in _diffs_of(r):
            counts[d] += 1
    return counts


def _remove_row_diffs(diff_counts, row):
    for d in _diffs_of(row):
        diff_counts[d] -= 1
        if diff_counts[d] == 0:
            del diff_counts[d]


def _add_row_diffs(diff_counts, row):
    for d in _diffs_of(row):
        diff_counts[d] = diff_counts.get(d, 0) + 1


def _row_rebuild(k, diff_counts, rng, max_scope, attempts=10):
    """Rebuild a row greedily, preferring values that avoid conflicts."""
    best_row = None
    best_conflicts = float('inf')

    for _ in range(attempts):
        row = [0]
        local = set()
        x = 0
        ok = True
        for j in range(1, k + 1):
            found = False
            # Try small steps first, then larger
            step_max = 1
            if _ > 0:
                step_max = rng.randint(1, 6)
            for t in range(x + 1, min(x + 1 + step_max * 5, max_scope + 1)):
                conflicts = 0
                new_diffs = []
                for a in row:
                    d = t - a
                    new_diffs.append(d)
                    if d in local:
                        conflicts = 999
                        break
                    if diff_counts.get(d, 0) > 0:
                        conflicts += 1
                if conflicts == 0:
                    row.append(t)
                    local.update(new_diffs)
                    x = t
                    found = True
                    break
                elif conflicts <= 1 and _ < 4 and rng.random() < 0.2:
                    row.append(t)
                    local.update(new_diffs)
                    x = t
                    found = True
                    break
            if not found:
                # Fallback: find any t
                for t in range(x + 1, max_scope + 1):
                    ok2 = True
                    new_diffs = []
                    for a in row:
                        d = t - a
                        if d in local:
                            ok2 = False
                            break
                        new_diffs.append(d)
                    if ok2:
                        row.append(t)
                        local.update(new_diffs)
                        x = t
                        found = True
                        break
            if not found:
                ok = False
                break
        if ok and len(row) == k + 1:
            # Count conflicts
            sc = 0
            for d in _diffs_of(row):
                if diff_counts.get(d, 0) > 0:
                    sc += 1
            if sc < best_conflicts:
                best_conflicts = sc
                best_row = list(row)
            if best_conflicts == 0:
                break

    return best_row


def _sa_optimize(rows, k, iters, rng, deadline, polish=False):
    n_rows = len(rows)
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope

    diff_counts = _build_diff_counts(current_rows)
    # Collision count: sum of max(0, c-1) over diff counts
    collisions = sum(max(0, c - 1) for c in diff_counts.values())

    def get_cost():
        return _scope(current_rows) + 500 * collisions

    current_cost = get_cost()
    best_cost = current_cost

    if polish:
        T = 0.3
        cooling = 0.99995
        min_T = 0.005
    else:
        T = 6.0
        cooling = 0.9996
        min_T = 0.05

    for it in range(iters):
        if time.time() > deadline:
            break

        r = rng.random()

        if r < 0.12:
            # Row rebuild
            i = rng.randint(0, n_rows - 1)
            old_row = list(current_rows[i])
            _remove_row_diffs(diff_counts, old_row)
            old_coll = collisions
            collisions = sum(max(0, c - 1) for c in diff_counts.values())

            new_row = _row_rebuild(k, diff_counts, rng, current_scope + 5)
            if new_row is None:
                _add_row_diffs(diff_counts, old_row)
                collisions = sum(max(0, c - 1) for c in diff_counts.values())
                continue

            _add_row_diffs(diff_counts, new_row)
            collisions = sum(max(0, c - 1) for c in diff_counts.values())
            current_rows[i] = new_row
            new_cost = get_cost()

            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = _scope(current_rows)
                    best_rows = [list(r) for r in current_rows]
            else:
                _remove_row_diffs(diff_counts, new_row)
                _add_row_diffs(diff_counts, old_row)
                current_rows[i] = old_row
                collisions = sum(max(0, c - 1) for c in diff_counts.values())

        elif r < 0.30:
            # Scope compression: move the max element down
            max_val = current_scope
            max_row_idx = -1
            max_pos_idx = -1
            for ri in range(n_rows):
                for pi in range(k, 0, -1):
                    if current_rows[ri][pi] == max_val:
                        max_row_idx = ri
                        max_pos_idx = pi
                        break
                if max_row_idx >= 0:
                    break
            if max_row_idx < 0:
                continue

            i = max_row_idx
            j = max_pos_idx
            old_val = current_rows[i][j]
            prev_val = current_rows[i][j - 1]
            if old_val - prev_val <= 1:
                continue
            new_val = rng.randint(prev_val + 1, old_val - 1)
            if new_val <= prev_val or new_val >= old_val:
                continue

            _remove_row_diffs(diff_counts, current_rows[i])
            current_rows[i][j] = new_val
            _add_row_diffs(diff_counts, current_rows[i])
            collisions = sum(max(0, c - 1) for c in diff_counts.values())
            new_cost = get_cost()

            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = _scope(current_rows)
                    best_rows = [list(r) for r in current_rows]
            else:
                _remove_row_diffs(diff_counts, current_rows[i])
                current_rows[i][j] = old_val
                _add_row_diffs(diff_counts, current_rows[i])
                collisions = sum(max(0, c - 1) for c in diff_counts.values())

        else:
            # Single-element move
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(1, k)
            old_val = current_rows[i][j]
            prev_val = current_rows[i][j - 1]
            next_val = current_rows[i][j + 1] if j < k else current_scope + 10
            lo = prev_val + 1
            hi = next_val - 1
            if lo > hi:
                continue
            if rng.random() < 0.7:
                span = hi - lo
                delta = int(rng.gauss(0, max(1, span * 0.2)))
                new_val = old_val + delta
            else:
                new_val = rng.randint(lo, hi)
            if new_val == old_val or new_val < lo or new_val > hi:
                continue

            _remove_row_diffs(diff_counts, current_rows[i])
            current_rows[i][j] = new_val
            _add_row_diffs(diff_counts, current_rows[i])
            collisions = sum(max(0, c - 1) for c in diff_counts.values())
            new_cost = get_cost()

            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = _scope(current_rows)
                    best_rows = [list(r) for r in current_rows]
            else:
                _remove_row_diffs(diff_counts, current_rows[i])
                current_rows[i][j] = old_val
                _add_row_diffs(diff_counts, current_rows[i])
                collisions = sum(max(0, c - 1) for c in diff_counts.values())

        T = max(T * cooling, min_T)

    return best_scope, best_rows


def _polish_scope(rows, k, diff_counts, max_iters=200000):
    """Deterministic polish: try to reduce scope by moving elements down."""
    n_rows = len(rows)
    best_scope = _scope(rows)
    rows = [list(r) for r in rows]

    for _ in range(max_iters):
        improved = False
        # Find current max
        cur_max = _scope(rows)
        if cur_max <= best_scope:
            break
        # Try to reduce each occurrence of cur_max
        for ri in range(n_rows):
            for pi in range(k, 0, -1):
                if rows[ri][pi] == cur_max:
                    prev_val = rows[ri][pi - 1]
                    # Try to move down
                    for new_val in range(cur_max - 1, prev_val, -1):
                        # Check if moving works
                        old_row = list(rows[ri])
                        _remove_row_diffs(diff_counts, old_row)
                        rows[ri][pi] = new_val
                        _add_row_diffs(diff_counts, rows[ri])
                        # Check no conflicts
                        conflicts = 0
                        for d in _diffs_of(rows[ri]):
                            if diff_counts[d] > 1:
                                conflicts += 1
                        if conflicts == 0:
                            best_scope = min(best_scope, _scope(rows))
                            improved = True
                            break
                        else:
                            _remove_row_diffs(diff_counts, rows[ri])
                            rows[ri][pi] = old_row[pi]
                            _add_row_diffs(diff_counts, rows[ri])
                    if improved:
                        break
            if improved:
                break
        if not improved:
            break

    return best_scope, rows


def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 115.0

    def consider(rows):
        global _BEST
        if not _valid(rows, k):
            return
        s = _scope(rows)
        if _BEST is None or s < _BEST[0]:
            _BEST = (s, [list(r) for r in rows])

    rng = random.Random(42)

    # Generate initial candidates
    candidates = []

    # Standard greedy
    candidates.append(_fast_greedy(k, n_rows, rng, spread=False))
    # Spread greedy
    candidates.append(_fast_greedy(k, n_rows, rng, spread=True))

    # DFS-based construction
    used = 0
    rows_dfs = []
    for _ in range(n_rows):
        row = _dfs_row(k, 200, used, rng, max_attempts=20)
        if row is None:
            # Fallback to greedy for this row
            x = 0
            row = [0]
            while len(row) < k + 1:
                x += 1
                new = 0
                for a in row:
                    new |= (1 << (x - a))
                if (used & new) == 0:
                    row.append(x)
                    used |= new
        else:
            for d in _diffs_of(row):
                used |= (1 << d)
        rows_dfs.append(row)
    candidates.append(rows_dfs)

    # Randomized greedy
    for _ in range(3):
        used_r = 0
        rows_r = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                found = False
                max_step = rng.randint(1, 8)
                for t in range(x + 1, x + 1 + max_step * 3):
                    new = 0
                    for a in row:
                        new |= (1 << (t - a))
                    if (used_r & new) == 0:
                        row.append(t)
                        used_r |= new
                        x = t
                        found = True
                        break
                if not found:
                    t = x + 1
                    new = 0
                    for a in row:
                        new |= (1 << (t - a))
                    while used_r & new:
                        t += 1
                        new = 0
                        for a in row:
                            new |= (1 << (t - a))
                    row.append(t)
                    used_r |= new
                    x = t
            rows_r.append(row)
        candidates.append(rows_r)

    # Sort by scope
    candidates.sort(key=_scope)

    # Run SA on top candidates
    for idx, cand in enumerate(candidates[:3]):
        if time.time() > _DEADLINE:
            break
        remaining = _DEADLINE - time.time()
        if idx == 0:
            sa_deadline = time.time() + remaining * 0.35
        elif idx == 1:
            sa_deadline = time.time() + remaining * 0.35
        else:
            sa_deadline = time.time() + remaining * 0.25
        iters = int((sa_deadline - time.time()) * 12000)
        iters = max(iters, 3000)
        s, rows = _sa_optimize(cand, k, iters, rng, sa_deadline)
        consider(rows)

    # Polish phase
    if _BEST is not None and time.time() < _DEADLINE:
        remaining_time = _DEADLINE - time.time()
        if remaining_time > 5:
            extra_iters = int(remaining_time * 15000)
            if extra_iters > 500:
                s, rows = _sa_optimize(_BEST[1], k, extra_iters, rng, _DEADLINE, polish=True)
                consider(rows)

    # Final deterministic polish
    if _BEST is not None and time.time() < _DEADLINE - 2:
        rows_p = [list(r) for r in _BEST[1]]
        dc = _build_diff_counts(rows_p)
        s_p, rows_p = _polish_scope(rows_p, k, dc, max_iters=100000)
        consider(rows_p)

    if _BEST is not None:
        return _BEST[1]
    return candidates[0]
