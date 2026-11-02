import time
import random
import math


def _scope(rows):
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


def _fast_greedy(k, n_rows, spread=False):
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
                if spread:
                    x += max(1, (k - len(row) + 1) * 2 - 2)
        rows.append(row)
    return rows


def _diffs_for_row(row):
    diffs = []
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.append(row[j] - row[i])
    return diffs


def _sa_optimize(rows, k, deadline, rng, max_iters=200000):
    """
    Simulated annealing that minimizes scope while penalizing collisions.
    Fast conflict bookkeeping: a single integer counts total excess occurrences
    of any difference (collisions), updated incrementally per move.
    """
    n_rows = len(rows)
    current = [list(r) for r in rows]
    current_scope = _scope(current)

    # difference -> occurrence count
    diff_count = {}
    for r in current:
        for d in _diffs_for_row(r):
            diff_count[d] = diff_count.get(d, 0) + 1

    def _conflicts(dc):
        return sum(c - 1 for c in dc.values() if c > 1)

    collisions = _conflicts(diff_count)

    best = [list(r) for r in current]
    best_scope = current_scope
    best_collisions = collisions

    T = 60.0
    cooling = 0.99985
    min_T = 0.02

    for it in range(max_iters):
        if time.time() > deadline:
            break

        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)

        old_val = current[i][j]
        prev_val = current[i][j - 1]
        next_val = current[i][j + 1] if j < k else old_val + 100

        # Move selection: bias toward reducing the maximum entry (scope compression)
        move_type = rng.random()
        if move_type < 0.35 and old_val == current_scope:
            # Scope compression: pull the largest entry down
            new_val = rng.randint(prev_val + 1, max(prev_val + 1, old_val - 1))
            if new_val >= old_val:
                new_val = old_val - 1
            if new_val <= prev_val:
                continue
        elif move_type < 0.75:
            # Small perturbation
            delta = rng.randint(-3, 3)
            new_val = old_val + delta
        else:
            # Larger perturbation (can expand to escape local minima)
            new_val = rng.randint(prev_val + 1, max(prev_val + 1, old_val + 15))

        if new_val <= prev_val or (j < k and new_val >= next_val):
            continue

        current[i][j] = new_val
        new_row = current[i]
        new_diffs = _diffs_for_row(new_row)

        # Validity of the row itself (no internal duplicate differences)
        valid_row = len(set(new_diffs)) == len(new_diffs)

        if not valid_row:
            current[i][j] = old_val
            continue

        # Remove old row's differences
        old_diffs = _diffs_for_row(current[i])  # current[i] now holds new_row
        # recompute old diffs from stored old_val? cheaper: recompute before change.
        # We changed already; instead compute old_diffs from a temp.
        # (recompute is fine; row length small)
        # Restore temporarily to get old_diffs correctly
        current[i][j] = old_val
        old_diffs = _diffs_for_row(current[i])
        current[i][j] = new_val

        for d in old_diffs:
            c = diff_count[d] - 1
            if c == 0:
                del diff_count[d]
            else:
                diff_count[d] = c
        for d in new_diffs:
            diff_count[d] = diff_count.get(d, 0) + 1

        new_scope = max(current_scope, max(new_row))
        new_collisions = _conflicts(diff_count)

        # Cost: scope dominates; collisions penalized heavily to keep validity in reach
        scope_penalty = 5000 if current_scope > 120 else 3000 if current_scope > 110 else 2000
        old_cost = current_scope + scope_penalty * collisions
        new_cost = new_scope + scope_penalty * new_collisions

        if new_cost < old_cost:
            accept = True
        else:
            delta_cost = new_cost - old_cost
            accept = rng.random() < math.exp(-delta_cost / max(T, min_T))

        if accept:
            current_scope = new_scope
            collisions = new_collisions
            if new_scope < best_scope or (new_scope == best_scope and new_collisions < best_collisions):
                best_scope = new_scope
                best_collisions = new_collisions
                best = [list(r) for r in current]
        else:
            # Revert
            current[i][j] = old_val
            for d in new_diffs:
                c = diff_count[d] - 1
                if c == 0:
                    del diff_count[d]
                else:
                    diff_count[d] = c
            for d in old_diffs:
                diff_count[d] = diff_count.get(d, 0) + 1

        T = max(T * cooling, min_T)

    return best_scope, best


def _backtrack_scope(rows, k, scope, deadline):
    """
    Deterministic refinement: try to rebuild the construction with every entry
    <= scope. Returns a valid list of rows if found, else None.
    Uses a shared used-diff set across rows and backtracks per row.
    """
    used = set()
    rows_out = []

    def build_row(used):
        row = [0]
        row_diffs = set()

        def extend(last_val):
            if len(row) == k + 1:
                return True
            for x in range(last_val + 1, scope + 1):
                if time.time() > deadline:
                    return False
                new_diffs = set()
                ok = True
                for a in row:
                    d = x - a
                    if d in used or d in row_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                if not ok:
                    continue
                row.append(x)
                row_diffs.update(new_diffs)
                if extend(x):
                    return True
                row.pop()
                row_diffs.difference_update(new_diffs)
            return False

        if extend(0):
            return row
        return None

    def construct_all():
        if len(rows_out) == n_rows:
            return True
        if time.time() > deadline:
            return False
        row = build_row(used)
        if row is None:
            return False
        rd = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                rd.add(row[j] - row[i])
        if rd & used:
            return False
        rows_out.append(row)
        used.update(rd)
        if construct_all():
            return True
        rows_out.pop()
        used.difference_update(rd)
        return False

    if construct_all():
        return rows_out
    return None


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # explicit guard, 10s safety margin before 150s kill

    # 1) Build a diverse pool of initial candidates (greedy variants)
    candidates = []
    candidates.append(_fast_greedy(k, n_rows, spread=False))
    candidates.append(_fast_greedy(k, n_rows, spread=True))
    for seed in range(4):
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += rng.randint(1, 3)
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
        candidates.append(rows)

    # 2) Simulated annealing on the best candidate -> structural scope reduction
    best_rows = min(candidates, key=_scope)
    best_scope = _scope(best_rows)

    rng = random.Random(123)
    sa_scope, sa_rows = _sa_optimize(best_rows, k, deadline, rng)
    if sa_scope < best_scope:
        best_scope = sa_scope
        best_rows = sa_rows

    # 3) Deterministic refinement: tighten scope via backtracking search
    #    (this is the "refinement machinery" that can push below SA's result)
    for scope in range(best_scope - 1, max(best_scope - 40, 0), -1):
        if time.time() > deadline:
            break
        res = _backtrack_scope(best_rows, k, scope, deadline)
        if res is not None:
            best_rows = res
            best_scope = scope

    return best_rows
