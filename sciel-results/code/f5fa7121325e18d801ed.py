"""New design: bounded candidate-pool + randomized greedy + local-search repair."""
import time
import random
import math

# EVOLVE-BLOCK-START
def _greedy(n_rows, k):
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _row_diffs(row):
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s


def _gen_candidates(k, max_mark, cap):
    """Generate valid rows (k+1 terms, start 0, distinct internal diffs) up to max_mark,
    stop after `cap` found."""
    n = k + 1
    results = []
    def rec(start, cur, used):
        if len(results) >= cap:
            return
        if len(cur) == n:
            results.append(tuple(cur))
            return
        need = n - len(cur)
        if max_mark - start + 1 < need:
            return
        for v in range(start, max_mark + 1):
            if len(results) >= cap:
                return
            ok = True
            new_diffs = []
            for a in cur:
                d = v - a
                if d in used:
                    ok = False
                    break
                new_diffs.append(d)
            if not ok:
                continue
            new_used = used | set(new_diffs)
            rec(v + 1, cur + [v], new_used)
    rec(1, [0], set())
    return results


def construct(n_rows, k):
    start_time = time.time()
    DEADLINE = 115.0

    def time_left():
        return DEADLINE - (time.time() - start_time)

    total_diffs = n_rows * (k + 1) * k // 2
    m_low = int(math.ceil((1 + math.sqrt(1 + 8 * total_diffs)) / 2))
    m_hi = max(m_low + 12, 115)

    if time_left() < 8:
        return _greedy(n_rows, k)

    # Precompute candidates with a cap to bound time
    candidates = _gen_candidates(k, m_hi, cap=20000)
    if time_left() < 5:
        return _greedy(n_rows, k)

    candidates.sort(key=lambda r: (r[-1], r))

    from collections import defaultdict
    by_scope = defaultdict(list)
    for r in candidates:
        by_scope[r[-1]].append(r)
    scopes = sorted(by_scope.keys())
    for s in scopes:
        by_scope[s].sort()

    best_rows = None
    best_scope = float('inf')

    def try_greedy(seed):
        rng2 = random.Random(seed)
        # shuffle within each scope bucket for diversity
        local_by_scope = {s: list(by_scope[s]) for s in scopes}
        for s in scopes:
            rng2.shuffle(local_by_scope[s])
        used = set()
        rows = []
        for _ in range(n_rows):
            if time_left() < 2:
                return None
            placed = False
            for s in scopes:
                for r in local_by_scope[s]:
                    rd = _row_diffs(r)
                    if not (rd & used):
                        rows.append(list(r))
                        used |= rd
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                return None
        return rows

    for seed in range(30):
        if time_left() < 5:
            break
        rows = try_greedy(seed)
        if rows is not None:
            sc = max(r[-1] for r in rows)
            if sc < best_scope:
                best_scope = sc
                best_rows = [list(r) for r in rows]
            if best_scope <= m_low:
                break

    # Local search repair from partial states
    def local_search(rows, used, seed):
        rng3 = random.Random(seed)
        rows = [list(r) for r in rows]
        used = set(used)
        for _ in range(n_rows - len(rows)):
            if time_left() < 2:
                return None
            placed = False
            for s in scopes:
                for r in by_scope[s]:
                    rd = _row_diffs(r)
                    if not (rd & used):
                        rows.append(list(r))
                        used |= rd
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                improved = False
                for i in range(len(rows)):
                    old = rows[i]
                    old_diffs = _row_diffs(old)
                    used -= old_diffs
                    new_placed = False
                    for s in scopes:
                        for r in by_scope[s]:
                            rd = _row_diffs(r)
                            if not (rd & used):
                                rows[i] = list(r)
                                used |= rd
                                new_placed = True
                                improved = True
                                break
                        if new_placed:
                            break
                    if not new_placed:
                        rows[i] = old
                        used |= old_diffs
                    if improved:
                        break
                if not improved:
                    return None
        return rows

    for seed in range(10):
        if time_left() < 5:
            break
        rng4 = random.Random(1000 + seed)
        used = set()
        partial = []
        for _ in range(n_rows - 1):
            placed = False
            for s in scopes:
                for r in by_scope[s]:
                    rd = _row_diffs(r)
                    if not (rd & used):
                        partial.append(list(r))
                        used |= rd
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                break
        if len(partial) == n_rows - 1:
            full = local_search(partial, used, seed)
            if full is not None and len(full) == n_rows:
                sc = max(r[-1] for r in full)
                if sc < best_scope:
                    best_scope = sc
                    best_rows = [list(r) for r in full]

    if best_rows is not None:
        return best_rows
    return _greedy(n_rows, k)
# EVOLVE-BLOCK-END
