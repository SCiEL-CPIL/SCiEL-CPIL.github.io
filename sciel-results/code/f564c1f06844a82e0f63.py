import time
import random
import sys

sys.setrecursionlimit(100000)

_DEADLINE = 150.0
_start = time.time()


def _time_left():
    return _DEADLINE - 3.0 - (time.time() - _start)


def construct(n_rows, k):
    global _start, _DEADLINE
    _start = time.time()
    _DEADLINE = 150.0
    random.seed(1234567)

    rows = _greedy_init(n_rows, k)
    if not rows:
        return []
    best = max(max(r) for r in rows)
    best_rows = [r[:] for r in rows]

    t0 = time.time()
    while time.time() - t0 < 115.0:
        # Local search refinement
        cand = [r[:] for r in best_rows]
        cand_best = _local_search(cand, n_rows, k)
        if cand_best < best:
            best = cand_best
            best_rows = [r[:] for r in cand]

        # Occasional randomized backtracking restart
        if random.random() < 0.4:
            cand2 = _backtrack_random(n_rows, k, max_scope=best - 1, timeout=4.0)
            if cand2 is not None:
                sc = max(max(r) for r in cand2)
                if sc < best:
                    best = sc
                    best_rows = [r[:] for r in cand2]

    return best_rows


def _greedy_init(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _backtrack_random(n_rows, k, max_scope, timeout=4.0):
    """Backtracking with MRV and randomization, bounded by timeout."""
    t0 = time.time()
    used = set()
    rows = []

    def _build_row(row, k, used, rows, t0, timeout):
        if time.time() - t0 > timeout:
            return False
        if len(row) == k + 1:
            rows.append(row[:])
            if len(rows) == n_rows:
                return True
            return _build_row([0], k, used, rows, t0, timeout)

        last = row[-1]
        # Candidates with slight randomization
        candidates = list(range(last + 1, max_scope + 1))
        random.shuffle(candidates)
        # Sort by value + small random noise to prefer smaller
        candidates.sort(key=lambda v: v + random.random() * 3)

        for x in candidates:
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
                if _build_row(row, k, used, rows, t0, timeout):
                    return True
                row.pop()
                used -= new
        return False

    result = _build_row([0], k, used, rows, t0, timeout)
    if result:
        return rows
    return None


def _local_search(rows, n_rows, k):
    """Local search to reduce max entry via relocations and swaps."""
    def scope_of(rs):
        return max(max(r) for r in rs)

    def used_diffs(rs):
        s = set()
        for r in rs:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    s.add(r[j] - r[i])
        return s

    best = scope_of(rows)
    current = [r[:] for r in rows]

    for _ in range(3000):
        if time.time() - _start > _DEADLINE - 5.0:
            break

        # Try random relocations
        r_idx = random.randint(0, n_rows - 1)
        e_idx = random.randint(1, len(current[r_idx]) - 1)
        old_val = current[r_idx][e_idx]
        row = current[r_idx]

        for new_val in range(max(1, old_val - 4), min(old_val + 5, scope_of(current) + 1)):
            if new_val == old_val:
                continue
            if new_val <= row[e_idx - 1] or new_val >= row[e_idx + 1]:
                continue
            old_diffs = {abs(old_val - row[i]) for i in range(len(row)) if i != e_idx}
            new_diffs = {abs(new_val - row[i]) for i in range(len(row)) if i != e_idx}
            other = used_diffs(current) - old_diffs
            if new_diffs & other:
                continue
            row[e_idx] = new_val
            sc = scope_of(current)
            if sc < best:
                best = sc
                break
            else:
                row[e_idx] = old_val

        # Try swaps between rows
        r1 = random.randint(0, n_rows - 1)
        r2 = random.randint(0, n_rows - 1)
        if r1 == r2:
            continue
        e1 = random.randint(1, len(current[r1]) - 1)
        e2 = random.randint(1, len(current[r2]) - 1)
        v1 = current[r1][e1]
        v2 = current[r2][e2]
        row1, row2 = current[r1], current[r2]

        if (v2 <= row1[e1 - 1] or v2 >= row1[e1 + 1]):
            continue
        if (v1 <= row2[e2 - 1] or v1 >= row2[e2 + 1]):
            continue

        old_d1 = {abs(v1 - row1[i]) for i in range(len(row1)) if i != e1}
        old_d2 = {abs(v2 - row2[i]) for i in range(len(row2)) if i != e2}
        all_other = used_diffs(current) - old_d1 - old_d2
        new_d1 = {abs(v2 - row1[i]) for i in range(len(row1)) if i != e1}
        new_d2 = {abs(v1 - row2[i]) for i in range(len(row2)) if i != e2}

        if (new_d1 | new_d2) & all_other or (new_d1 & new_d2):
            continue

        row1[e1] = v2
        row2[e2] = v1
        sc = scope_of(current)
        if sc < best:
            best = sc
        else:
            row1[e1] = v1
            row2[e2] = v2

    return best
