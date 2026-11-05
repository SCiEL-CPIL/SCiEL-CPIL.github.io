import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Helpers
    def get_scope(rows):
        return max(max(row) for row in rows) if rows else 0

    def row_diffs(row):
        L = len(row)
        s = set()
        for i in range(L):
            ai = row[i]
            for j in range(i + 1, L):
                s.add(row[j] - ai)
        return s

    def validate(rows):
        seen = set()
        for row in rows:
            for i in range(len(row)):
                ai = row[i]
                for j in range(i + 1, len(row)):
                    d = row[j] - ai
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    # Greedy initial solution
    def greedy_initial():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                nd = {x - a for a in row}
                if nd & used:
                    continue
                row.append(x)
                used |= nd
            rows.append(row)
        return rows, used

    rows, used = greedy_initial()
    best_rows = [r[:] for r in rows]
    best_scope = get_scope(best_rows)
    current_scope = best_scope

    # Row rebuild under a hard max limit, with pressure toward smaller diffs
    def rebuild_row(forbidden, max_val, rng, pressure=0.6):
        row = [0]
        local_used = set()
        for _ in range(k):
            last = row[-1]
            cands = []
            # Search up to max_val, but stop after enough candidates to keep it fast
            for x in range(last + 1, max_val + 1):
                nd = {x - a for a in row}
                if nd & local_used:
                    continue
                if nd & forbidden:
                    continue
                # score: blend of x and max new diff
                mx = max(nd)
                score = (1.0 - pressure) * x + pressure * mx
                cands.append((score, x, nd))
                if len(cands) >= 60:  # enough diversity
                    break
            if not cands:
                return None
            cands.sort(key=lambda t: t[0])
            top = cands[:min(8, len(cands))]
            # weighted random pick favoring lower score
            ws = [1.0 / (1.0 + t[0] * 0.05) for t in top]
            tot = sum(ws)
            r = rng.random() * tot
            cum = 0.0
            chosen = top[0]
            for t, w in zip(top, ws):
                cum += w
                if r <= cum:
                    chosen = t
                    break
            _, x, nd = chosen
            row.append(x)
            local_used |= nd
        return row

    # Two-row rebuild
    def rebuild_two_rows(forbidden, max_val, rng):
        r1 = rebuild_row(forbidden, max_val, rng)
        if r1 is None:
            return None, None
        d1 = row_diffs(r1)
        forb2 = forbidden | d1
        r2 = rebuild_row(forb2, max_val, rng)
        if r2 is None:
            return None, None
        d2 = row_diffs(r2)
        if (d1 & forbidden) or (d2 & forbidden) or (d1 & d2):
            return None, None
        return r1, r2

    rng = random.Random(20250101)
    temp = 1.0
    cool = 0.998
    min_temp = 0.02

    # Main annealing loop
    while time.time() < deadline - 20.0:
        # Decide move type
        use_two = (n_rows >= 2) and (rng.random() < 0.25)

        if use_two:
            i = rng.randrange(n_rows)
            j = rng.randrange(n_rows)
            if i == j:
                j = (j + 1) % n_rows
            old_i = rows[i]
            old_j = rows[j]
            di = row_diffs(old_i)
            dj = row_diffs(old_j)

            # Remove their diffs
            used -= di
            used -= dj

            # Try to rebuild under current scope (allow equal)
            new_i, new_j = rebuild_two_rows(used, current_scope, rng)

            if new_i is None or new_j is None:
                used |= di
                used |= dj
                temp = max(min_temp, temp * cool)
                continue

            ndi = row_diffs(new_i)
            ndj = row_diffs(new_j)
            if (ndi & used) or (ndj & used) or (ndi & ndj):
                used |= di
                used |= dj
                temp = max(min_temp, temp * cool)
                continue

            rows[i] = new_i
            rows[j] = new_j
            used |= ndi
            used |= ndj
            new_scope = get_scope(rows)

            if new_scope < current_scope:
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in rows]
                temp = max(min_temp, temp * 1.05)  # slight reheat
            else:
                # Accept neutral/worse with annealing
                delta = new_scope - current_scope
                if delta <= 0 or (temp > min_temp and rng.random() < (0.15 if temp > 0.2 else 0.05)):
                    current_scope = new_scope
                    temp = max(min_temp, temp * cool)
                else:
                    # revert
                    rows[i] = old_i
                    rows[j] = old_j
                    used -= ndi
                    used -= ndj
                    used |= di
                    used |= dj
                    temp = max(min_temp, temp * cool)
        else:
            i = rng.randrange(n_rows)
            old_i = rows[i]
            di = row_diffs(old_i)
            used -= di

            new_i = rebuild_row(used, current_scope, rng)
            if new_i is None:
                used |= di
                temp = max(min_temp, temp * cool)
                continue

            ndi = row_diffs(new_i)
            if ndi & used:
                used |= di
                temp = max(min_temp, temp * cool)
                continue

            rows[i] = new_i
            used |= ndi
            new_scope = get_scope(rows)

            if new_scope < current_scope:
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in rows]
                temp = max(min_temp, temp * 1.05)
            else:
                delta = new_scope - current_scope
                if delta <= 0 or (temp > min_temp and rng.random() < (0.15 if temp > 0.2 else 0.05)):
                    current_scope = new_scope
                    temp = max(min_temp, temp * cool)
                else:
                    rows[i] = old_i
                    used -= ndi
                    used |= di
                    temp = max(min_temp, temp * cool)

    # Final polish: try to reduce scope by 1 using targeted rebuilds
    # Rebuild rows that contain the current max, under scope-1 limit
    if best_rows is None:
        best_rows = [r[:] for r in rows]
        best_scope = get_scope(best_rows)

    polish_deadline = min(deadline, start_time + 118.0)
    scope = best_scope
    # Work on a copy
    rows = [r[:] for r in best_rows]
    used = set()
    for r in rows:
        used |= row_diffs(r)

    while time.time() < polish_deadline:
        target = scope - 1
        if target < k + 1:
            break
        # Identify rows that hit the current max
        max_rows = [i for i, r in enumerate(rows) if max(r) == scope]
        if not max_rows:
            break

        # Try to rebuild one of those rows under target
        i = max_rows[0]
        old_i = rows[i]
        di = row_diffs(old_i)
        used -= di
        new_i = rebuild_row(used, target, rng, pressure=0.75)
        if new_i is None:
            used |= di
            continue
        ndi = row_diffs(new_i)
        if ndi & used:
            used |= di
            continue
        rows[i] = new_i
        used |= ndi
        new_scope = get_scope(rows)
        if new_scope < scope:
            scope = new_scope
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
        else:
            # revert
            rows[i] = old_i
            used -= ndi
            used |= di

    # Final validation
    if not validate(best_rows):
        # Fallback to a fresh greedy
        rows, _ = greedy_initial()
        best_rows = rows

    return best_rows
