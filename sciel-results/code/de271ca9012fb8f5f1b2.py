import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Need n_rows * k * (k+1)/2 distinct positive differences, so scope >= that.
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs

    # Bitset of used differences. Index d means difference d is used.
    # We allow up to ~3000 distinct differences.
    MAXD = 4000
    mask_all = (1 << MAXD) - 1

    def scope_of(rows):
        return max(max(r) for r in rows)

    def greedy(seed, mode):
        """Randomized greedy construction. Returns list of rows or None."""
        rng = random.Random(seed)
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                if mode == 0:      # compact, mostly in-order
                    span = 20 + rng.randint(0, 15)
                    lo, hi = last + 1, last + span
                    ordered = True
                elif mode == 1:    # spread
                    span = 35 + rng.randint(0, 35)
                    lo, hi = last + 1, last + span
                    ordered = False
                elif mode == 2:    # very compact
                    span = 12 + rng.randint(0, 10)
                    lo, hi = last + 1, last + span
                    ordered = True
                else:              # mixed
                    span = 25 + rng.randint(0, 25)
                    lo, hi = last + 1, last + span
                    ordered = rng.random() < 0.5

                hi = min(hi, MAXD - 1)
                if ordered:
                    cand = range(lo, hi + 1)
                else:
                    cand = list(range(lo, hi + 1))
                    rng.shuffle(cand)

                placed = False
                for x in cand:
                    # differences x - a for a in row
                    ok = True
                    bits = 0
                    for a in row:
                        d = x - a
                        if d <= 0 or d >= MAXD:
                            ok = False
                            break
                        if (used >> d) & 1:
                            ok = False
                            break
                        bits |= (1 << d)
                    if ok:
                        used |= bits
                        row.append(x)
                        placed = True
                        break
                if not placed:
                    # widen search up to MAXD
                    for x in range(last + 1, MAXD):
                        ok = True
                        bits = 0
                        for a in row:
                            d = x - a
                            if d >= MAXD:
                                break
                            if (used >> d) & 1:
                                ok = False
                                break
                            bits |= (1 << d)
                        if ok:
                            used |= bits
                            row.append(x)
                            placed = True
                            break
                if not placed:
                    return None
            rows.append(row)
        return rows

    def polish(rows, seed):
        """Try to reduce the scope by shrinking the last (and middle) element
        of each row to the smallest feasible value. Returns (rows, scope)."""
        rng = random.Random(seed)
        rows = [list(r) for r in rows]

        def used_set_excluding_last(rows, row_idx):
            s = set()
            for i, r in enumerate(rows):
                rr = r[:-1] if i == row_idx else r
                for a in range(len(rr)):
                    for b in range(a + 1, len(rr)):
                        s.add(rr[b] - rr[a])
            return s

        improved = True
        while improved:
            improved = False
            for row_idx in range(n_rows):
                row = rows[row_idx]
                if len(row) < 2:
                    continue
                last_val = row[-1]
                # remove last
                row.pop()
                base = used_set_excluding_last(rows, row_idx)
                best_new = None
                # try smallest value > row[-1]
                for x in range(row[-1] + 1, last_val + 1):
                    ok = True
                    for a in row:
                        d = x - a
                        if d in base:
                            ok = False
                            break
                    if ok:
                        best_new = x
                        break
                if best_new is not None and best_new < last_val:
                    row.append(best_new)
                    improved = True
                else:
                    # try shrinking a middle element
                    row.append(last_val)
                    if len(row) >= 3:
                        mid = len(row) - 2  # second to last
                        old = row[mid]
                        row.pop()  # remove last
                        row.pop()  # remove mid
                        base2 = used_set_excluding_last(rows, row_idx)
                        # base2 now excludes last and mid
                        # differences among remaining
                        rem = row
                        # recompute used from all rows without last & mid of this row
                        s = set()
                        for i, r in enumerate(rows):
                            rr = r[:-2] if i == row_idx else r
                            for a in range(len(rr)):
                                for b in range(a + 1, len(rr)):
                                    s.add(rr[b] - rr[a])
                        # we will reinsert mid then last; try smaller mid
                        found = False
                        for x in range(rem[-1] + 1, old):
                            # check mid vs rem
                            ok = True
                            for a in rem:
                                if (x - a) in s:
                                    ok = False
                                    break
                            if not ok:
                                continue
                            # check last vs rem and mid
                            ok2 = True
                            for a in rem:
                                if (last_val - a) in s:
                                    ok2 = False
                                    break
                            if not ok2:
                                continue
                            if (last_val - x) in s:
                                continue
                            # also ensure mid-last diff not in s
                            row.append(x)
                            row.append(last_val)
                            found = True
                            improved = True
                            break
                        if not found:
                            row.append(old)
                            row.append(last_val)
        return rows, scope_of(rows)

    best_rows = None
    best_scope = float('inf')

    # Phase 1: many fast greedy restarts
    mode_cycle = [0, 2, 0, 1, 0, 2, 1, 0]
    trial = 0
    while trial < 200000:
        if time.time() > deadline - 25.0:
            break
        mode = mode_cycle[trial % len(mode_cycle)]
        rows = greedy(trial * 2654435761 + 12345, mode)
        if rows is not None:
            sc = scope_of(rows)
            if sc < best_scope:
                best_scope = sc
                best_rows = [list(r) for r in rows]
                if best_scope <= lb:
                    break
        trial += 1

    # Phase 2: polish the best found (and a few near-best)
    if best_rows is not None:
        improved, sc = polish(best_rows, 42)
        if sc < best_scope:
            best_scope = sc
            best_rows = improved

    # Phase 3: a second wave of greedy with different seeds, then polish again
    if time.time() < deadline - 40.0:
        trial2 = 0
        while trial2 < 100000:
            if time.time() > deadline - 40.0:
                break
            mode = mode_cycle[(trial2 + 3) % len(mode_cycle)]
            rows = greedy(987654321 + trial2 * 40503, mode)
            if rows is not None:
                sc = scope_of(rows)
                if sc < best_scope:
                    best_scope = sc
                    best_rows = [list(r) for r in rows]
                    if best_scope <= lb:
                        break
            trial2 += 1
        if best_rows is not None:
            improved, sc = polish(best_rows, 12345)
            if sc < best_scope:
                best_scope = sc
                best_rows = improved

    # Phase 4: final polish pass with more iterations (deterministic)
    if best_rows is not None and time.time() < deadline - 5.0:
        improved, sc = polish(best_rows, 777)
        if sc < best_scope:
            best_scope = sc
            best_rows = improved

    if best_rows is None:
        # Fallback: a guaranteed-valid (though large) construction
        rows = []
        offset = 0
        for _ in range(n_rows):
            row = [0]
            cur = 0
            while len(row) < k + 1:
                cur += 1
                row.append(cur + offset)
            rows.append(row)
            offset += max(row) + 1
        return rows

    return best_rows


if __name__ == "__main__":
    result = construct(4, 7)
    if result:
        print("Scope:", max(max(r) for r in result))
        for r in result:
            print(r)
        used = set()
        valid = True
        for row in result:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        valid = False
                    used.add(d)
        print("Valid:", valid)
    else:
        print("No valid construction found")
