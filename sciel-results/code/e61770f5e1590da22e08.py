import time
import random

_DEADLINE = 144.0  # safety margin under the 150 s wall-clock kill


def construct(n_rows, k):
    start = time.time()
    total_diffs_per_row = k * (k + 1) // 2  # C(k+1, 2)
    lb = n_rows * total_diffs_per_row       # counting lower bound on the largest entry

    def elapsed():
        return time.time() - start

    def greedy_once():
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                bits = 0
                ok = True
                for a in row:
                    b = 1 << (x - a)
                    if bits & used:
                        ok = False
                        break
                    bits |= b
                if ok:
                    row.append(x)
                    used |= bits
            rows.append(row)
        return rows, max(r[-1] for r in rows)

    base_rows, base_scope = greedy_once()
    best_rows = base_rows
    best_scope = base_scope

    # Precompute bitmask for each difference value (1..k)
    diff_bit = [0] * (k + 1)
    for d in range(1, k + 1):
        diff_bit[d] = 1 << d

    def backtrack(rows_so_far, used, cap, rng):
        """Build remaining rows so every entry is <= cap. Return list of rows or None."""
        if len(rows_so_far) == n_rows:
            return list(rows_so_far)
        if elapsed() > _DEADLINE:
            return None

        row = [0]
        row_used = 0
        n = k + 1
        # strong lookahead: enough distinct differences left to place?
        if ((1 << (cap + 1)) - 1) & ~used:
            rem = ((1 << (cap + 1)) - 1) & ~used
            rem.bit_count()
        else:
            return None

        def build_row():
            nonlocal row, row_used
            if len(row) == n:
                rows_so_far.append(list(row))
                result = backtrack(rows_so_far, used | row_used, cap, rng)
                if result is not None:
                    return result
                rows_so_far.pop()
                return None

            last = row[-1]
            lo = last + 1
            hi = cap
            if lo > hi:
                return None
            # candidates must keep enough remaining distinct differences available
            # to finish the row (need (n - len(row)) more entries -> need those
            # differences plus the ones they generate; simple capacity check):
            need = (n - len(row))
            if need <= 0:
                return None
            # gather candidate x in [lo, hi]
            # order: randomize to escape traps, but keep feasibility check
            cands = list(range(lo, hi + 1))
            rng.shuffle(cands)
            # feasibility prefilter: x must leave room (x - last) <= k and enough
            # distinct diffs remain above used|row_used to finish.
            for x in cands:
                if x - last > k:
                    continue
                bits = 0
                ok = True
                for a in row:
                    b = diff_bit[x - a]
                    if (bits | row_used | used) & b:
                        ok = False
                        break
                    bits |= b
                if not ok:
                    continue
                # capacity lookahead: after placing x, can we still place
                # (need-1) more entries within cap using fresh differences?
                if need - 1 > 0:
                    rem = ((1 << (cap + 1)) - 1) & ~(used | row_used | bits)
                    if rem.bit_count() < (need - 1) * 1:
                        # at least (need-1) fresh diffs must exist
                        if rem.bit_count() < (need - 1):
                            continue
                row.append(x)
                row_used |= bits
                res = build_row()
                if res is not None:
                    return res
                row.pop()
                row_used ^= bits
            return None

        return build_row()

    target = best_scope - 1
    attempt = 0
    while target >= lb and elapsed() < _DEADLINE:
        found = False
        for _ in range(8):
            if elapsed() > _DEADLINE:
                break
            rng = random.Random(attempt)
            attempt += 1
            rows_found = backtrack([], 0, target, rng)
            if rows_found is not None:
                # verify
                all_diffs = set()
                valid = True
                for r in rows_found:
                    if len(r) != k + 1:
                        valid = False
                        break
                    for i in range(len(r)):
                        for j in range(i + 1, len(r)):
                            d = r[j] - r[i]
                            if d in all_diffs:
                                valid = False
                                break
                            all_diffs.add(d)
                    if not valid:
                        break
                if valid:
                    actual_scope = max(r[-1] for r in rows_found)
                    if actual_scope < best_scope:
                        best_scope = actual_scope
                        best_rows = rows_found
                        target = best_scope - 1
                        found = True
                        break
        if found:
            continue
        # could not beat current best at this target; stop
        break

    return best_rows
