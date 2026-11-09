"""Structural change: global randomized local search (simulated annealing) over all rows,
minimizing the largest entry, with collision penalty for invalid intermediate states."""
import random
import time

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 140.0  # safety margin before 150s kill
    rng = random.Random(20240517)

    def row_diffs(row):
        d = []
        for i in range(len(row)):
            ai = row[i]
            for j in range(i + 1, len(row)):
                d.append(row[j] - ai)
        return d

    def full_cost(rows):
        # largest entry + penalty for duplicate differences across all rows
        scope = max(max(r) for r in rows)
        seen = set()
        dups = 0
        for r in rows:
            for i in range(len(r)):
                ai = r[i]
                for j in range(i + 1, len(r)):
                    diff = r[j] - ai
                    if diff in seen:
                        dups += 1
                    else:
                        seen.add(diff)
        return scope, dups

    def is_valid(rows):
        seen = set()
        for r in rows:
            for i in range(len(r)):
                ai = r[i]
                for j in range(i + 1, len(r)):
                    diff = r[j] - ai
                    if diff in seen:
                        return False
                    seen.add(diff)
        return True

    # ---- Greedy seed (valid starting point) ----
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = max(max(r) for r in best_rows)

    # ---- Local search ----
    rows = [list(r) for r in best_rows]
    cur_scope, cur_dups = full_cost(rows)
    cur_cost = cur_scope * 1000 + cur_dups  # scope dominates

    T = 50.0
    cooling = 0.9999
    n_marks = n_rows * (k + 1)

    while time.time() < deadline:
        # pick a row and a mark index to perturb (skip index 0, which is 0)
        ri = rng.randrange(n_rows)
        row = rows[ri]
        mi = rng.randrange(1, len(row))  # 1..k

        move_type = rng.random()
        new_row = list(row)

        if move_type < 0.5:
            # shift mark by small delta, keep order
            delta = rng.choice([-1, -1, 1, 1, -2, 2, -3, 3])
            v = new_row[mi] + delta
            if v <= new_row[mi - 1] or v >= new_row[mi + 1]:
                continue
            new_row[mi] = v
        elif move_type < 0.75:
            # swap with a neighbor mark (preserve ordering after swap)
            if mi + 1 < len(row):
                a, b = new_row[mi], new_row[mi + 1]
                if b - a > 1:  # only if there's room to swap meaningfully
                    new_row[mi], new_row[mi + 1] = b, a
                    # ensure still increasing (swap of two increasing values keeps increasing)
            else:
                continue
        else:
            # random re-insert: pick a new value in a plausible window around current
            lo = new_row[mi - 1] + 1
            hi = new_row[mi + 1] - 1
            if hi < lo:
                continue
            span = hi - lo
            # bias toward current but allow larger jumps
            if span <= 6:
                v = rng.randint(lo, hi)
            else:
                v = new_row[mi] + rng.randint(-span // 2, span // 2)
                v = max(lo, min(hi, v))
            new_row[mi] = v

        rows[ri] = new_row
        new_scope, new_dups = full_cost(rows)
        new_cost = new_scope * 1000 + new_dups

        delta_cost = new_cost - cur_cost
        if delta_cost <= 0:
            cur_cost, cur_scope, cur_dups = new_cost, new_scope, new_dups
            if new_dups == 0 and new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in rows]
        else:
            # accept with probability
            if T > 1e-6 and rng.random() < (1.0 / (1.0 + __import__('math').exp(delta_cost / max(T, 1e-9)))):
                cur_cost, cur_scope, cur_dups = new_cost, new_scope, new_dups
        T *= cooling

    # Final: return the best valid solution found
    if is_valid(best_rows):
        return best_rows
    # Fallback: greedy is always valid
    return [list(r) for r in greedy()]
# EVOLVE-BLOCK-END
