import time


def construct(n_rows, k):
    start = time.time()
    DEADLINE = start + 110.0          # safety margin before the 150 s kill
    C = k * (k + 1) // 2              # differences per row = C(k+1, 2)
    LB = n_rows * C                   # counting lower bound on the scope

    # ---- fast bitset greedy baseline (always valid, gives an initial scope) ----
    def greedy():
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            row_used = 0
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    nm = 0
                    for a in row:
                        nm |= 1 << (x - a)
                    if (nm & used) == 0 and (nm & row_used) == 0:
                        row.append(x)
                        row_used |= nm
                        used |= nm
                        break
                    x += 1
            rows.append(row)
        return rows, max(max(r) for r in rows)

    rows0, M0 = greedy()
    best_rows, best_S = rows0, M0

    # ---- exact-cover / DLX-style backtracking over a fixed scope S ----
    # used  : bitset of differences already claimed by completed rows
    # row_used : bitset of differences already inside the row being built
    # Feasibility: assign all rows with marks in [1, S], all differences distinct.
    hi = min(M0, 170)
    NODE_BUDGET = 400000

    for S in range(LB, hi + 1):
        if time.time() > DEADLINE - 5:
            break
        used = 0
        nodes = 0
        budget_hit = False
        rows_acc = []

        def bt(row_idx, row_vals, row_used):
            nonlocal used, nodes, budget_hit
            if budget_hit:
                return False
            nodes += 1
            if nodes > NODE_BUDGET or time.time() > DEADLINE - 2:
                budget_hit = True
                return False
            if row_idx == n_rows:
                return True
            if len(row_vals) < k + 1:
                # counting prune: enough unused differences <= S must remain
                need = (n_rows - row_idx) * C - row_used.bit_count()
                avail = S - (used | row_used).bit_count()
                if avail < need:
                    return False
                last = row_vals[-1]
                for x in range(last + 1, S + 1):
                    nm = 0
                    for a in row_vals:
                        nm |= 1 << (x - a)
                    if (nm & used) or (nm & row_used):
                        continue
                    row_vals.append(x)
                    if bt(row_idx, row_vals, row_used | nm):
                        return True
                    row_vals.pop()
                return False
            else:
                # row complete: commit its differences, start the next row
                rows_acc.append(list(row_vals))
                used |= row_used
                if bt(row_idx + 1, [0], 0):
                    return True
                rows_acc.pop()
                used ^= row_used
                return False

        if bt(0, [0], 0):
            best_rows, best_S = rows_acc, S
            break   # smallest feasible scope found (ascending) -> optimal in range

    return best_rows
