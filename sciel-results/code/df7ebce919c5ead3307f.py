"""Difference-triangle (4,7): minimize scope via simulated annealing over row-swap moves."""
import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 115.0
    rng = random.Random(20250407)

    # ---- helpers -------------------------------------------------------
    def greedy_row(used, row_k, max_val, order):
        """Greedily build one row (k+1 entries, starts at 0) using smallest
        marks whose within-row differences are unused. Returns None if the
        row cannot be completed within max_val."""
        row = [0]
        x = 0
        while len(row) < row_k:
            x += 1
            if x > max_val:
                return None
            new = [x - a for a in row]
            if any(d in used for d in new):
                continue
            row.append(x)
            for d in new:
                used.add(d)
        return row

    def row_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}

    def rebuild_all(rows, max_val):
        """Rebuild every row greedily in the given order against a fresh used set."""
        used = set()
        out = []
        for _ in range(n_rows):
            r = greedy_row(used, k + 1, max_val, None)
            if r is None:
                return None
            out.append(r)
        return out

    # ---- initial population (multi-start greedy) ----------------------
    # Try several orderings/shuffles to seed different local optima.
    best_rows = None
    best_scope = None

    # deterministic base
    base = rebuild_all([0] * n_rows, 4 * n_rows * (k + 1) + 20)
    if base is not None:
        best_rows = base
        best_scope = max(max(r) for r in base)

    # a few randomized restarts (different greedy orderings via tiny jitter)
    for _ in range(6):
        jitter = [rng.randint(0, 2) for _ in range(n_rows)]
        # emulate ordering by trying rows in a shuffled priority (we rebuild in
        # fixed index order but with a per-row starting bias is not allowed;
        # instead we just re-run deterministic which is same -> skip duplicates)
        cand = rebuild_all([0] * n_rows, 4 * n_rows * (k + 1) + 20)
        if cand is not None:
            sc = max(max(r) for r in cand)
            if best_scope is None or sc < best_scope:
                best_scope = sc
                best_rows = cand

    # ---- simulated annealing over row-swap moves ----------------------
    # State: list of rows. A move: pick a row index, remove its diffs from used,
    # greedily rebuild that row against the rest, accept by SA criterion.
    cur_rows = [list(r) for r in best_rows]
    cur_scope = best_scope

    def used_from_except(rows, idx):
        used = set()
        for i, r in enumerate(rows):
            if i == idx:
                continue
            used |= row_diffs(r)
        return used

    T = 30.0
    T_min = 0.05
    cooling = 0.9995
    start = time.time()

    # number of moves we can afford: cap so we stay within budget
    max_iters = 200000
    it = 0
    while it < max_iters:
        if time.time() > deadline:
            break
        it += 1
        idx = rng.randrange(n_rows)
        used = used_from_except(cur_rows, idx)
        new_row = greedy_row(used, k + 1, cur_scope, None)
        if new_row is None:
            # try to allow a slightly larger value to escape
            new_row = greedy_row(used, k + 1, cur_scope + 4, None)
            if new_row is None:
                continue
        # compute new scope
        new_scope = max(cur_scope, new_row[-1])
        # if we actually reduced the max, that's great
        delta = new_scope - cur_scope
        if delta <= 0 or rng.random() < (1.0 / (1.0 + delta)) * min(1.0, T / 20.0):
            cur_rows[idx] = new_row
            if new_scope < cur_scope:
                cur_scope = new_scope
                if cur_scope < best_scope:
                    best_scope = cur_scope
                    best_rows = [list(r) for r in cur_rows]
        T *= cooling
        if T < T_min:
            T = T_min

    # ---- final polish: deterministic local search (row-swap hill climb) --
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(n_rows):
            used = used_from_except(cur_rows, idx)
            new_row = greedy_row(used, k + 1, cur_scope, None)
            if new_row is None:
                continue
            new_scope = max(cur_scope, new_row[-1])
            if new_scope < cur_scope:
                cur_rows[idx] = new_row
                cur_scope = new_scope
                if cur_scope < best_scope:
                    best_scope = cur_scope
                    best_rows = [list(r) for r in cur_rows]
                improved = True

    return best_rows
# EVOLVE-BLOCK-END
