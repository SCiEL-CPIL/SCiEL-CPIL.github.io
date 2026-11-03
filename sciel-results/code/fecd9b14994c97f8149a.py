import time, random, math

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 115.0
    row_len = k + 1

    def row_diffs(row):
        d = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                d.add(row[j] - ri)
        return d

    def greedy_row(used, rng):
        row = [0]
        x = 0
        while len(row) < row_len:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        return row

    def find_row_dfs(used, best_max, S_cap, node_budget, dl, rng, prefer_small=False):
        """DFS to find a valid row avoiding `used` differences, max < best_max.
        Returns (row, diffs) or None."""
        best = [None]
        best_max_val = [best_max]
        best_sum = [float('inf')]
        nodes = [0]

        def dfs(row, rdiffs, last):
            if time.time() > dl or nodes[0] >= node_budget:
                return
            nodes[0] += 1
            if len(row) == row_len:
                m = row[-1]
                s = sum(row)
                if m < best_max_val[0] or (m == best_max_val[0] and s < best_sum[0]):
                    best_max_val[0] = m
                    best_sum[0] = s
                    best[0] = row[:]
                return
            hi = min(best_max_val[0] - 1, S_cap)
            lo = last + 1
            if lo > hi:
                return
            candidates = list(range(lo, hi + 1))
            if prefer_small:
                # Sort by how "tight" the new differences are (prefer smaller gaps)
                candidates.sort(key=lambda x: x - row[-1])
            else:
                # Randomize order for diversity
                rng.shuffle(candidates)
            for x in candidates:
                new = {x - a for a in row}
                if new & used or new & rdiffs:
                    continue
                row.append(x)
                rdiffs |= new
                dfs(row, rdiffs, x)
                rdiffs -= new
                row.pop()
                if nodes[0] >= node_budget or time.time() > dl:
                    return

        dfs([0], set(), 0)
        if best[0] is None:
            return None
        return best[0], row_diffs(best[0])

    def scope_of(rows):
        return max(r[-1] for r in rows)

    def sum_of(rows):
        return sum(sum(r) for r in rows)

    def energy(rows):
        # Primary: max entry; Secondary: sum of all entries (encourages tight packing)
        return scope_of(rows) * 10000 + sum_of(rows)

    def run_sa(seed, dl):
        rng = random.Random(seed)
        # Initial construction: greedy with some randomization
        used = set()
        rows = []
        for _ in range(n_rows):
            row = greedy_row(used, rng)
            used |= row_diffs(row)
            rows.append(row)

        best_rows = [r[:] for r in rows]
        best_energy = energy(rows)
        best_scope = scope_of(rows)

        # SA parameters
        T = 50.0
        T_min = 0.1
        cooling = 0.995
        step = 0
        total_steps = 5000

        while time.time() < dl and T > T_min and step < total_steps:
            step += 1
            # Pick a random row to re-optimize
            idx = rng.randrange(n_rows)
            old_row = rows[idx]
            old_diffs = row_diffs(old_row)
            used = set()
            for r in rows:
                used |= row_diffs(r)
            used -= old_diffs

            cur_max = old_row[-1]
            # Try to find a better row for this position
            # Use a smaller budget per step since we do many steps
            budget = 30000
            res = find_row_dfs(used, cur_max + 5, cur_max + 20, budget, dl, rng, prefer_small=True)

            if res is not None:
                new_row, new_diffs = res
                rows[idx] = new_row
                new_energy = energy(rows)
                old_energy = energy(best_rows)  # compare to current state
                # Actually compare current state before and after
                # Let's recompute: we changed rows[idx], so current energy is energy(rows)
                # Previous energy was old_energy_prev
                # Simpler: track current energy
                # Let me restructure: compute energy before and after
                pass

            # Revert if not accepted
            # Actually, let me restructure the SA loop more carefully

        # The above SA loop is messy. Let me redo it properly.
        return best_scope, best_rows

    def run_sa_clean(seed, dl):
        rng = random.Random(seed)
        # Initial construction
        used = set()
        rows = []
        for _ in range(n_rows):
            row = greedy_row(used, rng)
            used |= row_diffs(row)
            rows.append(row)

        best_rows = [r[:] for r in rows]
        best_scope = scope_of(rows)
        cur_energy = energy(rows)

        T = 80.0
        T_min = 0.5
        cooling = 0.997

        while time.time() < dl and T > T_min:
            # Pick a random row to re-optimize
            idx = rng.randrange(n_rows)
            old_row = rows[idx]
            old_diffs = row_diffs(old_row)

            # Compute used differences excluding this row
            used = set()
            for i, r in enumerate(rows):
                if i != idx:
                    used |= row_diffs(r)

            cur_max = old_row[-1]
            # Try to find a row with max <= cur_max (or slightly larger for exploration)
            # Primary goal: reduce max; secondary: reduce sum
            search_max = cur_max  # try to find something with max <= cur_max
            budget = 50000

            res = find_row_dfs(used, search_max, search_max + 10, budget, dl, rng, prefer_small=True)

            if res is not None:
                new_row, new_diffs = res
                # Compute new energy
                rows[idx] = new_row
                new_energy = energy(rows)
                delta_e = new_energy - cur_energy
                # Accept if better, or with probability exp(-delta/T) if worse
                if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                    cur_energy = new_energy
                    new_scope = scope_of(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [r[:] for r in rows]
                else:
                    rows[idx] = old_row
            else:
                # Couldn't find a replacement; try greedy as fallback
                # Only accept if it doesn't worsen scope
                pass

            T *= cooling

        return best_scope, best_rows

    try:
        overall = None
        seed = 0
        first = True
        while time.time() < DEADLINE - 5:
            if first:
                dl = min(DEADLINE - 3, time.time() + 60)
                first = False
            else:
                dl = min(DEADLINE - 3, time.time() + 20)
            sc, rows = run_sa_clean(seed, dl)
            if overall is None or sc < overall[0]:
                overall = (sc, rows)
            seed += 1
        if overall is not None:
            return overall[1]
    except Exception:
        pass

    # Last-resort valid greedy
    used = set()
    rows = []
    rng = random.Random(0)
    for _ in range(n_rows):
        row = greedy_row(used, rng)
        used |= row_diffs(row)
        rows.append(row)
    return rows
