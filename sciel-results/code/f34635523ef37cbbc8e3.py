import time
import random
import sys

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 138.0  # safety margin before 150s kill

    # Number of differences per row
    d = k * (k + 1) // 2
    total_diffs = n_rows * d

    def get_scope(rows):
        return max(max(r) for r in rows)

    def full_validate_scope(rows):
        """Return scope if valid, else None."""
        seen = set()
        scope = 0
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    diff = row[j] - ri
                    if diff in seen:
                        return None
                    seen.add(diff)
            scope = max(scope, max(row))
        return scope

    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    rows = greedy_init()
    best_rows = [r[:] for r in rows]
    best_scope = get_scope(rows)

    rng = random.Random(42)

    def row_diffs_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ri)
        return s

    def targeted_descent(rows, budget):
        """Try to reduce the scope by shrinking the largest mark in the row that has it.
        Returns (improved_rows, new_scope) or (None, None) if no improvement found within budget."""
        start = time.time()
        cur = [r[:] for r in rows]
        cur_scope = get_scope(cur)
        improved = False

        while time.time() - start < budget:
            # Find the row(s) containing the current max
            max_val = max(max(r) for r in cur)
            max_rows = [r for r in range(n_rows) if max(cur[r]) == max_val]
            r = rng.choice(max_rows)
            row = cur[r]
            # The last element is the max of this row
            pos = len(row) - 1
            old_val = row[pos]
            lower = row[pos - 1] + 1

            if old_val <= lower:
                continue

            # Try a set of candidate values: prefer values just above the lower bound
            # to maximize the reduction, but also sample a few random ones
            candidates = set()
            # Sequential near the bottom (best reduction)
            for v in range(lower, min(lower + 20, old_val)):
                candidates.add(v)
            # Some random samples in the middle
            for _ in range(15):
                candidates.add(rng.randint(lower, old_val - 1))

            for new_val in sorted(candidates, key=lambda x: -x):  # try larger reductions first
                if new_val >= old_val:
                    continue
                row[pos] = new_val
                scope = full_validate_scope(cur)
                if scope is not None:
                    if scope < cur_scope:
                        cur_scope = scope
                        improved = True
                        break
                row[pos] = old_val

            if not improved:
                # Try reducing a non-last element in the max row
                pos2 = rng.randint(1, len(row) - 2)
                old2 = row[pos2]
                lo2 = row[pos2 - 1] + 1
                hi2 = row[pos2 + 1] - 1
                if hi2 <= lo2:
                    continue
                for _ in range(8):
                    nv = rng.randint(lo2, hi2)
                    if nv == old2:
                        continue
                    row[pos2] = nv
                    scope = full_validate_scope(cur)
                    if scope is not None and scope < cur_scope:
                        cur_scope = scope
                        improved = True
                        break
                    row[pos2] = old2
                if not improved:
                    break

        if improved:
            return cur, cur_scope
        return None, None

    def hill_climb(cur, budget):
        """Pure hill climbing: only accept moves that reduce scope or maintain it.
        More aggressive local search after SA."""
        start = time.time()
        cur_scope = get_scope(cur)
        if cur_scope is None:
            return cur, cur_scope
        no_improve = 0

        while time.time() - start < budget:
            ri = rng.randrange(n_rows)
            pos = rng.randint(1, len(cur[ri]) - 1)
            old_val = cur[ri][pos]
            lo = cur[ri][pos - 1] + 1
            hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else cur_scope + 10
            if hi <= lo:
                continue
            new_val = rng.randint(lo, hi)
            if new_val == old_val:
                continue

            cur[ri][pos] = new_val
            new_scope = full_validate_scope(cur)

            if new_scope is not None:
                if new_scope < cur_scope:
                    cur_scope = new_scope
                    no_improve = 0
                else:
                    cur[ri][pos] = old_val
                    no_improve += 1
            else:
                cur[ri][pos] = old_val
                no_improve += 1

            if no_improve > 500:
                break

        return cur, cur_scope

    # Simulated annealing with restarts
    max_restarts = 400
    steps_per_restart = 60000

    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break

        # Start from best or perturbed
        if restart == 0:
            cur = [r[:] for r in best_rows]
        else:
            cur = [r[:] for r in best_rows]
            # Perturb: random row, random mark, random new value
            n_perturb = rng.randint(1, 4)
            for _ in range(n_perturb):
                ri = rng.randrange(n_rows)
                pos = rng.randint(1, len(cur[ri]) - 1)
                lo = cur[ri][pos - 1] + 1
                hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else best_scope + 15
                if hi > lo:
                    cur[ri][pos] = rng.randint(lo, hi)

        cur_scope = full_validate_scope(cur)
        if cur_scope is None:
            continue

        if cur_scope < best_scope:
            best_scope = cur_scope
            best_rows = [r[:] for r in cur]

        # Annealing with adaptive temperature
        T_start = max(0.1, 1.5 - restart / max_restarts)
        T = T_start
        for step in range(steps_per_restart):
            if time.time() > deadline - 10.0:
                break

            ri = rng.randrange(n_rows)
            pos = rng.randint(1, len(cur[ri]) - 1)
            old_val = cur[ri][pos]
            lo = cur[ri][pos - 1] + 1
            hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else best_scope + 25
            if hi <= lo:
                continue
            new_val = rng.randint(lo, hi)
            if new_val == old_val:
                continue

            cur[ri][pos] = new_val
            new_scope = full_validate_scope(cur)

            if new_scope is not None:
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in cur]
                delta = new_scope - cur_scope
                if delta <= 0 or rng.random() < (2.718281828 ** (-delta / max(T, 0.01))):
                    cur_scope = new_scope
                else:
                    cur[ri][pos] = old_val
            else:
                # Invalid: reject
                cur[ri][pos] = old_val

            T *= 0.99998

        # After annealing, run targeted descent
        if time.time() < deadline - 8.0:
            desc_rows, desc_scope = targeted_descent(cur, budget=1.5)
            if desc_rows is not None and desc_scope < best_scope:
                best_scope = desc_scope
                best_rows = desc_rows

        # Hill climb polish
        if time.time() < deadline - 5.0:
            hc_rows, hc_scope = hill_climb(best_rows, budget=1.0)
            if hc_scope is not None and hc_scope < best_scope:
                best_scope = hc_scope
                best_rows = hc_rows

    # Final polish: more aggressive targeted descent on best
    if time.time() < deadline - 2.0:
        desc_rows, desc_scope = targeted_descent(best_rows, budget=min(5.0, deadline - time.time() - 2.0))
        if desc_rows is not None and desc_scope < best_scope:
            best_scope = desc_scope
            best_rows = desc_rows

    # Final hill climb
    if time.time() < deadline - 1.0:
        hc_rows, hc_scope = hill_climb(best_rows, budget=min(3.0, deadline - time.time() - 1.0))
        if hc_scope is not None and hc_scope < best_scope:
            best_scope = hc_scope
            best_rows = hc_rows

    if best_rows is None:
        best_rows = greedy_init()

    return best_rows
