"""Difference triangle set: minimize the largest entry (scope)."""
import time
import random
from itertools import combinations


def construct(n_rows, k):
    start = time.time()
    deadline = start + 120.0  # safety margin under the 120 s evaluator limit
    rng = random.Random(12345)

    def scope_of(rows):
        return max(max(r) for r in rows)

    def all_diffs(row):
        s = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                s.add(row[j] - row[i])
        return s

    def valid(rows):
        seen = set()
        for r in rows:
            d = all_diffs(r)
            if d & seen:
                return False
            seen |= d
        return True

    # --- Greedy fallback (always produces a valid set) ---
    def greedy_rows(n, k, seed=0):
        rng2 = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n):
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

    best = greedy_rows(n_rows, k, 0)
    best_scope = scope_of(best)

    # --- Build candidate rows up to a bound, then solve via conflict graph ---
    def candidate_rows_for_bound(M, n, k, max_candidates=None):
        """All increasing subsets of size k+1 starting at 0 with max <= M."""
        cand = []
        if max_candidates is not None:
            # sample to bound memory/time
            pass
        for combo in combinations(range(1, M + 1), k):
            row = [0] + list(combo)
            cand.append(row)
        if max_candidates is not None and len(cand) > max_candidates:
            rng.shuffle(cand)
            cand = cand[:max_candidates]
        return cand

    def solve_greedy_mis(cand, n, k, seed=0):
        """Greedy maximal independent set: pick rows with disjoint difference sets.
        Heuristic: prefer rows with smaller max entry and fewer 'conflicts'."""
        rng2 = random.Random(seed)
        # Precompute difference sets
        diffsets = [all_diffs(r) for r in cand]
        # Sort by (max entry, random tiebreak) to bias toward small scope
        order = list(range(len(cand)))
        rng2.shuffle(order)
        order.sort(key=lambda i: (max(cand[i]), rng2.random()))
        chosen = []
        used = set()
        for i in order:
            d = diffsets[i]
            if not (d & used):
                chosen.append(cand[i])
                used |= d
                if len(chosen) >= n:
                    break
        if len(chosen) < n:
            return None
        return chosen[:n]

    # Iteratively increase the bound M from a lower bound until we find a full set.
    # Lower bound: counting argument. Each row uses C(k+1,2) distinct differences,
    # total n_rows*C(k+1,2) differences, each <= M, so M >= n_rows*C(k+1,2) roughly.
    C = k * (k + 1) // 2  # C(k+1,2)
    lower = max(k, (n_rows * C + 1) // 2)  # rough
    # A slightly tighter simple bound: differences are 1..M, need n_rows*C distinct, so M >= n_rows*C
    lower = max(lower, n_rows * C)
    # But that's usually too high; use a gentler start and climb.
    lower = max(k, int((n_rows * C) ** 0.6) + k)

    M = lower
    step = 1
    attempts = 0
    max_attempts = 200000
    while time.time() < deadline - 10.0:
        attempts += 1
        if attempts > max_attempts:
            break
        # Cap M to avoid combinatorial explosion; if too big, we already have a valid best.
        if M > best_scope + 50:
            break
        try:
            cand = candidate_rows_for_bound(M, n_rows, k, max_candidates=200000)
        except MemoryError:
            M += step
            continue
        if len(cand) < n_rows:
            M += step
            continue
        # Try several random greedy MIS seeds
        found = None
        for seed in range(20):
            if time.time() > deadline - 10.0:
                break
            res = solve_greedy_mis(cand, n_rows, k, seed)
            if res is not None:
                s = scope_of(res)
                if found is None or s < found[0]:
                    found = (s, res)
        if found is not None:
            s, res = found
            if s < best_scope:
                best = res
                best_scope = s
            # If we hit the lower bound, we're near-optimal; stop.
            if best_scope <= lower:
                break
            M += step
            step = max(1, step // 2)  # refine
        else:
            M += step

    # --- Local improvement: try to reduce the scope of the best set ---
    def try_improve_scope(rows, n, k, seed=0):
        rng2 = random.Random(seed)
        rows = [list(r) for r in rows]
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # For each row, try to find a smaller-max row with disjoint diffs.
            for ri in range(len(rows)):
                others = rows[:ri] + rows[ri + 1:]
                used = set()
                for r in others:
                    used |= all_diffs(r)
                cur_max = max(rows[ri])
                # Try to find any row with max < cur_max disjoint from used
                found_row = None
                # Search over smaller bounds
                for new_max in range(k, cur_max):
                    if time.time() > deadline - 5.0:
                        return None
                    # sample some candidate rows with this max
                    # To keep it fast, try a few random combos
                    for _ in range(50):
                        combo = rng2.sample(range(1, new_max + 1), k - 1) if k > 1 else []
                        row = [0] + sorted(combo) + [new_max] if k > 1 else [0, new_max]
                        if len(set(row)) != len(row):
                            continue
                        row.sort()
                        if row[0] != 0 or row[-1] != new_max:
                            continue
                        if len(row) != k + 1:
                            continue
                        d = all_diffs(row)
                        if not (d & used):
                            found_row = row
                            break
                    if found_row is not None:
                        break
                if found_row is not None:
                    rows[ri] = found_row
                    improved = True
        return rows

    for seed in range(10):
        if time.time() > deadline - 5.0:
            break
        imp = try_improve_scope(best, n_rows, k, seed)
        if imp is not None and valid(imp):
            s = scope_of(imp)
            if s < best_scope:
                best = imp
                best_scope = s

    return best
