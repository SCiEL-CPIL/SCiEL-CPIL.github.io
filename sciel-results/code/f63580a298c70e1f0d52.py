"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _row_diffs(row):
    s = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ai)
    return s

def _gen_candidate(rng, k, max_val, used, bias):
    # Build one row of k+1 increasing ints starting at 0 with all internal diffs
    # disjoint from `used` (global). Returns (row, diff_set) or None.
    row = [0]
    local = set()
    target_len = k + 1
    # Try to fill the remaining k slots.
    for _ in range(k):
        # candidate range: from row[-1]+1 up to max_val
        lo = row[-1] + 1
        hi = max_val
        if lo > hi:
            return None
        # sample candidates; bias toward larger values to spread out
        tried = 0
        max_tries = 60
        while tried < max_tries:
            if bias > 0.5:
                # skewed toward high end
                t = rng.random()
                x = lo + int((hi - lo + 1) * (t ** (1.0 / bias))) - 1
                if x < lo:
                    x = lo
                if x > hi:
                    x = hi
            else:
                x = rng.randint(lo, hi)
            tried += 1
            new = {x - a for a in row}
            if not (new & used) and not (new & local):
                row.append(x)
                local |= new
                break
        else:
            return None
    if len(row) != target_len:
        return None
    return row, local

def construct(n_rows, k):
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin under 120s
    rng = random.Random(12345)

    # Baseline: deterministic greedy (fast, always valid)
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    best_rows = rows
    best_scope = max(max(r) for r in best_rows)

    # Candidate-row generation + greedy packing (structural change)
    # We generate many candidate rows, then greedily pick n_rows non-conflicting
    # ones minimizing the max entry.
    # Start with a reasonable max_val estimate from the baseline.
    base_max = best_scope

    # Try several max_val targets from base_max down toward the counting bound.
    # counting bound ~ n_rows * k*(k+1)//2 + ...; use a simple lower estimate
    count_lb = n_rows * k * (k + 1) // 2
    # We'll try targets from base_max-1 down to max(count_lb, base_max - 40)
    # but only a handful to stay within budget.

    attempts = 0
    max_attempts = 400000
    # iterate over candidate max_val in descending order
    for target in range(base_max - 1, max(count_lb, base_max - 45) - 1, -1):
        if time.time() > deadline:
            break
        if attempts >= max_attempts:
            break
        # Try to pack n_rows rows with max entry <= target
        # Generate a pool of candidate rows, then greedy-pack.
        pool = []
        pool_diffs = []
        # Generate pool of candidates; stop early if pool is large enough
        pool_size_target = n_rows * 6
        gen_tries = 0
        max_gen = 4000
        while len(pool) < pool_size_target and gen_tries < max_gen:
            gen_tries += 1
            bias = 0.5 + 0.3 * rng.random()
            cand = _gen_candidate(rng, k, target, set(), bias)
            if cand is None:
                continue
            row, ds = cand
            # check internal validity (should hold by construction)
            pool.append(row)
            pool_diffs.append(ds)
            attempts += 1
            if time.time() > deadline:
                break

        if len(pool) < n_rows:
            continue

        # Greedy pack: repeatedly pick the candidate with smallest max entry
        # that doesn't conflict with already-chosen rows.
        chosen = []
        chosen_diffs = set()
        chosen_mask = [False] * len(pool)
        # Sort pool by max entry ascending for tie-breaking
        order = sorted(range(len(pool)), key=lambda i: max(pool[i]))
        for idx in order:
            if chosen_mask[idx]:
                continue
            ds = pool_diffs[idx]
            if ds & chosen_diffs:
                continue
            chosen.append(pool[idx])
            chosen_mask[idx] = True
            chosen_diffs |= ds
            if len(chosen) == n_rows:
                break

        if len(chosen) == n_rows:
            scope = max(max(r) for r in chosen)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in chosen]
            # If we hit the counting lower bound, we can stop
            if best_scope <= count_lb:
                break
            # If we found a solution at this target, try a smaller target
            # (the loop naturally decrements)
        else:
            # Could not pack at this target; smaller targets unlikely to help
            # but we still decrement; to save time, break after a few failures
            pass

    # Final validity check
    all_diffs = set()
    for r in best_rows:
        d = _row_diffs(r)
        if d & all_diffs:
            # fallback to baseline (which is valid)
            return rows
        all_diffs |= d
    return best_rows
# EVOLVE-BLOCK-END
