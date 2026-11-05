import random
import time

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s limit

    def time_up():
        return time.time() > deadline

    # Helper to compute all positive differences in a row
    def get_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def get_max(rows):
        return max(max(r) for r in rows)

    # 1. Greedy Initial Solution with pressure heuristic
    def greedy_initial():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used.update(new_diffs)
            rows.append(row)
        return rows, used

    rows, used_diffs = greedy_initial()
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in rows)
    current_max = best_max

    # Pressure-based rebuild: prefer candidates that minimize max new difference
    def rebuild_row_pressure(k, forbidden_diffs, rng, max_val_limit):
        """
        Rebuild a row using pressure heuristic:
        At each step, among valid candidates, prefer those that minimize
        the maximum difference introduced (to pack differences lower).
        """
        row = [0]
        local_used = set()

        for step in range(k):
            last = row[-1]
            valid_candidates = []
            # Collect valid candidates, but limit search for speed
            # We look at candidates in increasing order of x
            for x in range(last + 1, max_val_limit + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & local_used) and not (new_diffs & forbidden_diffs):
                    # Pressure: max new difference
                    max_new = max(new_diffs)
                    valid_candidates.append((x, new_diffs, max_new))
                    # Collect a few candidates for diversity
                    if len(valid_candidates) >= 8:
                        break

            if not valid_candidates:
                return None

            # Sort by pressure (max_new), then by x
            valid_candidates.sort(key=lambda c: (c[2], c[0]))

            # Pick from top candidates with bias towards lower pressure
            # Use a weighted selection: first candidate has highest weight
            n_cand = len(valid_candidates)
            if n_cand == 1:
                chosen = valid_candidates[0]
            else:
                # Geometric weights favoring low pressure
                weights = [1.0 / (i + 1) ** 1.5 for i in range(n_cand)]
                total_w = sum(weights)
                r = rng.random() * total_w
                cumulative = 0
                chosen = valid_candidates[0]
                for i, w in enumerate(weights):
                    cumulative += w
                    if r <= cumulative:
                        chosen = valid_candidates[i]
                        break

            x, new_diffs, _ = chosen
            row.append(x)
            local_used.update(new_diffs)

        return row

    # Standard randomized greedy rebuild (for diversity)
    def rebuild_row_random(k, forbidden_diffs, rng, max_val_limit):
        row = [0]
        local_used = set()

        for step in range(k):
            last = row[-1]
            valid_candidates = []
            for x in range(last + 1, max_val_limit + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & local_used) and not (new_diffs & forbidden_diffs):
                    valid_candidates.append((x, new_diffs))
                    if len(valid_candidates) >= 5:
                        break

            if not valid_candidates:
                return None

            # Random pick from top few
            x, new_diffs = valid_candidates[rng.randint(0, len(valid_candidates) - 1)]
            row.append(x)
            local_used.update(new_diffs)

        return row

    rng = random.Random(42)

    no_improve_count = 0
    iteration = 0

    while time.time() < deadline:
        iteration += 1
        # Pick a row to rebuild
        i = rng.randint(0, n_rows - 1)

        old_row = rows[i]
        old_diffs = get_diffs(old_row)

        # Remove from used_diffs
        used_diffs.difference_update(old_diffs)

        new_row = None
        # First, try to improve (find row with max < current_max)
        target_max = current_max - 1
        if target_max > k:
            # Use pressure-based rebuild first (better packing)
            new_row = rebuild_row_pressure(k, used_diffs, rng, target_max)
            if new_row is None:
                new_row = rebuild_row_random(k, used_diffs, rng, target_max)

        # If failed, attempt to maintain (neutral move)
        if new_row is None:
            new_row = rebuild_row_pressure(k, used_diffs, rng, current_max)
            if new_row is None:
                new_row = rebuild_row_random(k, used_diffs, rng, current_max)

        if new_row is None:
            # Rebuild failed completely. Restore old row.
            used_diffs.update(old_diffs)
            no_improve_count += 1
            continue

        new_diffs = get_diffs(new_row)

        # Verify no conflicts
        if new_diffs & used_diffs:
            used_diffs.update(old_diffs)
            no_improve_count += 1
            continue

        # Calculate new global max
        rows[i] = new_row
        new_global_max = get_max(rows)

        if new_global_max < current_max:
            # Improvement!
            used_diffs.update(new_diffs)
            current_max = new_global_max
            best_rows = [r[:] for r in rows]
            no_improve_count = 0
        else:
            # No improvement
            # Accept neutral moves with small probability to allow shuffling
            if rng.random() < 0.15:
                used_diffs.update(new_diffs)
                no_improve_count += 1
            else:
                # Reject: restore old row
                rows[i] = old_row
                used_diffs.update(old_diffs)
                no_improve_count += 1

        # If stuck for a long time, try a more aggressive restart
        if no_improve_count > 500 and rng.random() < 0.1:
            # Reset to best and try different random seed
            rows = [r[:] for r in best_rows]
            used_diffs = set()
            for r in rows:
                used_diffs.update(get_diffs(r))
            current_max = best_max
            no_improve_count = 0
            rng = random.Random(rng.randint(0, 2**31))

        # Lower bound check
        lb = n_rows * k * (k + 1) // 2
        if current_max == lb:
            break

    return best_rows

# EVOLVE-BLOCK-END
