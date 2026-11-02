"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Construct a difference triangle set: n_rows rows of k+1 increasing integers
    starting at 0, with all pairwise differences distinct across all rows.
    Minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit

    # Total number of differences needed
    total_diffs = n_rows * k * (k + 1) // 2

    def row_diffs_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                s.add(row[j] - row[i])
        return s

    def compute_all_used(rows):
        used = set()
        for r in rows:
            used |= row_diffs_set(r)
        return used

    def global_max(rows):
        return max(max(r) for r in rows)

    def max_except(rows, exclude_idx):
        m = 0
        for i, r in enumerate(rows):
            if i != exclude_idx:
                if r[-1] > m:
                    m = r[-1]
        return m

    def validate(rows):
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            diffs = []
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
                    diffs.append(d)
            if len(diffs) != len(set(diffs)):
                return False
        return True

    def greedy_init(rng=None, randomize=False):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    if not randomize or rng is None or rng.random() < 0.7:
                        row.append(x)
                        used |= new_diffs
                    x += 1
                else:
                    x += 1
                if randomize and rng is not None and rng.random() < 0.1:
                    x += rng.randint(1, 5)
            rows.append(row)
        return rows

    def random_valid_row(forbidden, max_limit, rng, max_candidates=15):
        """Find a random valid row with max entry <= max_limit."""
        if max_limit < k:
            return None
        row = [0]
        used_in_row = set()

        for step in range(k):
            candidates = []
            start = row[-1] + 1
            end = max_limit + 1

            # Scan for valid candidates
            for x in range(start, end):
                new_diffs = {x - a for a in row}
                if new_diffs & forbidden:
                    continue
                if new_diffs & used_in_row:
                    continue
                candidates.append(x)
                if len(candidates) >= max_candidates:
                    break

            if not candidates:
                return None

            # Weight toward smaller candidates to keep the row compact
            if len(candidates) > 1 and rng.random() < 0.7:
                half = max(1, len(candidates) // 2)
                chosen_x = rng.choice(candidates[:half])
            else:
                chosen_x = rng.choice(candidates)
            row.append(chosen_x)
            used_in_row |= {chosen_x - a for a in row[:-1]}

        return row

    def local_search_minimize(rows, deadline, rng):
        """Element-level local search to minimize each row's entries."""
        best = [r[:] for r in rows]
        best_val = global_max(best)
        best_used = compute_all_used(best)
        
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            improved = False
            for ri in range(n_rows):
                if time.time() > deadline - 2:
                    break
                forbidden = best_used - row_diffs_set(best[ri])
                # Try to find a row with strictly smaller max
                new_row = random_valid_row(forbidden, best_val - 1, rng, max_candidates=30)
                if new_row is not None:
                    other_max = max_except(best, ri)
                    new_max = max(other_max, new_row[-1])
                    if new_max < best_val:
                        best[ri] = new_row
                        best_val = new_max
                        best_used = compute_all_used(best)
                        improved = True
            if not improved:
                break
        return best, best_val, best_used

    # Start with greedy
    rng = random.Random(42)
    rows = greedy_init(rng)
    best = [r[:] for r in rows]
    best_val = global_max(best)
    best_used = compute_all_used(best)

    cur = [r[:] for r in best]
    cur_val = best_val
    cur_row_diffs = [row_diffs_set(r) for r in cur]
    cur_used = set(best_used)
    cur_row_maxes = [max(r) for r in cur]

    # Main search loop: iterative local search with restarts
    # Use multi-start strategy
    n_starts = 5
    for start_i in range(n_starts):
        if time.time() > deadline - 10:
            break

        if start_i == 0:
            cur = [r[:] for r in best]
            cur_val = best_val
            cur_used = set(best_used)
            cur_row_diffs = [row_diffs_set(r) for r in cur]
            cur_row_maxes = [max(r) for r in cur]
            rng_local = random.Random(42)
        else:
            rng_local = random.Random(42 + start_i * 1000)
            cur = greedy_init(rng_local, randomize=True)
            if not cur:
                continue
            cur_val = global_max(cur)
            cur_used = compute_all_used(cur)
            cur_row_diffs = [row_diffs_set(r) for r in cur]
            cur_row_maxes = [max(r) for r in cur]

        T = 15.0
        T_min = 0.1
        cool = 0.9995

        no_improve = 0
        max_no_improve = 50000

        while time.time() < deadline - 5 and no_improve < max_no_improve:
            # Preferentially target the row that achieves the global max
            if rng_local.random() < 0.7:
                max_rows = [i for i in range(n_rows) if cur_row_maxes[i] == cur_val]
                if max_rows:
                    ri = rng_local.choice(max_rows)
                else:
                    max_rows = [i for i in range(n_rows) if cur_row_maxes[i] >= cur_val - 3]
                    ri = rng_local.choice(max_rows) if max_rows else rng_local.randrange(n_rows)
            elif rng_local.random() < 0.5:
                # Target second-highest rows
                sorted_rows = sorted(range(n_rows), key=lambda i: cur_row_maxes[i], reverse=True)
                ri = rng_local.choice(sorted_rows[:max(1, n_rows // 3)])
            else:
                ri = rng_local.randrange(n_rows)

            forbidden = cur_used - cur_row_diffs[ri]

            # Try to find a new row
            # First try with current max (or slightly less for improvement)
            target_max = cur_val
            if rng_local.random() < 0.4:
                target_max = cur_val - 1  # Try to improve

            new_row = random_valid_row(forbidden, target_max, rng_local, max_candidates=15)

            if new_row is None:
                # Try with slightly larger limit
                new_row = random_valid_row(forbidden, cur_val + 5, rng_local, max_candidates=10)

            if new_row is None:
                no_improve += 1
                T = max(T_min, T * cool)
                continue

            other_max = max_except(cur, ri)
            new_max = max(other_max, new_row[-1])
            delta_e = new_max - cur_val

            accept = False
            if delta_e < 0:
                accept = True
            elif delta_e == 0:
                if rng_local.random() < 0.4:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta_e / T)
                    if rng_local.random() < prob:
                        accept = True

            if accept:
                old_diffs = cur_row_diffs[ri]
                cur_used -= old_diffs
                new_diffs = row_diffs_set(new_row)
                cur_used |= new_diffs
                cur_row_diffs[ri] = new_diffs

                cur[ri] = new_row
                cur_row_maxes[ri] = new_row[-1]
                cur_val = new_max

                if cur_val < best_val:
                    best_val = cur_val
                    best = [r[:] for r in cur]
                    best_used = set(cur_used)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

            T = max(T_min, T * cool)

            # Restart from best if stuck
            if no_improve > 5000:
                cur = [r[:] for r in best]
                cur_val = best_val
                cur_used = set(best_used)
                cur_row_diffs = [row_diffs_set(r) for r in cur]
                cur_row_maxes = [max(r) for r in cur]
                T = 15.0
                no_improve = 0

        # After each start, do local search polish
        if time.time() < deadline - 10:
            polished, pol_val, pol_used = local_search_minimize(best, deadline, rng_local)
            if pol_val < best_val:
                best_val = pol_val
                best = polished
                best_used = pol_used

    # Final polish
    if time.time() < deadline - 3:
        polished, pol_val, pol_used = local_search_minimize(best, deadline, random.Random(99))
        if pol_val < best_val:
            best_val = pol_val
            best = polished
            best_used = pol_used

    # Validate
    if not validate(best):
        best = greedy_init(random.Random(42))

    return best

# EVOLVE-BLOCK-END
