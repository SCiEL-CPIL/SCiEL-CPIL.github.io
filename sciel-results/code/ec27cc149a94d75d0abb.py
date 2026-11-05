import random
import time
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs

    # Helper functions
    def get_max_entry(rows):
        return max(max(row) for row in rows)

    def get_row_diffs_set(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def get_all_diffs_set(rows):
        diffs = set()
        for row in rows:
            diffs |= get_row_diffs_set(row)
        return diffs

    def is_valid_row(row, global_used_set):
        row_diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in row_diffs:
                    return False, set()
                row_diffs.add(d)
        if row_diffs & global_used_set:
            return False, set()
        return True, row_diffs

    # Fast greedy initial solution
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

    # Rebuild a single row with pressure-based candidate scoring
    def rebuild_row(k, forbidden_diffs, rng, max_val_limit, pressure_weight=1.0):
        row = [0]
        local_used = set()

        for step in range(k):
            last = row[-1]
            valid_candidates = []

            # Limit search to avoid excessive time
            max_candidates_to_check = 60
            checked = 0

            for x in range(last + 1, max_val_limit + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & local_used) and not (new_diffs & forbidden_diffs):
                    max_new_diff = max(new_diffs)
                    score = x * (1.0 - pressure_weight) + max_new_diff * pressure_weight
                    valid_candidates.append((x, new_diffs, score))
                    checked += 1
                    if checked >= max_candidates_to_check:
                        break

            if not valid_candidates:
                return None

            # Sort by score and pick from top few with some randomness
            valid_candidates.sort(key=lambda t: t[2])
            n_top = min(8, len(valid_candidates))
            top_candidates = valid_candidates[:n_top]

            # Weighted random selection favoring lower scores
            weights = [1.0 / (1.0 + c[2] * 0.05) for c in top_candidates]
            total_w = sum(weights)
            r = rng.random() * total_w
            cumulative = 0.0
            chosen = top_candidates[0]
            for c, w in zip(top_candidates, weights):
                cumulative += w
                if r <= cumulative:
                    chosen = c
                    break

            x, new_diffs, _ = chosen
            row.append(x)
            local_used.update(new_diffs)

            if x > max_val_limit:
                return None

        return row

    # Rebuild two rows simultaneously
    def rebuild_two_rows(k, forbidden_diffs, rng, max_val_limit):
        row_i_new = rebuild_row(k, forbidden_diffs, rng, max_val_limit)
        if row_i_new is None:
            return None, None

        diffs_i = get_row_diffs_set(row_i_new)
        forbidden_j = forbidden_diffs | diffs_i
        row_j_new = rebuild_row(k, forbidden_j, rng, max_val_limit)
        if row_j_new is None:
            return None, None

        return row_i_new, row_j_new

    # Main optimization loop with simulated annealing
    def optimize(rows, used_diffs, current_max):
        rng = random.Random(42)
        best_rows = [r[:] for r in rows]
        best_max = current_max
        current_max = current_max

        # Simulated annealing parameters
        initial_temp = 1.0
        cooling_rate = 0.998
        min_temp = 0.01
        temp = initial_temp

        no_improve_count = 0
        total_moves = 0

        while time.time() < deadline:
            elapsed = time.time() - start_time
            remaining = deadline - elapsed

            # Check if we should switch to polish mode (last 10 seconds)
            polish_mode = remaining < 10.0

            if polish_mode:
                target_max = current_max - 1
                accept_neutral = False
                pressure_weight = 0.8
            else:
                target_max = current_max - 1
                accept_neutral = rng.random() < (0.2 if temp > 0.1 else 0.05)
                pressure_weight = 0.6

            # Decide whether to do single or multi-row rebuild
            use_multi = (not polish_mode) and (rng.random() < 0.25) and (n_rows >= 2)

            if use_multi:
                # Pick two distinct rows
                i = rng.randint(0, n_rows - 1)
                j = rng.randint(0, n_rows - 1)
                if i == j:
                    j = (j + 1) % n_rows

                old_row_i = rows[i]
                old_row_j = rows[j]
                old_diffs_i = get_row_diffs_set(old_row_i)
                old_diffs_j = get_row_diffs_set(old_row_j)

                # Remove both from used_diffs
                used_diffs.difference_update(old_diffs_i)
                used_diffs.difference_update(old_diffs_j)

                # Try to rebuild both
                limit = target_max if target_max > k else current_max
                new_row_i, new_row_j = rebuild_two_rows(k, used_diffs, rng, limit)

                if new_row_i is None or new_row_j is None:
                    # Failed, restore
                    used_diffs.update(old_diffs_i)
                    used_diffs.update(old_diffs_j)
                    no_improve_count += 1
                    temp *= cooling_rate
                    total_moves += 1
                    continue

                new_diffs_i = get_row_diffs_set(new_row_i)
                new_diffs_j = get_row_diffs_set(new_row_j)

                # Verify no conflicts
                if (new_diffs_i & used_diffs) or (new_diffs_j & used_diffs) or (new_diffs_i & new_diffs_j):
                    used_diffs.update(old_diffs_i)
                    used_diffs.update(old_diffs_j)
                    no_improve_count += 1
                    temp *= cooling_rate
                    total_moves += 1
                    continue

                # Calculate new global max
                rows[i] = new_row_i
                rows[j] = new_row_j
                new_global_max = get_max_entry(rows)

                if new_global_max < current_max:
                    used_diffs.update(new_diffs_i)
                    used_diffs.update(new_diffs_j)
                    current_max = new_global_max
                    best_rows = [r[:] for r in rows]
                    best_max = current_max
                    no_improve_count = 0
                    temp = max(temp * 1.15, min_temp)
                else:
                    if accept_neutral:
                        used_diffs.update(new_diffs_i)
                        used_diffs.update(new_diffs_j)
                        no_improve_count += 1
                    else:
                        # Reject
                        rows[i] = old_row_i
                        rows[j] = old_row_j
                        used_diffs.update(old_diffs_i)
                        used_diffs.update(old_diffs_j)
                        no_improve_count += 1
                    temp *= cooling_rate
                total_moves += 1

            else:
                # Single row rebuild
                i = rng.randint(0, n_rows - 1)
                old_row = rows[i]
                old_diffs = get_row_diffs_set(old_row)

                # Remove from used_diffs
                used_diffs.difference_update(old_diffs)

                limit = target_max if target_max > k else current_max
                new_row = rebuild_row(k, used_diffs, rng, limit)

                if new_row is None:
                    # Failed, restore
                    used_diffs.update(old_diffs)
                    no_improve_count += 1
                    temp *= cooling_rate
                    total_moves += 1
                    continue

                new_diffs = get_row_diffs_set(new_row)

                # Verify no conflicts
                if new_diffs & used_diffs:
                    used_diffs.update(old_diffs)
                    no_improve_count += 1
                    temp *= cooling_rate
                    total_moves += 1
                    continue

                # Calculate new global max
                rows[i] = new_row
                new_global_max = get_max_entry(rows)

                if new_global_max < current_max:
                    used_diffs.update(new_diffs)
                    current_max = new_global_max
                    best_rows = [r[:] for r in rows]
                    best_max = current_max
                    no_improve_count = 0
                    temp = max(temp * 1.15, min_temp)
                else:
                    if accept_neutral:
                        used_diffs.update(new_diffs)
                        no_improve_count += 1
                    else:
                        # Reject
                        rows[i] = old_row
                        used_diffs.update(old_diffs)
                        no_improve_count += 1
                    temp *= cooling_rate
                total_moves += 1

            # Early exit if we've hit the lower bound
            if current_max <= lower_bound:
                break

            # If no improvement for a long time, reheat
            if no_improve_count > 100:
                temp = max(temp * 2.0, min_temp)
                no_improve_count = 0

        return best_rows, best_max

    # Final validation
    def validate(rows):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i] >= row[i + 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    # 1. Initial Solution: Fast Greedy
    rows, used_diffs = greedy_initial()
    current_max = get_max_entry(rows)

    # 2. Optimization Loop: Simulated Annealing
    best_rows, best_max = optimize(rows, used_diffs, current_max)

    # 3. Final Polish: Try to squeeze out one more improvement
    if time.time() < deadline - 2.0:
        rng = random.Random(99)
        polish_rows = [r[:] for r in best_rows]
        polish_diffs = get_all_diffs_set(polish_rows)
        polish_max = best_max

        for _ in range(50):
            if time.time() > deadline - 2.0:
                break
            i = rng.randint(0, n_rows - 1)
            old_diffs = get_row_diffs_set(polish_rows[i])
            polish_diffs.difference_update(old_diffs)

            target = polish_max - 1
            if target > k:
                new_row = rebuild_row(k, polish_diffs, rng, target, pressure_weight=0.9)
                if new_row is not None:
                    new_diffs = get_row_diffs_set(new_row)
                    if not (new_diffs & polish_diffs):
                        polish_rows[i] = new_row
                        polish_diffs.update(new_diffs)
                        new_max = get_max_entry(polish_rows)
                        if new_max < best_max:
                            best_rows = polish_rows
                            best_max = new_max
                            break
            polish_diffs.update(old_diffs)

    # Final Validation
    if not validate(best_rows):
        # Fallback to a known valid structure if something went wrong
        rows_fallback, _ = greedy_initial()
        best_rows = rows_fallback

    return best_rows
