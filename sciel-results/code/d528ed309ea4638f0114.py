import random
import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit

    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs

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

    def greedy_row(max_val, global_used_set):
        row = [0]
        current_diffs = set()
        while len(row) < k + 1:
            last = row[-1]
            found = False
            for v in range(last + 1, max_val + 1):
                conflict = False
                new_diffs = set()
                for u in row:
                    d = v - u
                    if d in current_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                if conflict:
                    continue
                if new_diffs & global_used_set:
                    continue
                row.append(v)
                current_diffs |= new_diffs
                found = True
                break
            if not found:
                return None
        return row

    def random_row(max_val, global_used_set, attempts=1):
        for _ in range(attempts):
            row = [0]
            current_diffs = set()
            success = True
            for _ in range(k):
                last = row[-1]
                candidates = list(range(last + 1, max_val + 1))
                random.shuffle(candidates)
                found = False
                for v in candidates:
                    conflict = False
                    new_diffs = set()
                    for u in row:
                        d = v - u
                        if d in current_diffs:
                            conflict = True
                            break
                        new_diffs.add(d)
                    if conflict:
                        continue
                    if new_diffs & global_used_set:
                        continue
                    row.append(v)
                    current_diffs |= new_diffs
                    found = True
                    break
                if not found:
                    success = False
                    break
            if success:
                valid, _ = is_valid_row(row, global_used_set)
                if valid:
                    return row
        return None

    def backtrack_solve(rows, row_idx, max_val, time_limit):
        if time.time() > time_limit:
            return False
        if row_idx == n_rows:
            return True

        used_before = set()
        for i in range(row_idx):
            used_before |= get_row_diffs_set(rows[i])

        candidates = []

        g_row = greedy_row(max_val, used_before)
        if g_row is not None:
            candidates.append(g_row)

        for _ in range(2):
            if time.time() > time_limit:
                return False
            row = random_row(max_val, used_before, attempts=1)
            if row is not None:
                candidates.append(row)

        if not candidates:
            return False

        random.shuffle(candidates)

        for c_row in candidates:
            rows[row_idx] = c_row
            if backtrack_solve(rows, row_idx + 1, max_val, time_limit):
                return True
        return False

    def find_solution_for_scope(M, time_limit):
        for attempt in range(5):
            if time.time() > time_limit:
                return None
            rows = [None] * n_rows
            if backtrack_solve(rows, 0, M, time_limit):
                return rows
        return None

    # 1. Initial Solution: Greedy with large scope
    init_scope = 250
    rows = [None] * n_rows
    success = False

    if time.time() < deadline - 20:
        for _ in range(5):
            if time.time() > deadline - 20:
                break
            rows = [None] * n_rows
            if backtrack_solve(rows, 0, init_scope, deadline - 20):
                success = True
                break

    if not success:
        rows = []
        global_used = set()
        for i in range(n_rows):
            row = greedy_row(500, global_used)
            if row is None:
                row = [0] + list(range(1, k + 1))
            rows.append(row)
            global_used |= get_row_diffs_set(row)

    best_rows = [list(r) for r in rows]
    best_max = get_max_entry(best_rows)
    current_rows = [list(r) for r in rows]
    current_max = best_max

    # 2. Optimization Loop: Binary Search for Minimum Max
    low = lower_bound
    high = best_max

    search_start_time = time.time()
    remaining_time = deadline - search_start_time - 5.0

    if remaining_time > 10:
        while low < high and (time.time() - search_start_time) < remaining_time - 5:
            mid = (low + high) // 2
            check_time_limit = time.time() + (remaining_time - (time.time() - search_start_time)) * 0.5
            sol = find_solution_for_scope(mid, check_time_limit)
            if sol is not None:
                best_rows = [list(r) for r in sol]
                best_max = get_max_entry(best_rows)
                high = mid
            else:
                low = mid + 1

        if best_rows:
            current_rows = best_rows
            current_max = best_max

    # 3. Local Search to Polish
    step = 0
    no_improve_count = 0

    while time.time() < deadline - 2.0:
        step += 1

        row_idx = random.randint(0, n_rows - 1)

        global_used_except = set()
        for i in range(n_rows):
            if i != row_idx:
                global_used_except |= get_row_diffs_set(current_rows[i])

        curr_row_max = current_rows[row_idx][-1]

        target = curr_row_max - 1

        if target < k:
            continue

        new_row = None
        for _ in range(10):
            if time.time() > deadline - 2.0:
                break
            new_row = random_row(target, global_used_except, attempts=1)
            if new_row is not None:
                break

        if new_row is not None:
            valid, _ = is_valid_row(new_row, global_used_except)
            if valid:
                current_rows[row_idx] = new_row
                new_global_max = get_max_entry(current_rows)
                if new_global_max < best_max:
                    best_max = new_global_max
                    best_rows = [list(r) for r in current_rows]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        else:
            no_improve_count += 1

    # 4. Simulated Annealing Polish (from inspiration)
    # This is a structural change: accepting worse solutions to escape local minima
    rng = random.Random(42)
    temp = 1.0
    cooling_rate = 0.998
    min_temp = 0.01

    # Rebuild the current rows and used diffs
    used_diffs = get_all_diffs_set(current_rows)
    rows_sa = [list(r) for r in current_rows]
    best_rows_sa = [list(r) for r in current_rows]
    best_max_sa = get_max_entry(rows_sa)

    def rebuild_row_sa(k, forbidden_diffs, rng, max_val_limit, pressure_weight=1.0):
        row = [0]
        local_used = set()

        for step in range(k):
            last = row[-1]
            valid_candidates = []

            max_candidates_to_check = 50
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

            valid_candidates.sort(key=lambda t: t[2])

            n_top = min(5, len(valid_candidates))
            top_candidates = valid_candidates[:n_top]

            weights = [1.0 / (1.0 + c[2] * 0.1) for c in top_candidates]
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

    # SA loop
    while time.time() < deadline - 2.0:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed

        polish_mode = remaining < 15.0

        if polish_mode:
            target_max = current_max - 1
            accept_neutral = False
        else:
            target_max = current_max - 1
            accept_neutral = rng.random() < (0.15 if temp > 0.1 else 0.05)

        # Decide whether to do single or multi-row rebuild
        use_multi = (not polish_mode) and (rng.random() < 0.2) and (n_rows >= 2)

        if use_multi:
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(0, n_rows - 1)
            if i == j:
                j = (j + 1) % n_rows

            old_row_i = rows_sa[i]
            old_row_j = rows_sa[j]
            old_diffs_i = get_row_diffs_set(old_row_i)
            old_diffs_j = get_row_diffs_set(old_row_j)

            used_diffs.difference_update(old_diffs_i)
            used_diffs.difference_update(old_diffs_j)

            # Try to rebuild both
            new_row_i = rebuild_row_sa(k, used_diffs, rng, target_max if target_max > k else current_max)
            if new_row_i is None:
                used_diffs.update(old_diffs_i)
                used_diffs.update(old_diffs_j)
                temp *= cooling_rate
                continue

            diffs_i = get_row_diffs_set(new_row_i)
            forbidden_j = used_diffs | diffs_i
            new_row_j = rebuild_row_sa(k, forbidden_j, rng, target_max if target_max > k else current_max)
            if new_row_j is None:
                used_diffs.update(old_diffs_i)
                used_diffs.update(old_diffs_j)
                temp *= cooling_rate
                continue

            new_diffs_j = get_row_diffs_set(new_row_j)

            if (diffs_i & used_diffs) or (new_diffs_j & used_diffs) or (diffs_i & new_diffs_j):
                used_diffs.update(old_diffs_i)
                used_diffs.update(old_diffs_j)
                temp *= cooling_rate
                continue

            rows_sa[i] = new_row_i
            rows_sa[j] = new_row_j
            new_global_max = get_max_entry(rows_sa)

            if new_global_max < best_max_sa:
                used_diffs.update(diffs_i)
                used_diffs.update(new_diffs_j)
                best_max_sa = new_global_max
                best_rows_sa = [r[:] for r in rows_sa]
                temp = max(temp * 1.1, min_temp)
            else:
                if accept_neutral:
                    used_diffs.update(diffs_i)
                    used_diffs.update(new_diffs_j)
                    temp *= cooling_rate
                else:
                    rows_sa[i] = old_row_i
                    rows_sa[j] = old_row_j
                    used_diffs.update(old_diffs_i)
                    used_diffs.update(old_diffs_j)
                    temp *= cooling_rate
        else:
            # Single row rebuild
            i = rng.randint(0, n_rows - 1)
            old_row_i = rows_sa[i]
            old_diffs_i = get_row_diffs_set(old_row_i)

            used_diffs.difference_update(old_diffs_i)

            new_row_i = rebuild_row_sa(k, used_diffs, rng, target_max if target_max > k else current_max)
            if new_row_i is None:
                used_diffs.update(old_diffs_i)
                temp *= cooling_rate
                continue

            new_diffs_i = get_row_diffs_set(new_row_i)

            if new_diffs_i & used_diffs:
                used_diffs.update(old_diffs_i)
                temp *= cooling_rate
                continue

            rows_sa[i] = new_row_i
            new_global_max = get_max_entry(rows_sa)

            if new_global_max < best_max_sa:
                used_diffs.update(new_diffs_i)
                best_max_sa = new_global_max
                best_rows_sa = [r[:] for r in rows_sa]
                temp = max(temp * 1.1, min_temp)
            else:
                if accept_neutral:
                    used_diffs.update(new_diffs_i)
                    temp *= cooling_rate
                else:
                    rows_sa[i] = old_row_i
                    used_diffs.update(old_diffs_i)
                    temp *= cooling_rate

        temp = max(temp * cooling_rate, min_temp)

    # Update best if SA found something better
    if best_max_sa < best_max:
        best_rows = best_rows_sa
        best_max = best_max_sa

    # Final Validation
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

    if not validate(best_rows):
        rows_fallback = []
        gu = set()
        for i in range(n_rows):
            r = greedy_row(1000, gu)
            if r is None:
                r = [0] + list(range(1, k + 1))
            rows_fallback.append(r)
            gu |= get_row_diffs_set(r)
        best_rows = rows_fallback

    return best_rows
