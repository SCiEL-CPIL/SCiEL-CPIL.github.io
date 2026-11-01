import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135.0

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        return max(max(r) for r in rows)

    def validate(rows):
        used = 0
        for r in rows:
            if len(r) != row_len:
                return False
            if r[0] != 0:
                return False
            for i in range(1, len(r)):
                if r[i] <= r[i-1]:
                    return False
            for i in range(len(r)):
                for j in range(i):
                    d = r[i] - r[j]
                    bit = 1 << (d - 1)
                    if used & bit:
                        return False
                    used |= bit
        return True

    def greedy_init(rand_prob=0.1, window=20, seed=None):
        if seed is not None:
            random.seed(seed)
        used_diffs = 0
        rows = []

        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                candidates = []
                x = last + 1
                limit = 200
                while x <= limit and len(candidates) < window:
                    valid = True
                    for a in row:
                        d = x - a
                        if used_diffs & (1 << (d - 1)):
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                    x += 1

                if not candidates:
                    while True:
                        x += 1
                        valid = True
                        for a in row:
                            d = x - a
                            if used_diffs & (1 << (d - 1)):
                                valid = False
                                break
                        if valid:
                            candidates.append(x)
                            break

                if random.random() < rand_prob and len(candidates) > 1:
                    x = random.choice(candidates)
                else:
                    x = candidates[0]

                row.append(x)
                for a in row[:-1]:
                    d = x - a
                    used_diffs |= (1 << (d - 1))

            rows.append(row)

        return rows

    def local_search(rows, time_limit):
        start = time.time()
        best_rows = [r[:] for r in rows]
        best_scope = get_scope(best_rows)
        stuck_counter = 0
        no_improve_counter = 0

        while time.time() < start + time_limit:
            # Find global max element
            max_val = -1
            max_r_idx = -1
            max_c_idx = -1
            for r_idx, r in enumerate(best_rows):
                for c_idx, val in enumerate(r):
                    if val > max_val:
                        max_val = val
                        max_r_idx = r_idx
                        max_c_idx = c_idx

            if max_c_idx == 0:
                break

            current_val = max_val
            prev_val = best_rows[max_r_idx][max_c_idx - 1]

            # Build bitset of all diffs
            all_diffs = 0
            for r in best_rows:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        all_diffs |= (1 << (r[j] - r[i] - 1))

            # Diffs involving target element
            remove_set = 0
            row = best_rows[max_r_idx]
            for i in range(max_c_idx):
                remove_set |= (1 << (current_val - row[i] - 1))
            for j in range(max_c_idx + 1, len(row)):
                remove_set |= (1 << (row[j] - current_val - 1))

            fixed_diffs = all_diffs & ~remove_set

            found_improvement = False
            for new_val in range(current_val - 1, prev_val, -1):
                valid = True
                new_diffs = 0
                for i in range(max_c_idx):
                    d = new_val - row[i]
                    bit = 1 << (d - 1)
                    if fixed_diffs & bit:
                        valid = False
                        break
                    if new_diffs & bit:
                        valid = False
                        break
                    new_diffs |= bit
                if not valid:
                    continue
                for j in range(max_c_idx + 1, len(row)):
                    d = row[j] - new_val
                    bit = 1 << (d - 1)
                    if fixed_diffs & bit:
                        valid = False
                        break
                    if new_diffs & bit:
                        valid = False
                        break
                    new_diffs |= bit
                if not valid:
                    continue

                row[max_c_idx] = new_val
                found_improvement = True
                new_scope = get_scope(best_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    stuck_counter = 0
                    no_improve_counter = 0
                break

            if found_improvement:
                continue

            # Try to reduce other elements
            improved_any = False
            for r_idx in range(n_rows):
                if improved_any:
                    break
                for c_idx in range(1, row_len):
                    if improved_any:
                        break
                    if r_idx == max_r_idx and c_idx == max_c_idx:
                        continue
                    curr_val = best_rows[r_idx][c_idx]
                    prev_v = best_rows[r_idx][c_idx - 1]
                    next_v = best_rows[r_idx][c_idx + 1] if c_idx < row_len - 1 else curr_val + 1

                    all_diffs = 0
                    for r in best_rows:
                        for i in range(len(r)):
                            for j in range(i + 1, len(r)):
                                all_diffs |= (1 << (r[j] - r[i] - 1))

                    remove_set = 0
                    row = best_rows[r_idx]
                    for i in range(c_idx):
                        remove_set |= (1 << (curr_val - row[i] - 1))
                    for j in range(c_idx + 1, len(row)):
                        remove_set |= (1 << (row[j] - curr_val - 1))

                    fixed_diffs = all_diffs & ~remove_set

                    for new_val in range(curr_val - 1, prev_v, -1):
                        valid = True
                        new_diffs = 0
                        for i in range(c_idx):
                            d = new_val - row[i]
                            bit = 1 << (d - 1)
                            if fixed_diffs & bit:
                                valid = False
                                break
                            if new_diffs & bit:
                                valid = False
                                break
                            new_diffs |= bit
                        if not valid:
                            continue
                        for j in range(c_idx + 1, len(row)):
                            d = row[j] - new_val
                            bit = 1 << (d - 1)
                            if fixed_diffs & bit:
                                valid = False
                                break
                            if new_diffs & bit:
                                valid = False
                                break
                            new_diffs |= bit
                        if not valid:
                            continue

                        row[c_idx] = new_val
                        improved_any = True
                        break

            if improved_any:
                new_scope = get_scope(best_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    stuck_counter = 0
                    no_improve_counter = 0
                continue

            # Stuck: perturb
            no_improve_counter += 1
            stuck_counter += 1

            if stuck_counter > 10 and random.random() < 0.5:
                # Row reconstruction
                r_idx = random.randint(0, n_rows - 1)
                other_rows = [best_rows[i] for i in range(n_rows) if i != r_idx]

                fixed_diffs = 0
                for r in other_rows:
                    for i in range(len(r)):
                        for j in range(i + 1, len(r)):
                            fixed_diffs |= (1 << (r[j] - r[i] - 1))

                success = False
                for attempt in range(10):
                    temp_row = [0]
                    temp_used = 0
                    while len(temp_row) < row_len:
                        last = temp_row[-1]
                        candidates = []
                        x = last + 1
                        limit = min(best_scope, last + 80)
                        while x <= limit and len(candidates) < 30:
                            valid = True
                            for a in temp_row:
                                d = x - a
                                bit = 1 << (d - 1)
                                if (fixed_diffs | temp_used) & bit:
                                    valid = False
                                    break
                            if valid:
                                candidates.append(x)
                            x += 1

                        if not candidates:
                            while True:
                                x += 1
                                valid = True
                                for a in temp_row:
                                    d = x - a
                                    bit = 1 << (d - 1)
                                    if (fixed_diffs | temp_used) & bit:
                                        valid = False
                                        break
                                if valid:
                                    candidates.append(x)
                                    break

                        if random.random() < 0.3 and len(candidates) > 1:
                            x = random.choice(candidates)
                        else:
                            x = candidates[0]

                        temp_row.append(x)
                        for a in temp_row[:-1]:
                            d = x - a
                            temp_used |= (1 << (d - 1))

                    if max(temp_row) <= best_scope:
                        best_rows[r_idx] = temp_row
                        success = True
                        break

                if success:
                    stuck_counter = 0
                    no_improve_counter = 0
                    new_scope = get_scope(best_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
            else:
                # Single element perturbation
                r_idx = random.randint(0, n_rows - 1)
                c_idx = random.randint(1, row_len - 1)

                curr_val = best_rows[r_idx][c_idx]
                prev_v = best_rows[r_idx][c_idx - 1]
                next_v = best_rows[r_idx][c_idx + 1] if c_idx < row_len - 1 else curr_val + 50

                all_diffs = 0
                for r in best_rows:
                    for i in range(len(r)):
                        for j in range(i + 1, len(r)):
                            all_diffs |= (1 << (r[j] - r[i] - 1))

                remove_set = 0
                row = best_rows[r_idx]
                for i in range(c_idx):
                    remove_set |= (1 << (curr_val - row[i] - 1))
                for j in range(c_idx + 1, len(row)):
                    remove_set |= (1 << (row[j] - curr_val - 1))

                fixed_diffs = all_diffs & ~remove_set

                low = prev_v + 1
                high = next_v - 1 if c_idx < row_len - 1 else curr_val + 50

                if low > high:
                    continue

                applied = False
                for _ in range(20):
                    cand = random.randint(low, high)
                    if cand == curr_val:
                        continue

                    valid = True
                    new_diffs = 0
                    for i in range(c_idx):
                        d = cand - row[i]
                        bit = 1 << (d - 1)
                        if fixed_diffs & bit:
                            valid = False
                            break
                        if new_diffs & bit:
                            valid = False
                            break
                        new_diffs |= bit
                    if not valid:
                        continue
                    for j in range(c_idx + 1, len(row)):
                        d = row[j] - cand
                        bit = 1 << (d - 1)
                        if fixed_diffs & bit:
                            valid = False
                            break
                        if new_diffs & bit:
                            valid = False
                            break
                        new_diffs |= bit
                    if not valid:
                        continue

                    row[c_idx] = cand
                    applied = True
                    break

                if applied:
                    stuck_counter = 0

        return best_rows, best_scope

    # Phase 1: Diverse initialization
    best_rows = None
    best_scope = float('inf')

    num_inits = 20
    init_time_limit = 5.0
    start_init = time.time()

    for i in range(num_inits):
        if time.time() > start_init + init_time_limit:
            break
        rows = greedy_init(rand_prob=0.03 + 0.01 * i, window=30, seed=i)
        scope = get_scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = rows

    # Phase 2: Local search
    local_time = 50.0
    if time.time() > deadline - 70:
        local_time = deadline - time.time() - 15

    if local_time > 10 and best_rows:
        refined_rows, refined_scope = local_search(best_rows, local_time)
        if validate(refined_rows):
            best_rows = refined_rows
            best_scope = refined_scope

    # Phase 2.5: Polish pass
    if best_rows and time.time() < deadline - 20:
        polish_time = min(12.0, deadline - time.time() - 10)
        if polish_time > 3:
            polish_rows = [r[:] for r in best_rows]
            polish_scope = best_scope

            start_p = time.time()
            improved = True
            while improved and time.time() < start_p + polish_time:
                improved = False
                for r_idx in range(n_rows):
                    for c_idx in range(1, row_len):
                        if time.time() > start_p + polish_time:
                            break
                        curr_val = polish_rows[r_idx][c_idx]
                        prev_v = polish_rows[r_idx][c_idx - 1]
                        next_v = polish_rows[r_idx][c_idx + 1] if c_idx < row_len - 1 else curr_val + 1

                        all_diffs = 0
                        for r in polish_rows:
                            for i in range(len(r)):
                                for j in range(i + 1, len(r)):
                                    all_diffs |= (1 << (r[j] - r[i] - 1))

                        remove_set = 0
                        row = polish_rows[r_idx]
                        for i in range(c_idx):
                            remove_set |= (1 << (curr_val - row[i] - 1))
                        for j in range(c_idx + 1, len(row)):
                            remove_set |= (1 << (row[j] - curr_val - 1))

                        fixed_diffs = all_diffs & ~remove_set

                        for new_val in range(curr_val - 1, prev_v, -1):
                            valid = True
                            new_diffs = 0
                            for i in range(c_idx):
                                d = new_val - row[i]
                                bit = 1 << (d - 1)
                                if fixed_diffs & bit:
                                    valid = False
                                    break
                                if new_diffs & bit:
                                    valid = False
                                    break
                                new_diffs |= bit
                            if not valid:
                                continue
                            for j in range(c_idx + 1, len(row)):
                                d = row[j] - new_val
                                bit = 1 << (d - 1)
                                if fixed_diffs & bit:
                                    valid = False
                                    break
                                if new_diffs & bit:
                                    valid = False
                                    break
                                new_diffs |= bit
                            if not valid:
                                continue

                            row[c_idx] = new_val
                            improved = True
                            new_scope = max(max(r) for r in polish_rows)
                            if new_scope < polish_scope:
                                polish_scope = new_scope
                            break
                    if time.time() > start_p + polish_time:
                        break

            if validate(polish_rows):
                if polish_scope < best_scope:
                    best_rows = polish_rows
                    best_scope = polish_scope

    # Phase 3: Backtracking to beat current best
    current_target = best_scope - 1
    seed_counter = 0

    def backtracking_search(n_rows, k, target_scope, time_limit, seed=0):
        random.seed(seed)
        start = time.time()

        rows = [[0] * row_len for _ in range(n_rows)]
        used_diffs = 0

        def solve(row_idx, pos):
            if time.time() > start + time_limit:
                return False

            if row_idx == n_rows:
                return True

            if pos == row_len:
                return solve(row_idx + 1, 1)

            prev_val = rows[row_idx][pos - 1]
            lower = prev_val + 1
            upper = target_scope

            if lower > upper:
                return False

            candidates = []
            scan_limit = min(upper, lower + 100)
            for val in range(lower, scan_limit + 1):
                valid = True
                for i in range(pos):
                    d = val - rows[row_idx][i]
                    if used_diffs & (1 << (d - 1)):
                        valid = False
                        break
                if valid:
                    candidates.append(val)
                    if len(candidates) >= 20:
                        break

            if not candidates:
                for val in range(scan_limit + 1, upper + 1):
                    valid = True
                    for i in range(pos):
                        d = val - rows[row_idx][i]
                        if used_diffs & (1 << (d - 1)):
                            valid = False
                            break
                    if valid:
                        candidates.append(val)
                        if len(candidates) >= 20:
                            break

            if not candidates:
                return False

            random.shuffle(candidates)
            candidates.sort()

            for val in candidates:
                new_diffs = 0
                for i in range(pos):
                    d = val - rows[row_idx][i]
                    new_diffs |= (1 << (d - 1))
                used_diffs |= new_diffs

                rows[row_idx][pos] = val

                if solve(row_idx, pos + 1):
                    return True

                rows[row_idx][pos] = 0
                used_diffs ^= new_diffs

            return False

        if solve(0, 1):
            return [r[:] for r in rows]
        return None

    while time.time() < deadline - 5:
        if current_target < 112:
            break

        search_time = 4.0
        if time.time() > deadline - 12:
            search_time = deadline - time.time() - 2

        if search_time < 1.0:
            break

        result = backtracking_search(n_rows, k, current_target, search_time, seed=seed_counter)
        seed_counter += 1

        if result and validate(result):
            new_scope = get_scope(result)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = result
                current_target = best_scope - 1
        else:
            if time.time() < deadline - 8:
                result_retry = backtracking_search(n_rows, k, current_target, 4.0, seed=seed_counter)
                seed_counter += 1
                if result_retry and validate(result_retry):
                    new_scope = get_scope(result_retry)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = result_retry
                        current_target = best_scope - 1
                break
            else:
                break

    return best_rows

if __name__ == "__main__":
    rows = construct(4, 7)
    print("Scope:", get_scope(rows))
    diffs = set()
    valid = True
    for r in rows:
        if r[0] != 0:
            valid = False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d in diffs:
                    valid = False
                diffs.add(d)
    print("Valid:", valid)
