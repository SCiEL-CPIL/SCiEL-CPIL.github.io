import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 132.0  # Safety margin before 150s kill

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2

    def get_scope(rows):
        return max(max(r) for r in rows)

    def validate(rows):
        used = set()
        for r in rows:
            if len(r) != row_len:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                if i > 0 and r[i] <= r[i-1]:
                    return False
                for j in range(i):
                    d = r[i] - r[j]
                    if d <= 0:
                        return False
                    if d in used:
                        return False
                    used.add(d)
        return True

    def greedy_init(rand_prob=0.12, window=25, seed=None):
        if seed is not None:
            random.seed(seed)
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                candidates = []
                x = last + 1
                limit = 250
                while x <= limit and len(candidates) < window:
                    valid = True
                    for a in row:
                        if (x - a) in used_diffs:
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
                            if (x - a) in used_diffs:
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
                    used_diffs.add(x - a)
            rows.append(row)
        return rows

    def local_search(rows, time_limit):
        start = time.time()
        best_rows = [r[:] for r in rows]
        best_scope = get_scope(best_rows)

        def get_all_diffs(rows):
            diffs = set()
            for r in rows:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        diffs.add(r[j] - r[i])
            return diffs

        def try_move(r_idx, c_idx, val, fixed_diffs, row):
            if val <= row[c_idx - 1]:
                return False
            if c_idx < row_len - 1 and val >= row[c_idx + 1]:
                return False
            for i in range(c_idx):
                if (val - row[i]) in fixed_diffs:
                    return False
            for j in range(c_idx + 1, row_len):
                if (row[j] - val) in fixed_diffs:
                    return False
            new_diffs = set()
            for i in range(c_idx):
                d = val - row[i]
                if d in new_diffs:
                    return False
                new_diffs.add(d)
            for j in range(c_idx + 1, row_len):
                d = row[j] - val
                if d in new_diffs:
                    return False
                new_diffs.add(d)
            return True

        def reconstruct_row(r_idx, time_limit):
            """Reconstruct a single row to escape local minima."""
            other_rows = [r[:] for i, r in enumerate(best_rows) if i != r_idx]
            other_diffs = set()
            for r in other_rows:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        other_diffs.add(r[j] - r[i])

            new_row = [0]
            for c in range(1, row_len):
                prev = new_row[-1]
                found = False
                # Try values from prev+1 up to some limit
                for val in range(prev + 1, prev + 200):
                    valid = True
                    for i in range(len(new_row)):
                        d = val - new_row[i]
                        if d in other_diffs or d in [x for x in other_diffs]:
                            valid = False
                            break
                    if valid:
                        # Check internal diffs of new_row
                        internal_ok = True
                        for i in range(len(new_row)):
                            d = val - new_row[i]
                            for j in range(i + 1, len(new_row)):
                                if d == new_row[j] - new_row[i]:
                                    internal_ok = False
                                    break
                            if not internal_ok:
                                break
                        if internal_ok:
                            new_row.append(val)
                            found = True
                            break
                if not found:
                    # If stuck, just extend greedily
                    x = prev + 1
                    while True:
                        valid = True
                        for i in range(len(new_row)):
                            d = x - new_row[i]
                            if d in other_diffs:
                                valid = False
                                break
                        if valid:
                            internal_ok = True
                            for i in range(len(new_row)):
                                d = x - new_row[i]
                                for j in range(i + 1, len(new_row)):
                                    if d == new_row[j] - new_row[i]:
                                        internal_ok = False
                                        break
                                if not internal_ok:
                                    break
                            if internal_ok:
                                new_row.append(x)
                                break
                        x += 1
            best_rows[r_idx] = new_row
            return new_row

        while time.time() < start + time_limit:
            # Phase A: Targeted descent on the global max
            max_val = -1
            max_r = -1
            max_c = -1
            for r_idx, r in enumerate(best_rows):
                for c_idx, val in enumerate(r):
                    if val > max_val:
                        max_val = val
                        max_r = r_idx
                        max_c = c_idx

            if max_c == 0:
                break

            row = best_rows[max_r]
            all_diffs = get_all_diffs(best_rows)
            remove_set = set()
            for i in range(max_c):
                remove_set.add(max_val - row[i])
            for j in range(max_c + 1, row_len):
                remove_set.add(row[j] - max_val)
            fixed_diffs = all_diffs - remove_set

            moved = False
            for new_val in range(max_val - 1, row[max_c - 1], -1):
                if try_move(max_r, max_c, new_val, fixed_diffs, row):
                    row[max_c] = new_val
                    moved = True
                    best_scope = get_scope(best_rows)
                    break

            if moved:
                continue

            # Phase B: Try all elements for descent
            improved = False
            for r_idx in range(n_rows):
                for c_idx in range(1, row_len):
                    if r_idx == max_r and c_idx == max_c:
                        continue
                    row = best_rows[r_idx]
                    curr = row[c_idx]
                    all_diffs = get_all_diffs(best_rows)
                    remove_set = set()
                    for i in range(c_idx):
                        remove_set.add(curr - row[i])
                    for j in range(c_idx + 1, row_len):
                        remove_set.add(row[j] - curr)
                    fixed_diffs = all_diffs - remove_set

                    for new_val in range(curr - 1, row[c_idx - 1], -1):
                        if try_move(r_idx, c_idx, new_val, fixed_diffs, row):
                            row[c_idx] = new_val
                            improved = True
                            best_scope = get_scope(best_rows)
                            break
                    if improved:
                        break
                if improved:
                    break

            if improved:
                continue

            # Phase C: Perturbation - random move or row reconstruction
            if random.random() < 0.3:
                # Reconstruct a random row
                r_idx = random.randint(0, n_rows - 1)
                reconstruct_row(r_idx, 1.0)
            else:
                # Random perturbation of a random element
                r_idx = random.randint(0, n_rows - 1)
                c_idx = random.randint(1, row_len - 1)
                row = best_rows[r_idx]
                curr = row[c_idx]
                all_diffs = get_all_diffs(best_rows)
                remove_set = set()
                for i in range(c_idx):
                    remove_set.add(curr - row[i])
                for j in range(c_idx + 1, row_len):
                    remove_set.add(row[j] - curr)
                fixed_diffs = all_diffs - remove_set

                low = row[c_idx - 1] + 1
                high = row[c_idx + 1] - 1 if c_idx < row_len - 1 else curr + 30
                for _ in range(20):
                    if time.time() > start + time_limit:
                        break
                    cand = random.randint(low, high)
                    if cand == curr:
                        continue
                    if try_move(r_idx, c_idx, cand, fixed_diffs, row):
                        row[c_idx] = cand
                        break

        # Final polish: exhaustive single-element optimization
        for _ in range(3):
            if time.time() > start + time_limit:
                break
            for r_idx in range(n_rows):
                for c_idx in range(1, row_len):
                    if time.time() > start + time_limit:
                        break
                    row = best_rows[r_idx]
                    curr = row[c_idx]
                    all_diffs = get_all_diffs(best_rows)
                    remove_set = set()
                    for i in range(c_idx):
                        remove_set.add(curr - row[i])
                    for j in range(c_idx + 1, row_len):
                        remove_set.add(row[j] - curr)
                    fixed_diffs = all_diffs - remove_set
                    for new_val in range(curr - 1, row[c_idx - 1], -1):
                        if try_move(r_idx, c_idx, new_val, fixed_diffs, row):
                            row[c_idx] = new_val
                            break

        return best_rows, get_scope(best_rows)

    def mrv_backtracking(n_rows, k, target_scope, time_limit, seed=0):
        random.seed(seed)
        start = time.time()

        rows = [[-1] * row_len for _ in range(n_rows)]
        for r in range(n_rows):
            rows[r][0] = 0

        used_diffs = set()

        def add_diff(d):
            used_diffs.add(d)

        def remove_diff(d):
            used_diffs.discard(d)

        def get_candidates(r, c, max_val):
            if rows[r][c] != -1:
                return [rows[r][c]]
            prev_val = rows[r][c - 1] if c > 0 else -1
            lower = prev_val + 1
            upper = min(max_val, target_scope)
            if lower > upper:
                return []
            candidates = []
            scan_limit = min(upper, lower + 60)
            for val in range(lower, scan_limit + 1):
                valid = True
                for i in range(c):
                    if rows[r][i] == -1:
                        continue
                    if (val - rows[r][i]) in used_diffs:
                        valid = False
                        break
                if valid:
                    candidates.append(val)
                    if len(candidates) >= 12:
                        break
            if not candidates and scan_limit < upper:
                for val in range(scan_limit + 1, upper + 1):
                    valid = True
                    for i in range(c):
                        if rows[r][i] == -1:
                            continue
                        if (val - rows[r][i]) in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(val)
                        if len(candidates) >= 12:
                            break
            return candidates

        def count_unfilled():
            count = 0
            for r in range(n_rows):
                for c in range(1, row_len):
                    if rows[r][c] == -1:
                        count += 1
            return count

        def solve():
            if time.time() > start + time_limit:
                return False
            if count_unfilled() == 0:
                return True

            best_pos = None
            best_candidates = None
            best_count = float('inf')

            for r in range(n_rows):
                for c in range(1, row_len):
                    if rows[r][c] == -1:
                        cands = get_candidates(r, c, target_scope)
                        if len(cands) == 0:
                            return False
                        if len(cands) < best_count:
                            best_count = len(cands)
                            best_pos = (r, c)
                            best_candidates = cands
                            if best_count == 1:
                                break
                if best_count == 1:
                    break

            if best_pos is None:
                return True

            r, c = best_pos
            random.shuffle(best_candidates)

            for val in best_candidates:
                rows[r][c] = val
                new_diffs = []
                for i in range(c):
                    if rows[r][i] != -1:
                        d = val - rows[r][i]
                        new_diffs.append(d)
                        add_diff(d)

                if solve():
                    return True

                rows[r][c] = -1
                for d in new_diffs:
                    remove_diff(d)

            return False

        if solve():
            return [r[:] for r in rows]
        return None

    # Phase 1: Greedy baseline
    best_rows = greedy_init(0.12, 25, seed=42)
    best_scope = get_scope(best_rows)

    # Phase 2: Local search refinement
    local_time = 50.0
    elapsed = time.time() - start_time
    if elapsed > deadline - 60:
        local_time = max(10, deadline - elapsed - 30)

    if local_time > 5:
        refined_rows, refined_scope = local_search(best_rows, local_time)
        if validate(refined_rows):
            best_rows = refined_rows
            best_scope = refined_scope

    # Phase 3: MRV backtracking to try to beat current best
    current_target = best_scope - 1
    seed_counter = 0

    while time.time() < deadline - 5:
        if current_target < 112:
            break

        search_time = 8.0
        remaining = deadline - time.time() - 3
        if remaining < search_time:
            search_time = max(1.0, remaining)

        if search_time < 1.0:
            break

        result = mrv_backtracking(n_rows, k, current_target, search_time, seed=seed_counter)
        seed_counter += 1

        if result:
            if validate(result):
                new_scope = get_scope(result)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = result
                    current_target = best_scope - 1
                else:
                    break
            else:
                break
        else:
            if time.time() < deadline - 8:
                result_retry = mrv_backtracking(n_rows, k, current_target, 5.0, seed=seed_counter)
                seed_counter += 1
                if result_retry and validate(result_retry):
                    new_scope = get_scope(result_retry)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = result_retry
                        current_target = best_scope - 1
                    else:
                        break
                else:
                    break
            else:
                break

    # Phase 4: Final polish
    if time.time() < deadline - 2:
        polish_time = min(10.0, deadline - time.time() - 2)
        if polish_time > 2:
            polished_rows, polished_scope = local_search(best_rows, polish_time)
            if validate(polished_rows):
                best_rows = polished_rows
                best_scope = polished_scope

    return best_rows

if __name__ == "__main__":
    rows = construct(4, 7)
    print("Scope:", max(max(r) for r in rows))
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
