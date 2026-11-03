import time
import random
import math

def construct(n_rows, k):
    row_len = k + 1
    deadline = time.time() + 118.0
    best_max = float('inf')
    best_rows = None

    # --- Fast helpers ---
    def get_diffs(row):
        diffs = []
        L = len(row)
        for i in range(L):
            ri = row[i]
            for j in range(i + 1, L):
                diffs.append(row[j] - ri)
        return diffs

    def row_max(row):
        return row[-1]

    def sum_entries(rows):
        return sum(sum(r) for r in rows)

    # --- Greedy Baseline ---
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    baseline_rows = greedy_construct()
    best_rows = baseline_rows
    best_max = max(max(r) for r in best_rows)
    best_sum = sum_entries(best_rows)

    # --- SA with collision tracking ---
    def sa_optimize(initial_rows, time_limit, bound):
        rows = [r[:] for r in initial_rows]
        n = n_rows
        L = row_len

        row_diffs = [get_diffs(r) for r in rows]

        diff_counts = {}
        collision_count = 0
        for rd in row_diffs:
            for d in rd:
                if d in diff_counts:
                    if diff_counts[d] == 1:
                        collision_count += 1
                    diff_counts[d] += 1
                else:
                    diff_counts[d] = 1

        def compute_max_val():
            return max(r[-1] for r in rows)

        def compute_sum():
            return sum(sum(r) for r in rows)

        current_max_val = compute_max_val()
        current_sum = compute_sum()
        # Cost: collisions dominate, then max, then sum (for tie-breaking)
        current_cost = collision_count * 100000 + current_max_val * 100 + current_sum
        best_local_cost = current_cost
        best_local_rows = [r[:] for r in rows]
        best_local_max = current_max_val
        best_local_sum = current_sum

        T = 50.0
        cooling = 0.9997
        start_time = time.time()

        while time.time() - start_time < time_limit:
            r_idx = random.randint(0, n - 1)
            i_idx = random.randint(1, L - 1)

            row = rows[r_idx]
            old_val = row[i_idx]

            lower = row[i_idx - 1] + 1
            upper = row[i_idx + 1] - 1 if i_idx < L - 1 else bound + 15
            upper = min(upper, bound + 25)

            if lower > upper:
                continue

            new_val = random.randint(lower, upper)
            if new_val == old_val:
                continue

            # Remove old diffs
            old_diffs = row_diffs[r_idx]
            for d in old_diffs:
                cnt = diff_counts[d]
                if cnt == 2:
                    collision_count -= 1
                diff_counts[d] = cnt - 1
                if cnt - 1 == 0:
                    del diff_counts[d]

            # Apply change
            row[i_idx] = new_val
            new_diffs = get_diffs(row)
            row_diffs[r_idx] = new_diffs

            # Add new diffs
            for d in new_diffs:
                if d in diff_counts:
                    if diff_counts[d] == 1:
                        collision_count += 1
                    diff_counts[d] += 1
                else:
                    diff_counts[d] = 1

            new_max_val = compute_max_val()
            new_sum = compute_sum()
            new_cost = collision_count * 100000 + new_max_val * 100 + new_sum

            if new_cost < current_cost:
                current_cost = new_cost
                current_max_val = new_max_val
                current_sum = new_sum
                if new_cost < best_local_cost:
                    best_local_cost = new_cost
                    best_local_rows = [r[:] for r in rows]
                    best_local_max = new_max_val
                    best_local_sum = new_sum
            else:
                delta = new_cost - current_cost
                if delta == 0:
                    accept = random.random() < 0.8
                elif delta < 2000:
                    accept = random.random() < math.exp(-delta / T)
                else:
                    accept = False

                if not accept:
                    # Revert
                    row[i_idx] = old_val
                    for d in new_diffs:
                        cnt = diff_counts[d]
                        if cnt == 2:
                            collision_count -= 1
                        diff_counts[d] = cnt - 1
                        if cnt - 1 == 0:
                            del diff_counts[d]
                    for d in old_diffs:
                        if d in diff_counts:
                            if diff_counts[d] == 1:
                                collision_count += 1
                            diff_counts[d] += 1
                        else:
                            diff_counts[d] = 1
                    row_diffs[r_idx] = old_diffs

            T *= cooling
            if T < 0.1:
                T = 0.1

        return best_local_rows, best_local_max, best_local_sum, best_local_cost

    # --- CSP Backtracking ---
    def csp_construct(target_M, time_limit):
        used = set()
        rows = []
        start_time = time.time()

        def dfs(row_idx):
            if time.time() - start_time > time_limit:
                return False
            if row_idx == n_rows:
                return True

            attempts = 0
            while attempts < 120:
                attempts += 1
                if target_M < k:
                    return False

                # Try to find a valid row by sampling
                pts = random.sample(range(1, target_M + 1), k)
                pts.sort()
                row = [0] + pts

                diffs = get_diffs(row)
                if len(set(diffs)) != len(diffs):
                    continue
                if any(d in used for d in diffs):
                    continue

                rows.append(row)
                used.update(diffs)

                if dfs(row_idx + 1):
                    return True
                else:
                    rows.pop()
                    for d in diffs:
                        used.discard(d)

            return False

        if dfs(0):
            return rows
        return None

    # Main Loop
    while time.time() < deadline - 5.0:
        # 1. Try CSP
        for target in range(best_max, best_max - 25, -1):
            if time.time() > deadline - 10.0:
                break
            csp_time = min(2.5, deadline - time.time() - 10.0)
            if csp_time <= 0:
                break

            result = csp_construct(target, csp_time)
            if result is not None:
                new_max = max(r[-1] for r in result)
                if new_max < best_max:
                    best_rows = result
                    best_max = new_max
                    best_sum = sum_entries(best_rows)
                    break

        # 2. Polish with SA — multiple short runs with perturbations
        remaining = deadline - time.time() - 3.0
        if remaining <= 0:
            break

        # Do 2 SA rounds with different perturbations
        for sa_round in range(2):
            sa_time = min(8.0, remaining / (2 - sa_round) - 1.0)
            if sa_time <= 0.5:
                break

            init_rows = [r[:] for r in best_rows]
            n_perturb = random.randint(2, min(4, n_rows * (row_len - 1)))
            for _ in range(n_perturb):
                r_idx = random.randint(0, n_rows - 1)
                i_idx = random.randint(1, row_len - 1)
                row = init_rows[r_idx]
                lower = row[i_idx - 1] + 1
                upper = row[i_idx + 1] - 1 if i_idx < row_len - 1 else best_max + 8
                if lower <= upper:
                    init_rows[r_idx][i_idx] = random.randint(lower, upper)

            local_rows, local_max, local_sum, local_cost = sa_optimize(init_rows, sa_time, best_max)

            # Accept if no collisions and better max, or same max but lower sum
            if local_cost < 100000:
                if local_max < best_max:
                    best_rows = local_rows
                    best_max = local_max
                    best_sum = local_sum
                elif local_max == best_max and local_sum < best_sum:
                    best_rows = local_rows
                    best_sum = local_sum

    return best_rows
