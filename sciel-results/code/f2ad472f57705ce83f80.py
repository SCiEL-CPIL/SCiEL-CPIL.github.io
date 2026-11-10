import time
import random
import sys

sys.setrecursionlimit(100000)

DEADLINE = 0
TIME_LIMIT = 110


def construct(n_rows, k):
    global DEADLINE
    start_time = time.time()
    DEADLINE = start_time + TIME_LIMIT

    # Baseline via greedy
    baseline_rows = greedy_construct(n_rows, k)
    if not baseline_rows:
        return []
    baseline_scope = max(max(row) for row in baseline_rows)

    # --- Strategy A: Generate-All + Local Swap (strong for small n_rows, k) ---
    if n_rows <= 10 and k <= 7:
        # Generate all valid rows up to baseline_scope within a time budget
        candidates = generate_all_valid_rows(baseline_scope, k, time_budget=18)

        if candidates is not None and len(candidates) > 0:
            # Sort by max value then mask sparsity
            candidates.sort(key=lambda x: (x[0], x[1].bit_count()))

            # Current best: use the smallest-max valid rows greedily
            current_rows = [list(r) for r in baseline_rows]
            current_masks = []
            for r in current_rows:
                mask = 0
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        mask |= (1 << (r[j] - r[i]))
                current_masks.append(mask)

            total_mask = 0
            for m in current_masks:
                total_mask |= m

            best_scope = baseline_scope
            improved = True

            # Local search: repeatedly swap the max row for a better candidate
            while improved and time.time() < DEADLINE - 5:
                improved = False
                # Pick row with largest max value
                max_idx = -1
                max_val = -1
                for i, r in enumerate(current_rows):
                    if r[-1] > max_val:
                        max_val = r[-1]
                        max_idx = i

                if max_idx == -1:
                    break

                others_mask = total_mask ^ current_masks[max_idx]

                # Try to find a disjoint candidate with smaller max
                found = False
                for cand in candidates:
                    cand_max, cand_mask, cand_row = cand
                    if cand_max >= max_val:
                        break
                    if (cand_mask & others_mask) == 0:
                        total_mask = (total_mask ^ current_masks[max_idx]) | cand_mask
                        current_rows[max_idx] = list(cand_row)
                        current_masks[max_idx] = cand_mask
                        improved = True
                        found = True
                        break

                # Also try swapping any row (not just max) for a better one
                if not found:
                    for i in range(n_rows):
                        if i == max_idx:
                            continue
                        i_val = current_rows[i][-1]
                        others_m = total_mask ^ current_masks[i]
                        for cand in candidates:
                            cand_max, cand_mask, cand_row = cand
                            if cand_max >= i_val:
                                break
                            if (cand_mask & others_m) == 0:
                                total_mask = (total_mask ^ current_masks[i]) | cand_mask
                                current_rows[i] = list(cand_row)
                                current_masks[i] = cand_mask
                                improved = True
                                break
                        if improved:
                            break

                if not found and not improved:
                    # Try random swap among same max to escape local optimum
                    if random.random() < 0.3:
                        swap_idx = random.randint(0, n_rows - 1)
                        oth = total_mask ^ current_masks[swap_idx]
                        best_cand = None
                        for cand in candidates:
                            cand_max, cand_mask, cand_row = cand
                            if (cand_mask & oth) == 0 and cand_max <= current_rows[swap_idx][-1]:
                                if best_cand is None or cand[0] < best_cand[0]:
                                    best_cand = cand
                        if best_cand is not None and best_cand[0] < current_rows[swap_idx][-1]:
                            total_mask = (total_mask ^ current_masks[swap_idx]) | best_cand[1]
                            current_rows[swap_idx] = list(best_cand[2])
                            current_masks[swap_idx] = best_cand[1]
                            improved = True

            final_scope = max(max(r) for r in current_rows)
            if final_scope < best_scope:
                best_scope = final_scope
                best_result = current_rows
            else:
                best_result = current_rows

            # Return if we've reached the counting lower bound
            lb = n_rows * k * (k + 1) // 2
            if n_rows == 7 and k == 5:
                lb = 105
            if best_scope <= lb:
                return best_result

    # --- Strategy B: Iterative Deepening DFS for each M ---
    generic_lb = n_rows * k * (k + 1) // 2
    if n_rows == 7 and k == 5:
        lower_bound = 105
    else:
        lower_bound = generic_lb

    start_m = max(lower_bound, generic_lb)
    best_rows = baseline_rows
    best_scope = baseline_scope

    for M in range(start_m, best_scope + 1):
        if time.time() > DEADLINE:
            break
        # Give more time budget to smaller M (more likely to succeed)
        remaining = max(1.0, DEADLINE - time.time())
        time_budget = min(25.0, remaining * 0.8)

        sol = solve_m(n_rows, k, M, time_budget)
        if sol is not None:
            return sol

    return best_rows


def greedy_construct(n_rows, k):
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def generate_all_valid_rows(M_limit, k, time_budget=18):
    """
    Generate all valid rows of length k+1 starting at 0, with max element <= M_limit.
    Returns list of (max_val, mask, row_tuple) or None if timeout.
    """
    start = time.time()
    timeout = time_budget

    rows = []
    bit_of = [1 << d for d in range(M_limit + 1)]

    def backtrack(current_row, current_mask, last_val):
        if time.time() - start > timeout:
            return False
        if len(current_row) == k + 1:
            rows.append((current_row[-1], current_mask, tuple(current_row)))
            return True

        for x in range(last_val + 1, M_limit + 1):
            new_mask = 0
            valid = True
            for a in current_row:
                d = x - a
                bit = bit_of[d]
                if current_mask & bit:
                    valid = False
                    break
                new_mask |= bit

            if valid:
                current_row.append(x)
                if not backtrack(current_row, current_mask | new_mask, x):
                    return False
                current_row.pop()

        return True

    if backtrack([0], 0, 0):
        return rows
    return None


def solve_m(n_rows, k, M, time_budget):
    """
    Try to find n_rows valid rows with max entry <= M.
    Uses DFS with randomized restarts.
    """
    start_time = time.time()
    deadline = start_time + min(time_budget, DEADLINE - start_time)

    if deadline < start_time + 0.5:
        return None

    bit_of = [1 << d for d in range(M + 1)]
    used_mask = 0
    rows = []

    def build_row(current_row, current_row_mask, rows_remaining):
        nonlocal used_mask

        if len(current_row) == k + 1:
            rows.append(list(current_row))
            used_mask |= current_row_mask
            return True

        last_val = current_row[-1]
        candidates = []

        for x in range(last_val + 1, M + 1):
            if used_mask & bit_of[x]:
                continue
            new_mask = 0
            valid = True
            for a in current_row[1:]:
                d = x - a
                bit = bit_of[d]
                if (used_mask | current_row_mask) & bit:
                    valid = False
                    break
                new_mask |= bit

            if valid:
                new_mask |= bit_of[x]
                candidates.append((x, new_mask))

        if not candidates:
            return False

        return False  # Placeholder; actual DFS below

    def dfs_row(current_row, current_row_mask, rows_remaining, order):
        nonlocal used_mask
        if time.time() > deadline:
            return False

        if len(current_row) == k + 1:
            rows.append(list(current_row))
            used_mask |= current_row_mask
            return True

        last_val = current_row[-1]
        candidates = []

        for x in range(last_val + 1, M + 1):
            if used_mask & bit_of[x]:
                continue
            new_mask = 0
            valid = True
            for a in current_row[1:]:
                d = x - a
                bit = bit_of[d]
                if (used_mask | current_row_mask) & bit:
                    valid = False
                    break
                new_mask |= bit

            if valid:
                new_mask |= bit_of[x]
                candidates.append((x, new_mask))

        if not candidates:
            return False

        # Shuffle for randomized search
        if order != 'deterministic':
            random.shuffle(candidates)

        for x, new_mask in candidates:
            current_row.append(x)
            if dfs_row(current_row, current_row_mask | new_mask, rows_remaining - 1, order):
                return True
            current_row.pop()

        return False

    def run(order):
        nonlocal used_mask, rows
        used_mask = 0
        rows = []
        for i in range(n_rows):
            if time.time() > deadline:
                return False
            if not dfs_row([0], 0, n_rows - i - 1, order):
                return False
        return True

    # Deterministic pass
    if run('deterministic'):
        return rows

    # Randomized restarts
    max_restarts = max(3, int((deadline - time.time()) / max(0.1, (deadline - start_time) / 20)))
    for _ in range(max_restarts):
        if time.time() > deadline:
            return None
        if run('random'):
            return rows

    return None


# Entry point
def construct(n_rows, k):
    global DEADLINE
    start_time = time.time()
    DEADLINE = start_time + TIME_LIMIT

    # Baseline via greedy
    baseline_rows = greedy_construct(n_rows, k)
    if not baseline_rows:
        return []
    baseline_scope = max(max(row) for row in baseline_rows)

    # --- Strategy A: Generate-All + Local Swap ---
    if n_rows <= 10 and k <= 7:
        candidates = generate_all_valid_rows(baseline_scope, k, time_budget=18)

        if candidates is not None and len(candidates) > 0:
            candidates.sort(key=lambda x: (x[0], x[1].bit_count()))

            current_rows = [list(r) for r in baseline_rows]
            current_masks = []
            for r in current_rows:
                mask = 0
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        mask |= (1 << (r[j] - r[i]))
                current_masks.append(mask)

            total_mask = 0
            for m in current_masks:
                total_mask |= m

            best_scope = baseline_scope
            improved = True

            while improved and time.time() < DEADLINE - 5:
                improved = False
                max_idx = -1
                max_val = -1
                for i, r in enumerate(current_rows):
                    if r[-1] > max_val:
                        max_val = r[-1]
                        max_idx = i

                if max_idx == -1:
                    break

                others_mask = total_mask ^ current_masks[max_idx]
                found = False

                for cand in candidates:
                    cand_max, cand_mask, cand_row = cand
                    if cand_max >= max_val:
                        break
                    if (cand_mask & others_mask) == 0:
                        total_mask = (total_mask ^ current_masks[max_idx]) | cand_mask
                        current_rows[max_idx] = list(cand_row)
                        current_masks[max_idx] = cand_mask
                        improved = True
                        found = True
                        break

                if not found:
                    for i in range(n_rows):
                        if i == max_idx:
                            continue
                        i_val = current_rows[i][-1]
                        others_m = total_mask ^ current_masks[i]
                        for cand in candidates:
                            cand_max, cand_mask, cand_row = cand
                            if cand_max >= i_val:
                                break
                            if (cand_mask & others_m) == 0:
                                total_mask = (total_mask ^ current_masks[i]) | cand_mask
                                current_rows[i] = list(cand_row)
                                current_masks[i] = cand_mask
                                improved = True
                                break
                        if improved:
                            break

                if not improved and random.random() < 0.3:
                    swap_idx = random.randint(0, n_rows - 1)
                    oth = total_mask ^ current_masks[swap_idx]
                    best_cand = None
                    for cand in candidates:
                        cand_max, cand_mask, cand_row = cand
                        if (cand_mask & oth) == 0 and cand_max < current_rows[swap_idx][-1]:
                            if best_cand is None or cand[0] < best_cand[0]:
                                best_cand = cand
                    if best_cand is not None:
                        total_mask = (total_mask ^ current_masks[swap_idx]) | best_cand[1]
                        current_rows[swap_idx] = list(best_cand[2])
                        current_masks[swap_idx] = best_cand[1]
                        improved = True

            final_scope = max(max(r) for r in current_rows)
            if final_scope < best_scope:
                best_scope = final_scope
                best_result = current_rows
            else:
                best_result = current_rows

            lb = n_rows * k * (k + 1) // 2
            if n_rows == 7 and k == 5:
                lb = 105
            if best_scope <= lb:
                return best_result

    # --- Strategy B: Iterative Deepening DFS ---
    generic_lb = n_rows * k * (k + 1) // 2
    if n_rows == 7 and k == 5:
        lower_bound = 105
    else:
        lower_bound = generic_lb

    start_m = max(lower_bound, generic_lb)
    best_rows = baseline_rows
    best_scope = baseline_scope

    for M in range(start_m, best_scope + 1):
        if time.time() > DEADLINE:
            break
        remaining = max(1.0, DEADLINE - time.time())
        time_budget = min(25.0, remaining * 0.8)
        sol = solve_m(n_rows, k, M, time_budget)
        if sol is not None:
            return sol

    return best_rows
