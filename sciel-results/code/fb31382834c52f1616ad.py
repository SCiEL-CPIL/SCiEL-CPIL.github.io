import time
import random
import sys
import numpy as np


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0

    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs - 1
    k_plus_1 = k + 1

    # 1. Fast greedy baseline
    best_solution = _greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 2. Local compaction (left-to-right + right-to-left)
    best_solution = _local_compact(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 3. Polish pass
    best_solution = _polish_scope(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 4. Backtracking search for scope reduction
    attempt = 0
    while time.time() < deadline - 8.0:
        target_scope = best_scope - 1
        if target_scope < lower_bound:
            break

        found = False
        for _ in range(30):
            if time.time() > deadline - 8.0:
                break
            solution = _backtracking_search(n_rows, k, target_scope, deadline, attempt)
            if solution:
                new_scope = max(max(row) for row in solution)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_solution = solution
                    found = True
                    # Improve the found solution
                    improved = _local_compact(best_solution, n_rows, k, deadline)
                    new_scope = max(max(row) for row in improved)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = improved
                    polished = _polish_scope(best_solution, n_rows, k, deadline)
                    new_scope = max(max(row) for row in polished)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = polished
                    if best_scope <= lower_bound:
                        return best_solution
                    break
            attempt += 1

        if not found and target_scope <= lower_bound + 3:
            break

    # 5. Simulated annealing polish
    best_solution = _simulated_anneal(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    # Final compaction
    best_solution = _local_compact(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    return best_solution


def _greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _build_used_set(rows, k):
    L = k + 1
    used = set()
    for row in rows:
        for i in range(L):
            for j in range(i + 1, L):
                used.add(row[j] - row[i])
    return used


def _get_diffs_for_pos(row, pos, L):
    diffs = set()
    for i in range(L):
        if i < pos:
            diffs.add(row[pos] - row[i])
        elif i > pos:
            diffs.add(row[i] - row[pos])
    return diffs


def _local_compact(rows, n_rows, k, time_limit):
    """Iteratively compact each element to its smallest feasible value,
    alternating left-to-right and right-to-left passes."""
    work = [list(r) for r in rows]
    used_set = _build_used_set(work, k)
    L = k + 1

    def _pull_left_to_right(row_idx):
        row = work[row_idx]
        any_changed = False
        for pos in range(1, L):
            old_val = row[pos]
            lower = row[pos - 1] + 1
            if lower >= old_val:
                continue
            old_diffs = _get_diffs_for_pos(row, pos, L)
            for new_val in range(lower, old_val):
                if pos < L - 1 and new_val >= row[pos + 1]:
                    break
                new_diffs = set()
                for i in range(L):
                    if i < pos:
                        new_diffs.add(new_val - row[i])
                    elif i > pos:
                        new_diffs.add(row[i] - new_val)
                other = used_set - old_diffs
                if new_diffs & other:
                    continue
                for d in old_diffs:
                    used_set.discard(d)
                for d in new_diffs:
                    used_set.add(d)
                row[pos] = new_val
                any_changed = True
                break
        return any_changed

    def _pull_right_to_left(row_idx):
        row = work[row_idx]
        any_changed = False
        for pos in range(L - 1, 0, -1):
            old_val = row[pos]
            lower = row[pos - 1] + 1
            if lower >= old_val:
                continue
            old_diffs = _get_diffs_for_pos(row, pos, L)
            for new_val in range(lower, old_val):
                if pos < L - 1 and new_val >= row[pos + 1]:
                    break
                new_diffs = set()
                for i in range(L):
                    if i < pos:
                        new_diffs.add(new_val - row[i])
                    elif i > pos:
                        new_diffs.add(row[i] - new_val)
                other = used_set - old_diffs
                if new_diffs & other:
                    continue
                for d in old_diffs:
                    used_set.discard(d)
                for d in new_diffs:
                    used_set.add(d)
                row[pos] = new_val
                any_changed = True
                break
        return any_changed

    # Alternate passes
    for iteration in range(60):
        if time.time() > time_limit - 5.0:
            break
        any_changed = False
        # Left-to-right
        for row_idx in range(n_rows):
            if _pull_left_to_right(row_idx):
                any_changed = True
        # Right-to-left
        for row_idx in range(n_rows - 1, -1, -1):
            if _pull_right_to_left(row_idx):
                any_changed = True
        if not any_changed:
            break

    # Perturbation escape: nudge a random element up, then compact
    rng = random.Random(42)
    for iteration in range(20):
        if time.time() > time_limit - 5.0:
            break
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, L - 1)
        original = work[row_idx][pos]
        offset = rng.randint(2, 8)
        new_val = original + offset
        if new_val > 250:
            continue
        old_diffs = _get_diffs_for_pos(work[row_idx], pos, L)
        new_diffs = set()
        ok = True
        for i in range(L):
            if i < pos:
                new_diffs.add(new_val - work[row_idx][i])
            elif i > pos:
                new_diffs.add(work[row_idx][i] - new_val)
        other = used_set - old_diffs
        if new_diffs & other:
            continue
        for d in old_diffs:
            used_set.discard(d)
        for d in new_diffs:
            used_set.add(d)
        work[row_idx][pos] = new_val
        # Compact back
        for _ in range(5):
            changed = False
            for r in range(n_rows):
                if _pull_left_to_right(r):
                    changed = True
            for r in range(n_rows - 1, -1, -1):
                if _pull_right_to_left(r):
                    changed = True
            if not changed:
                break

    return work


def _backtracking_search(n_rows, k, max_scope, deadline, seed_offset=0):
    """Backtracking with boolean array for used differences."""
    rng = random.Random(seed_offset * 7919 + 31)
    L = k + 1
    rows = []
    used_diffs = [False] * (max_scope + 1)

    def build_row(pos, current_mark, row_so_far):
        if time.time() > deadline:
            return None
        if pos == L:
            return list(row_so_far)

        max_x = max_scope - (L - pos - 1)
        if max_x <= current_mark:
            return None

        valid_candidates = []
        for x in range(current_mark + 1, max_x + 1):
            conflict = False
            for a in row_so_far:
                d = x - a
                if d > max_scope or used_diffs[d]:
                    conflict = True
                    break
            if not conflict:
                valid_candidates.append(x)
            if len(valid_candidates) >= 50:
                break

        if not valid_candidates:
            return None

        n_smallest = min(15, len(valid_candidates))
        candidates_to_try = valid_candidates[:n_smallest]
        if len(valid_candidates) > n_smallest and rng.random() < 0.3:
            extra = rng.sample(valid_candidates[n_smallest:], min(5, len(valid_candidates) - n_smallest))
            candidates_to_try = candidates_to_try + extra
        rng.shuffle(candidates_to_try)

        for x in candidates_to_try:
            mask_diffs = []
            for a in row_so_far:
                d = x - a
                mask_diffs.append(d)
                used_diffs[d] = True

            res = build_row(pos + 1, x, row_so_far + [x])
            if res is not None:
                return res

            for d in mask_diffs:
                used_diffs[d] = False

        return None

    for i in range(n_rows):
        row = build_row(1, 0, [0])
        if row is None:
            return None
        rows.append(row)
        for j in range(L):
            for i2 in range(j + 1, L):
                d = row[i2] - row[j]
                if d <= max_scope:
                    used_diffs[d] = True

    return rows


def _polish_scope(solution, n_rows, k, deadline):
    """High-precision polish with swaps and targeted perturbations."""
    if not solution:
        return solution

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1
    used_diffs = _build_used_set(best, k)

    def _pull_all():
        any_changed = False
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos in range(1, k_plus_1):
                old_val = row[pos]
                lower = row[pos - 1] + 1
                if lower >= old_val:
                    continue
                old_diffs = _get_diffs_for_pos(row, pos, k_plus_1)
                for new_val in range(lower, old_val):
                    if pos < k_plus_1 - 1 and new_val >= row[pos + 1]:
                        break
                    new_diffs = set()
                    for i in range(k_plus_1):
                        if i < pos:
                            new_diffs.add(new_val - row[i])
                        elif i > pos:
                            new_diffs.add(row[i] - new_val)
                    other = used_diffs - old_diffs
                    if new_diffs & other:
                        continue
                    for d in old_diffs:
                        used_diffs.discard(d)
                    for d in new_diffs:
                        used_diffs.add(d)
                    row[pos] = new_val
                    any_changed = True
                    break
        return any_changed

    def _try_swap(row_idx, pos1, pos2):
        row = best[row_idx]
        v1, v2 = row[pos1], row[pos2]
        if v1 == v2 or v1 >= v2:
            return False
        if pos1 > 0 and v2 <= row[pos1 - 1]:
            return False
        if pos2 < k_plus_1 - 1 and v1 >= row[pos2 + 1]:
            return False
        new_diffs = set()
        ok = True
        for i in range(pos1):
            d = v2 - row[i]
            if d <= 0:
                ok = False
                break
            new_diffs.add(d)
        if ok:
            for i in range(pos1 + 1, pos2):
                d = v2 - row[i]
                if d <= 0 or d in used_diffs:
                    ok = False
                    break
                new_diffs.add(d)
        if ok:
            for i in range(pos2 + 1, k_plus_1):
                d = row[i] - v2
                if d <= 0 or d in used_diffs:
                    ok = False
                    break
                new_diffs.add(d)
        if ok:
            for i in range(pos1 + 1, pos2):
                d = v1 - row[i]
                if d <= 0 or d in used_diffs:
                    ok = False
                    break
                new_diffs.add(d)
        if ok:
            for i in range(pos2 + 1, k_plus_1):
                d = row[i] - v1
                if d <= 0 or d in used_diffs:
                    ok = False
                    break
                new_diffs.add(d)
        if not ok:
            return False
        old_diffs = set()
        for i in range(pos1):
            old_diffs.add(v1 - row[i])
        for i in range(pos1 + 1, pos2):
            old_diffs.add(v1 - row[i])
        for i in range(pos2 + 1, k_plus_1):
            old_diffs.add(row[i] - v1)
        for i in range(pos1):
            old_diffs.add(v2 - row[i])
        for i in range(pos1 + 1, pos2):
            old_diffs.add(v2 - row[i])
        for i in range(pos2 + 1, k_plus_1):
            old_diffs.add(row[i] - v2)
        old_diffs.add(v2 - v1)
        if not (new_diffs & (used_diffs - old_diffs)):
            used_diffs.difference_update(old_diffs)
            used_diffs.update(new_diffs)
            row[pos1], row[pos2] = v2, v1
            return True
        return False

    rng = random.Random(99)
    for round_num in range(40):
        if time.time() > deadline - 5.0:
            break
        _pull_all()
        if round_num % 3 == 2:
            row_idx = round_num % n_rows
            row = best[row_idx]
            for pos1 in range(1, k_plus_1):
                for pos2 in range(pos1 + 1, k_plus_1):
                    if _try_swap(row_idx, pos1, pos2):
                        break
                else:
                    continue
                break
        if round_num % 2 == 1:
            row_idx = rng.randint(0, n_rows - 1)
            row = best[row_idx]
            pos = rng.randint(1, k_plus_1 - 1)
            original = row[pos]
            for offset in [1, 2, 3, 5]:
                new_val = original + offset
                if new_val > 200:
                    break
                old_diffs = _get_diffs_for_pos(row, pos, k_plus_1)
                new_diffs = set()
                for i in range(k_plus_1):
                    if i < pos:
                        new_diffs.add(new_val - row[i])
                    elif i > pos:
                        new_diffs.add(row[i] - new_val)
                other = used_diffs - old_diffs
                if new_diffs & other:
                    continue
                for d in old_diffs:
                    used_diffs.discard(d)
                for d in new_diffs:
                    used_diffs.add(d)
                row[pos] = new_val
                break
        _pull_all()

    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    return [list(row) for row in solution]


def _simulated_anneal(solution, n_rows, k, deadline):
    """Lightweight simulated annealing to escape local minima."""
    if not solution:
        return solution

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    L = k + 1
    used_set = _build_used_set(best, k)
    rng = random.Random(123)

    current = [list(row) for row in best]
    current_scope = best_scope
    T = 10.0
    T_min = 0.1
    cooling = 0.95

    steps = 0
    max_steps = 2000
    while T > T_min and steps < max_steps:
        if time.time() > deadline - 5.0:
            break
        steps += 1
        # Random move: pick a row, pick a position, try a small delta
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, L - 1)
        delta = rng.choice([-2, -1, 1, 2, 3])
        new_val = current[row_idx][pos] + delta
        if new_val <= current[row_idx][pos - 1] or (pos < L - 1 and new_val >= current[row_idx][pos + 1]):
            continue

        old_diffs = _get_diffs_for_pos(current[row_idx], pos, L)
        new_diffs = set()
        for i in range(L):
            if i < pos:
                new_diffs.add(new_val - current[row_idx][i])
            elif i > pos:
                new_diffs.add(current[row_idx][i] - new_val)
        other = used_set - old_diffs
        if new_diffs & other:
            continue

        # Apply
        for d in old_diffs:
            used_set.discard(d)
        for d in new_diffs:
            used_set.add(d)
        current[row_idx][pos] = new_val
        new_scope = max(max(row) for row in current)

        if new_scope < current_scope:
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best = [list(row) for row in current]
        else:
            # Accept with probability exp(-delta/T)
            if rng.random() < np.exp(-(new_scope - current_scope) / T):
                current_scope = new_scope
            else:
                # Revert
                for d in new_diffs:
                    used_set.discard(d)
                for d in old_diffs:
                    used_set.add(d)
                current[row_idx][pos] = current[row_idx][pos] - delta

        T *= cooling

    # Final compact
    result = _local_compact(best, n_rows, k, deadline)
    return result
