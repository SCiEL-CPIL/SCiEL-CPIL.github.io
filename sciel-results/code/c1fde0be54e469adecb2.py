import time
import random
import math
import bisect
import heapq
from collections import defaultdict


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    num_diffs_per_row = (k + 1) * k // 2

    best_scope = float('inf')
    best_rows = None

    # Phase 1: Generate initial solutions with best-fit greedy
    seed = 0
    max_initial_seeds = 40
    while time.time() < deadline - 55 and seed < max_initial_seeds:
        rows = _greedy_construct(n_rows, k, seed)
        if rows is None:
            seed += 1
            continue
        scope = _scope(rows)

        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]

        seed += 1

    if best_rows is None:
        best_rows = _greedy_construct(n_rows, k, 0)
        if best_rows is None:
            # Last resort fallback
            return [[0, i] for i in range(k + 1)]
        best_scope = _scope(best_rows)

    # Phase 2: Element-level simulated annealing with bitmask tracking
    if time.time() < deadline - 15:
        improved = _sa_element_moves(best_rows, n_rows, k, start_time, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved

    # Phase 3: Final polish - targeted reduction of max element
    if time.time() < deadline - 3 and best_rows is not None:
        polished = _polish_max(best_rows, n_rows, k, start_time, deadline)
        if polished is not None:
            pol_scope = _scope(polished)
            if pol_scope < best_scope:
                best_scope = pol_scope
                best_rows = polished

    return best_rows


def _scope(rows):
    if not rows:
        return 0
    return max(row[-1] for row in rows if row)


def _all_diffs(rows):
    diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
    return diffs


def _compute_row_masks(rows):
    """Compute bitmask for each row's differences."""
    row_masks = []
    for row in rows:
        m = 0
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                m |= (1 << (row[j] - ri))
        row_masks.append(m)
    return row_masks


def _compute_global_mask(row_masks):
    g = 0
    for m in row_masks:
        g |= m
    return g


def _greedy_construct(n_rows, k, seed):
    """Best-fit greedy using interval fragmentation minimization."""
    rng = random.Random(seed)
    used = set()
    rows = []

    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            last_x = row[-1]
            limit = 500

            x = last_x + 1
            best_score = float('inf')
            best_x = None
            candidates = []

            while x <= limit:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    score = _interval_score(new_diffs, used, x)
                    candidates.append((score, x))
                    if score < best_score:
                        best_score = score
                        best_x = x
                x += 1

            if best_x is None:
                x = limit + 1
                while x <= limit * 3:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        score = _interval_score(new_diffs, used, x)
                        candidates.append((score, x))
                        if score < best_score:
                            best_score = score
                            best_x = x
                    x += 1

            if best_x is None:
                return None

            candidates.sort(key=lambda t: t[0])
            top_n = min(5, len(candidates))
            chosen_x = candidates[rng.randint(0, top_n - 1)][1]

            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used.update(new_diffs)

        rows.append(row)

    return rows


def _interval_score(new_diffs, used, x):
    """Score based on interval fragmentation."""
    if not used:
        return float(len(new_diffs))

    new_count = 0
    for d in new_diffs:
        has_neighbor = False
        for delta in [1, 2, 3]:
            if (d - delta) in used or (d + delta) in used:
                has_neighbor = True
                break
        if not has_neighbor:
            new_count += 1

    return float(new_count) + 0.01 * x


def _sa_element_moves(initial_rows, n_rows, k, start_time, deadline):
    """
    Simulated annealing with element-level moves and bitmask-based validity.
    Much faster per-step than row-regeneration approach.
    """
    rng = random.Random(42)
    rows = [list(r) for r in initial_rows]
    current_scope = _scope(rows)
    best_scope = current_scope
    best_rows = [list(r) for r in rows]

    row_masks = _compute_row_masks(rows)
    global_mask = _compute_global_mask(row_masks)

    T = 50.0
    T_min = 0.1
    cool_rate = 0.9997
    steps_since_improve = 0
    max_no_improve = 5000

    while time.time() < deadline - 5 and T > T_min:
        # Pick a random row and position (not first)
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, k)
        row = rows[row_idx]
        old_val = row[pos]

        # Determine valid range
        low = row[pos - 1] + 1
        if pos < k:
            high = row[pos + 1] - 1
        else:
            high = current_scope + 10

        if low >= high:
            T *= cool_rate
            continue

        # Bias toward lower values to reduce scope
        if old_val > low + 2 and rng.random() < 0.75:
            search_high = min(old_val - 1, high)
            if low > search_high:
                T *= cool_rate
                continue
            new_val = rng.randint(low, search_high)
        else:
            new_val = rng.randint(low, high)

        if new_val == old_val:
            T *= cool_rate
            continue

        # Try the move using bitmask
        success, new_global_mask, new_row_mask = _try_move(
            rows, row_idx, pos, new_val, global_mask, row_masks
        )

        if not success:
            T *= cool_rate
            continue

        # Compute new scope
        new_scope = current_scope
        if pos == k and new_val < row[-1]:
            # Last element decreased; check if it's still the max
            new_scope = max(r[-1] for r in rows)
        elif pos == k and new_val > row[-1]:
            new_scope = new_val

        delta = new_scope - current_scope

        if delta <= 0:
            # Accept improvement
            rows[row_idx][pos] = new_val
            row_masks[row_idx] = new_row_mask
            global_mask = new_global_mask
            current_scope = new_scope
            steps_since_improve = 0
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        else:
            # Metropolis acceptance
            if T > 1e-9:
                prob = math.exp(-delta / T)
            else:
                prob = 0.0
            if rng.random() < prob:
                rows[row_idx][pos] = new_val
                row_masks[row_idx] = new_row_mask
                global_mask = new_global_mask
                current_scope = new_scope
                steps_since_improve += 1
            else:
                steps_since_improve += 1

        T *= cool_rate

        # If stuck, perturb
        if steps_since_improve > max_no_improve:
            # Try a random perturbation: move a random element in a random row
            for _ in range(20):
                p_row = rng.randint(0, n_rows - 1)
                p_pos = rng.randint(1, k)
                p_row_data = rows[p_row]
                p_old = p_row_data[p_pos]
                p_low = p_row_data[p_pos - 1] + 1
                if p_pos < k:
                    p_high = p_row_data[p_pos + 1] - 1
                else:
                    p_high = current_scope + 10
                if p_low >= p_high:
                    continue
                p_new = rng.randint(p_low, p_high)
                if p_new == p_old:
                    continue
                s, ngm, nrm = _try_move(rows, p_row, p_pos, p_new, global_mask, row_masks)
                if s:
                    rows[p_row][p_pos] = p_new
                    row_masks[p_row] = nrm
                    global_mask = ngm
                    current_scope = _scope(rows)
                    break
            steps_since_improve = 0
            T = max(T, 1.0)

    return best_rows


def _try_move(rows, row_idx, pos, new_val, global_mask, row_masks):
    """
    Try to move element at rows[row_idx][pos] to new_val.
    Returns (success, new_global_mask, new_row_mask) or (False, None, None).
    """
    row = rows[row_idx]
    n = len(row)

    if pos > 0 and new_val <= row[pos - 1]:
        return False, None, None
    if pos < n - 1 and new_val >= row[pos + 1]:
        return False, None, None

    old_val = row[pos]
    if new_val == old_val:
        return False, None, None

    # Tentatively apply
    row[pos] = new_val

    # Check within-row uniqueness
    diffs_in_row = set()
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            d = row[j] - ri
            if d in diffs_in_row:
                row[pos] = old_val
                return False, None, None
            diffs_in_row.add(d)

    # Compute new row mask
    new_row_mask = 0
    for d in diffs_in_row:
        new_row_mask |= (1 << d)

    # Check against other rows
    other_mask = global_mask & ~row_masks[row_idx]
    if new_row_mask & other_mask:
        row[pos] = old_val
        return False, None, None

    new_global_mask = (global_mask & ~row_masks[row_idx]) | new_row_mask
    row[pos] = old_val  # revert; caller will apply

    return True, new_global_mask, new_row_mask


def _polish_max(rows, n_rows, k, start_time, deadline):
    """
    Final polish: targeted local search to reduce the maximum element.
    Uses element-level moves focused on the row with the largest entry.
    """
    rng = random.Random(123)
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)

    row_masks = _compute_row_masks(rows)
    global_mask = _compute_global_mask(row_masks)

    for iteration in range(200):
        if time.time() > deadline - 3:
            break

        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]

        # Try to reduce elements in the max row, especially the last one
        improved = False
        for pos in range(k, 0, -1):
            row = rows[max_idx]
            old_val = row[pos]
            if pos == k:
                # Try to reduce the last element
                low = row[pos - 1] + 1
                high = old_val - 1
            else:
                low = row[pos - 1] + 1
                high = row[pos + 1] - 1

            if low >= high:
                continue

            # Try a few random lower values
            for _ in range(5):
                new_val = rng.randint(low, high)
                if new_val == old_val:
                    continue
                success, ngm, nrm = _try_move(
                    rows, max_idx, pos, new_val, global_mask, row_masks
                )
                if success:
                    new_scope = _scope(rows)
                    if new_scope < best_scope:
                        rows[max_idx][pos] = new_val
                        row_masks[max_idx] = nrm
                        global_mask = ngm
                        best_scope = new_scope
                        best_rows = [list(r) for r in rows]
                        improved = True
                        break
            if improved:
                break

        if not improved:
            # Try reducing other rows to free up difference space
            for _ in range(3):
                r_idx = rng.randint(0, n_rows - 1)
                if r_idx == max_idx:
                    continue
                p_pos = rng.randint(1, k)
                row = rows[r_idx]
                old_val = row[p_pos]
                low = row[p_pos - 1] + 1
                if p_pos < k:
                    high = row[p_pos + 1] - 1
                else:
                    high = row[p_pos] - 1  # Try to reduce
                if low >= high:
                    continue
                new_val = rng.randint(low, high)
                if new_val == old_val:
                    continue
                success, ngm, nrm = _try_move(
                    rows, r_idx, p_pos, new_val, global_mask, row_masks
                )
                if success:
                    new_scope = _scope(rows)
                    if new_scope < best_scope:
                        rows[r_idx][p_pos] = new_val
                        row_masks[r_idx] = nrm
                        global_mask = ngm
                        best_scope = new_scope
                        best_rows = [list(r) for r in rows]
                        improved = True
                        break
            if improved:
                continue

        # If no improvement after several iterations, stop
        if iteration > 30 and not improved:
            break

    return best_rows if best_scope < _scope([list(r) for r in rows]) else None


def _is_valid(rows, n_rows, k):
    """Check if all differences are distinct across all rows."""
    all_diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in all_diffs:
                    return False
                all_diffs.add(d)
    return len(all_diffs) == n_rows * (k + 1) * k // 2
