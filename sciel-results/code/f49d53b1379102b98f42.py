"""Difference Triangle Set (4,7) solver.

Structural change: use a bitset-based difference tracking for speed,
combined with a targeted backtracking scope-reduction search.

The key insight from the inspiration is that bitset acceleration enables
targeted backtracking scope reduction, allowing explicit search for lower
scopes than random SA can reliably find. We combine this with the local
compaction machinery from the current program.

Approach:
1. Fast greedy baseline.
2. Aggressive local compaction (pull to smallest feasible).
3. Bitset-accelerated backtracking to try to reduce scope by 1, 2, ...
4. Simulated annealing as a fallback improvement.
"""
import time
import random
import sys


def _check_valid(rows, k):
    """Return True iff all positive differences within rows are distinct."""
    used = set()
    for row in rows:
        L = len(row)
        for i in range(L):
            for j in range(i + 1, L):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _max_entry(rows):
    return max(max(r) for r in rows)


def _greedy_construct(n_rows, k, rng=None):
    """Greedy ruler construction with optional randomization."""
    if rng is None:
        rng = random.Random(0)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if rng.random() < 0.02:
                x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _greedy_variant(n_rows, k, rng, skip_p=0.05, jump_p=0.03, shuffle_rows=False, bias_p=0.0, bias_range=5):
    """Greedy with configurable parameters."""
    used = set()
    rows = []
    order = list(range(n_rows))
    if shuffle_rows:
        rng.shuffle(order)
    row_results = {}
    for r_idx in order:
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if rng.random() < jump_p:
                x += rng.randint(1, 4)
            if rng.random() < skip_p:
                continue
            if bias_p > 0 and rng.random() < bias_p:
                x += rng.randint(1, bias_range)
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        if len(row) != k + 1:
            return None
        row_results[r_idx] = row
    for r_idx in range(n_rows):
        rows.append(row_results[r_idx])
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


def _can_set(row, pos, new_val, used_set, L):
    if pos > 0 and new_val <= row[pos - 1]:
        return False
    if pos < L - 1 and new_val >= row[pos + 1]:
        return False
    old_diffs = _get_diffs_for_pos(row, pos, L)
    new_diffs = set()
    for i in range(L):
        if i < pos:
            new_diffs.add(new_val - row[i])
        elif i > pos:
            new_diffs.add(row[i] - new_val)
    other = used_set - old_diffs
    if new_diffs & other:
        return False
    return True


def _apply_set(row, pos, new_val, used_set, L):
    old_diffs = _get_diffs_for_pos(row, pos, L)
    new_diffs = set()
    for i in range(L):
        if i < pos:
            new_diffs.add(new_val - row[i])
        elif i > pos:
            new_diffs.add(row[i] - new_val)
    for d in old_diffs:
        used_set.discard(d)
    for d in new_diffs:
        used_set.add(d)
    row[pos] = new_val


def _fast_compact_row(row, used_set, L):
    """Compact a single row in-place, return True if any change made."""
    changed = False
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
            changed = True
            break
    return changed


def _local_compact(rows, n_rows, k, time_limit):
    """Iteratively compact each element to its smallest feasible value."""
    work = [list(r) for r in rows]
    used_set = _build_used_set(work, k)
    L = k + 1
    improved = True

    while improved and time.time() < time_limit:
        improved = False
        for r_idx in range(n_rows):
            row = work[r_idx]
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
                    improved = True
                    break
    return work


def _backtracking_scope_search(n_rows, k, target_scope, deadline, seed_offset=0):
    """
    Bitset-accelerated backtracking to find a solution with max entry <= target_scope.
    Uses a boolean array for used differences for O(1) membership checks.
    """
    rng = random.Random(seed_offset * 7919 + 31)
    L = k + 1
    rows = []
    
    used_diffs = [False] * (target_scope + 1)
    used_count = 0

    def build_row(pos, current_mark, row_so_far):
        nonlocal used_count
        if time.time() > deadline:
            return None

        if pos == L:
            return list(row_so_far)

        # Prune: need at least (L - pos - 1) more values, each at least 1 apart
        max_x = target_scope - (L - pos - 1)
        if max_x <= current_mark:
            return None

        # Find valid candidates with early termination
        valid_candidates = []
        for x in range(current_mark + 1, max_x + 1):
            conflict = False
            for a in row_so_far:
                d = x - a
                if used_diffs[d]:
                    conflict = True
                    break
            if not conflict:
                valid_candidates.append(x)
            # Limit candidate list for speed
            if len(valid_candidates) >= 60:
                break

        if not valid_candidates:
            return None

        # Bias towards smaller values
        n_smallest = min(20, len(valid_candidates))
        candidates_to_try = valid_candidates[:n_smallest]

        # Occasionally add larger candidates for diversity
        if len(valid_candidates) > n_smallest and rng.random() < 0.3:
            extra = rng.sample(valid_candidates[n_smallest:], min(5, len(valid_candidates) - n_smallest))
            candidates_to_try = candidates_to_try + extra

        rng.shuffle(candidates_to_try)

        for x in candidates_to_try:
            # Mark new differences
            new_diffs = []
            for a in row_so_far:
                d = x - a
                new_diffs.append(d)
                used_diffs[d] = True
            used_count += len(new_diffs)

            res = build_row(pos + 1, x, row_so_far + [x])
            if res is not None:
                return res

            # Unmark differences
            for d in new_diffs:
                used_diffs[d] = False
            used_count -= len(new_diffs)

        return None

    for i in range(n_rows):
        row = build_row(1, 0, [0])
        if row is None:
            # Unmark all differences from completed rows
            for r in rows:
                for j in range(L):
                    for ii in range(j + 1, L):
                        d = r[ii] - r[j]
                        if used_diffs[d]:
                            used_diffs[d] = False
                            used_count -= 1
            return None
        rows.append(row)
        # Mark all differences in this row as used
        for j in range(L):
            for ii in range(j + 1, L):
                d = row[ii] - row[j]
                if not used_diffs[d]:
                    used_diffs[d] = True
                    used_count += 1

    return rows


def _improve_scope_bitset(solution, n_rows, k, deadline):
    """
    Use bitset-accelerated backtracking to try to reduce scope by 1, 2, ...
    """
    if not solution:
        return solution

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)

    # Try to reduce scope step by step
    for reduction in range(1, 15):
        if time.time() > deadline:
            break

        target_scope = best_scope - reduction
        # Theoretical lower bound check
        num_diffs = n_rows * k * (k + 1) // 2
        if target_scope < num_diffs - 1:
            break

        # Try multiple restarts for this target scope
        found = False
        for attempt in range(30):
            if time.time() > deadline:
                break

            result = _backtracking_scope_search(n_rows, k, target_scope, deadline, seed_offset=attempt)
            if result is not None:
                if _check_valid(result, k):
                    new_scope = max(max(row) for row in result)
                    if new_scope < best_scope:
                        best = result
                        best_scope = new_scope
                        found = True
                        break

        if not found:
            # If we couldn't reduce by this amount, stop
            if reduction <= 3:
                continue
            else:
                break

    return best


def _simulated_annealing(rows, n_rows, k, rng, time_limit):
    """Simulated annealing to improve the solution."""
    work = [list(r) for r in rows]
    L = k + 1
    used_set = _build_used_set(work, k)
    current_max = _max_entry(work)
    best_rows = [list(r) for r in work]
    best_max = current_max

    start_temp = max(1, current_max // 8)
    end_temp = 0.05
    cooling = 0.9995

    t = start_temp
    stagnation_counter = 0
    max_stagnation = 2000

    while time.time() < time_limit and t > end_temp:
        move_type = rng.random()

        if move_type < 0.35:
            # Move 1: Try to decrease a random element
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, L - 1)
            row = work[r_idx]
            old_val = row[pos]
            lower = row[pos - 1] + 1
            if lower >= old_val:
                t *= cooling
                continue
            new_val = rng.randint(lower, old_val - 1)
            if new_val >= old_val:
                t *= cooling
                continue
            if _can_set(row, pos, new_val, used_set, L):
                _apply_set(row, pos, new_val, used_set, L)
                _fast_compact_row(row, used_set, L)
                new_max = _max_entry(work)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in work]
                    current_max = best_max
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

        elif move_type < 0.65:
            # Move 2: Increase a random element (perturbation), then recompact
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, L - 1)
            row = work[r_idx]
            old_val = row[pos]
            upper_limit = old_val + rng.randint(1, max(2, current_max // 3))
            new_val = rng.randint(old_val + 1, upper_limit)
            if new_val <= old_val:
                t *= cooling
                continue
            if _can_set(row, pos, new_val, used_set, L):
                _apply_set(row, pos, new_val, used_set, L)
                _fast_compact_row(row, used_set, L)
                new_max = _max_entry(work)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in work]
                    current_max = best_max
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

        elif move_type < 0.85:
            # Move 3: Full recompact of all rows
            for r_idx in range(n_rows):
                _fast_compact_row(work[r_idx], used_set, L)
            new_max = _max_entry(work)
            if new_max < best_max:
                best_max = new_max
                best_rows = [list(r) for r in work]
                current_max = best_max
                stagnation_counter = 0
            else:
                stagnation_counter += 1

        else:
            # Move 4: Swap two elements between different rows
            r1 = rng.randint(0, n_rows - 1)
            r2 = rng.randint(0, n_rows - 1)
            if r1 != r2:
                p1 = rng.randint(1, L - 1)
                p2 = rng.randint(1, L - 1)
                row1 = work[r1]
                row2 = work[r2]
                v1 = row1[p1]
                v2 = row2[p2]

                old_diffs1 = _get_diffs_for_pos(row1, p1, L)
                old_diffs2 = _get_diffs_for_pos(row2, p2, L)

                if p1 > 0 and v2 <= row1[p1 - 1]:
                    pass
                elif p1 < L - 1 and v2 >= row1[p1 + 1]:
                    pass
                else:
                    if p2 > 0 and v1 <= row2[p2 - 1]:
                        pass
                    elif p2 < L - 1 and v1 >= row2[p2 + 1]:
                        pass
                    else:
                        new_diffs1 = set()
                        for i in range(L):
                            if i < p1:
                                new_diffs1.add(v2 - row1[i])
                            elif i > p1:
                                new_diffs1.add(row1[i] - v2)

                        new_diffs2 = set()
                        for i in range(L):
                            if i < p2:
                                new_diffs2.add(v1 - row2[i])
                            elif i > p2:
                                new_diffs2.add(row2[i] - v1)

                        other = used_set - old_diffs1 - old_diffs2
                        if not (new_diffs1 & other) and not (new_diffs2 & other) and not (new_diffs1 & new_diffs2):
                            for d in old_diffs1:
                                used_set.discard(d)
                            for d in old_diffs2:
                                used_set.discard(d)
                            for d in new_diffs1:
                                used_set.add(d)
                            for d in new_diffs2:
                                used_set.add(d)
                            row1[p1] = v2
                            row2[p2] = v1
                            _fast_compact_row(row1, used_set, L)
                            _fast_compact_row(row2, used_set, L)
                            new_max = _max_entry(work)
                            if new_max < best_max:
                                best_max = new_max
                                best_rows = [list(r) for r in work]
                                current_max = best_max
                                stagnation_counter = 0
                            else:
                                stagnation_counter += 1

        if stagnation_counter > max_stagnation:
            fresh = _greedy_variant(n_rows, k, rng, skip_p=rng.random() * 0.15,
                                    jump_p=rng.random() * 0.08, shuffle_rows=True,
                                    bias_p=rng.random() * 0.3, bias_range=rng.randint(3, 8))
            if fresh is not None and _check_valid(fresh, k):
                fresh_max = _max_entry(fresh)
                if fresh_max < best_max:
                    best_max = fresh_max
                    best_rows = [list(r) for r in fresh]
                    current_max = best_max
                    work = [list(r) for r in fresh]
                    used_set = _build_used_set(work, k)
                    t = start_temp
                    stagnation_counter = 0

        t *= cooling

    return best_rows


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    # 1) Deterministic greedy baseline
    best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))
    best_max = _max_entry(best_rows)

    # 2) Many randomized greedy restarts
    rng = random.Random(42)
    greedy_deadline = deadline - 60
    trial = 0
    while time.time() < greedy_deadline:
        trial += 1
        skip_p = rng.random() * 0.15
        jump_p = rng.random() * 0.08
        shuffle = rng.random() < 0.6
        bias_p = rng.random() * 0.4
        bias_range = rng.randint(3, 8)
        rows = _greedy_variant(n_rows, k, rng, skip_p=skip_p, jump_p=jump_p,
                               shuffle_rows=shuffle, bias_p=bias_p, bias_range=bias_range)
        if rows is not None and _check_valid(rows, k):
            rmax = _max_entry(rows)
            if rmax < best_max:
                best_rows = rows
                best_max = rmax

    # 3) Local compaction
    compact_deadline = deadline - 40
    if time.time() < compact_deadline:
        compacted = _local_compact(best_rows, n_rows, k, compact_deadline)
        if _check_valid(compacted, k):
            cmax = _max_entry(compacted)
            if cmax < best_max:
                best_rows = compacted
                best_max = cmax

    # 4) Bitset-accelerated backtracking scope reduction
    backtrack_deadline = deadline - 20
    if time.time() < backtrack_deadline:
        improved = _improve_scope_bitset(best_rows, n_rows, k, backtrack_deadline)
        if _check_valid(improved, k):
            imax = _max_entry(improved)
            if imax < best_max:
                best_rows = improved
                best_max = imax

    # 5) Simulated annealing
    sa_deadline = deadline - 2
    if time.time() < sa_deadline:
        sa_result = _simulated_annealing(best_rows, n_rows, k, rng, sa_deadline)
        if _check_valid(sa_result, k):
            smax = _max_entry(sa_result)
            if smax < best_max:
                best_rows = sa_result
                best_max = smax

    # 6) Final validation
    if not _check_valid(best_rows, k):
        best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))

    return best_rows
