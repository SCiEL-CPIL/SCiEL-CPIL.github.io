import time
import random
import sys
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0

    # Theoretical lower bound
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs - 1

    k_plus_1 = k + 1

    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 2. Local improvement to reduce scope
    best_solution = _improve_scope(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 3. Polish pass for further compaction
    best_solution = _polish_scope(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 4. Try to find better solutions with randomized backtracking
    attempt = 0
    while time.time() < deadline - 5.0:
        target_scope = best_scope - 1
        if target_scope < lower_bound:
            break

        found = False
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break

            solution = _backtracking_search(n_rows, k, target_scope, deadline, attempt)
            if solution:
                new_scope = max(max(row) for row in solution)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_solution = solution
                    found = True
                    improved = _improve_scope(best_solution, n_rows, k, deadline)
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

        if not found and target_scope <= lower_bound + 5:
            if target_scope < lower_bound:
                break

    # Final polish
    best_solution = _polish_scope(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    return best_solution


def greedy_construct(n_rows, k):
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


def _backtracking_search(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    Backtracking search to build n_rows of k+1 increasing integers starting at 0,
    with all pairwise differences distinct across all rows, and max element <= max_scope.
    Uses a bitset (Python int) for fast difference checking.
    """
    rng = random.Random(seed_offset * 12345 + 678)
    rows = []
    k_plus_1 = k + 1

    # Use a Python int as a bitset for used differences
    # Bit d is set if difference d is used
    used_bits = 0

    def build_row(pos, current_mark, row_so_far):
        nonlocal used_bits
        if time.time() > deadline:
            return None

        if pos == k_plus_1:
            return list(row_so_far)

        # Find valid candidates
        valid_candidates = []
        # Prune upper bound: we need at least (k_plus_1 - pos - 1) more elements,
        # each at least 1 greater than previous, so max possible x is:
        max_x = max_scope - (k_plus_1 - pos - 1)
        if max_x <= current_mark:
            return None

        for x in range(current_mark + 1, max_x + 1):
            # Check if any new difference is already used using bitset
            conflict = False
            for a in row_so_far:
                d = x - a
                if (used_bits >> d) & 1:
                    conflict = True
                    break
            if not conflict:
                valid_candidates.append(x)

        if not valid_candidates:
            return None

        # Randomize the order of candidates to escape local minima
        # But bias towards smaller values to keep scope small
        n_smallest = min(12, len(valid_candidates))
        smallest = valid_candidates[:n_smallest]

        # With some probability, also consider larger candidates
        if len(valid_candidates) > n_smallest and rng.random() < 0.3:
            extra = rng.sample(valid_candidates[n_smallest:], min(6, len(valid_candidates) - n_smallest))
            candidates_to_try = smallest + extra
        else:
            candidates_to_try = smallest

        rng.shuffle(candidates_to_try)

        for x in candidates_to_try:
            # Mark new differences in bitset
            new_diffs = []
            for a in row_so_far:
                d = x - a
                new_diffs.append(d)
                used_bits |= (1 << d)

            res = build_row(pos + 1, x, row_so_far + [x])
            if res is not None:
                return res

            # Unmark differences
            for d in new_diffs:
                used_bits &= ~(1 << d)

        return None

    for i in range(n_rows):
        row = build_row(1, 0, [0])
        if row is None:
            return None
        rows.append(row)
        # Mark all differences in this row as used
        for j in range(k_plus_1):
            for i2 in range(j + 1, k_plus_1):
                used_bits |= (1 << (row[i2] - row[j]))

    return rows


def _improve_scope(solution, n_rows, k, deadline):
    """
    Try to reduce the scope of a valid solution by local search.
    We try to shift elements down to reduce the maximum value.
    """
    if not solution:
        return solution

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1

    # Collect all used differences
    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])

    # Try to reduce scope by shifting elements
    # Strategy: for each row, try to shift elements down
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos in range(k_plus_1 - 1, 0, -1):  # from end to start
                # Try to shift this element down
                original = row[pos]
                # Only try to shift if it's the max or could reduce max
                for new_val in range(row[pos - 1] + 1, original):
                    # Check if new value creates conflicts
                    new_diffs = {new_val - row[i] for i in range(pos)}
                    # Also need to check diffs with elements after pos
                    for j in range(pos + 1, k_plus_1):
                        new_diffs.add(row[j] - new_val)

                    # Remove old diffs
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(original - row[i])
                    for j in range(pos + 1, k_plus_1):
                        old_diffs.add(row[j] - original)

                    # Check if new_diffs conflict with (used_diffs - old_diffs)
                    if new_diffs & (used_diffs - old_diffs):
                        continue

                    # Check if new_diffs are all positive
                    if any(d <= 0 for d in new_diffs):
                        continue

                    # Apply the change
                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    row[pos] = new_val
                    improved = True
                    break

    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        return [list(row) for row in solution]


def _polish_scope(solution, n_rows, k, deadline):
    """
    Aggressive polishing: repeatedly try single-element shifts to minimize scope.
    This is a tighter local search that can find small improvements.
    """
    if not solution:
        return solution

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1

    # Collect all used differences
    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])

    # Iteratively try to reduce each element
    for iteration in range(20):
        if time.time() > deadline - 2.0:
            break

        any_improved = False
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos in range(k_plus_1 - 1, 0, -1):
                original = row[pos]
                # Try shifting down by 1, 2, ... up to original - row[pos-1] - 1
                for new_val in range(original - 1, row[pos - 1], -1):
                    # Check if new value creates conflicts
                    new_diffs = set()
                    for i in range(pos):
                        new_diffs.add(new_val - row[i])
                    for j in range(pos + 1, k_plus_1):
                        new_diffs.add(row[j] - new_val)

                    # Remove old diffs
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(original - row[i])
                    for j in range(pos + 1, k_plus_1):
                        old_diffs.add(row[j] - original)

                    # Check if new_diffs conflict with (used_diffs - old_diffs)
                    if new_diffs & (used_diffs - old_diffs):
                        continue

                    # Check if new_diffs are all positive
                    if any(d <= 0 for d in new_diffs):
                        continue

                    # Apply the change
                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    row[pos] = new_val
                    any_improved = True
                    break

        if not any_improved:
            break

    return best
