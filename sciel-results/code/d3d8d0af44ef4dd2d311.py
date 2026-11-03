import time
import random
import sys
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    # Theoretical lower bound: need n_rows * k*(k+1)/2 distinct positive differences
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs - 1  # differences are 1..M, so max entry >= num_diffs - 1

    k_plus_1 = k + 1

    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 2. Pull-to-smallest compaction (structural: rebuild each row greedily)
    best_solution = _pull_compact(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 3. Kick-and-pull: escape local minima by perturbing a row then re-pulling
    best_solution = _kick_and_pull(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 4. Randomized backtracking for tighter target scopes
    attempt = 0
    while time.time() < deadline - 5.0:
        target_scope = best_scope - 1
        if target_scope < lower_bound:
            break

        found = False
        for _ in range(60):
            if time.time() > deadline - 5.0:
                break

            solution = _backtracking_search(n_rows, k, target_scope, deadline, attempt)
            if solution:
                new_scope = max(max(row) for row in solution)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_solution = solution
                    found = True
                    # Compact the new solution
                    improved = _pull_compact(best_solution, n_rows, k, deadline)
                    new_scope = max(max(row) for row in improved)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = improved
                    kicked = _kick_and_pull(best_solution, n_rows, k, deadline)
                    new_scope = max(max(row) for row in kicked)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = kicked
                    if best_scope <= lower_bound:
                        return best_solution
                    break
            attempt += 1

        if not found and target_scope <= lower_bound + 3:
            break

    # Final compaction
    best_solution = _pull_compact(best_solution, n_rows, k, deadline)
    best_solution = _kick_and_pull(best_solution, n_rows, k, deadline)

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


def _row_diffs(row):
    """Return the set of all positive pairwise differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _pull_compact(solution, n_rows, k, deadline):
    """
    Structural compaction: rebuild each row from left to right, pulling every
    element to its smallest feasible value given the current global difference
    set. This is a different representation than single-element shifts: it
    re-derives each row as a whole, so it can collapse multiple elements at
    once and reach a tighter local optimum.
    """
    if not solution:
        return solution

    best = [list(row) for row in solution]
    k_plus_1 = k + 1

    # Global difference set (all positive diffs across all rows)
    used_diffs = set()
    for row in best:
        used_diffs |= _row_diffs(row)

    def pull_row(row_idx, used_diffs):
        """Rebuild row row_idx greedily from left to right to smallest feasible."""
        row = best[row_idx]
        # Remove this row's current diffs from the global set
        row_d = _row_diffs(row)
        used_diffs -= row_d

        new_row = [0]
        # For each subsequent position, find smallest x > new_row[-1] such that
        # all new diffs (x - a for a in new_row) are not in used_diffs and positive.
        # Also must leave room for the remaining elements.
        for pos in range(1, k_plus_1):
            remaining = k_plus_1 - pos  # elements after this one
            x = new_row[-1] + 1
            while True:
                # Check feasibility of placing x here
                feasible = True
                for a in new_row:
                    d = x - a
                    if d <= 0 or d in used_diffs:
                        feasible = False
                        break
                if feasible:
                    # Tentatively place
                    new_row.append(x)
                    for a in new_row[:-1]:
                        used_diffs.add(x - a)
                    break
                x += 1
                # Safety: if x grows too large, we may be stuck; but we only pull
                # to smallest, so it should be bounded by original scope.
                if x > 100000:
                    # Fallback: keep original row
                    new_row = list(row)
                    used_diffs |= _row_diffs(row)
                    return
        best[row_idx] = new_row
        used_diffs |= _row_diffs(new_row)

    # Pass 1: pull each row to smallest
    for row_idx in range(n_rows):
        if time.time() > deadline - 2.0:
            break
        pull_row(row_idx, used_diffs)

    # Pass 2: re-pull in reverse order (different order can find tighter packing)
    for row_idx in range(n_rows - 1, -1, -1):
        if time.time() > deadline - 2.0:
            break
        pull_row(row_idx, used_diffs)

    return best


def _kick_and_pull(solution, n_rows, k, deadline):
    """
    Escape local minima: occasionally perturb a row (add a random small offset
    to a middle element, or swap two elements' relative spacing), then re-run
    pull-to-smallest. Accept only if scope improved. This gives the pull-to-
    smallest representation the ability to move between basins, which pure
    greedy pull cannot.
    """
    if not solution:
        return solution

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1
    rng = random.Random(987654321)

    for iteration in range(40):
        if time.time() > deadline - 2.0:
            break

        # Pick a random row to kick
        row_idx = rng.randrange(n_rows)
        row = list(best[row_idx])

        # Kick: pick a random position (not 0) and nudge it, or swap spacing
        pos = rng.randrange(1, k_plus_1)
        kick_type = rng.random()
        if kick_type < 0.5:
            # Add a random positive offset to this element (and shift later ones to keep increasing)
            offset = rng.randint(1, 3)
            row[pos] += offset
            for j in range(pos + 1, k_plus_1):
                if row[j] <= row[j - 1]:
                    row[j] = row[j - 1] + 1
        else:
            # Swap the "gap" at pos with the gap at pos+1 (perturb spacing)
            if pos + 1 < k_plus_1:
                # Increase gap at pos, decrease gap at pos+1 if possible
                new_val = row[pos] + 1
                if new_val < row[pos + 1] - 1:
                    row[pos] = new_val

        # Re-pull this row to smallest, keeping other rows fixed
        used_diffs = set()
        for i, r in enumerate(best):
            used_diffs |= _row_diffs(r)
        # Remove the original row's diffs
        used_diffs -= _row_diffs(best[row_idx])

        # Pull the kicked row
        new_row = [0]
        stuck = False
        for pos2 in range(1, k_plus_1):
            x = new_row[-1] + 1
            while True:
                feasible = True
                for a in new_row:
                    d = x - a
                    if d <= 0 or d in used_diffs:
                        feasible = False
                        break
                if feasible:
                    new_row.append(x)
                    for a in new_row[:-1]:
                        used_diffs.add(x - a)
                    break
                x += 1
                if x > 100000:
                    stuck = True
                    break
            if stuck:
                break

        if not stuck and len(new_row) == k_plus_1:
            # Check if scope improved
            cand_scope = max(new_row)
            if cand_scope <= best_scope:
                # Also verify no global conflict (should be guaranteed by pull)
                best[row_idx] = new_row
                new_total_scope = max(max(r) for r in best)
                if new_total_scope < best_scope:
                    best_scope = new_total_scope

    return best


def _backtracking_search(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    Backtracking search with stronger pruning and randomized candidate ordering.
    Uses a boolean array for fast difference checking.
    """
    rng = random.Random(seed_offset * 12345 + 678)
    rows = []
    k_plus_1 = k + 1

    max_diff = max_scope
    used_diffs = [False] * (max_diff + 1)
    used_count = 0
    total_needed = n_rows * k * (k + 1) // 2

    def build_row(pos, current_mark, row_so_far):
        nonlocal used_count
        if time.time() > deadline:
            return None

        if pos == k_plus_1:
            return list(row_so_far)

        # Prune upper bound: need (k_plus_1 - pos - 1) more elements each >=1 apart
        max_x = max_scope - (k_plus_1 - pos - 1)
        if max_x <= current_mark:
            return None

        # Find valid candidates
        valid_candidates = []
        for x in range(current_mark + 1, max_x + 1):
            conflict = False
            for a in row_so_far:
                d = x - a
                if d > max_diff or used_diffs[d]:
                    conflict = True
                    break
            if not conflict:
                valid_candidates.append(x)
            if len(valid_candidates) >= 50:
                break

        if not valid_candidates:
            return None

        # Bias towards smaller values, with some randomness
        n_smallest = min(15, len(valid_candidates))
        candidates_to_try = valid_candidates[:n_smallest]

        if len(valid_candidates) > n_smallest and rng.random() < 0.25:
            extra = rng.sample(valid_candidates[n_smallest:], min(8, len(valid_candidates) - n_smallest))
            candidates_to_try = candidates_to_try + extra

        rng.shuffle(candidates_to_try)

        for x in candidates_to_try:
            new_diffs = []
            for a in row_so_far:
                d = x - a
                new_diffs.append(d)
                used_diffs[d] = True
            used_count += len(new_diffs)

            res = build_row(pos + 1, x, row_so_far + [x])
            if res is not None:
                return res

            for d in new_diffs:
                used_diffs[d] = False
            used_count -= len(new_diffs)

        return None

    for i in range(n_rows):
        row = build_row(1, 0, [0])
        if row is None:
            return None
        rows.append(row)
        for j in range(k_plus_1):
            for i2 in range(j + 1, k_plus_1):
                d = row[i2] - row[j]
                if not used_diffs[d]:
                    used_diffs[d] = True
                    used_count += 1

    return rows
