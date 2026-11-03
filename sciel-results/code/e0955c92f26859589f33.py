"""Difference Triangle Set (4,7) solver using bitmask backtracking with local search refinement."""
import time
import random
import math

def _row_diffs(row):
    """Compute all positive differences within a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            diffs.add(row[j] - ai)
    return diffs

def _check_valid(rows):
    """Check if all positive differences across all rows are distinct."""
    all_diffs = set()
    for row in rows:
        d = _row_diffs(row)
        if d & all_diffs:
            return False
        all_diffs |= d
    return True

def _max_entry(rows):
    """Return the maximum entry across all rows."""
    return max(max(r) for r in rows)

def _greedy_construct(n_rows, k):
    """Deterministic greedy: build each row by scanning upward, picking smallest valid value."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 1
        while len(row) < k + 1:
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
            x += 1
        rows.append(row)
    return rows

def _backtracking_search(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    Backtracking search to build n_rows of k+1 increasing integers starting at 0,
    with all pairwise differences distinct across all rows, and max element <= max_scope.
    Uses bitmask for fast difference checking.
    """
    rng = random.Random(seed_offset * 12345 + 678)
    rows = []
    k_plus_1 = k + 1
    used_mask = 0  # Python int, arbitrary precision

    def build_row(pos, current_mark, row_so_far, mask):
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
            # Check if any new difference is already used using bitmask
            conflict = False
            for a in row_so_far:
                d = x - a
                if d > max_scope:
                    conflict = True
                    break
                if (mask >> d) & 1:
                    conflict = True
                    break
            if not conflict:
                valid_candidates.append(x)

        if not valid_candidates:
            return None

        # Randomize the order of candidates to escape local minima
        # But bias towards smaller values to keep scope small
        n_smallest = min(15, len(valid_candidates))
        smallest = valid_candidates[:n_smallest]

        # With some probability, also consider larger candidates
        if len(valid_candidates) > n_smallest and rng.random() < 0.25:
            extra = rng.sample(valid_candidates[n_smallest:], min(8, len(valid_candidates) - n_smallest))
            candidates_to_try = smallest + extra
        else:
            candidates_to_try = smallest

        rng.shuffle(candidates_to_try)

        for x in candidates_to_try:
            # Mark new differences using bitmask
            mask_change = 0
            for a in row_so_far:
                d = x - a
                mask_change |= (1 << d)

            res = build_row(pos + 1, x, row_so_far + [x], mask | mask_change)
            if res is not None:
                return res

        return None

    for r in range(n_rows):
        if time.time() > deadline:
            return None
        row = build_row(0, -1, [0], used_mask)
        if row is None:
            return None
        rows.append(row)
        # Update mask with differences from this row
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                used_mask |= (1 << d)

    return rows

def _try_replace_row_bitmask(rows, row_idx, k, used_mask_others, max_scope, rng, max_tries=200):
    """
    Try to find a valid replacement for rows[row_idx] using bitmask.
    used_mask_others: bitmask of all differences used by other rows.
    Returns (new_row, new_mask) or None.
    """
    k_plus_1 = k + 1

    def build_row(pos, current_mark, row_so_far, local_mask):
        if time.time() > deadline_global:
            return None

        if pos == k_plus_1:
            return list(row_so_far)

        max_x = max_scope - (k_plus_1 - pos - 1)
        if max_x <= current_mark:
            return None

        valid_candidates = []
        for x in range(current_mark + 1, max_x + 1):
            conflict = False
            for a in row_so_far:
                d = x - a
                if d > max_scope:
                    conflict = True
                    break
                if ((used_mask_others >> d) & 1) or ((local_mask >> d) & 1):
                    conflict = True
                    break
            if not conflict:
                valid_candidates.append(x)

        if not valid_candidates:
            return None

        n_smallest = min(10, len(valid_candidates))
        candidates_to_try = valid_candidates[:n_smallest]
        if len(valid_candidates) > n_smallest and rng.random() < 0.3:
            extra = rng.sample(valid_candidates[n_smallest:], min(5, len(valid_candidates) - n_smallest))
            candidates_to_try = candidates_to_try + extra
        rng.shuffle(candidates_to_try)

        for x in candidates_to_try:
            mask_change = 0
            for a in row_so_far:
                d = x - a
                mask_change |= (1 << d)
            res = build_row(pos + 1, x, row_so_far + [x], local_mask | mask_change)
            if res is not None:
                return res
        return None

    for _ in range(max_tries):
        row = build_row(0, -1, [0], 0)
        if row is not None:
            # Compute the mask for this row
            row_mask = 0
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    row_mask |= (1 << d)
            return row, row_mask
    return None

def _improve_scope(rows, n_rows, k, deadline):
    """Local search to reduce scope by replacing rows with smaller ones."""
    global deadline_global
    deadline_global = deadline
    best_rows = [list(r) for r in rows]
    best_scope = _max_entry(best_rows)
    rng = random.Random(12345)

    # Build initial mask
    def compute_mask(rows):
        mask = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    mask |= (1 << d)
        return mask

    current_mask = compute_mask(best_rows)
    no_improve = 0
    max_no_improve = 50

    while time.time() < deadline - 2.0 and no_improve < max_no_improve:
        improved = False
        for row_idx in range(n_rows):
            # Compute mask of other rows
            mask_others = 0
            for j in range(n_rows):
                if j != row_idx:
                    row = best_rows[j]
                    for i in range(len(row)):
                        for jj in range(i + 1, len(row)):
                            d = row[jj] - row[i]
                            mask_others |= (1 << d)

            current_row = best_rows[row_idx]
            current_max = max(current_row)

            # Try to find a row with smaller max
            target_max = current_max - 1
            if target_max < k:  # Need at least k+1 elements, min max is k
                continue

            res = _try_replace_row_bitmask(best_rows, row_idx, k, mask_others, target_max, rng, max_tries=50)
            if res is not None:
                new_row, new_mask = res
                new_max = max(new_row)
                if new_max < current_max:
                    best_rows[row_idx] = new_row
                    improved = True

        if improved:
            no_improve = 0
            new_scope = _max_entry(best_rows)
            if new_scope < best_scope:
                best_scope = new_scope
        else:
            no_improve += 1

    return best_rows

def _polish_scope(rows, n_rows, k, deadline):
    """Final polish pass to squeeze out any remaining scope."""
    global deadline_global
    deadline_global = deadline
    best_rows = [list(r) for r in rows]
    best_scope = _max_entry(best_rows)
    rng = random.Random(54321)

    no_improve = 0
    max_no_improve = 30

    while time.time() < deadline - 1.0 and no_improve < max_no_improve:
        improved = False
        for row_idx in range(n_rows):
            mask_others = 0
            for j in range(n_rows):
                if j != row_idx:
                    row = best_rows[j]
                    for i in range(len(row)):
                        for jj in range(i + 1, len(row)):
                            d = row[jj] - row[i]
                            mask_others |= (1 << d)

            current_row = best_rows[row_idx]
            current_max = max(current_row)

            # Try to find a row with smaller max
            target_max = current_max - 1
            if target_max < k:
                continue

            res = _try_replace_row_bitmask(best_rows, row_idx, k, mask_others, target_max, rng, max_tries=30)
            if res is not None:
                new_row, new_mask = res
                new_max = max(new_row)
                if new_max < current_max:
                    best_rows[row_idx] = new_row
                    improved = True

        if improved:
            no_improve = 0
            new_scope = _max_entry(best_rows)
            if new_scope < best_scope:
                best_scope = new_scope
        else:
            no_improve += 1

    return best_rows

def construct(n_rows, k):
    """Main entry point: construct n_rows rows of k+1 increasing integers starting at 0."""
    global deadline_global
    t0 = time.time()
    deadline_global = t0 + 145.0  # Safety margin under 150s

    # Phase 1: Deterministic greedy baseline
    best_rows = _greedy_construct(n_rows, k)
    best_scope = _max_entry(best_rows)

    # Phase 2: Local improvement to reduce scope
    improved_rows = _improve_scope(best_rows, n_rows, k, deadline_global)
    improved_scope = _max_entry(improved_rows)
    if improved_scope < best_scope:
        best_rows = improved_rows
        best_scope = improved_scope

    # Phase 3: Polish pass
    polished_rows = _polish_scope(best_rows, n_rows, k, deadline_global)
    polished_scope = _max_entry(polished_rows)
    if polished_scope < best_scope:
        best_rows = polished_rows
        best_scope = polished_scope

    # Phase 4: Try backtracking search for even smaller scopes
    if time.time() < deadline_global - 10.0:
        target_scope = best_scope - 1
        lower_bound = n_rows * k * (k + 1) // 2 - 1
        while target_scope > lower_bound and time.time() < deadline_global - 5.0:
            found = False
            for attempt in range(20):
                if time.time() > deadline_global - 5.0:
                    break
                sol = _backtracking_search(n_rows, k, target_scope, deadline_global, attempt)
                if sol is not None:
                    new_scope = _max_entry(sol)
                    if new_scope < best_scope:
                        best_rows = sol
                        best_scope = new_scope
                        found = True
                        # Polish the found solution
                        polished = _polish_scope(best_rows, n_rows, k, deadline_global)
                        if _max_entry(polished) < best_scope:
                            best_rows = polished
                            best_scope = _max_entry(polished)
                        break
            if found:
                target_scope = best_scope - 1
            else:
                break

    # Final validity check
    if not _check_valid(best_rows):
        best_rows = _greedy_construct(n_rows, k)

    return best_rows
