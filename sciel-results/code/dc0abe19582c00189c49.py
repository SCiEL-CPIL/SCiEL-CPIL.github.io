"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k, seed=42):
    """
    Build initial solution greedily: for each row, pick the smallest valid
    next entry that doesn't create a difference collision.
    """
    rng = random.Random(seed)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            # Try candidates from last+1 upward
            for x in range(row[-1] + 1, row[-1] + 500):
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
                # Check Sidon: all pairwise diffs distinct
                # Since row is increasing, diffs x-a for a in row are all distinct
                # automatically (because a values are distinct). But we need
                # to check that the new diffs don't duplicate each other.
                # Actually, for a fixed x and distinct a's, x-a are distinct.
                # So the only constraint is no collision with used.
                row.append(x)
                used |= new_diffs
                break
            else:
                # Should not happen for reasonable parameters
                row.append(row[-1] + 1)
                used.add(row[-1] - row[-2])
        rows.append(row)
    return rows

def _row_diffs_set(row):
    """Compute the set of all pairwise differences in a row."""
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _compute_all_used(rows):
    """Compute the union of all difference sets across all rows."""
    used = set()
    for r in rows:
        used |= _row_diffs_set(r)
    return used

def _global_max(rows):
    """Return the maximum entry across all rows."""
    return max(max(r) for r in rows)

def _is_move_valid(cur, ri, pi, new_val, cur_used, cur_row_diffs, k):
    """
    Check if moving position pi in row ri to new_val is valid.
    Returns the new diff set for row ri if valid, else None.
    """
    row = cur[ri]
    # Check strict increase
    if pi > 0 and new_val <= row[pi - 1]:
        return None
    if pi < k and new_val >= row[pi + 1]:
        return None
    if pi == 0 and new_val != 0:
        return None  # First element must be 0

    # Build candidate row
    cand_row = row[:]
    cand_row[pi] = new_val

    # Compute new diffs for this row
    new_diffs = _row_diffs_set(cand_row)

    # Check Sidon property: all diffs in the row must be distinct
    # For a set of integers, pairwise differences can repeat.
    # We need to verify they're all distinct.
    diff_list = []
    n = len(cand_row)
    for i in range(n):
        for j in range(i + 1, n):
            diff_list.append(cand_row[j] - cand_row[i])
    if len(diff_list) != len(set(diff_list)):
        return None

    # Check no collision with other rows' diffs
    other_diffs = cur_used - cur_row_diffs[ri]
    if new_diffs & other_diffs:
        return None

    return new_diffs

def _try_move(cur, ri, pi, new_val, cur_used, cur_row_diffs, k, n_rows):
    """
    Try to move position pi in row ri to new_val.
    Returns (cand_row, new_diffs, new_max) if valid, else None.
    """
    new_diffs = _is_move_valid(cur, ri, pi, new_val, cur_used, cur_row_diffs, k)
    if new_diffs is None:
        return None

    cand_row = cur[ri][:]
    cand_row[pi] = new_val

    # Compute new global max
    other_max = 0
    for i, r in enumerate(cur):
        if i != ri:
            other_max = max(other_max, max(r))
    cand_max = max(other_max, max(cand_row))

    return (cand_row, new_diffs, cand_max)

def _rebuild_row_greedy(ri, cur, cur_used, cur_row_diffs, k, max_limit=None):
    """
    Rebuild row ri greedily given all other rows are fixed.
    Returns the new row and its diff set if successful, else (None, None).
    """
    other_diffs = cur_used - cur_row_diffs[ri]
    row = [0]
    while len(row) < k + 1:
        found = False
        start = row[-1] + 1
        end = max_limit if max_limit else start + 200
        for x in range(start, end):
            new_diffs = {x - a for a in row}
            if new_diffs & other_diffs:
                continue
            # Check Sidon within the partial row + x
            # For now, just check no collision with other_diffs
            # Full Sidon check will be done at the end
            row.append(x)
            found = True
            break
        if not found:
            return None, None

    # Verify full Sidon property
    diff_list = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diff_list.append(row[j] - row[i])
    if len(diff_list) != len(set(diff_list)):
        return None, None

    new_diffs = set(diff_list)
    # Verify no collision with other_diffs
    if new_diffs & other_diffs:
        return None, None

    return row, new_diffs

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s/150s limit

    # Initialize with greedy
    rows = _greedy_init(n_rows, k, seed=42)
    best = [r[:] for r in rows]
    best_val = _global_max(best)
    best_used = _compute_all_used(best)

    # Current state
    cur = [r[:] for r in best]
    cur_val = best_val
    cur_used = set(best_used)
    cur_row_diffs = [_row_diffs_set(r) for r in cur]

    # Simulated annealing parameters
    T = 50.0
    T_min = 0.01
    cool = 0.99997

    no_improve_steps = 0
    max_no_improve = 80000
    step = 0

    rng = random.Random(12345)

    while time.time() < deadline and no_improve_steps < max_no_improve:
        step += 1

        # Pick a random row and position
        ri = rng.randrange(n_rows)
        pi = rng.randrange(0, k + 1)
        if pi == 0:
            # First position is always 0, so skip or pick another
            pi = rng.randrange(1, k + 1)
        old_val = cur[ri][pi]

        # Generate candidate moves
        candidates = []

        # Try small decreases
        for d in range(1, 15):
            new_val = old_val - d
            if pi > 0 and new_val <= cur[ri][pi - 1]:
                break
            candidates.append(new_val)

        # Try small increases
        for d in range(1, 15):
            new_val = old_val + d
            if pi < k and new_val >= cur[ri][pi + 1]:
                break
            candidates.append(new_val)

        # Occasionally try larger jumps
        if rng.random() < 0.3:
            for d in range(15, 40):
                new_val = old_val - d
                if pi > 0 and new_val > cur[ri][pi - 1]:
                    candidates.append(new_val)
                new_val = old_val + d
                if pi < k and new_val < cur[ri][pi + 1]:
                    candidates.append(new_val)

        # Occasionally try row re-solve
        if rng.random() < 0.12:
            # Try to rebuild this row greedily with a tighter bound
            other_max = 0
            for i in range(n_rows):
                if i != ri:
                    other_max = max(other_max, max(cur[i]))
            # Try to find a row with max <= cur_val - 1
            target_max = cur_val - 1
            if target_max < k:
                target_max = cur_val
            new_row, new_diffs = _rebuild_row_greedy(ri, cur, cur_used, cur_row_diffs, k, max_limit=target_max)
            if new_row is not None:
                cand_max = max(other_max, max(new_row))
                delta_e = cand_max - cur_val
                accept = False
                if delta_e < 0:
                    accept = True
                elif delta_e == 0:
                    if rng.random() < 0.2:
                        accept = True
                else:
                    if T > T_min:
                        prob = 1.0 / (1.0 + delta_e / max(T, 0.01))
                        if rng.random() < prob:
                            accept = True

                if accept:
                    cur_used -= cur_row_diffs[ri]
                    cur_used |= new_diffs
                    cur[ri] = new_row
                    cur_row_diffs[ri] = new_diffs
                    cur_val = cand_max

                    if cur_val < best_val:
                        best_val = cur_val
                        best = [r[:] for r in cur]
                        best_used = set(cur_used)
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1

                    T = max(T_min, T * cool)
                    continue

        rng.shuffle(candidates)

        moved = False
        for new_val in candidates:
            result = _try_move(cur, ri, pi, new_val, cur_used, cur_row_diffs, k, n_rows)
            if result is None:
                continue

            cand_row, new_diffs, cand_max = result
            delta_e = cand_max - cur_val
            accept = False
            if delta_e < 0:
                accept = True
            elif delta_e == 0:
                if rng.random() < 0.25:
                    accept = True
            else:
                if T > T_min:
                    prob = 1.0 / (1.0 + delta_e / max(T, 0.01))
                    if rng.random() < prob:
                        accept = True

            if accept:
                cur_used -= cur_row_diffs[ri]
                cur_used |= new_diffs
                cur[ri] = cand_row
                cur_row_diffs[ri] = new_diffs
                cur_val = cand_max

                if cur_val < best_val:
                    best_val = cur_val
                    best = [r[:] for r in cur]
                    best_used = set(cur_used)
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1

                moved = True
                break

        if not moved:
            no_improve_steps += 1

        T = max(T_min, T * cool)

        # Periodic restart from best
        if no_improve_steps > 25000:
            cur = [r[:] for r in best]
            cur_val = best_val
            cur_used = set(best_used)
            cur_row_diffs = [_row_diffs_set(r) for r in cur]
            T = 50.0
            no_improve_steps = 0

    # Final polish: try to rebuild each row to minimize its max
    for _ in range(5):
        for ri in range(n_rows):
            other_max = 0
            for i in range(n_rows):
                if i != ri:
                    other_max = max(other_max, max(cur[i]))
            target_max = best_val - 1
            if target_max < k:
                target_max = best_val
            new_row, new_diffs = _rebuild_row_greedy(ri, cur, cur_used, cur_row_diffs, k, max_limit=target_max)
            if new_row is not None:
                cand_max = max(other_max, max(new_row))
                if cand_max < best_val:
                    cur_used -= cur_row_diffs[ri]
                    cur_used |= new_diffs
                    cur[ri] = new_row
                    cur_row_diffs[ri] = new_diffs
                    best_val = cand_max
                    best = [r[:] for r in cur]
                    best_used = set(cur_used)

    # Validate best
    if not _validate(best, n_rows, k):
        best = _greedy_init(n_rows, k, seed=42)

    return best

def _validate(rows, n_rows, k):
    """Check all rows are valid and all differences are globally distinct."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        # Check Sidon
        diff_list = []
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diff_list.append(row[j] - row[i])
        if len(diff_list) != len(set(diff_list)):
            return False
        for d in diff_list:
            if d in used:
                return False
            used.add(d)
    return True

# EVOLVE-BLOCK-END
