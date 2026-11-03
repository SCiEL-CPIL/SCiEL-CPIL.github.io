"""Combined program: CSP-style initialization with incremental-diff SA optimization."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _get_scope(rows):
    """Return the largest entry in any row."""
    return max(max(row) for row in rows)

def _is_valid(rows):
    """Check if all differences are distinct across all rows."""
    seen = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _row_diffs(row):
    """Compute all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _greedy_init(n_rows, k, diversity=0.0):
    """
    Greedy initialization with optional diversity.
    diversity: probability of skipping a few valid values to create more spread.
    Returns (rows, used_set).
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                continue
            # With some probability, skip to create diversity
            if diversity > 0 and random.random() < diversity:
                skipped = False
                for _ in range(random.randint(1, 3)):
                    x += 1
                    new_diffs = {x - a for a in row}
                    if new_diffs & used:
                        continue
                    row.append(x)
                    used |= new_diffs
                    skipped = True
                    break
                if not skipped:
                    # Fall through to take current x if valid
                    if not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
                    else:
                        continue
            else:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows, used

def _rebuild_row(forbidden, k, max_scope, deadline):
    """
    Rebuild a row to fit within max_scope, avoiding forbidden differences.
    Uses DFS with pruning. Returns (row, diffs) or (None, None).
    """
    n = k + 1
    best_row = None
    best_last = max_scope + 1

    def dfs(row, local_used, current_max):
        nonlocal best_row, best_last
        if time.time() > deadline:
            return False

        if len(row) == n:
            if row[-1] < best_last:
                best_last = row[-1]
                best_row = list(row)
            return True

        last = row[-1]
        remaining = n - len(row)
        # Must leave room for remaining elements
        if max_scope - last < remaining:
            return False

        min_next = last + 1
        # Upper bound: leave at least (remaining) room
        max_next = max_scope - (remaining - 1) if remaining > 1 else max_scope

        for x in range(min_next, max_next + 1):
            new_diffs = {x - a for a in row}
            if new_diffs & local_used:
                continue
            if new_diffs & forbidden:
                continue

            row.append(x)
            local_used |= new_diffs

            if dfs(row, local_used, x):
                return True

            row.pop()
            local_used -= new_diffs

        return False

    dfs([0], set(), 0)

    if best_row is not None:
        final_diffs = _row_diffs(best_row)
        return best_row, final_diffs
    return None, None

def _sa_optimize(rows, n_rows, k, deadline):
    """
    Simulated annealing with incremental diff updates.
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _get_scope(rows)

    # Build used set
    used = set()
    for row in rows:
        used |= _row_diffs(row)

    current_scope = best_scope
    T = 50.0
    cooling = 0.999
    min_T = 0.1

    stagnation = 0
    max_stagnation = 5000

    while T > min_T and time.time() < deadline - 3:
        if stagnation > max_stagnation:
            # Restart with higher temperature
            T = max(T, 30.0)
            stagnation = 0

        move_type = random.random()

        if move_type < 0.5:
            # Single element move
            r_idx = random.randint(0, n_rows - 1)
            row = rows[r_idx]
            pos = random.randint(1, len(row) - 1)
            old_val = row[pos]

            # Bias towards reducing scope
            if row[pos] >= best_scope - 5:
                new_val = random.randint(1, max(1, old_val - 1)) if old_val > 1 else 1
            else:
                new_val = random.randint(1, max(1, current_scope + 5))

            if new_val == old_val:
                T *= cooling
                continue

            if pos > 0 and new_val <= row[pos - 1]:
                T *= cooling
                continue
            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                T *= cooling
                continue

            # Compute old diffs involving pos
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(old_val - row[i])
            for j in range(pos + 1, len(row)):
                old_diffs.add(row[j] - old_val)

            # Compute new diffs involving pos
            new_diffs = set()
            for i in range(pos):
                new_diffs.add(new_val - row[i])
            for j in range(pos + 1, len(row)):
                new_diffs.add(row[j] - new_val)

            # Check conflict
            remaining = used - old_diffs
            if new_diffs & remaining:
                T *= cooling
                continue

            # Apply move
            row[pos] = new_val
            used = (used - old_diffs) | new_diffs
            new_scope = _get_scope(rows)
            delta = new_scope - current_scope

            if delta <= 0 or random.random() < math.exp(-delta / T):
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                # Revert
                row[pos] = old_val
                used = (used - new_diffs) | old_diffs

        elif move_type < 0.8:
            # Row rebuild: try to rebuild a random row to fit in current scope
            r_idx = random.randint(0, n_rows - 1)
            old_row = rows[r_idx]
            old_diffs = _row_diffs(old_row)
            remaining_used = used - old_diffs

            target_scope = min(current_scope, best_scope)
            result = _rebuild_row(remaining_used, k, target_scope, deadline)
            if result[0] is not None:
                new_row, new_diffs = result
                if not (new_diffs & remaining_used):
                    rows[r_idx] = new_row
                    used = remaining_used | new_diffs
                    new_scope = _get_scope(rows)
                    delta = new_scope - current_scope

                    if delta <= 0 or random.random() < math.exp(-delta / T):
                        current_scope = new_scope
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [list(r) for r in rows]
                            stagnation = 0
                        else:
                            stagnation += 1
                    else:
                        # Revert
                        rows[r_idx] = old_row
                        used = (used - new_diffs) | old_diffs
            else:
                stagnation += 1

        else:
            # Swap two elements in the same row
            r_idx = random.randint(0, n_rows - 1)
            row = rows[r_idx]
            if len(row) < 3:
                T *= cooling
                continue
            pos1, pos2 = random.sample(range(1, len(row)), 2)
            if pos1 > pos2:
                pos1, pos2 = pos2, pos1

            # Check if swap maintains increasing order
            v1, v2 = row[pos1], row[pos2]
            if v1 == v2:
                T *= cooling
                continue

            # For swap to maintain order, we need:
            # If pos1 and pos2 are not adjacent, the elements between must be
            # between v1 and v2 in the right order after swap.
            # Simple check: after swap, is the row still increasing?
            if pos2 - pos1 == 1:
                # Adjacent: need row[pos1-1] < v2 and v1 < row[pos2+1] (if exists)
                if pos1 > 0 and row[pos1 - 1] >= v2:
                    T *= cooling
                    continue
                if pos2 < len(row) - 1 and v1 >= row[pos2 + 1]:
                    T *= cooling
                    continue
            else:
                # Non-adjacent: check all constraints
                if pos1 > 0 and row[pos1 - 1] >= v2:
                    T *= cooling
                    continue
                if pos2 < len(row) - 1 and v1 >= row[pos2 + 1]:
                    T *= cooling
                    continue
                # Check intermediate elements
                valid = True
                for p in range(pos1 + 1, pos2):
                    if row[p] <= v1 or row[p] >= v2:
                        valid = False
                        break
                if not valid:
                    T *= cooling
                    continue

            # Compute old diffs
            old_diffs = set()
            for i in range(pos1):
                old_diffs.add(v1 - row[i])
            for j in range(pos1 + 1, len(row)):
                if j != pos2:
                    old_diffs.add(row[j] - v1)
            old_diffs.add(v2 - v1)
            for i in range(pos2):
                if i != pos1:
                    old_diffs.add(v2 - row[i])
            for j in range(pos2 + 1, len(row)):
                old_diffs.add(row[j] - v2)

            # Compute new diffs after swap
            new_diffs = set()
            for i in range(pos1):
                new_diffs.add(v2 - row[i])
            for j in range(pos1 + 1, len(row)):
                if j != pos2:
                    new_diffs.add(row[j] - v2)
            new_diffs.add(v1 - v2)
            for i in range(pos2):
                if i != pos1:
                    new_diffs.add(v1 - row[i])
            for j in range(pos2 + 1, len(row)):
                new_diffs.add(row[j] - v1)

            if new_diffs & (used - old_diffs):
                T *= cooling
                continue

            # Apply swap
            row[pos1], row[pos2] = v2, v1
            used = (used - old_diffs) | new_diffs
            new_scope = _get_scope(rows)
            delta = new_scope - current_scope

            if delta <= 0 or random.random() < math.exp(-delta / T):
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                # Revert
                row[pos1], row[pos2] = v1, v2
                used = (used - new_diffs) | old_diffs

        T *= cooling

    return best_rows

def construct(n_rows, k):
    deadline = time.time() + 145.0  # Safety margin before 150s limit

    # Start with greedy solution, with some diversity
    rows, used = _greedy_init(n_rows, k, diversity=0.3)

    # Verify validity
    if not _is_valid(rows):
        rows, used = _greedy_init(n_rows, k, diversity=0.0)

    # Try to improve initial solution with row rebuilding
    initial_scope = _get_scope(rows)
    for attempt in range(200):
        if time.time() > deadline - 30:
            break
        target_scope = initial_scope - 1
        if target_scope < 0:
            break

        # Find row with max entry
        max_row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        if rows[max_row_idx][-1] <= target_scope:
            continue

        old_row = rows[max_row_idx]
        old_diffs = _row_diffs(old_row)
        remaining_used = used - old_diffs

        result = _rebuild_row(remaining_used, k, target_scope, deadline)
        if result[0] is not None:
            new_row, new_diffs = result
            if not (new_diffs & remaining_used):
                rows[max_row_idx] = new_row
                used = remaining_used | new_diffs
                initial_scope = _get_scope(rows)

    # Optimize with SA
    rows = _sa_optimize(rows, n_rows, k, deadline)

    # Final validation
    if not _is_valid(rows):
        return _greedy_init(n_rows, k, diversity=0.0)[0]

    return rows
# EVOLVE-BLOCK-END
