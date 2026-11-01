"""Difference triangle set (4,7): minimize largest entry via Simulated Annealing."""
import time
import random
import math

_DEADLINE_MARGIN = 15.0  # seconds to spare before the 150s kill


def _row_diffs(row):
    """Set of positive differences within a single row."""
    d = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            d.add(row[j] - ai)
    return d


def _all_diffs(rows):
    """Combined set of all positive differences across all rows."""
    used = set()
    for row in rows:
        used |= _row_diffs(row)
    return used


def _is_valid(rows):
    """Check all positive differences are distinct across all rows."""
    used = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                d = row[j] - ai
                if d in used:
                    return False
                used.add(d)
    return True


def _max_entry(rows):
    return max(max(r) for r in rows)


def _greedy_init(n_rows, k):
    """Greedy construction to get a valid starting point."""
    used, rows = set(), []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _row_diffs_set(row):
    """Return set of diffs for a row (recomputed)."""
    d = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            d.add(row[j] - ai)
    return d


def construct(n_rows, k):
    t_start = time.time()
    deadline = t_start + 150.0 - _DEADLINE_MARGIN

    # Start with greedy solution
    rows = _greedy_init(n_rows, k)
    best_rows = [r[:] for r in rows]
    best_max = _max_entry(best_rows)

    # Precompute diff sets for quick validity check
    # We'll maintain a global used set and per-row diff sets
    def compute_state(rows):
        """Return (used_set, valid_bool, max_entry)."""
        used = set()
        valid = True
        mx = 0
        for row in rows:
            n = len(row)
            for i in range(n):
                ai = row[i]
                for j in range(i + 1, n):
                    d = row[j] - ai
                    if d in used:
                        valid = False
                    else:
                        used.add(d)
            if row[-1] > mx:
                mx = row[-1]
        return used, valid, mx

    used, valid, cur_max = compute_state(rows)
    cur_max = best_max

    rng = random.Random(42)
    T = 50.0
    T_min = 0.1
    cooling = 0.9995

    step = 0
    while time.time() < deadline and T > T_min:
        # Pick a random row and a random mark (not the first, not necessarily last)
        ri = rng.randrange(n_rows)
        # Pick a position to move: any position except index 0
        # We'll pick a position and a new value
        pos = rng.randrange(1, k + 1)
        row = rows[ri]

        # Compute current diffs for this row
        old_diffs = _row_diffs_set(row)

        # Try moving mark at position pos to a new value
        # New value should keep the row strictly increasing
        lo = row[pos - 1] + 1
        hi = row[pos + 1] - 1 if pos < k else lo + int(T * 3) + 20

        if hi < lo:
            hi = lo

        # Sample new value
        if hi - lo < 50:
            new_val = rng.randrange(lo, hi + 1)
        else:
            # Gaussian-ish around current
            center = row[pos]
            spread = max(5, int(T * 2))
            new_val = int(rng.gauss(center, spread))
            new_val = max(lo, min(hi, new_val))

        if new_val == row[pos]:
            continue

        # Check if new value creates conflicts within this row
        # Compute new diffs for this row
        old_val = row[pos]
        # New row would be: row with row[pos] = new_val
        # New diffs involving pos:
        new_diffs_in_row = set()
        valid_in_row = True
        for j in range(pos):
            d = new_val - row[j]
            if d <= 0:
                valid_in_row = False
                break
            new_diffs_in_row.add(d)
        if not valid_in_row:
            continue
        for j in range(pos + 1, k + 1):
            d = row[j] - new_val
            if d <= 0:
                valid_in_row = False
                break
            new_diffs_in_row.add(d)
        if not valid_in_row:
            continue

        # Diffs NOT involving pos remain the same
        # Diffs involving pos: old ones are old_val - row[j] for j<pos, row[j] - old_val for j>pos
        # New ones: new_val - row[j] for j<pos, row[j] - new_val for j>pos

        # Check if any new diff conflicts with other rows' diffs
        # We need the set of diffs from other rows
        # To be efficient, let's just check new_diffs_in_row against (used - old_diffs)
        other_diffs = used - old_diffs
        conflict = new_diffs_in_row & other_diffs
        if conflict:
            # This move creates conflicts -> reject (or accept with prob if we track count)
            # For simplicity, reject if conflicts
            # But SA should allow some bad moves... let's count conflicts
            n_conflicts = len(conflict)
            # Accept with probability exp(-n_conflicts / T) if n_conflicts small
            if n_conflicts > 3:
                continue
            if rng.random() > math.exp(-n_conflicts / T):
                continue

        # Perform the move
        row[pos] = new_val
        new_diffs_full = _row_diffs_set(row)

        # Recompute used
        # used = (used - old_diffs) | new_diffs_full
        # But we need to check global validity
        # Let's just check if new_diffs_full has any conflict with other_diffs
        global_conflict = new_diffs_full & other_diffs

        new_max = max(max(r) for r in rows)

        if not global_conflict and new_max < best_max:
            # Valid and better
            best_rows = [r[:] for r in rows]
            best_max = new_max
            used = (used - old_diffs) | new_diffs_full
            cur_max = new_max
        else:
            # Accept with SA probability
            # Cost = number of conflicts + max(0, new_max - best_max)
            cost = len(global_conflict) + max(0, new_max - best_max)
            if cost == 0:
                # Neutral or better in max
                used = (used - old_diffs) | new_diffs_full
                cur_max = new_max
            else:
                if rng.random() < math.exp(-cost / T):
                    used = (used - old_diffs) | new_diffs_full
                    cur_max = new_max
                else:
                    # Revert
                    row[pos] = old_val
                    # used stays the same
                    continue

        T *= cooling
        step += 1

    # Return the best valid solution found
    if _is_valid(best_rows):
        return best_rows
    # Fallback: return greedy (always valid)
    return _greedy_init(n_rows, k)
