"""Difference triangle set (7,5): minimize the largest entry.
Combined approach: greedy baseline + simulated annealing (coordinated multi-row) + local search polish."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _all_diffs(rows):
    """Return the complete set of all differences across all rows."""
    used = set()
    for row in rows:
        used |= _row_diffs(row)
    return used


def _valid(rows):
    """Check rows are valid: start at 0, strictly increasing, globally distinct diffs."""
    all_d = set()
    for row in rows:
        if not row or row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        rd = _row_diffs(row)
        if rd & all_d:
            return False
        all_d |= rd
    return True


def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)


def _greedy_construct(n_rows, k, seed_offset=0):
    """Greedy construction: pick smallest valid value at each step."""
    rng = random.Random(42 + seed_offset)
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
            row.append(x)
            used |= new_diffs
        rows.append(row)
    return rows


def _greedy_randomized(n_rows, k, seed_offset=0, max_step=12):
    """Greedy with randomization: try random steps, fall back to sequential."""
    rng = random.Random(42 + seed_offset)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            found = False
            for _attempt in range(40):
                step = rng.randint(1, max_step)
                x = row[-1] + step
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    found = True
                    break
            if not found:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        break
                    x += 1
            row.append(x)
            used |= new_diffs
        rows.append(row)
    return rows


def _simulated_annealing(rows, k, deadline, rng, t0=30.0, cooling=0.9998, t_min=0.05):
    """Simulated annealing: coordinated single-element moves across all rows.
    Accepts worsening moves with temperature-based probability to escape local optima."""
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    used = _all_diffs(rows)
    current_scope = best_scope
    n_rows = len(rows)
    row_len = k + 1
    T = t0

    while time.time() < deadline and T > t_min:
        # Pick random row and position
        ri = rng.randint(0, n_rows - 1)
        row = rows[ri]
        pos = rng.randint(1, row_len - 1)

        old_val = row[pos]
        low = row[pos - 1] + 1
        if pos + 1 < row_len:
            high = row[pos + 1] - 1
        else:
            high = old_val + 20  # allow some expansion for last element

        if low >= high:
            T *= cooling
            continue

        # Sample new value: bias toward smaller values (to reduce scope)
        if rng.random() < 0.6:
            # Try to reduce
            new_val = rng.randint(low, min(high, old_val + 5))
        else:
            new_val = rng.randint(low, high)

        if new_val == old_val:
            T *= cooling
            continue

        # Compute old diffs involving position pos
        old_diffs = set()
        for i in range(pos):
            old_diffs.add(old_val - row[i])
        for i in range(pos + 1, row_len):
            old_diffs.add(row[i] - old_val)

        # Compute new diffs
        new_diffs = set()
        for i in range(pos):
            new_diffs.add(new_val - row[i])
        for i in range(pos + 1, row_len):
            new_diffs.add(row[i] - new_val)

        # Check conflicts with other rows' diffs
        conflict = new_diffs & (used - old_diffs)
        if conflict:
            T *= cooling
            continue

        # Compute new scope
        if pos == row_len - 1:
            new_scope = max(_scope(rows[:ri] + rows[ri + 1:]), new_val) if n_rows > 1 else new_val
        else:
            new_scope = max(max(row[:pos] + [new_val] + row[pos + 1:]), _scope(rows[:ri] + rows[ri + 1:])) if n_rows > 1 else max(row[:pos] + [new_val] + row[pos + 1:])

        delta = new_scope - current_scope

        if delta <= 0 or rng.random() < math.exp(-delta / T):
            # Accept
            row[pos] = new_val
            used -= old_diffs
            used |= new_diffs
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]

        T *= cooling

    return best_rows


def _local_search_reduce(rows, k, deadline, rng):
    """Targeted local search: aggressively try to reduce the maximum element(s)."""
    rows = [list(r) for r in rows]
    used = _all_diffs(rows)
    best_scope = _scope(rows)
    row_len = k + 1

    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_scope = _scope(rows)

        for ri in range(len(rows)):
            if time.time() > deadline:
                return [list(r) for r in rows]
            row = rows[ri]
            for pos in range(row_len - 1, 0, -1):
                if time.time() > deadline:
                    return [list(r) for r in rows]
                old_val = row[pos]
                if old_val < current_scope - 1:
                    continue

                low = row[pos - 1] + 1
                if pos + 1 < row_len:
                    high = row[pos + 1] - 1
                else:
                    high = old_val - 1

                if low > high:
                    continue

                # Try values from high down to low (prefer smallest reduction that works)
                for new_val in range(high, low - 1, -1):
                    if new_val >= old_val:
                        continue

                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(old_val - row[i])
                    for i in range(pos + 1, row_len):
                        old_diffs.add(row[i] - old_val)

                    new_diffs = set()
                    for i in range(pos):
                        new_diffs.add(new_val - row[i])
                    for i in range(pos + 1, row_len):
                        new_diffs.add(row[i] - new_val)

                    conflict = new_diffs & (used - old_diffs)
                    if conflict:
                        continue

                    # Accept
                    row[pos] = new_val
                    used -= old_diffs
                    used |= new_diffs
                    improved = True
                    break

    return rows


def _anneal_restart(rows, k, deadline, rng, t0=15.0, cooling=0.9999, t_min=0.02):
    """Shorter SA restart with different parameters for diversity."""
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    used = _all_diffs(rows)
    current_scope = best_scope
    n_rows = len(rows)
    row_len = k + 1
    T = t0

    while time.time() < deadline and T > t_min:
        ri = rng.randint(0, n_rows - 1)
        row = rows[ri]
        pos = rng.randint(1, row_len - 1)

        old_val = row[pos]
        low = row[pos - 1] + 1
        if pos + 1 < row_len:
            high = row[pos + 1] - 1
        else:
            high = old_val + 15

        if low >= high:
            T *= cooling
            continue

        new_val = rng.randint(low, high)
        if new_val == old_val:
            T *= cooling
            continue

        old_diffs = set()
        for i in range(pos):
            old_diffs.add(old_val - row[i])
        for i in range(pos + 1, row_len):
            old_diffs.add(row[i] - old_val)

        new_diffs = set()
        for i in range(pos):
            new_diffs.add(new_val - row[i])
        for i in range(pos + 1, row_len):
            new_diffs.add(row[i] - new_val)

        conflict = new_diffs & (used - old_diffs)
        if conflict:
            T *= cooling
            continue

        # Quick scope estimate
        new_row_max = max(row[:pos] + [new_val] + row[pos + 1:])
        other_max = max(max(r) for r in rows[:ri] + rows[ri + 1:]) if n_rows > 1 else 0
        new_scope = max(new_row_max, other_max)
        delta = new_scope - current_scope

        if delta <= 0 or rng.random() < math.exp(-delta / T):
            row[pos] = new_val
            used -= old_diffs
            used |= new_diffs
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]

        T *= cooling

    return best_rows


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    rng = random.Random(42)

    # Phase 1: Greedy baseline (fast, <0.1s)
    rows = _greedy_construct(n_rows, k)
    best_rows = rows
    best_scope = _scope(rows)

    # Phase 2: Multiple randomized greedy starts, keep best
    for seed in range(1, 8):
        if time.time() > deadline - 100.0:
            break
        r2 = _greedy_randomized(n_rows, k, seed_offset=seed * 7)
        s2 = _scope(r2)
        if s2 < best_scope:
            best_scope = s2
            best_rows = r2

    # Phase 3: Main simulated annealing (bulk of time budget)
    sa_deadline = deadline - 25.0
    if time.time() < sa_deadline:
        sa_result = _simulated_annealing(best_rows, k, sa_deadline, rng, t0=30.0, cooling=0.9998, t_min=0.05)
        sa_scope = _scope(sa_result)
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = sa_result

    # Phase 4: Second SA restart with different parameters
    if time.time() < deadline - 15.0:
        sa2_deadline = deadline - 15.0
        sa2_result = _anneal_restart(best_rows, k, sa2_deadline, rng, t0=10.0, cooling=0.99995, t_min=0.01)
        sa2_scope = _scope(sa2_result)
        if sa2_scope < best_scope:
            best_scope = sa2_scope
            best_rows = sa2_result

    # Phase 5: Local search polish (targeted reduction of max element)
    if time.time() < deadline - 3.0:
        polished = _local_search_reduce(best_rows, k, deadline - 3.0, rng)
        pol_scope = _scope(polished)
        if pol_scope < best_scope:
            best_scope = pol_scope
            best_rows = polished

    return best_rows

# EVOLVE-BLOCK-END
