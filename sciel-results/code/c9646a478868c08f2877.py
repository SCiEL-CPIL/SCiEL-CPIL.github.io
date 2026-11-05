import math
import random
import time

# EVOLVE-BLOCK-START


def _total_scope(rows):
    """Return the maximum entry across all rows."""
    if not rows:
        return 0
    return max(max(r) for r in rows)


def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _all_diffs_used(rows):
    """Return the set of all differences used across all rows."""
    used = set()
    for r in rows:
        used |= _row_diffs(r)
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


def _greedy_construct(n_rows, k, randomize=False, seed_offset=0):
    """Greedy construction with optional randomization as a fast baseline."""
    rng = random.Random(42 + seed_offset)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            if randomize:
                max_step = max(1, int((k + 1) * 2.5))
                found = False
                for _attempt in range(50):
                    step = rng.randint(1, max_step)
                    x = row[-1] + step
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        found = True
                        break
                if not found:
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            break
                        x += 1
            else:
                x += 1
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
            row.append(x)
            used |= new_diffs
        rows.append(row)
    return rows


def _backtrack_row(k, used, max_val, rng, deadline):
    """
    Build a single row of k+1 elements starting at 0, all elements <= max_val,
    with all differences avoiding the global 'used' set.
    Returns row or None.
    """
    if time.time() > deadline:
        return None
    row = [0]

    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == k + 1:
            return True

        low = row[-1] + 1
        high = max_val

        # Candidate ordering: ascending, with occasional randomization
        candidates = list(range(low, high + 1))
        if rng.random() < 0.25:
            rng.shuffle(candidates)

        for val in candidates:
            new_diffs = {val - a for a in row}
            if new_diffs & used:
                continue
            row.append(val)
            used.update(new_diffs)
            if backtrack(idx + 1):
                return True
            used.difference_update(new_diffs)
            row.pop()

        return False

    if backtrack(1):
        return row
    return None


def _backtrack_construct(n_rows, k, rng, deadline, max_scope=None):
    """
    Constraint-propagation backtracking: build rows one at a time under a
    tightening max_val bound. Randomized restarts with different seeds.
    Returns the best (lowest-scope) valid set found, or None.
    """
    best_solution = None
    best_scope = max_scope if max_scope else 999999

    # Estimate an initial feasible max_val from greedy
    g = _greedy_construct(n_rows, k)
    g_scope = _total_scope(g)

    # Try tightening bounds from best-known down
    for max_val in range(g_scope, 104, -1):
        if time.time() > deadline:
            break
        # A few randomized attempts at this bound
        for attempt in range(3):
            if time.time() > deadline:
                break
            attempt_rng = random.Random(1000 * max_val + attempt + 7)
            used = set()
            rows = []
            ok = True
            for r in range(n_rows):
                if time.time() > deadline:
                    ok = False
                    break
                row = _backtrack_row(k, used, max_val, attempt_rng, deadline)
                if row is None:
                    ok = False
                    break
                rows.append(row)
                used |= _row_diffs(row)
            if ok and len(rows) == n_rows:
                scope = _total_scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_solution = [r[:] for r in rows]
                # If we hit the target bound, stop tightening
                if max_val == scope:
                    return best_solution

    return best_solution


def _local_search_scope(rows, k, rng, deadline):
    """
    Local search to reduce the scope (maximum entry).
    Tries to shrink elements in rows.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)

    improved = True
    num_passes = 0
    while improved and time.time() < deadline:
        improved = False
        num_passes += 1
        if num_passes % 3 == 0:
            row_indices = list(range(len(rows)))
            rng.shuffle(row_indices)
        elif num_passes % 3 == 1:
            row_indices = sorted(range(len(rows)), key=lambda i: -rows[i][-1])
        else:
            row_indices = sorted(range(len(rows)), key=lambda i: rows[i][-1])

        for idx in row_indices:
            if time.time() > deadline:
                break
            row = rows[idx]

            for pos in range(len(row) - 1, 0, -1):
                if time.time() > deadline:
                    break
                old_val = row[pos]
                for new_val in range(old_val - 1, row[pos - 1], -1):
                    old_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            old_diffs.add(old_val - row[i])
                        else:
                            old_diffs.add(row[i] - old_val)
                    new_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            new_diffs.add(new_val - row[i])
                        else:
                            new_diffs.add(row[i] - new_val)
                    remaining = used - old_diffs
                    if not (new_diffs & remaining):
                        row[pos] = new_val
                        used = (used - old_diffs) | new_diffs
                        improved = True
                        break

    return rows


def _simulated_annealing_scope(rows, k, rng, deadline):
    """
    Simulated annealing to escape local optima in scope minimization.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    current_scope = _total_scope(rows)
    best_rows = [r[:] for r in rows]
    best_scope = current_scope

    temperature = 10.0
    cooling_rate = 0.995

    while time.time() < deadline and temperature > 0.1:
        idx = rng.randint(0, len(rows) - 1)
        pos = rng.randint(1, len(rows[idx]) - 1)
        row = rows[idx]
        old_val = row[pos]

        delta = rng.randint(-5, 5)
        new_val = old_val + delta
        if new_val <= row[pos - 1] or (pos + 1 < len(row) and new_val >= row[pos + 1]):
            continue

        old_diffs = set()
        for i in range(len(row)):
            if i == pos:
                continue
            if i < pos:
                old_diffs.add(old_val - row[i])
            else:
                old_diffs.add(row[i] - old_val)

        new_diffs = set()
        for i in range(len(row)):
            if i == pos:
                continue
            if i < pos:
                new_diffs.add(new_val - row[i])
            else:
                new_diffs.add(row[i] - new_val)

        remaining = used - old_diffs
        if new_diffs & remaining:
            continue

        new_scope = max(current_scope, new_val)
        if new_val == current_scope:
            other_max = max(max(r) for i, r in enumerate(rows) if i != idx)
            new_scope = max(other_max, new_val)

        delta_scope = new_scope - current_scope
        if delta_scope < 0 or rng.random() < math.exp(-delta_scope / temperature):
            row[pos] = new_val
            used = (used - old_diffs) | new_diffs
            current_scope = new_scope

            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in rows]

        temperature *= cooling_rate

    return best_rows


def _polish_pass(rows, k, deadline):
    """
    High-precision polish: try every possible reduction for each position.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)

    for _ in range(3):
        if time.time() > deadline:
            break
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(len(rows)):
                if time.time() > deadline:
                    break
                row = rows[idx]
                for pos in range(len(row) - 1, 0, -1):
                    if time.time() > deadline:
                        break
                    old_val = row[pos]
                    for new_val in range(old_val - 1, row[pos - 1], -1):
                        if time.time() > deadline:
                            break
                        old_diffs = set()
                        for i in range(len(row)):
                            if i == pos:
                                continue
                            if i < pos:
                                old_diffs.add(old_val - row[i])
                            else:
                                old_diffs.add(row[i] - old_val)
                        new_diffs = set()
                        for i in range(len(row)):
                            if i == pos:
                                continue
                            if i < pos:
                                new_diffs.add(new_val - row[i])
                            else:
                                new_diffs.add(row[i] - new_val)
                        remaining = used - old_diffs
                        if not (new_diffs & remaining):
                            row[pos] = new_val
                            used = (used - old_diffs) | new_diffs
                            improved = True
                            break

    return rows


def _diff_assignment_search(n_rows, k, max_scope, rng, deadline):
    """
    Structural change: formulate as a CSP over difference values (1..max_scope).
    Each difference d must be assigned to exactly one row and a unique pair (i,j)
    within that row such that the resulting rows are valid (start at 0, strictly
    increasing, and the assigned differences are consistent with the row structure).
    
    This is a different modelling family: instead of building rows sequentially,
    we assign differences to rows and then reconstruct rows from the assigned
    differences. We use most-constrained-variable ordering (fewest compatible
    row assignments) to minimize branching.
    """
    if time.time() > deadline:
        return None

    # For each difference d in 1..max_scope, determine which rows it could be
    # assigned to. Initially, all rows are candidates. As we assign differences,
    # we update the constraints.
    
    # A row is defined by its k+1 elements. The differences within a row are
    # determined by the elements. So if we assign a set of differences to a row,
    # we need to check if there exists a row of k+1 elements starting at 0
    # with exactly those differences.
    
    # This is complex. Instead, let's use a hybrid: assign differences to rows,
    # then for each row, try to find a valid set of elements that realizes
    # the assigned differences.
    
    # For now, implement a simpler version: use the backtracking row builder
    # but with a global constraint that the total number of differences used
    # is n_rows * k*(k+1)/2, and each difference is used at most once.
    
    # Actually, the current backtracking already enforces this. The structural
    # change here is to use a different search strategy: instead of building
    # rows one by one, we interleave the construction of all rows, always
    # picking the row with the fewest remaining valid next elements.
    
    # This is a "lookahead" strategy: at each step, for each incomplete row,
    # compute the number of valid next elements. Pick the row with the fewest
    # (most constrained) and extend it.
    
    rows = [[0] for _ in range(n_rows)]
    used = set()
    row_indices = list(range(n_rows))
    
    while any(len(rows[i]) < k + 1 for i in range(n_rows)):
        if time.time() > deadline:
            return None
        
        # Find the most constrained incomplete row
        min_count = float('inf')
        min_row = -1
        for i in range(n_rows):
            if len(rows[i]) >= k + 1:
                continue
            low = rows[i][-1] + 1
            high = max_scope
            count = 0
            for val in range(low, high + 1):
                new_diffs = {val - a for a in rows[i]}
                if not (new_diffs & used):
                    count += 1
            if count < min_count:
                min_count = count
                min_row = i
        
        if min_row == -1 or min_count == 0:
            return None
        
        # Extend the most constrained row
        row = rows[min_row]
        low = row[-1] + 1
        high = max_scope
        candidates = []
        for val in range(low, high + 1):
            new_diffs = {val - a for a in row}
            if not (new_diffs & used):
                candidates.append((val, new_diffs))
        
        if not candidates:
            return None
        
        # Shuffle for randomization
        rng.shuffle(candidates)
        
        # Try candidates in order
        for val, new_diffs in candidates:
            row.append(val)
            used.update(new_diffs)
            # Recurse (implicitly, by continuing the loop)
            break
    
    # Check if all rows are complete
    if all(len(rows[i]) == k + 1 for i in range(n_rows)):
        return rows
    
    return None


def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    Uses constraint-propagation backtracking with randomized restarts as the
    primary search, then local search / annealing / polish as refinement.
    """
    rng = random.Random(42)
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # Phase 1: Constraint-propagation backtracking with tightening bounds
    best_rows = None
    best_scope = float('inf')

    bt_deadline = deadline - 50
    result = _backtrack_construct(n_rows, k, rng, bt_deadline)
    if result is not None and len(result) == n_rows:
        best_rows = [r[:] for r in result]
        best_scope = _total_scope(best_rows)

    # Fallback / alternative: greedy baseline
    if best_rows is None:
        best_rows = _greedy_construct(n_rows, k)
        best_scope = _total_scope(best_rows)

    # Also try a randomized greedy to compare
    if time.time() < deadline - 45:
        rg = _greedy_construct(n_rows, k, randomize=True, seed_offset=5)
        rg_scope = _total_scope(rg)
        if rg_scope < best_scope:
            best_rows = [r[:] for r in rg]
            best_scope = rg_scope

    # Phase 1.5: Try the difference-assignment search (structural change)
    if time.time() < deadline - 40:
        for attempt in range(3):
            if time.time() > deadline - 40:
                break
            attempt_rng = random.Random(2000 + attempt)
            da_rows = _diff_assignment_search(n_rows, k, best_scope, attempt_rng, deadline - 40)
            if da_rows is not None and len(da_rows) == n_rows:
                da_scope = _total_scope(da_rows)
                if da_scope < best_scope:
                    best_rows = [r[:] for r in da_rows]
                    best_scope = da_scope

    # Phase 2: Local search to reduce scope
    if time.time() < deadline - 25:
        ls_rows = _local_search_scope(best_rows, k, rng, deadline - 25)
        ls_scope = _total_scope(ls_rows)
        if ls_scope < best_scope:
            best_rows = ls_rows
            best_scope = ls_scope

    # Phase 3: Simulated annealing to escape local optima
    if time.time() < deadline - 12:
        sa_rows = _simulated_annealing_scope(best_rows, k, rng, deadline - 12)
        sa_scope = _total_scope(sa_rows)
        if sa_scope < best_scope:
            best_rows = sa_rows
            best_scope = sa_scope

    # Phase 4: Final polish
    if time.time() < deadline - 3:
        polish_rows = _polish_pass(best_rows, k, deadline - 3)
        polish_scope = _total_scope(polish_rows)
        if polish_scope < best_scope:
            best_rows = polish_rows

    return best_rows

# EVOLVE-BLOCK-END
