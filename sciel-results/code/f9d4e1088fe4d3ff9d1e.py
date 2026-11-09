"""Difference triangle set (7,5) – optimized greedy + local search + simulated annealing."""
import time
import random
import math

_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START


def _dead():
    return time.time() >= _DEADLINE


def _diffs_of_row(row):
    """All positive differences a_j - a_i for i<j in one row."""
    d = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            d.add(row[j] - ai)
    return d


def _scope(rows):
    return max(max(r) for r in rows)


def _valid(rows, n_rows, k):
    if len(rows) != n_rows:
        return False
    used = set()
    for r in rows:
        if len(r) != k + 1 or r[0] != 0:
            return False
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                if r[j] <= r[i]:
                    return False
        for d in _diffs_of_row(r):
            if d in used:
                return False
            used.add(d)
    return True


def _build_diff_array(rows, max_val):
    """Build a difference count array for all rows."""
    size = max_val + 2
    diff_counts = [0] * (size + 1)
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d <= size:
                    diff_counts[d] += 1
    return diff_counts


def _interleaved_greedy(n_rows, k, seed=0):
    """Interleaved greedy: build all rows simultaneously, one element at a time."""
    random.seed(seed)
    rows = [[0] for _ in range(n_rows)]
    used = set()
    
    for step in range(1, k + 1):
        order = list(range(n_rows))
        random.shuffle(order)
        for r in order:
            row = rows[r]
            x = row[-1]
            while True:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    break
    
    return rows


def _greedy_row(used, k, max_mark, rnd):
    """Greedy place one row of length k+1 starting at 0."""
    row = [0]
    local = set()
    x = 0
    while len(row) < k + 1:
        x += 1
        if x > max_mark:
            return None, local, False
        new = []
        bad = False
        for a in row:
            dv = x - a
            if dv in used or dv in local:
                bad = True
                break
            new.append(dv)
        if not bad:
            row.append(x)
            local.update(new)
    return row, local, True


def _local_search_reduce(rows, n_rows, k, time_limit=30.0):
    """
    Local search that tries to reduce each element to its minimum valid value.
    Uses efficient difference counting.
    """
    start_time = time.time()
    current_rows = [list(r) for r in rows]
    
    max_val = max(max(r) for r in current_rows)
    diff_size = max_val + 50
    diff_counts = [0] * (diff_size + 1)
    
    for row in current_rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d >= len(diff_counts):
                    diff_counts.extend([0] * (d - len(diff_counts) + 1))
                diff_counts[d] += 1
    
    def get_scope():
        return max(max(r) for r in current_rows)
    
    best_rows = [list(r) for r in current_rows]
    best_scope = get_scope()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for r in range(n_rows):
            if (time.time() - start_time) > time_limit:
                break
            row = current_rows[r]
            for pos in range(len(row) - 1, 0, -1):
                if (time.time() - start_time) > time_limit:
                    break
                old_val = row[pos]
                lower_bound = row[pos - 1] + 1
                
                if lower_bound >= old_val:
                    continue
                
                # Precompute old differences
                old_diffs = [abs(old_val - row[i]) for i in range(len(row)) if i != pos]
                
                # Remove old diffs from counts
                for d in old_diffs:
                    if d < len(diff_counts):
                        diff_counts[d] -= 1
                
                # Try values from lower_bound upward until we find a valid one
                found = False
                for new_val in range(lower_bound, old_val):
                    valid = True
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        d = abs(new_val - row[i])
                        if d < len(diff_counts) and diff_counts[d] > 0:
                            valid = False
                            break
                    if valid:
                        row[pos] = new_val
                        new_diffs = [abs(new_val - row[i]) for i in range(len(row)) if i != pos]
                        for d in new_diffs:
                            if d < len(diff_counts):
                                diff_counts[d] += 1
                        improved = True
                        found = True
                        break
                
                if not found:
                    for d in old_diffs:
                        if d < len(diff_counts):
                            diff_counts[d] += 1
    
    current_scope = get_scope()
    if current_scope < best_scope:
        best_scope = current_scope
        best_rows = [list(r) for r in current_rows]
        
    return best_rows


def _simulated_annealing(rows, n_rows, k, time_budget, seed=42):
    """
    Simulated annealing with row replacement and element shifts.
    Uses efficient difference tracking.
    """
    rnd = random.Random(seed)
    best = [r[:] for r in rows]
    best_scope = _scope(best)
    current = [r[:] for r in best]
    current_scope = best_scope

    def compute_used(rows):
        used = set()
        for r in rows:
            used |= _diffs_of_row(r)
        return used

    used = compute_used(current)
    max_val = max(max(r) for r in current)
    diff_counts = _build_diff_array(current, max_val)

    def update_diff_counts(old_row, new_row, diff_counts):
        """Update diff counts by removing old row's diffs and adding new row's diffs."""
        n = len(old_row)
        for i in range(n):
            for j in range(i + 1, n):
                d = old_row[j] - old_row[i]
                if d < len(diff_counts):
                    diff_counts[d] -= 1
        n = len(new_row)
        for i in range(n):
            for j in range(i + 1, n):
                d = new_row[j] - new_row[i]
                if d >= len(diff_counts):
                    diff_counts.extend([0] * (d - len(diff_counts) + 1))
                diff_counts[d] += 1

    def check_validity(new_row, ri, diff_counts, used):
        """Check if new_row can replace current[ri] without conflicts."""
        # Remove current row's diffs temporarily
        old_row = current[ri]
        n = len(old_row)
        temp_counts = diff_counts[:]
        for i in range(n):
            for j in range(i + 1, n):
                d = old_row[j] - old_row[i]
                if d < len(temp_counts):
                    temp_counts[d] -= 1
        
        # Check new row's diffs
        n = len(new_row)
        for i in range(n):
            for j in range(i + 1, n):
                d = new_row[j] - new_row[i]
                if d >= len(temp_counts) or temp_counts[d] > 0:
                    return False
        return True

    deadline = time.time() + time_budget
    T = 20.0
    cooling = 0.998

    while time.time() < deadline and T > 0.1:
        if _dead():
            break

        move_type = rnd.random()
        if move_type < 0.5:
            # Move type 1: Replace one row
            ri = rnd.randint(0, n_rows - 1)
            old_row = current[ri]
            old_diffs = _diffs_of_row(old_row)
            used_without = used - old_diffs

            max_mark = min(current_scope + 5, best_scope + 5)
            new_row = None
            for attempt in range(20):
                row = [0]
                local = set()
                x = 0
                success = True
                while len(row) < k + 1:
                    x += 1
                    if x > max_mark:
                        success = False
                        break
                    if rnd.random() < 0.05 and x < max_mark:
                        x += 1
                        if x > max_mark:
                            success = False
                            break
                    bad = False
                    for a in row:
                        dv = x - a
                        if dv in used_without or dv in local:
                            bad = True
                            break
                    if not bad:
                        row.append(x)
                        local.update({x - a for a in row[:-1]})
                if success and len(row) == k + 1:
                    new_row = row
                    break

            if new_row is None:
                T *= cooling
                continue

            new_diffs = _diffs_of_row(new_row)
            if new_diffs & used_without:
                T *= cooling
                continue

            other_max = max(max(r) for i, r in enumerate(current) if i != ri)
            new_scope = max(other_max, max(new_row))
            delta = new_scope - current_scope
            if delta <= 0 or rnd.random() < math.exp(-delta / T):
                update_diff_counts(old_row, new_row, diff_counts)
                used = used_without | new_diffs
                current[ri] = new_row
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best = [r[:] for r in current]
        else:
            # Move type 2: Shift a single element down
            ri = rnd.randint(0, n_rows - 1)
            row = current[ri]
            pi = rnd.randint(1, len(row) - 1)
            old_val = row[pi]
            
            # Try shifting down by 1 or 2
            shift = rnd.choice([1, 1, 2])
            new_val = old_val - shift
            if new_val <= row[pi - 1]:
                T *= cooling
                continue
            if pi < len(row) - 1 and new_val >= row[pi + 1]:
                T *= cooling
                continue

            new_row = row[:]
            new_row[pi] = new_val
            
            # Check validity using diff counts
            if not check_validity(new_row, ri, diff_counts, used):
                T *= cooling
                continue

            other_max = max(max(r) for i, r in enumerate(current) if i != ri)
            new_scope = max(other_max, max(new_row))
            delta = new_scope - current_scope
            if delta <= 0 or rnd.random() < math.exp(-delta / T):
                update_diff_counts(row, new_row, diff_counts)
                used = used - _diffs_of_row(row) | _diffs_of_row(new_row)
                current[ri] = new_row
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best = [r[:] for r in current]

        T *= cooling

    return best, best_scope


def _final_polish(rows, n_rows, k, time_limit=10.0, allow_up=False):
    """
    Final high-precision polish: repeatedly try to reduce each element to its minimum valid value.
    If allow_up is True, also try moving elements up (useful to escape local minima).
    """
    start_time = time.time()
    current_rows = [list(r) for r in rows]
    
    max_val = max(max(r) for r in current_rows)
    diff_size = max_val + 60
    diff_counts = [0] * diff_size
    
    for row in current_rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d >= len(diff_counts):
                    diff_counts.extend([0] * (d - len(diff_counts) + 1))
                diff_counts[d] += 1
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for r in range(n_rows):
            if (time.time() - start_time) > time_limit:
                break
            row = current_rows[r]
            for pos in range(len(row) - 1, 0, -1):
                old_val = row[pos]
                lower_bound = row[pos - 1] + 1
                upper_bound = row[pos + 1] - 1 if pos < len(row) - 1 else old_val
                
                # Try to reduce
                if lower_bound < old_val:
                    old_diffs = [abs(old_val - row[i]) for i in range(len(row)) if i != pos]
                    for d in old_diffs:
                        if d < len(diff_counts):
                            diff_counts[d] -= 1
                    
                    found = False
                    for new_val in range(lower_bound, old_val):
                        valid = True
                        for i in range(len(row)):
                            if i == pos:
                                continue
                            d = abs(new_val - row[i])
                            if d < len(diff_counts) and diff_counts[d] > 0:
                                valid = False
                                break
                        if valid:
                            row[pos] = new_val
                            new_diffs = [abs(new_val - row[i]) for i in range(len(row)) if i != pos]
                            for d in new_diffs:
                                if d < len(diff_counts):
                                    diff_counts[d] += 1
                            improved = True
                            found = True
                            break
                    
                    if not found:
                        for d in old_diffs:
                            if d < len(diff_counts):
                                diff_counts[d] += 1
                        old_val = row[pos]  # re-read in case we moved it
                        lower_bound = row[pos - 1] + 1
                        upper_bound = row[pos + 1] - 1 if pos < len(row) - 1 else old_val
                
                # Try to move up (escape local minima)
                if allow_up and upper_bound > old_val:
                    old_diffs = [abs(old_val - row[i]) for i in range(len(row)) if i != pos]
                    for d in old_diffs:
                        if d < len(diff_counts):
                            diff_counts[d] -= 1
                    
                    found = False
                    for new_val in range(old_val + 1, upper_bound + 1):
                        valid = True
                        for i in range(len(row)):
                            if i == pos:
                                continue
                            d = abs(new_val - row[i])
                            if d >= len(diff_counts):
                                diff_counts.extend([0] * (d - len(diff_counts) + 1))
                            if diff_counts[d] > 0:
                                valid = False
                                break
                        if valid:
                            row[pos] = new_val
                            new_diffs = [abs(new_val - row[i]) for i in range(len(row)) if i != pos]
                            for d in new_diffs:
                                if d < len(diff_counts):
                                    diff_counts[d] += 1
                            improved = True
                            found = True
                            break
                    
                    if not found:
                        for d in old_diffs:
                            if d < len(diff_counts):
                                diff_counts[d] += 1
    
    return current_rows


def _multi_start_sa(rows, n_rows, k, time_budget, num_starts=3):
    """Run simulated annealing multiple times with different seeds and keep the best."""
    best_rows = [r[:] for r in rows]
    best_scope = _scope(best_rows)
    
    time_per_start = time_budget / num_starts
    
    for i in range(num_starts):
        if _dead():
            break
        remaining = time_budget - i * time_per_start
        if remaining < 5.0:
            break
        sa_result, sa_scope = _simulated_annealing(
            best_rows, n_rows, k, 
            time_budget=min(time_per_start, remaining),
            seed=43 + i * 7
        )
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = sa_result
    
    return best_rows, best_scope


def _csp_branch_and_bound(rows, n_rows, k, max_scope, time_budget, seed=99):
    """
    CSP-style branch-and-bound: try to find a valid configuration where every row's
    largest entry is <= max_scope. Uses forward checking with a difference-count array
    to prune conflicts early. Rows are ordered by current max (most constrained first).
    
    Returns (rows, scope) of the best found, or (None, inf) if no valid config found.
    """
    rnd = random.Random(seed)
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Order rows: most constrained (largest current max) first
    order = sorted(range(n_rows), key=lambda r: -max(rows[r]))
    
    # Build diff counts for the "background" (all rows not yet being reconstructed)
    # We'll reconstruct all rows, so start with empty counts and build up.
    diff_counts = [0] * (max_scope + 5)
    
    # For each row in order, greedily reconstruct with forward checking.
    # If a row can't be placed, backtrack by trying a different construction.
    
    def try_build_row(used_diffs, max_mark, rnd_local):
        """Try to build a row of k+1 increasing ints starting at 0, all diffs in [1, max_mark],
        avoiding used_diffs and self-conflicts. Returns row or None."""
        row = [0]
        local = set()
        x = 0
        while len(row) < k + 1:
            x += 1
            if x > max_mark:
                return None
            bad = False
            for a in row:
                dv = x - a
                if dv in used_diffs or dv in local:
                    bad = True
                    break
            if not bad:
                row.append(x)
                local.update({x - a for a in row[:-1]})
        return row
    
    # Multi-attempt: try to build all rows in order, one at a time, using the running
    # difference set. If we get stuck, retry with a different seed for the stuck row.
    best_found = None
    best_found_scope = float('inf')
    
    num_attempts = 50
    for attempt in range(num_attempts):
        if time.time() > deadline:
            break
        if _dead():
            break
        
        rnd_attempt = random.Random(seed + attempt * 131)
        current_rows = [None] * n_rows
        used = set()
        success = True
        
        for r in order:
            # Try to build row r avoiding `used`
            max_mark = max_scope
            row = try_build_row(used, max_mark, rnd_attempt)
            if row is None:
                # Retry with slight randomization: skip some values
                for retry in range(10):
                    row = [0]
                    local = set()
                    x = 0
                    ok = True
                    while len(row) < k + 1:
                        x += 1
                        if x > max_mark:
                            ok = False
                            break
                        if rnd_attempt.random() < 0.08 and x < max_mark:
                            x += 1
                            if x > max_mark:
                                ok = False
                                break
                        bad = False
                        for a in row:
                            dv = x - a
                            if dv in used or dv in local:
                                bad = True
                                break
                        if not bad:
                            row.append(x)
                            local.update({x - a for a in row[:-1]})
                    if ok:
                        break
                if row is None or len(row) != k + 1:
                    success = False
                    break
            
            current_rows[r] = row
            used |= _diffs_of_row(row)
        
        if success:
            scope = max(max(r) for r in current_rows)
            if scope < best_found_scope:
                best_found_scope = scope
                best_found = [list(r) for r in current_rows]
            # If we hit the bound exactly, we can stop early
            if scope <= max_scope:
                break
    
    if best_found is not None:
        return best_found, best_found_scope
    return None, float('inf')


def construct(n_rows, k):
    """Construct difference triangle set with n_rows rows of k+1 increasing integers."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_rows = None
    best_scope = float('inf')
    
    # 1. Generate diverse initial solutions using interleaved greedy
    max_greedy_seeds = 10
    
    for seed in range(max_greedy_seeds):
        if time.time() > deadline - 100.0:
            break
            
        rows = _interleaved_greedy(n_rows, k, seed=seed)
        sc = max(max(r) for r in rows)
        
        if sc < best_scope:
            best_scope = sc
            best_rows = [list(r) for r in rows]
            
    if best_rows is None:
        # Fallback: simple sequential greedy
        rows = []
        used = set()
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # 2. Optimization Phase
    time_remaining = deadline - time.time()
    
    # Phase 2a: Local search to reduce scope
    if time_remaining > 20.0:
        t_ls = min(20.0, time_remaining * 0.2)
        optimized = _local_search_reduce(best_rows, n_rows, k, time_limit=t_ls)
        sc = max(max(r) for r in optimized)
        if sc < best_scope:
            best_scope = sc
            best_rows = optimized
    
    # Phase 2b: Multi-start simulated annealing
    time_remaining = deadline - time.time()
    if time_remaining > 15.0:
        t_sa = min(55.0, time_remaining * 0.5)
        sa_result, sa_scope = _multi_start_sa(best_rows, n_rows, k, time_budget=t_sa, num_starts=3)
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = sa_result
    
    # Phase 2c: CSP branch-and-bound to tighten the scope bound
    time_remaining = deadline - time.time()
    if time_remaining > 10.0:
        t_csp = min(25.0, time_remaining * 0.5)
        # Try to find a config with scope = best_scope - 1 (or lower)
        target = best_scope - 1
        csp_result, csp_scope = _csp_branch_and_bound(
            best_rows, n_rows, k, max_scope=target, time_budget=t_csp, seed=200
        )
        if csp_result is not None and csp_scope < best_scope:
            best_scope = csp_scope
            best_rows = csp_result
        # Try one more step down if we improved
        time_remaining = deadline - time.time()
        if time_remaining > 5.0 and best_scope < target:
            t_csp2 = min(10.0, time_remaining * 0.6)
            target2 = best_scope - 1
            csp_result2, csp_scope2 = _csp_branch_and_bound(
                best_rows, n_rows, k, max_scope=target2, time_budget=t_csp2, seed=500
            )
            if csp_result2 is not None and csp_scope2 < best_scope:
                best_scope = csp_scope2
                best_rows = csp_result2
    
    # Phase 2d: Final polish (with up-moves to escape local minima)
    time_remaining = deadline - time.time()
    if time_remaining > 5.0:
        t_polish = min(12.0, time_remaining * 0.8)
        polished = _final_polish(best_rows, n_rows, k, time_limit=t_polish, allow_up=True)
        sc = max(max(r) for r in polished)
        if sc < best_scope:
            best_scope = sc
            best_rows = polished
    
    return best_rows
