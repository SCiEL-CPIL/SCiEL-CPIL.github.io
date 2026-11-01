"""Difference triangle set: minimize largest entry for n_rows rows of k+1 increasing ints from 0."""
import time
import random

_START_TIME = None
_DEADLINE = 145.0  # Safety margin before 150s kill

def _elapsed():
    if _START_TIME is None:
        return 0
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _elapsed()

def _compute_diffs_set(row):
    """Return set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _validate_solution(rows, n_rows, k):
    """Check that all differences across all rows are distinct and rows are valid."""
    seen = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(k):
            if row[i] >= row[i+1]:
                return False
        d = _compute_diffs_set(row)
        if len(d) != k * (k + 1) // 2:
            return False
        if d & seen:
            return False
        seen |= d
    return True

def _greedy_sequential(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _greedy_round_robin(n_rows, k, max_iter=100000):
    """Round-robin: at each step, pick the row that is most constrained and extend it."""
    used = set()
    rows = [[0] for _ in range(n_rows)]
    total_needed = n_rows * (k + 1)
    placed = n_rows
    for _ in range(max_iter):
        if placed == total_needed:
            break
        if _time_left() < 2.0:
            break
        best_row = -1
        best_count = float('inf')
        best_candidate = None
        for r in range(n_rows):
            if len(rows[r]) >= k + 1:
                continue
            last = rows[r][-1]
            count = 0
            candidate = None
            # Look ahead to find smallest valid candidate
            for x in range(last + 1, last + 500):
                new = {x - a for a in rows[r]}
                if not (new & used):
                    count += 1
                    if candidate is None:
                        candidate = x
                    if count > 10: # Enough to estimate constraint
                        break
            if candidate is None:
                # Try larger if none found in immediate range
                x = last + 500
                while True:
                    new = {x - a for a in rows[r]}
                    if not (new & used):
                        count = 1
                        candidate = x
                        break
                    x += 1
            if count < best_count:
                best_count = count
                best_row = r
                best_candidate = candidate
                if count == 1:
                    break
        if best_row == -1:
            break
        x = best_candidate
        rows[best_row].append(x)
        for a in rows[best_row]:
            used.add(x - a)
        placed += 1
    for r in range(n_rows):
        if len(rows[r]) < k + 1:
            return None
    return rows

def _greedy_random_order(n_rows, k, rng):
    """Build rows in a random order."""
    used = set()
    rows = [None] * n_rows
    order = list(range(n_rows))
    rng.shuffle(order)
    for r in order:
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows[r] = row
    return rows

def _local_search_optimize(rows, n_rows, k, max_iterations=5000):
    """
    Attempt to reduce the maximum value by moving elements down.
    Returns the optimized rows.
    """
    rows = [list(r) for r in rows]
    # Precompute used diffs
    used = set()
    for r in rows:
        used |= _compute_diffs_set(r)
    
    current_max = max(max(r) for r in rows)
    best_rows = [list(r) for r in rows]
    best_max = current_max
    
    # Map value to list of (row_idx, pos_idx)
    # This helps find which element to move
    # But since we only care about the global max, we can just scan.
    
    for _ in range(max_iterations):
        if _time_left() < 5.0:
            break
            
        current_max = max(max(r) for r in rows)
        if current_max < best_max:
            best_max = current_max
            best_rows = [list(r) for r in rows]
        
        # Find all occurrences of current_max
        max_positions = []
        for r in range(n_rows):
            if rows[r][-1] == current_max:
                max_positions.append((r, len(rows[r]) - 1))
        
        if not max_positions:
            break
            
        moved = False
        # Try to move one of the max elements to a smaller value
        random.shuffle(max_positions)
        for r_idx, pos_idx in max_positions:
            if _time_left() < 5.0:
                break
            row = rows[r_idx]
            old_val = row[pos_idx]
            # We want to find x < old_val such that x > row[pos_idx-1]
            # And the new diffs don't conflict with (used - old_diffs)
            
            old_diffs = {old_val - a for a in row[:pos_idx]}
            # Remove old diffs from global used set temporarily
            # Note: We must be careful if old_diffs overlaps with other rows' diffs?
            # No, by problem definition, all diffs are distinct. So old_diffs are unique to this row/element pair?
            # Actually, a diff d in row r is unique across ALL rows.
            # So removing old_diffs from 'used' is safe as long as we only consider this row's contribution.
            
            # However, 'used' is the global set.
            # Let's create a local view: used_except_old = used - old_diffs
            # We need to check if new_diffs intersects with used_except_old.
            
            # To speed up, we can just iterate x downwards.
            # Range: row[pos_idx-1] + 1 to old_val - 1
            # We prefer smaller x to reduce max more aggressively? 
            # Actually, any x < old_val reduces the max if it's the only max.
            # If there are multiple maxes, reducing one doesn't reduce global max unless it was the unique max.
            # But it might allow others to be reduced later.
            
            # Let's try to find the smallest valid x > row[pos_idx-1]
            # This minimizes the row's spread, potentially helping other rows.
            # Or should we try to keep it close to old_val to minimize perturbation?
            # Usually, packing tightly (smaller x) is better for minimizing global max.
            
            found_x = None
            # Optimization: Only check a limited range?
            # The valid x must be > row[pos_idx-1].
            # Let's scan from row[pos_idx-1] + 1 upwards?
            # No, we want to replace old_val.
            # If we pick a small x, we might free up large numbers.
            # Let's try scanning from row[pos_idx-1] + 1 up to old_val - 1.
            
            # To avoid O(N^2) check, we can precompute the set of "forbidden" diffs for this position?
            # Forbidden diffs are:
            # 1. Diffs already in 'used' (excluding old_diffs)
            # 2. Diffs internal to the row prefix (row[:pos_idx])
            
            prefix_diffs = _compute_diffs_set(row[:pos_idx])
            # The new diffs will be {x - a for a in row[:pos_idx]}
            # These must not be in (used - old_diffs)
            # And they must not be in prefix_diffs (which is guaranteed if x > row[pos_idx-1] and row is increasing? 
            # Wait. If x > row[pos_idx-1], then x - a > row[pos_idx-1] - a for all a < row[pos_idx-1].
            # The max diff in prefix is row[pos_idx-1] - 0.
            # So new diffs are strictly larger than any diff in prefix?
            # No. x - row[0] is the largest new diff.
            # row[pos_idx-1] - row[0] is the largest prefix diff.
            # If x > row[pos_idx-1], then x - row[0] > row[pos_idx-1] - row[0].
            # So new diffs are disjoint from prefix_diffs automatically?
            # Yes, because for any a in row[:pos_idx], x - a > row[pos_idx-1] - a >= max(prefix_diffs)?
            # Actually, max(prefix_diffs) is row[pos_idx-1] - row[0].
            # Min(new_diffs) is x - row[pos_idx-1] > 0.
            # Is it possible that x - a_i == a_j - a_k?
            # Since x > row[pos_idx-1] >= a_j, x - a_i > a_j - a_i.
            # This logic holds: new diffs are disjoint from internal prefix diffs.
            
            # So we only need to check against (used - old_diffs).
            
            # Let's iterate x from row[pos_idx-1] + 1 to old_val - 1
            # We want to find ANY valid x.
            # To maximize chance of success, maybe try random x's?
            # Or try the smallest x (tightest packing)?
            
            # Let's try smallest x first.
            lower_bound = row[pos_idx - 1] + 1
            upper_bound = old_val - 1
            
            # If the range is huge, we might waste time.
            # But usually, the solution is dense.
            
            for x in range(lower_bound, upper_bound + 1):
                new_diffs = {x - a for a in row[:pos_idx]}
                # Check conflict with global used (excluding old_diffs)
                # We can check if (new_diffs & (used - old_diffs)) is empty
                # To speed up, we can check if any d in new_diffs is in used AND d not in old_diffs
                conflict = False
                for d in new_diffs:
                    if d in used and d not in old_diffs:
                        conflict = True
                        break
                if not conflict:
                    # Valid move
                    # Update row
                    row[pos_idx] = x
                    # Update used set
                    used -= old_diffs
                    used |= new_diffs
                    moved = True
                    break
            
            if moved:
                break # Restart the loop to re-evaluate max and positions
                
    return best_rows

def _perturb(rows, n_rows, k, rng):
    """
    Perturb the solution by re-greedy-ing a random suffix of a random row.
    This helps escape local minima.
    """
    rows = [list(r) for r in rows]
    # Pick a random row
    r = rng.randint(0, n_rows - 1)
    # Pick a random cut point (keep first 'cut' elements)
    # Must keep at least 1 element (0)
    cut = rng.randint(1, k) # Keep 1 to k elements
    # Re-greedy the rest
    # We need to know which diffs are used by OTHER rows
    used_by_others = set()
    for i in range(n_rows):
        if i != r:
            used_by_others |= _compute_diffs_set(rows[i])
    
    # Also, the prefix of row r uses some diffs.
    # The new suffix must not conflict with used_by_others OR the prefix diffs?
    # Actually, the total set of diffs for row r must be disjoint from used_by_others.
    # The prefix diffs are fixed.
    # So the new suffix diffs must be disjoint from (used_by_others U prefix_diffs).
    
    prefix = rows[r][:cut]
    prefix_diffs = _compute_diffs_set(prefix)
    forbidden = used_by_others | prefix_diffs
    
    # Re-greedy from prefix
    new_suffix = []
    last_val = prefix[-1]
    # We need to add k + 1 - cut elements
    needed = (k + 1) - cut
    
    # Greedy construction for the suffix
    # We start with x = last_val + 1
    # But we must ensure the *entire* row's diffs are valid.
    # The greedy approach:
    # Current row: prefix + new_suffix
    # When adding x, check {x - a for a in current_row} against forbidden.
    
    current_row = list(prefix)
    for _ in range(needed):
        x = last_val + 1
        while True:
            new_diffs = {x - a for a in current_row}
            if not (new_diffs & forbidden):
                current_row.append(x)
                last_val = x
                # Update forbidden? 
                # The new diffs become part of the "used" for subsequent elements in this row.
                # So we should add them to forbidden?
                # Yes, because they are now "used" by this row.
                forbidden |= new_diffs
                break
            x += 1
            
    rows[r] = current_row
    return rows

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    rng = random.Random(42)
    
    # 1. Generate initial solutions
    candidates = []
    
    # Sequential
    seq_rows = _greedy_sequential(n_rows, k)
    candidates.append(seq_rows)
    
    # Round Robin
    rr_rows = _greedy_round_robin(n_rows, k)
    if rr_rows:
        candidates.append(rr_rows)
        
    # Random Order (multiple trials)
    for _ in range(5):
        if _time_left() < 10.0:
            break
        ro_rows = _greedy_random_order(n_rows, k, rng)
        if ro_rows:
            candidates.append(ro_rows)
            
    # 2. Optimize each candidate using Iterated Local Search
    best_rows = None
    best_max = float('inf')
    
    for init_rows in candidates:
        if _time_left() < 15.0:
            break
            
        # Run ILS
        current_rows = [list(r) for r in init_rows]
        current_max = max(max(r) for r in current_rows)
        
        # Local Search Phase 1: Pure Hill Climbing
        optimized_rows = _local_search_optimize(current_rows, n_rows, k)
        opt_max = max(max(r) for r in optimized_rows)
        
        if opt_max < best_max:
            best_max = opt_max
            best_rows = [list(r) for r in optimized_rows]
            
        # If we have time, try Perturbation + Local Search
        if _time_left() > 30.0:
            for _ in range(5): # 5 ILS iterations
                if _time_left() < 10.0:
                    break
                # Perturb
                perturbed_rows = _perturb(optimized_rows, n_rows, k, rng)
                # Optimize
                new_optimized = _local_search_optimize(perturbed_rows, n_rows, k)
                new_max = max(max(r) for r in new_optimized)
                
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in new_optimized]
                    optimized_rows = new_optimized
                else:
                    # Keep the better of the two for next perturbation?
                    # Or just restart from the perturbed one?
                    # Standard ILS: accept perturbed if it leads to improvement after optimization.
                    # Here we always optimize, so we compare the optimized results.
                    # If no improvement, we might want to keep the previous best as the base for next perturbation?
                    # Or perturb the current optimized one again?
                    # Let's perturb the current optimized one.
                    pass 

    # Final Validation and Return
    if best_rows is None:
        # Fallback to sequential if something went wrong
        return _greedy_sequential(n_rows, k)
        
    # Ensure validity
    if _validate_solution(best_rows, n_rows, k):
        return best_rows
    else:
        # Fallback
        return _greedy_sequential(n_rows, k)
