"""
Strategy: Simulated Annealing with a global difference-set representation.

Instead of constructing rows one-by-one (which creates a sequential bottleneck
and poor global coordination), we:
1. Start with a random valid configuration (greedy construction with randomization)
2. Use simulated annealing to minimize the scope (largest entry)
3. Moves: pick a random row and a random non-first element, try to move it
   to a new position (lower or higher), checking global difference uniqueness
4. Accept worse moves with probability exp(-delta/T) to escape local optima
5. The key structural difference: we optimize the ENTIRE configuration jointly
   rather than building rows sequentially, and we use temperature-based
   acceptance to explore the space more effectively.
"""
import time
import random
import math
import sys


def _row_diffs_list(row):
    """Return list of all positive differences within a row (i<j)."""
    n = len(row)
    diffs = []
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.append(row[j] - ri)
    return diffs


def _row_mask(row):
    """Bitmask of differences in a row."""
    m = 0
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            m |= (1 << (row[j] - ri))
    return m


def _scope(rows):
    return max(r[-1] for r in rows)


def _is_valid(rows):
    """Check all positive differences are globally distinct."""
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs_list(row))
    return len(all_diffs) == len(set(all_diffs))


def _greedy_construct(n_rows, num_elements, rng, max_scope_hint=None):
    """
    Greedy construction: build rows one by one, picking the smallest valid value
    for each position with some randomization.
    """
    rows = []
    used_mask = 0
    
    for row_idx in range(n_rows):
        row = [0]
        for pos in range(1, num_elements):
            found = False
            # Try values from low to high with some randomization
            if max_scope_hint:
                high = max_scope_hint
            else:
                high = 200  # reasonable upper bound
            
            # Shuffle candidate values but bias toward smaller
            candidates = list(range(row[-1] + 1, high + 1))
            
            # Weighted random: smaller values more likely
            if len(candidates) > 30:
                # Take a random sample, biased toward smaller
                sample_size = min(25, len(candidates))
                # Use exponential weighting
                weights = [1.0 / (1.0 + 0.5 * i) for i in range(len(candidates))]
                selected = []
                pool = list(range(len(candidates)))
                for _ in range(sample_size):
                    total_w = sum(weights[i] for i in pool)
                    r = rng.random() * total_w
                    cum = 0
                    for idx, pi in enumerate(pool):
                        cum += weights[pi]
                        if r <= cum:
                            selected.append(candidates[pi])
                            pool.remove(pi)
                            break
                candidates = sorted(selected)
            
            for val in candidates:
                new_diffs = {val - prev for prev in row}
                # Check no duplicate within row
                if len(new_diffs) != len(set(new_diffs)):
                    continue
                # Check no conflict with used diffs
                new_mask = 0
                for d in new_diffs:
                    new_mask |= (1 << d)
                if new_mask & used_mask:
                    continue
                # Also check within-row consistency
                existing_diffs = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        existing_diffs.add(row[j] - row[i])
                if new_diffs & existing_diffs:
                    continue
                
                row.append(val)
                used_mask |= new_mask
                found = True
                break
            
            if not found:
                # If we couldn't find a value, extend the search range
                row.append(row[-1] + 10 + rng.randint(0, 20))
                # Recompute used_mask (this might create conflicts, but we'll fix later)
                new_diffs = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        new_diffs.add(row[j] - row[i])
                for d in new_diffs:
                    used_mask |= (1 << d)
        
        rows.append(row)
    
    return rows


def _compute_all_diffs(rows):
    """Return (global_mask, row_masks) for all rows."""
    row_masks = []
    global_mask = 0
    for row in rows:
        m = _row_mask(row)
        row_masks.append(m)
        global_mask |= m
    return global_mask, row_masks


def _try_move_element(rows, row_idx, pos, new_val, global_mask, row_masks):
    """
    Try to move element at rows[row_idx][pos] to new_val.
    Returns (success, new_global_mask, new_row_mask) or (False, None, None).
    """
    row = rows[row_idx]
    n = len(row)
    
    # Check bounds: must maintain strictly increasing order
    if pos > 0 and new_val <= row[pos - 1]:
        return False, None, None
    if pos < n - 1 and new_val >= row[pos + 1]:
        return False, None, None
    
    # Compute new differences for this row
    old_val = row[pos]
    if new_val == old_val:
        return False, None, None
    
    # Tentatively apply
    row[pos] = new_val
    
    # Check within-row uniqueness
    diffs_in_row = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            if d in diffs_in_row:
                row[pos] = old_val
                return False, None, None
            diffs_in_row.add(d)
    
    # Compute new row mask
    new_row_mask = 0
    for d in diffs_in_row:
        new_row_mask |= (1 << d)
    
    # Check against other rows
    other_mask = global_mask & ~row_masks[row_idx]
    if new_row_mask & other_mask:
        row[pos] = old_val
        return False, None, None
    
    # Success
    new_global_mask = (global_mask & ~row_masks[row_idx]) | new_row_mask
    row[pos] = old_val  # revert; caller will apply
    
    return True, new_global_mask, new_row_mask


def _simulated_annealing(rows, num_elements, rng, time_limit, initial_temp=50.0):
    """
    Simulated annealing to minimize scope.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    global_mask, row_masks = _compute_all_diffs(current_rows)
    
    n_rows = len(current_rows)
    n_elem = num_elements
    
    temp = initial_temp
    cooling = 0.9995
    min_temp = 0.5
    
    iterations = 0
    max_iter = 200000
    
    while time.time() < time_limit and temp > min_temp and iterations < max_iter:
        iterations += 1
        
        # Pick a random row and position (not first)
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, n_elem - 1)
        row = current_rows[row_idx]
        old_val = row[pos]
        
        # Determine valid range
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < n_elem - 1 else current_scope + 10
        
        if low >= high:
            continue
        
        # Bias toward lower values (to reduce scope)
        if old_val > low + 2 and rng.random() < 0.7:
            search_high = min(old_val - 1, high)
            if low > search_high:
                continue
            new_val = rng.randint(low, search_high)
        else:
            # Allow some upward moves for exploration
            new_val = rng.randint(low, high)
        
        if new_val == old_val:
            continue
        
        # Try the move
        success, new_global_mask, new_row_mask = _try_move_element(
            current_rows, row_idx, pos, new_val, global_mask, row_masks
        )
        
        if not success:
            continue
        
        # Apply the move
        current_rows[row_idx][pos] = new_val
        old_row_mask = row_masks[row_idx]
        row_masks[row_idx] = new_row_mask
        global_mask = new_global_mask
        
        new_scope = _scope(current_rows)
        delta = new_scope - current_scope
        
        if delta <= 0 or rng.random() < math.exp(-delta / temp):
            # Accept
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
        else:
            # Reject: revert
            current_rows[row_idx][pos] = old_val
            row_masks[row_idx] = old_row_mask
            global_mask = (global_mask & ~new_row_mask) | old_row_mask
        
        temp *= cooling
    
    return best_rows, best_scope


def _local_search_refine(rows, num_elements, rng, time_limit):
    """
    Simple local search: try to reduce scope by moving elements down.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    global_mask, row_masks = _compute_all_diffs(current_rows)
    n_rows = len(current_rows)
    n_elem = num_elements
    
    max_iter = 100000
    stagnation = 0
    max_stagnation = 5000
    
    for it in range(max_iter):
        if time.time() > time_limit:
            break
        if stagnation > max_stagnation:
            break
        
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, n_elem - 1)
        row = current_rows[row_idx]
        old_val = row[pos]
        
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < n_elem - 1 else current_scope
        
        if low >= old_val:
            continue
        
        # Try to move down
        new_val = rng.randint(low, old_val - 1)
        
        success, new_global_mask, new_row_mask = _try_move_element(
            current_rows, row_idx, pos, new_val, global_mask, row_masks
        )
        
        if success:
            current_rows[row_idx][pos] = new_val
            old_row_mask = row_masks[row_idx]
            row_masks[row_idx] = new_row_mask
            global_mask = new_global_mask
            
            new_scope = _scope(current_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
                stagnation = 0
            else:
                stagnation += 1
        else:
            stagnation += 1
    
    return best_rows, best_scope


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = start_time + 135.0  # safety margin before 150s kill
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))
    
    rng = random.Random(42)
    
    # Phase 1: Generate initial solution
    best_rows = None
    best_scope = float('inf')
    
    # Try multiple greedy constructions with different seeds
    for seed in range(10):
        if time.time() > DEADLINE - 80:
            break
        rng_local = random.Random(seed * 7 + 1)
        rows = _greedy_construct(n_rows, num_elements, rng_local)
        if _is_valid(rows):
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = rows
    
    # If no valid solution found, use a simple sequential fallback
    if best_rows is None:
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        best_rows = rows
        best_scope = _scope(best_rows)
    
    # Phase 2: Simulated Annealing
    remaining = DEADLINE - 30
    if remaining > 30:
        improved_rows, improved_scope = _simulated_annealing(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 40,
            initial_temp=max(10.0, best_scope * 0.1)
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Phase 3: Local search refinement
    if time.time() < DEADLINE - 20:
        improved_rows, improved_scope = _local_search_refine(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 15
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Phase 4: Another SA round with lower temperature if time permits
    if time.time() < DEADLINE - 25 and best_scope > counting_bound:
        improved_rows, improved_scope = _simulated_annealing(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 15,
            initial_temp=5.0
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Final validation
    if _is_valid(best_rows):
        return best_rows
    else:
        # Fallback: sequential construction
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
