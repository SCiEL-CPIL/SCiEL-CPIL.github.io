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
            if max_scope_hint:
                high = max_scope_hint
            else:
                high = 200
            
            candidates = list(range(row[-1] + 1, high + 1))
            
            if len(candidates) > 30:
                sample_size = min(25, len(candidates))
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
                if len(new_diffs) != len(set(new_diffs)):
                    continue
                new_mask = 0
                for d in new_diffs:
                    new_mask |= (1 << d)
                if new_mask & used_mask:
                    continue
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
                row.append(row[-1] + 10 + rng.randint(0, 20))
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
    
    if pos > 0 and new_val <= row[pos - 1]:
        return False, None, None
    if pos < n - 1 and new_val >= row[pos + 1]:
        return False, None, None
    
    old_val = row[pos]
    if new_val == old_val:
        return False, None, None
    
    row[pos] = new_val
    
    diffs_in_row = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            if d in diffs_in_row:
                row[pos] = old_val
                return False, None, None
            diffs_in_row.add(d)
    
    new_row_mask = 0
    for d in diffs_in_row:
        new_row_mask |= (1 << d)
    
    other_mask = global_mask & ~row_masks[row_idx]
    if new_row_mask & other_mask:
        row[pos] = old_val
        return False, None, None
    
    new_global_mask = (global_mask & ~row_masks[row_idx]) | new_row_mask
    row[pos] = old_val
    
    return True, new_global_mask, new_row_mask


def _error_repair(rows, num_elements, rng, time_limit):
    """
    Error repair: repeatedly find the element causing the most conflicts and
    shift it to a position that minimizes conflicts. This is a local repair
    heuristic that targets specific "error" locations rather than random moves.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    global_mask, row_masks = _compute_all_diffs(current_rows)
    n_rows = len(current_rows)
    n_elem = num_elements
    
    max_iter = 50000
    stagnation = 0
    max_stagnation = 3000
    
    for it in range(max_iter):
        if time.time() > time_limit:
            break
        if stagnation > max_stagnation:
            break
        
        # Find the element with the most "pressure" (highest value that could be lowered)
        best_row_idx = None
        best_pos = None
        best_gain = 0
        
        # Sample a few random candidates to find a good repair target
        for _ in range(5):
            row_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, n_elem - 1)
            row = current_rows[row_idx]
            old_val = row[pos]
            
            low = row[pos - 1] + 1
            # How much can we lower this element?
            potential_gain = old_val - low
            if potential_gain > best_gain:
                best_gain = potential_gain
                best_row_idx = row_idx
                best_pos = pos
        
        if best_row_idx is None or best_gain <= 0:
            stagnation += 1
            continue
        
        row_idx = best_row_idx
        pos = best_pos
        row = current_rows[row_idx]
        old_val = row[pos]
        
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < n_elem - 1 else current_scope
        
        if low >= old_val:
            stagnation += 1
            continue
        
        # Try several candidate values, biased toward lower
        tried = 0
        for _ in range(10):
            if tried >= 10:
                break
            tried += 1
            # Bias toward lower values
            if rng.random() < 0.7:
                new_val = rng.randint(low, max(low, old_val - 1))
            else:
                new_val = rng.randint(low, high)
            
            if new_val >= old_val:
                continue
            
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
                break
        else:
            stagnation += 1
    
    return best_rows, best_scope


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
        
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, n_elem - 1)
        row = current_rows[row_idx]
        old_val = row[pos]
        
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < n_elem - 1 else current_scope + 10
        
        if low >= high:
            continue
        
        if old_val > low + 2 and rng.random() < 0.7:
            search_high = min(old_val - 1, high)
            if low > search_high:
                continue
            new_val = rng.randint(low, search_high)
        else:
            new_val = rng.randint(low, high)
        
        if new_val == old_val:
            continue
        
        success, new_global_mask, new_row_mask = _try_move_element(
            current_rows, row_idx, pos, new_val, global_mask, row_masks
        )
        
        if not success:
            continue
        
        current_rows[row_idx][pos] = new_val
        old_row_mask = row_masks[row_idx]
        row_masks[row_idx] = new_row_mask
        global_mask = new_global_mask
        
        new_scope = _scope(current_rows)
        delta = new_scope - current_scope
        
        if delta <= 0 or rng.random() < math.exp(-delta / temp):
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
        else:
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
    DEADLINE = start_time + 135.0
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))
    
    rng = random.Random(42)
    
    # Phase 1: Generate initial solution
    best_rows = None
    best_scope = float('inf')
    
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
    
    # Phase 2: Error repair (new targeted approach)
    if time.time() < DEADLINE - 60:
        repaired_rows, repaired_scope = _error_repair(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 60
        )
        if repaired_scope < best_scope:
            best_scope = repaired_scope
            best_rows = repaired_rows
    
    # Phase 3: Simulated Annealing
    if time.time() < DEADLINE - 40:
        improved_rows, improved_scope = _simulated_annealing(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 40,
            initial_temp=max(10.0, best_scope * 0.1)
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Phase 4: Local search refinement
    if time.time() < DEADLINE - 25:
        improved_rows, improved_scope = _local_search_refine(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 25
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Phase 5: Final error repair pass
    if time.time() < DEADLINE - 15 and best_scope > counting_bound:
        repaired_rows, repaired_scope = _error_repair(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 15
        )
        if repaired_scope < best_scope:
            best_scope = repaired_scope
            best_rows = repaired_rows
    
    # Final validation
    if _is_valid(best_rows):
        return best_rows
    else:
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
