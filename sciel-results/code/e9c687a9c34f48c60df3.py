import time
import random
import math
import sys

def _row_diffs_list(row):
    """List of all positive differences in a row."""
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
    seen = set()
    for row in rows:
        for d in _row_diffs_list(row):
            if d in seen:
                return False
            seen.add(d)
    return True

def _compute_all_diffs(rows):
    row_masks = []
    global_mask = 0
    for row in rows:
        m = _row_mask(row)
        row_masks.append(m)
        global_mask |= m
    return global_mask, row_masks

def _spectral_assignment(n_rows, num_elements, target_scope, rng):
    """
    Build an initial solution using a cost-based assignment heuristic:
    For each row, greedily assign values minimizing a "spectral cost"
    that balances small differences (scarce) vs. keeping the row compact.
    This is a proxy for Hungarian-style optimal packing without the O(n^3) cost.
    """
    rows = []
    used_mask = 0
    # Precompute pair indices for each row
    pair_indices = []
    for pos in range(1, num_elements):
        pair_indices.append(pos)
    
    for _ in range(n_rows):
        row = [0]
        for pos in range(1, num_elements):
            best_val = None
            best_cost = float('inf')
            # Try values from row[-1]+1 up to target_scope
            for val in range(row[-1] + 1, target_scope + 1):
                new_diffs = set(val - prev for prev in row)
                if len(new_diffs) != len(set(new_diffs)):
                    continue
                new_mask = 0
                for d in new_diffs:
                    new_mask |= (1 << d)
                if new_mask & used_mask:
                    continue
                # Spectral cost: sum of squared differences (encourages balance)
                # plus a penalty for large values
                cost = sum(d * d for d in new_diffs) + 0.1 * val * val
                if cost < best_cost:
                    best_cost = cost
                    best_val = val
                    best_mask = new_mask
            if best_val is not None:
                row.append(best_val)
                used_mask |= best_mask
            else:
                # Fallback: jump far
                val = row[-1] + 20 + rng.randint(0, 30)
                row.append(val)
                for d in (val - prev for prev in row[:-1]):
                    used_mask |= (1 << d)
        rows.append(row)
    return rows

def _constrained_swap_search(rows, num_elements, rng, time_limit, max_swaps=100000):
    """
    Constrained swap local search:
    - Randomly pick two positions (possibly in different rows)
    - Try swapping their values
    - Accept only if global difference uniqueness is preserved
    - Track best scope
    """
    start = time.time()
    cur = [r[:] for r in rows]
    best = [r[:] for r in cur]
    best_scope = _scope(cur)
    cur_scope = best_scope
    gmask, rmasks = _compute_all_diffs(cur)
    n_rows = len(cur)
    n_elem = num_elements
    swaps = 0
    
    while time.time() < time_limit and swaps < max_swaps:
        swaps += 1
        # Pick two random positions
        r1 = rng.randint(0, n_rows - 1)
        r2 = rng.randint(0, n_rows - 1)
        p1 = rng.randint(1, n_elem - 1)
        p2 = rng.randint(1, n_elem - 1)
        
        if r1 == r2 and p1 == p2:
            continue
        
        v1 = cur[r1][p1]
        v2 = cur[r2][p2]
        
        if v1 == v2:
            continue
        
        # Tentatively swap
        cur[r1][p1] = v2
        cur[r2][p2] = v1
        
        # Check validity of affected rows
        ok = True
        affected = set()
        if r1 == r2:
            affected.add(r1)
        else:
            affected.add(r1)
            affected.add(r2)
        
        for ri in affected:
            row = cur[ri]
            # Check within-row uniqueness
            diffs = set()
            for i in range(n_elem):
                for j in range(i + 1, n_elem):
                    d = row[j] - row[i]
                    if d in diffs:
                        ok = False
                        break
                    diffs.add(d)
                if not ok:
                    break
            if not ok:
                break
            # Check global uniqueness
            new_mask = 0
            for d in diffs:
                new_mask |= (1 << d)
            other = gmask & ~rmasks[ri]
            if new_mask & other:
                ok = False
                break
        
        if not ok:
            # Revert
            cur[r1][p1] = v1
            cur[r2][p2] = v2
            continue
        
        # Update masks
        old_masks = []
        for ri in affected:
            old_masks.append(rmasks[ri])
            rmasks[ri] = _row_mask(cur[ri])
        
        new_gmask = gmask
        for ri in affected:
            new_gmask &= ~old_masks[affected.index(ri)]
            new_gmask |= rmasks[ri]
        
        # Check if scope improved
        new_scope = _scope(cur)
        if new_scope < best_scope:
            best_scope = new_scope
            best = [r[:] for r in cur]
            gmask = new_gmask
            cur_scope = new_scope
        else:
            # Revert
            cur[r1][p1] = v1
            cur[r2][p2] = v2
            for ri in affected:
                rmasks[ri] = old_masks[affected.index(ri)]
            gmask = (gmask & ~rmasks[r1]) | _row_mask(cur[r1])
            if r1 != r2:
                gmask = (gmask & ~rmasks[r2]) | _row_mask(cur[r2])
    
    return best, best_scope

def _element_reduction_search(rows, num_elements, rng, time_limit):
    """
    Targeted element reduction:
    - Focus on the row with the largest scope
    - Try to reduce its last element by finding a valid smaller value
    """
    start = time.time()
    cur = [r[:] for r in rows]
    best = [r[:] for r in cur]
    best_scope = _scope(cur)
    cur_scope = best_scope
    gmask, rmasks = _compute_all_diffs(cur)
    n_rows = len(cur)
    n_elem = num_elements
    
    while time.time() < time_limit:
        # Find row with max scope
        max_row = max(range(n_rows), key=lambda i: cur[i][-1])
        max_val = cur[max_row][-1]
        
        if max_val <= best_scope - 1:
            break
        
        # Try to reduce the last element
        row = cur[max_row]
        old = row[-1]
        low = row[-2] + 1
        high = old - 1
        
        if low >= high:
            break
        
        # Try random values in [low, high]
        for _ in range(50):
            new_val = rng.randint(low, high)
            if new_val == old:
                continue
            
            row[-1] = new_val
            diffs = set()
            ok = True
            for i in range(n_elem):
                for j in range(i + 1, n_elem):
                    d = row[j] - row[i]
                    if d in diffs:
                        ok = False
                        break
                    diffs.add(d)
                if not ok:
                    break
            
            if ok:
                new_mask = 0
                for d in diffs:
                    new_mask |= (1 << d)
                other = gmask & ~rmasks[max_row]
                if new_mask & other:
                    ok = False
            
            if not ok:
                row[-1] = old
                continue
            
            # Accept
            old_mask = rmasks[max_row]
            rmasks[max_row] = _row_mask(row)
            gmask = (gmask & ~old_mask) | rmasks[max_row]
            new_scope = _scope(cur)
            if new_scope < best_scope:
                best_scope = new_scope
                best = [r[:] for r in cur]
            break
        
        if time.time() > time_limit - 5:
            break
    
    return best, best_scope

def _construct(n_rows, k):
    start_time = time.time()
    DEADLINE = start_time + 135.0
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Spectral assignment at progressively lower scopes
    for seed in range(8):
        if time.time() > DEADLINE - 60:
            break
        rng_local = random.Random(seed * 13 + 7)
        start_scope = min(200, counting_bound + 50)
        for scope in range(start_scope, max(start_scope - 30, counting_bound - 5), -1):
            if time.time() > DEADLINE - 55:
                break
            rows = _spectral_assignment(n_rows, num_elements, scope, rng_local)
            if rows is not None and _is_valid(rows):
                s = _scope(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = rows
                break
        if best_rows is not None and best_scope <= counting_bound + 10:
            break
    
    # Fallback: simple greedy if spectral failed
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
        best_scope = _scope(rows)
    
    # Phase 2: Constrained swap search
    if time.time() < DEADLINE - 40:
        improved, imp_scope = _constrained_swap_search(
            best_rows, num_elements, rng, DEADLINE - 35)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved
    
    # Phase 3: Element reduction search
    if time.time() < DEADLINE - 20:
        improved, imp_scope = _element_reduction_search(
            best_rows, num_elements, rng, DEADLINE - 15)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved
    
    # Phase 4: Final polish with another swap pass
    if time.time() < DEADLINE - 10:
        improved, imp_scope = _constrained_swap_search(
            best_rows, num_elements, rng, DEADLINE - 8)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved
    
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

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    return _construct(n_rows, k)
