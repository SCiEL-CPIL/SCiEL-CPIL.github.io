import time
import random
import math

_DEADLINE = 120.0

def _get_diffs(row):
    diffs = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i+1, n):
            diffs.add(row[j] - ai)
    return diffs

def _is_valid_row(row):
    n = len(row)
    if n < 2: return True
    diffs = set()
    for i in range(n):
        ai = row[i]
        for j in range(i+1, n):
            d = row[j] - ai
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _construct_greedy(n_rows, k, seed):
    random.seed(seed)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            found = False
            # Greedy: try smallest available
            for v in range(row[-1] + 1, row[-1] + 500):
                new_row = row + [v]
                if not _is_valid_row(new_row):
                    continue
                new_diffs = _get_diffs(new_row)
                if new_diffs & used:
                    continue
                row = new_row
                used |= new_diffs
                found = True
                break
            if not found:
                # Fallback: random jump
                while len(row) < k + 1:
                    v = row[-1] + random.randint(1, 50)
                    new_row = row + [v]
                    if _is_valid_row(new_row):
                        new_diffs = _get_diffs(new_row)
                        if not (new_diffs & used):
                            row = new_row
                            used |= new_diffs
                            break
                    # If this fails too, we have a problem.
                    # But for k=5, it's very unlikely to fail with range 500.
                    pass
        rows.append(row)
    return rows, used

def _scope(rows):
    return max(max(r) for r in rows)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + _DEADLINE - 5.0
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Initialization
    for s in range(20):
        rows, used = _construct_greedy(n_rows, k, s)
        if rows:
            sc = _scope(rows)
            if sc < best_scope:
                best_scope = sc
                best_rows = [list(r) for r in rows]
                if best_scope == 112:
                    return best_rows

    if best_rows is None:
        # Emergency fallback
        rows = []
        for i in range(n_rows):
            row = [0]
            for j in range(1, k+1):
                row.append((i * (k+1) + j) * 100)
            rows.append(row)
        best_rows = rows
        best_scope = max(max(r) for r in rows)

    # Phase 2: Simulated Annealing
    current_rows = [list(r) for r in best_rows]
    current_scope = best_scope
    row_diffs = [_get_diffs(r) for r in current_rows]
    current_used = set()
    for rd in row_diffs:
        current_used |= rd
        
    T = 50.0
    T_min = 0.5
    alpha = 0.9995
    
    while time.time() < deadline:
        r_idx = random.randint(0, n_rows - 1)
        pos_idx = random.randint(1, k)
        
        old_val = current_rows[r_idx][pos_idx]
        
        # Generate candidate
        # Bias to reduce
        if random.random() < 0.8:
            min_val = current_rows[r_idx][pos_idx-1] + 1
            max_val = old_val - 1
            if max_val < min_val:
                continue
            new_val = random.randint(min_val, max_val)
        else:
            min_val = current_rows[r_idx][pos_idx-1] + 1
            max_val = old_val + 5
            if pos_idx < k:
                max_val = min(max_val, current_rows[r_idx][pos_idx+1] - 1)
            if max_val < min_val:
                continue
            new_val = random.randint(min_val, max_val)
            
        if new_val == old_val:
            continue
            
        new_row = list(current_rows[r_idx])
        new_row[pos_idx] = new_val
        
        if not _is_valid_row(new_row):
            continue
            
        new_row_diffs = _get_diffs(new_row)
        
        # Check conflict with other rows
        # A difference d in new_row_diffs is invalid if d is in current_used 
        # AND d was NOT in the old row_diffs (meaning it's used by another row)
        # If d was in old_row_diffs, it's "free" to be used again (or not) in this row, 
        # as long as it doesn't conflict with OTHER rows.
        # So: conflict if d in (current_used - old_row_diffs)
        
        old_rd = row_diffs[r_idx]
        other_diffs = current_used - old_rd
        
        # Optimization: iterate smaller set
        if len(new_row_diffs) < len(other_diffs):
            conflict = False
            for d in new_row_diffs:
                if d in other_diffs:
                    conflict = True
                    break
        else:
            conflict = False
            for d in other_diffs:
                if d in new_row_diffs:
                    conflict = True
                    break
                    
        if conflict:
            continue
            
        # Valid move
        new_max = max(new_row)
        # Calculate new scope
        # We can just take max of new_max and max of other rows
        # But we don't have precomputed maxes for other rows.
        # Let's just compute max of all rows, it's O(n*k) = 35 ops.
        new_scope = new_max
        for i in range(n_rows):
            if i != r_idx:
                m = max(current_rows[i])
                if m > new_scope:
                    new_scope = m
                    
        delta = new_scope - current_scope
        
        if delta < 0:
            # Accept
            current_rows[r_idx] = new_row
            current_used = (current_used - old_rd) | new_row_diffs
            row_diffs[r_idx] = new_row_diffs
            current_scope = new_scope
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in current_rows]
                if best_scope == 112:
                    return best_rows
        else:
            if T > T_min:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_rows[r_idx] = new_row
                    current_used = (current_used - old_rd) | new_row_diffs
                    row_diffs[r_idx] = new_row_diffs
                    current_scope = new_scope
                    
        T *= alpha
        if T < T_min:
            T = 20.0 # Restart temperature
            
    return best_rows
