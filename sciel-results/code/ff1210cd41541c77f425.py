import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k, gap_factor=10):
    """
    Standard greedy initialization: each row takes the smallest next mark with no difference collision.
    """
    rows = []
    used = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    
    return rows

def _greedy_init_random(n_rows, k, max_attempts=50):
    """
    Randomized greedy initialization: try multiple random orderings of rows and keep best.
    """
    best_rows = None
    best_val = float('inf')
    
    for attempt in range(max_attempts):
        rows = []
        used = set()
        order = list(range(n_rows))
        random.shuffle(order)
        
        row_map = {}
        for r in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            row_map[r] = row
        
        rows = [row_map[r] for r in range(n_rows)]
        val = max(max(row) for row in rows)
        
        if val < best_val:
            best_val = val
            best_rows = [r[:] for r in rows]
    
    return best_rows

def _row_diffs_set(row):
    """Compute the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _compute_all_used(rows):
    """Compute the union of all differences across all rows."""
    all_used = set()
    for row in rows:
        all_used |= _row_diffs_set(row)
    return all_used

def _global_max(rows):
    """Compute the maximum entry across all rows."""
    return max(max(row) for row in rows)

def _build_static_diffs(row, k):
    """
    For each position pi (1..k), compute the set of differences in the row
    that do NOT involve position pi.
    """
    n = k + 1
    static = [None] * (k + 1)
    for pi in range(1, k + 1):
        s = set()
        for i in range(n):
            if i == pi:
                continue
            for j in range(i + 1, n):
                if j == pi:
                    continue
                s.add(row[j] - row[i])
        static[pi] = s
    return static

def _compute_new_diffs_fast(row, pi, new_val, static_diffs_pi):
    """
    Given a row with position pi replaced by new_val, compute the full diff set
    using the precomputed static diffs plus the k diffs involving pi.
    Returns (new_diffs_set, valid_internal).
    """
    n = len(row)
    new_diffs = set(static_diffs_pi)
    valid = True
    for i in range(n):
        if i == pi:
            continue
        if i < pi:
            d = new_val - row[i]
        else:
            d = row[i] - new_val
        if d in new_diffs:
            valid = False
            break
        new_diffs.add(d)
    return new_diffs, valid

def _find_valid_value(row, pi, all_used, old_diffs, k, max_search=200):
    """
    Find a valid value for position pi in row that doesn't conflict with all_used
    (excluding old_diffs) and maintains internal uniqueness.
    Returns (new_val, new_diffs) or (None, None).
    """
    old_val = row[pi]
    min_allowed = row[pi - 1] + 1
    max_allowed = (row[pi + 1] - 1) if pi < k else old_val + max_search
    other_diffs = all_used - old_diffs

    # First try decreasing values
    for new_val in range(old_val - 1, min_allowed - 1, -1):
        # Compute new diffs
        new_diffs = set()
        valid = True
        for i in range(len(row)):
            if i == pi:
                continue
            if i < pi:
                d = new_val - row[i]
            else:
                d = row[i] - new_val
            if d in new_diffs:
                valid = False
                break
            new_diffs.add(d)
        
        if not valid:
            continue
        
        # Check conflicts using set intersection
        if not (new_diffs & other_diffs):
            return new_val, new_diffs
    
    # Try increasing values
    for new_val in range(old_val + 1, max_allowed + 1):
        if pi < k and new_val >= row[pi + 1]:
            break
        
        new_diffs = set()
        valid = True
        for i in range(len(row)):
            if i == pi:
                continue
            if i < pi:
                d = new_val - row[i]
            else:
                d = row[i] - new_val
            if d in new_diffs:
                valid = False
                break
            new_diffs.add(d)
        
        if not valid:
            continue
        
        if not (new_diffs & other_diffs):
            return new_val, new_diffs
    
    return None, None

def _row_replacement_search(rows, all_used, k, time_limit):
    """
    Try to replace each row with a better one using DFS-based search.
    Returns improved rows if found.
    """
    start_time = time.time()
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    
    for ri in range(len(rows)):
        if time.time() > time_limit:
            break
        
        old_row = rows[ri]
        old_diffs = _row_diffs_set(old_row)
        other_diffs = set(all_used) - old_diffs
        
        # Try to find a better row using DFS
        best_new_row = None
        best_new_max = old_row[-1]
        
        def dfs_row(row, current_max):
            nonlocal best_new_row, best_new_max
            
            if time.time() > time_limit:
                return
            
            if len(row) == k + 1:
                if current_max < best_new_max:
                    best_new_max = current_max
                    best_new_row = row[:]
                return
            
            last = row[-1]
            existing_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    existing_diffs.add(row[j] - row[i])
            
            # Try values from last+1 up to best_new_max - 1
            max_try = best_new_max - 1 if best_new_max < float('inf') else last + 50
            for x in range(last + 1, max_try + 1):
                new_diffs = {x - a for a in row}
                
                # Check against other rows' diffs
                if new_diffs & other_diffs:
                    continue
                
                # Check internal uniqueness
                if new_diffs & existing_diffs:
                    continue
                
                row.append(x)
                dfs_row(row, x)
                row.pop()
                
                if best_new_row is not None and x >= best_new_max:
                    break
        
        dfs_row([0], 0)
        
        if best_new_row is not None and best_new_max < old_row[-1]:
            # Update rows and all_used
            rows[ri] = best_new_row
            all_used = set(all_used) - old_diffs
            all_used |= _row_diffs_set(best_new_row)
            current_val = _global_max(rows)
            if current_val < best_val:
                best_val = current_val
                best_rows = [r[:] for r in rows]
    
    return best_rows, all_used

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 108.0  # Safety margin for 120s limit
    
    # 1. Multi-start Initialization
    # Try both standard greedy and randomized greedy, keep the best
    rows_std = _greedy_init(n_rows, k)
    val_std = _global_max(rows_std)
    
    rows_rand = _greedy_init_random(n_rows, k, max_attempts=20)
    val_rand = _global_max(rows_rand)
    
    if val_rand < val_std:
        rows = rows_rand
    else:
        rows = rows_std
    
    # Track used differences for fast checking
    all_used = _compute_all_used(rows)
    
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    
    # Maintain row maxes for efficient global max computation
    row_maxes = [max(r) for r in rows]
    
    # 2. Row Replacement Phase (inspired by DFS-based row search)
    replacement_deadline = min(deadline, start_time + 30.0)
    while time.time() < replacement_deadline:
        best_rows, all_used = _row_replacement_search(
            [r[:] for r in best_rows], 
            _compute_all_used(best_rows), 
            k, 
            min(replacement_deadline, time.time() + 10.0)
        )
        new_val = _global_max(best_rows)
        if new_val < best_val:
            best_val = new_val
            row_maxes = [max(r) for r in best_rows]
        else:
            break
    
    # 3. Simulated Annealing
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    T = 200.0
    T_min = 0.5
    cool = 0.99995
    
    no_improve = 0
    max_no_improve = 30000
    steps = 0
    
    # Precompute row diffs and static diffs
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    
    sa_deadline = min(deadline, start_time + 90.0)
    while time.time() < sa_deadline and no_improve < max_no_improve:
        steps += 1
        
        # Pick a row and position to modify
        # Bias toward rows with larger max entries and last positions
        if random.random() < 0.6:
            # Pick row with largest max
            ri = max(range(n_rows), key=lambda i: row_maxes[i])
        else:
            ri = random.randrange(n_rows)
        
        if random.random() < 0.5:
            # Pick last position (most impactful for max)
            pi = k
        else:
            pi = random.randrange(1, k + 1)  # 1..k
        
        old_val = current_rows[ri][pi]
        
        # Generate candidates with bias toward reducing max
        candidates = []
        
        # Decreasing candidates - try more of these
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(50, old_val - min_allowed)
        for d in range(1, max_decrease + 1):
            new_v = old_val - d
            candidates.append(new_v)
            
        # Increasing candidates (for exploration) - fewer
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 30
        max_increase = min(15, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            new_v = old_val + d
            candidates.append(new_v)
            
        # Random jump (large perturbation)
        if random.random() < 0.15:
            low = current_rows[ri][pi - 1] + 1
            high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 20
            if low <= high:
                candidates.append(random.randint(low, high))
        
        if not candidates:
            no_improve += 1
            T = max(T_min, T * cool)
            continue
            
        random.shuffle(candidates)
        
        # Pre-fetch static diffs for this position
        static_pi = static_diffs[ri][pi]
        
        moved = False
        for new_val in candidates:
            # Basic validity check for this position
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            # Fast diff computation using static diffs
            new_diffs, valid_internal = _compute_new_diffs_fast(
                current_rows[ri], pi, new_val, static_pi)
            
            if not valid_internal:
                continue
                
            # Get old diffs of row ri
            old_diffs = row_diffs[ri]
            
            # Check for conflicts
            if new_diffs & (all_used - old_diffs):
                continue
                
            # Move is valid - compute new global max efficiently
            if pi == k:
                cand_row_max = new_val
            else:
                cand_row_max = current_rows[ri][k]
            
            # Update row_maxes temporarily
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max  # restore
            
            delta = new_global_max - current_val
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.2:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.1))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                # Update state
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri][pi] = new_val
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = max(row_maxes)
                
                # Rebuild static diffs for this row
                static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
                    
                moved = True
                break
        
        if not moved:
            no_improve += 1
            
        T = max(T_min, T * cool)
        
        # Periodic restart if stuck
        if no_improve > 15000:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = _compute_all_used(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            static_diffs = [_build_static_diffs(r, k) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 200.0
            no_improve = 0
    
    # 4. Final polishing phase with tighter tolerance
    polish_deadline = min(deadline, start_time + 112.0)
    T = 1.0
    T_min = 0.1
    cool = 0.99999
    no_improve = 0
    max_no_improve_polish = 10000
    
    # Reset state to best
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    while time.time() < polish_deadline and no_improve < max_no_improve_polish:
        # Pick a row and position - bias toward last position of max row
        if random.random() < 0.7:
            ri = max(range(n_rows), key=lambda i: row_maxes[i])
            pi = k
        else:
            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        # Only try decreasing moves in polish phase
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(30, old_val - min_allowed)
        
        candidates = []
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)
        
        if not candidates:
            no_improve += 1
            T = max(T_min, T * cool)
            continue
            
        random.shuffle(candidates)
        
        static_pi = static_diffs[ri][pi]
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            new_diffs, valid_internal = _compute_new_diffs_fast(
                current_rows[ri], pi, new_val, static_pi)
            
            if not valid_internal:
                continue
                
            old_diffs = row_diffs[ri]
            
            if new_diffs & (all_used - old_diffs):
                continue
                
            if pi == k:
                cand_row_max = new_val
            else:
                cand_row_max = current_rows[ri][k]
            
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.1:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.05))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri][pi] = new_val
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = max(row_maxes)
                
                static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
                    
                moved = True
                break
        
        if not moved:
            no_improve += 1
            
        T = max(T_min, T * cool)
    
    # 5. Final local search: try all positions in all rows with small decreases
    final_deadline = deadline
    for _ in range(5):  # Multiple passes
        if time.time() > final_deadline:
            break
        
        improved = True
        while improved and time.time() < final_deadline:
            improved = False
            for ri in range(n_rows):
                if time.time() > final_deadline:
                    break
                for pi in range(k, 0, -1):  # Start from last position
                    if time.time() > final_deadline:
                        break
                    
                    old_val = current_rows[ri][pi]
                    min_allowed = current_rows[ri][pi - 1] + 1
                    static_pi = static_diffs[ri][pi]
                    
                    # Try small decreases
                    for new_val in range(old_val - 1, min_allowed - 1, -1):
                        if pi < k and new_val >= current_rows[ri][pi + 1]:
                            break
                            
                        new_diffs, valid_internal = _compute_new_diffs_fast(
                            current_rows[ri], pi, new_val, static_pi)
                        
                        if not valid_internal:
                            continue
                        
                        old_diffs = row_diffs[ri]
                        
                        if new_diffs & (all_used - old_diffs):
                            continue
                        
                        # Valid move - check if it improves global max
                        if pi == k:
                            cand_row_max = new_val
                        else:
                            cand_row_max = current_rows[ri][k]
                        
                        old_row_max = row_maxes[ri]
                        row_maxes[ri] = cand_row_max
                        new_global_max = max(row_maxes)
                        row_maxes[ri] = old_row_max
                        
                        if new_global_max < current_val:
                            # Accept
                            all_used -= old_diffs
                            all_used |= new_diffs
                            current_rows[ri][pi] = new_val
                            row_diffs[ri] = new_diffs
                            row_maxes[ri] = cand_row_max
                            current_val = new_global_max
                            
                            static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                            
                            if current_val < best_val:
                                best_val = current_val
                                best_rows = [r[:] for r in current_rows]
                            
                            improved = True
                            break
    
    return best_rows

# EVOLVE-BLOCK-END
