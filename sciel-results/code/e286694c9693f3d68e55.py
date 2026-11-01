import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k):
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

def _greedy_init_target(n_rows, k, target_max):
    """
    Greedy initialization with a target maximum. Each row tries to stay within target_max.
    If it can't, it uses the smallest valid value (which may exceed target_max).
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
        # Try to compress this row to fit within target_max
        # Simple local search on the row
        for _ in range(200):
            improved = False
            for pi in range(k, 0, -1):
                min_allowed = row[pi-1] + 1
                for new_val in range(min_allowed, row[pi]):
                    if new_val == row[pi]:
                        continue
                    # Check internal validity
                    new_diffs = set()
                    valid = True
                    for i in range(pi):
                        d = row[pi] - row[i] if False else new_val - row[i]
                        new_diffs.add(d)
                    for i in range(pi+1, k+1):
                        d = row[i] - new_val
                        new_diffs.add(d)
                    # Check for internal conflicts
                    static_diffs = set()
                    for i in range(k+1):
                        if i == pi:
                            continue
                        for j in range(i+1, k+1):
                            if j == pi:
                                continue
                            static_diffs.add(row[j] - row[i])
                    if new_diffs & static_diffs:
                        continue
                    # Check external conflicts
                    if new_diffs & (used - _row_diffs_set(row)):
                        continue
                    # Valid move
                    used -= _row_diffs_set(row)
                    row[pi] = new_val
                    used |= new_diffs
                    improved = True
                    break
                if improved:
                    break
            if not improved:
                break
        rows.append(row)
    return rows

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

def _random_valid_row(k, used_diffs, max_val):
    """
    Try to build a random valid row of k+1 entries starting at 0,
    with all entries <= max_val, and no differences in used_diffs.
    Returns None if no valid row found.
    """
    for attempt in range(50):
        row = [0]
        valid = True
        for _ in range(k):
            # Pick a random next value
            low = row[-1] + 1
            high = max_val
            if low > high:
                valid = False
                break
            x = random.randint(low, high)
            new_diffs = {x - a for a in row}
            if new_diffs & used_diffs:
                valid = False
                break
            row.append(x)
        if valid:
            return row
    return None

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 108.0  # Safety margin for 120s limit
    
    # 1. Initialization - try multiple strategies
    rows = _greedy_init(n_rows, k)
    all_used = _compute_all_used(rows)
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    
    # Try greedy with target to see if we can do better
    greedy_target_max = best_val
    for _ in range(3):
        rows2 = _greedy_init_target(n_rows, k, greedy_target_max)
        val2 = _global_max(rows2)
        if val2 < best_val:
            best_rows = [r[:] for r in rows2]
            best_val = val2
            greedy_target_max = best_val
    
    # Maintain row maxes for efficient global max computation
    row_maxes = [max(r) for r in rows]
    
    # 2. Simulated Annealing with fast diff computation
    current_rows = [r[:] for r in rows]
    current_val = best_val
    T = 200.0
    T_min = 0.5
    cool = 0.99995
    
    no_improve = 0
    max_no_improve = 30000
    
    # Precompute row diffs and static diffs
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    
    # Phase 1: Main SA loop
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a row and position to modify - bias toward positions that affect max
        ri = random.randrange(n_rows)
        # Bias toward last position (which is the max of the row)
        if random.random() < 0.4:
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
            
            # Check for conflicts: new_diffs should not intersect with (all_used - old_diffs)
            conflict = False
            for d in new_diffs:
                if d in all_used and d not in old_diffs:
                    conflict = True
                    break
            
            if conflict:
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
                current_val = new_global_max
                
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
        
        # Periodic restart if stuck - use a smarter restart
        if no_improve > 8000:
            # Perturb the best solution slightly before restarting
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            
            # Apply small random perturbations to escape local minima
            for _ in range(5):
                ri = random.randrange(n_rows)
                pi = random.randrange(1, k + 1)
                old_val = current_rows[ri][pi]
                min_allowed = current_rows[ri][pi - 1] + 1
                max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else old_val + 10
                if min_allowed < max_allowed:
                    new_val = random.randint(min_allowed, max_allowed)
                    static_pi = _build_static_diffs(current_rows[ri], k)[pi]
                    new_diffs, valid_internal = _compute_new_diffs_fast(
                        current_rows[ri], pi, new_val, static_pi)
                    if valid_internal:
                        old_diffs = _row_diffs_set(current_rows[ri])
                        all_used_check = _compute_all_used(current_rows)
                        all_used_check -= old_diffs
                        if not (new_diffs & all_used_check):
                            current_rows[ri][pi] = new_val
            
            all_used = _compute_all_used(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            static_diffs = [_build_static_diffs(r, k) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 200.0
            no_improve = 0
    
    # Phase 2: Exhaustive polish - try all possible decreases for each position
    polish_deadline = min(deadline, start_time + 108.0)
    
    # Reset state to best
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    # Iterative improvement: keep trying to reduce the max
    improved = True
    while improved and time.time() < polish_deadline:
        improved = False
        for ri in range(n_rows):
            for pi in range(k, 0, -1):  # Start from last position (affects max)
                old_val = current_rows[ri][pi]
                min_allowed = current_rows[ri][pi - 1] + 1
                static_pi = static_diffs[ri][pi]
                old_diffs = row_diffs[ri]
                
                # Try all values from min_allowed to old_val - 1
                for new_val in range(min_allowed, old_val):
                    if pi < k and new_val >= current_rows[ri][pi + 1]:
                        continue
                    
                    new_diffs, valid_internal = _compute_new_diffs_fast(
                        current_rows[ri], pi, new_val, static_pi)
                    
                    if not valid_internal:
                        continue
                    
                    conflict = False
                    for d in new_diffs:
                        if d in all_used and d not in old_diffs:
                            conflict = True
                            break
                    
                    if conflict:
                        continue
                    
                    # Valid move - accept it (always improves or maintains)
                    if pi == k:
                        cand_row_max = new_val
                    else:
                        cand_row_max = current_rows[ri][k]
                    
                    old_row_max = row_maxes[ri]
                    row_maxes[ri] = cand_row_max
                    new_global_max = max(row_maxes)
                    row_maxes[ri] = old_row_max
                    
                    if new_global_max < current_val:
                        all_used -= old_diffs
                        all_used |= new_diffs
                        current_rows[ri][pi] = new_val
                        row_diffs[ri] = new_diffs
                        static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                        row_maxes[ri] = cand_row_max
                        current_val = new_global_max
                        improved = True
                        break
                    else:
                        # Same max - still accept to enable future improvements
                        if random.random() < 0.3:
                            all_used -= old_diffs
                            all_used |= new_diffs
                            current_rows[ri][pi] = new_val
                            row_diffs[ri] = new_diffs
                            static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                            row_maxes[ri] = cand_row_max
                            current_val = new_global_max
                            break
                # Check time
                if time.time() > polish_deadline:
                    break
            if time.time() > polish_deadline:
                break
    
    if current_val < best_val:
        best_val = current_val
        best_rows = [r[:] for r in current_rows]
    
    return best_rows

# EVOLVE-BLOCK-END
