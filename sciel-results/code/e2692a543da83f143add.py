import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k):
    """
    Greedy initialization using a boolean array for speed.
    """
    rows = []
    used = [False] * 600
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            valid = True
            new_diffs = []
            for a in row:
                d = x - a
                if d >= len(used):
                    used.extend([False] * (d - len(used) + 10))
                if used[d]:
                    valid = False
                    break
                new_diffs.append(d)
            if valid:
                row.append(x)
                for d in new_diffs:
                    used[d] = True
        rows.append(row)
    return rows

def _row_diffs_list(row):
    """Compute the list of all positive differences in a row."""
    diffs = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return diffs

def _row_diffs_set(row):
    """Compute the set of all positive differences in a row."""
    return set(_row_diffs_list(row))

def _compute_all_used_set(rows):
    """Compute the set of all differences across all rows."""
    all_used = set()
    for row in rows:
        all_used |= _row_diffs_set(row)
    return all_used

def _global_max(rows):
    """Compute the maximum entry across all rows."""
    return max(max(row) for row in rows)

def _check_conflicts(new_diffs, old_diffs, all_used):
    """
    Check if new_diffs conflict with all_used, excluding those in old_diffs.
    Returns True if there is a conflict.
    """
    for d in new_diffs:
        if d in all_used and d not in old_diffs:
            return True
    return False

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

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit
    
    # 1. Initialization with multi-start
    best_rows = None
    best_val = float('inf')
    
    # Try a few different greedy initializations with different random seeds
    for seed in [0, 1, 2]:
        random.seed(seed)
        rows = _greedy_init(n_rows, k)
        val = _global_max(rows)
        if val < best_val:
            best_val = val
            best_rows = [r[:] for r in rows]
    
    # State for SA
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    row_maxes = [max(r) for r in current_rows]
    
    all_used = _compute_all_used_set(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    
    T = 400.0
    T_min = 0.3
    cool = 0.99997
    
    no_improve = 0
    max_no_improve = 20000
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a row and position
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        # Generate candidates
        candidates = []
        
        # Decreasing candidates
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(50, old_val - min_allowed)
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)
            
        # Increasing candidates
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 30
        max_increase = min(20, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            candidates.append(old_val + d)
            
        # Random jump
        if random.random() < 0.2:
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
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            # Fast diff computation using static diffs
            new_diffs, valid_internal = _compute_new_diffs_fast(
                current_rows[ri], pi, new_val, static_pi)
            
            if not valid_internal:
                continue
                
            old_diffs = row_diffs[ri]
            
            # Check conflicts
            if _check_conflicts(new_diffs, old_diffs, all_used):
                continue
                
            # Move is valid
            if pi == k:
                cand_row_max = new_val
            else:
                cand_row_max = current_rows[ri][k]
            
            # Calculate delta
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.4:
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
        
        # Restart if stuck
        if no_improve > 10000:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = _compute_all_used_set(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            static_diffs = [_build_static_diffs(r, k) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 400.0
            no_improve = 0
    
    # 2. Polishing Phase: Try to force lower max values
    polish_deadline = min(deadline, start_time + 119.0)
    
    current_best = best_val
    for target_offset in range(1, 6):
        target_val = current_best - target_offset
        if target_val < 100:
            break
            
        if time.time() > polish_deadline - 2:
            break
            
        # Reset to best
        current_rows = [r[:] for r in best_rows]
        current_val = best_val
        all_used = _compute_all_used_set(current_rows)
        row_diffs = [_row_diffs_set(r) for r in current_rows]
        static_diffs = [_build_static_diffs(r, k) for r in current_rows]
        row_maxes = [max(r) for r in current_rows]
        
        T = 50.0
        T_min = 0.1
        cool = 0.9999
        no_improve = 0
        max_steps = 3000
        
        for _ in range(max_steps):
            if time.time() > polish_deadline - 2:
                break
                
            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            old_val = current_rows[ri][pi]
            
            min_allowed = current_rows[ri][pi - 1] + 1
            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else target_val
            
            if min_allowed > max_allowed:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
                
            new_val = random.randint(min_allowed, max_allowed)
            if new_val == old_val:
                continue
                
            # Fast diff computation
            static_pi = static_diffs[ri][pi]
            new_diffs, valid_internal = _compute_new_diffs_fast(
                current_rows[ri], pi, new_val, static_pi)
            
            if not valid_internal:
                continue
                
            old_diffs = row_diffs[ri]
            
            if _check_conflicts(new_diffs, old_diffs, all_used):
                continue
                
            if pi == k:
                cand_row_max = new_val
            else:
                cand_row_max = current_rows[ri][k]
                
            if cand_row_max > target_val:
                continue
                
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.5:
                    accept = True
            else:
                if new_global_max <= target_val:
                    prob = math.exp(-delta / max(T, 0.1))
                    if random.random() < prob:
                        accept = True
            
            if accept:
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
                    no_improve = 0
                    if best_val <= target_val:
                        break
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
            T = max(T_min, T * cool)
            
            if no_improve > 500:
                break
    
    return best_rows

# EVOLVE-BLOCK-END
