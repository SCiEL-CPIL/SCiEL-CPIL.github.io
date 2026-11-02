import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k):
    """
    Standard greedy initialization: each row takes the smallest next mark with no difference collision.
    Returns rows and the initial all_used bitmask.
    """
    rows = []
    used_bits = 0
    for r in range(n_rows):
        row = [0]
        x = 0
        row_bits = 0
        while len(row) < k + 1:
            x += 1
            # Check if x - a creates a collision
            valid = True
            new_diffs_bits = 0
            for a in row:
                d = x - a
                # Check if d is already in row_bits (internal collision)
                if (row_bits >> d) & 1:
                    valid = False
                    break
                # Check if d is already in used_bits (external collision)
                if (used_bits >> d) & 1:
                    valid = False
                    break
                new_diffs_bits |= (1 << d)
            
            if valid:
                row.append(x)
                row_bits |= new_diffs_bits
                used_bits |= new_diffs_bits
        
        rows.append(row)
    return rows, used_bits

def _row_diffs_bits(row):
    """Compute the bitmask of all positive differences in a row."""
    bits = 0
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            bits |= (1 << d)
    return bits

def _build_static_diffs_bits(row, k):
    """
    For each position pi (1..k), compute the bitmask of differences in the row
    that do NOT involve position pi.
    """
    n = k + 1
    static = [0] * (k + 1)
    for pi in range(1, k + 1):
        s = 0
        for i in range(n):
            if i == pi:
                continue
            for j in range(i + 1, n):
                if j == pi:
                    continue
                d = row[j] - row[i]
                s |= (1 << d)
        static[pi] = s
    return static

def _compute_new_diffs_bits_fast(row, pi, new_val, static_bits_pi):
    """
    Given a row with position pi replaced by new_val, compute the full diff bitmask
    using the precomputed static bits (diffs not involving pi) plus the k diffs
    involving pi. Returns (new_bits, valid_internal).
    """
    n = len(row)
    new_bits = static_bits_pi
    valid = True
    for i in range(n):
        if i == pi:
            continue
        if i < pi:
            d = new_val - row[i]
        else:
            d = row[i] - new_val
        
        # Check if this new diff collides with existing new_bits
        if (new_bits >> d) & 1:
            valid = False
            break
        
        new_bits |= (1 << d)
    return new_bits, valid

def _try_move(current_rows, ri, pi, new_val, row_diffs_bits, static_diffs_bits, all_used_bits, current_val, k):
    """
    Attempt a single move: set current_rows[ri][pi] = new_val.
    Returns (accepted, new_bits, cand_row_max) if accepted, else (False, None, None).
    Does NOT mutate state; caller must apply.
    """
    # Basic validity check for this position
    if new_val <= current_rows[ri][pi - 1]:
        return False, None, None
    if pi < k and new_val >= current_rows[ri][pi + 1]:
        return False, None, None
    
    # Fast diff computation using static bits
    static_pi = static_diffs_bits[ri][pi]
    new_bits, valid_internal = _compute_new_diffs_bits_fast(
        current_rows[ri], pi, new_val, static_pi)
    
    if not valid_internal:
        return False, None, None
    
    # Check for conflicts: new_bits should not intersect with (all_used - old_bits)
    old_bits = row_diffs_bits[ri]
    # Mask out the old row's bits from the global used bits
    other_used_bits = all_used_bits & ~old_bits
    
    if new_bits & other_used_bits:
        return False, None, None
    
    # Compute new row max
    if pi == k:
        cand_row_max = new_val
    else:
        cand_row_max = current_rows[ri][k]
    
    return True, new_bits, cand_row_max

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Initialization
    rows, initial_used_bits = _greedy_init(n_rows, k)
    
    best_rows = [r[:] for r in rows]
    best_val = max(max(r) for r in rows)
    
    # Maintain state
    current_rows = [r[:] for r in rows]
    current_val = best_val
    all_used_bits = initial_used_bits
    row_diffs_bits = [_row_diffs_bits(r) for r in current_rows]
    static_diffs_bits = [_build_static_diffs_bits(r, k) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    # 2. Simulated Annealing
    T = 200.0
    T_min = 0.5
    cool = 0.99995
    
    no_improve = 0
    max_no_improve = 30000
    
    while time.time() < deadline and no_improve < max_no_improve:
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)  # 1..k
        
        old_val = current_rows[ri][pi]
        
        # Generate candidates with bias toward reducing max
        candidates = []
        
        # Decreasing candidates
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(50, old_val - min_allowed)
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)
            
        # Increasing candidates (for exploration)
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 30
        max_increase = min(15, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            candidates.append(old_val + d)
            
        # Random jump
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
        
        moved = False
        for new_val in candidates:
            accepted, new_bits, cand_row_max = _try_move(
                current_rows, ri, pi, new_val, row_diffs_bits, static_diffs_bits, 
                all_used_bits, current_val, k)
            
            if not accepted:
                continue
            
            # Compute new global max
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
                all_used_bits = (all_used_bits & ~row_diffs_bits[ri]) | new_bits
                current_rows[ri][pi] = new_val
                row_diffs_bits[ri] = new_bits
                row_maxes[ri] = cand_row_max
                current_val = new_global_max
                
                # Rebuild static bits for this row
                static_diffs_bits[ri] = _build_static_diffs_bits(current_rows[ri], k)
                
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
        if no_improve > 12000:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used_bits = 0
            for r in current_rows:
                all_used_bits |= _row_diffs_bits(r)
            row_diffs_bits = [_row_diffs_bits(r) for r in current_rows]
            static_diffs_bits = [_build_static_diffs_bits(r, k) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 200.0
            no_improve = 0
    
    # 3. Targeted shrink phase
    polish_deadline = min(deadline, start_time + 115.0)
    
    # Reset state to best
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used_bits = 0
    for r in current_rows:
        all_used_bits |= _row_diffs_bits(r)
    row_diffs_bits = [_row_diffs_bits(r) for r in current_rows]
    static_diffs_bits = [_build_static_diffs_bits(r, k) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
    
    no_improve = 0
    max_no_improve_polish = 25000
    
    while time.time() < polish_deadline and no_improve < max_no_improve_polish:
        if max_row_indices and random.random() < 0.8:
            ri = random.choice(max_row_indices)
        else:
            ri = random.randrange(n_rows)
        
        if ri in max_row_indices and random.random() < 0.6:
            pi = k  # the last element
        else:
            pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(60, old_val - min_allowed)
        
        if max_decrease <= 0:
            no_improve += 1
            continue
        
        if max_decrease > 20:
            n_try = min(max_decrease, random.randint(10, 20))
            weights = [d for d in range(1, max_decrease + 1)]
            try_decreases = random.choices(weights, k=n_try)
            if max_decrease not in try_decreases:
                try_decreases.append(max_decrease)
        else:
            try_decreases = list(range(1, max_decrease + 1))
        
        candidates = [old_val - d for d in try_decreases]
        
        moved = False
        for new_val in candidates:
            accepted, new_bits, cand_row_max = _try_move(
                current_rows, ri, pi, new_val, row_diffs_bits, static_diffs_bits, 
                all_used_bits, current_val, k)
            
            if not accepted:
                continue
            
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.2
            else:
                accept = False
            
            if accept:
                all_used_bits = (all_used_bits & ~row_diffs_bits[ri]) | new_bits
                current_rows[ri][pi] = new_val
                row_diffs_bits[ri] = new_bits
                row_maxes[ri] = cand_row_max
                current_val = new_global_max
                
                static_diffs_bits[ri] = _build_static_diffs_bits(current_rows[ri], k)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                    max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
                else:
                    no_improve += 1
                    
                moved = True
                break
        
        if not moved:
            no_improve += 1
            
    # 4. Final quick SA pass
    final_deadline = deadline
    T = 0.15
    T_min = 0.01
    cool = 0.99999
    no_improve = 0
    
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used_bits = 0
    for r in current_rows:
        all_used_bits |= _row_diffs_bits(r)
    row_diffs_bits = [_row_diffs_bits(r) for r in current_rows]
    static_diffs_bits = [_build_static_diffs_bits(r, k) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    while time.time() < final_deadline:
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(25, old_val - min_allowed)
        
        if max_decrease <= 0:
            continue
        
        n_try = min(max_decrease, random.randint(5, 10))
        candidates = random.sample(range(1, max_decrease + 1), n_try)
        candidates = [old_val - d for d in candidates]
        
        moved = False
        for new_val in candidates:
            accepted, new_bits, cand_row_max = _try_move(
                current_rows, ri, pi, new_val, row_diffs_bits, static_diffs_bits, 
                all_used_bits, current_val, k)
            
            if not accepted:
                continue
            
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.1
            else:
                accept = False
            
            if accept:
                all_used_bits = (all_used_bits & ~row_diffs_bits[ri]) | new_bits
                current_rows[ri][pi] = new_val
                row_diffs_bits[ri] = new_bits
                row_maxes[ri] = cand_row_max
                current_val = new_global_max
                
                static_diffs_bits[ri] = _build_static_diffs_bits(current_rows[ri], k)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    
                moved = True
                break
    
    return best_rows

# EVOLVE-BLOCK-END
