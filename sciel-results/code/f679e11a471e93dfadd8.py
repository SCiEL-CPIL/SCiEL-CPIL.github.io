import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k):
    """Standard greedy initialization."""
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
    using precomputed static diffs plus the k diffs involving pi.
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

def _multi_position_move(row, positions, static_diffs, all_used, old_diffs):
    """
    Try to improve multiple positions simultaneously in a row.
    positions: list of (pi, new_val) pairs to apply together.
    Returns (new_row, new_diffs, valid) or (None, None, False).
    """
    n = len(row)
    new_row = row[:]
    for pi, new_val in positions:
        new_row[pi] = new_val
    
    # Check strict increasing
    for i in range(1, n):
        if new_row[i] <= new_row[i-1]:
            return None, None, False
    
    # Compute all diffs of new_row
    new_diffs = set()
    for i in range(n):
        for j in range(i+1, n):
            d = new_row[j] - new_row[i]
            if d in new_diffs:
                return None, None, False
            new_diffs.add(d)
    
    # Check conflicts with other rows
    for d in new_diffs:
        if d in all_used and d not in old_diffs:
            return None, None, False
    
    return new_row, new_diffs, True

def _row_replacement_greedy(n_rows, k, all_used, exclude_row_idx, current_rows, target_max):
    """
    Generate a new row for position exclude_row_idx using greedy that avoids
    all differences used by other rows. target_max is the desired max.
    """
    other_diffs = set()
    for i in range(n_rows):
        if i == exclude_row_idx:
            continue
        other_diffs |= _row_diffs_set(current_rows[i])
    
    # Try to build a row with max <= target_max
    best_row = None
    best_max = target_max + 100
    
    # Multiple random restarts for the greedy
    for attempt in range(20):
        row = [0]
        used_in_row = set()
        x = 0
        success = True
        while len(row) < k + 1:
            found = False
            x += 1
            if x > target_max:
                success = False
                break
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_in_row) and not (new_diffs & other_diffs):
                row.append(x)
                used_in_row |= new_diffs
                found = True
            if not found:
                # Try next x
                continue
        if success and len(row) == k + 1:
            row_max = max(row)
            if row_max < best_max:
                best_max = row_max
                best_row = row[:]
            if best_max <= target_max:
                break
    
    return best_row

def _multi_pos_sa_step(current_rows, row_diffs, all_used, row_maxes, k, 
                        n_positions, T, target_row):
    """
    Perform a multi-position simulated annealing step on target_row.
    Tries to simultaneously move n_positions random positions.
    Returns (accepted, new_rows, new_diffs, new_max) or (False, None, None, None).
    """
    n = k + 1
    ri = target_row
    
    # Pick random positions to modify (excluding position 0)
    positions_to_move = random.sample(range(1, k + 1), min(n_positions, k))
    
    # For each position, generate a candidate
    moves = []
    for pi in positions_to_move:
        old_val = current_rows[ri][pi]
        min_allowed = current_rows[ri][pi - 1] + 1
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_rows[ri][k] + 20
        
        # Bias toward decreasing
        if random.random() < 0.7:
            # Decrease
            max_dec = min(30, old_val - min_allowed)
            if max_dec <= 0:
                continue
            new_val = random.randint(min_allowed, old_val - 1)
        else:
            # Small increase or decrease
            max_inc = min(10, max_allowed - old_val)
            max_dec = min(10, old_val - min_allowed)
            if max_inc > 0 and max_dec > 0:
                new_val = random.randint(min_allowed, max_allowed)
            elif max_dec > 0:
                new_val = random.randint(min_allowed, old_val - 1)
            elif max_inc > 0:
                new_val = random.randint(old_val + 1, max_allowed)
            else:
                continue
        
        if new_val == old_val:
            continue
        moves.append((pi, new_val))
    
    if not moves:
        return False, None, None, None
    
    # Apply moves and check validity
    new_row = current_rows[ri][:]
    for pi, new_val in moves:
        new_row[pi] = new_val
    
    # Check strict increasing
    for i in range(1, n):
        if new_row[i] <= new_row[i-1]:
            return False, None, None, None
    
    # Compute new diffs
    new_diffs = set()
    for i in range(n):
        for j in range(i+1, n):
            d = new_row[j] - new_row[i]
            if d in new_diffs:
                return False, None, None, False
            new_diffs.add(d)
    
    # Check conflicts
    old_diffs = row_diffs[ri]
    for d in new_diffs:
        if d in all_used and d not in old_diffs:
            return False, None, None, None
    
    # Compute new global max
    new_row_max = max(new_row)
    old_row_max = row_maxes[ri]
    
    # Temporarily compute new global max
    if new_row_max < old_row_max:
        # Global max might decrease
        new_global_max = max(max(current_rows[i][k]) if i != ri else new_row_max 
                           for i in range(len(current_rows)))
    else:
        new_global_max = max(row_maxes) if new_row_max <= max(row_maxes) else new_row_max
    
    delta = new_global_max - max(row_maxes)
    
    # Acceptance
    accept = False
    if delta < 0:
        accept = True
    elif delta == 0:
        accept = random.random() < 0.15
    else:
        if T > 0.1:
            prob = math.exp(-delta / max(T, 0.1))
            if random.random() < prob:
                accept = True
    
    if accept:
        new_rows = [r[:] for r in current_rows]
        new_rows[ri] = new_row
        return True, new_rows, new_diffs, new_global_max
    
    return False, None, None, None

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin for 150s limit
    
    # 1. Initialization
    rows = _greedy_init(n_rows, k)
    all_used = _compute_all_used(rows)
    
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    row_maxes = [max(r) for r in rows]
    
    # 2. Multi-position Simulated Annealing (structural change)
    current_rows = [r[:] for r in rows]
    current_val = best_val
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    T = 150.0
    T_min = 0.3
    cool = 0.99997
    
    no_improve = 0
    max_no_improve = 50000
    
    # Number of positions to move simultaneously
    n_pos_base = 2
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Choose strategy: multi-position or single-position
        if random.random() < 0.6:
            # Multi-position move
            ri = random.randrange(n_rows)
            # Bias toward rows with higher max
            if random.random() < 0.4 and n_rows > 1:
                max_idx = row_maxes.index(max(row_maxes))
                if random.random() < 0.5:
                    ri = max_idx
            
            n_pos = random.randint(1, min(3, k))
            
            accepted, new_rows, new_diffs, new_max = _multi_pos_sa_step(
                current_rows, row_diffs, all_used, row_maxes, k, n_pos, T, ri)
            
            if accepted:
                # Update state
                old_diffs = row_diffs[ri]
                all_used -= old_diffs
                all_used |= new_diffs
                
                current_rows = new_rows
                row_diffs[ri] = new_diffs
                row_maxes[ri] = max(new_rows[ri])
                current_val = new_max
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
        else:
            # Single-position move (finer granularity)
            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            
            old_val = current_rows[ri][pi]
            min_allowed = current_rows[ri][pi - 1] + 1
            max_decrease = min(40, old_val - min_allowed)
            
            if max_decrease <= 0:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
            
            candidates = []
            for d in range(1, max_decrease + 1):
                candidates.append(old_val - d)
            
            # Some increasing for exploration
            if pi < k:
                max_inc = min(10, current_rows[ri][pi+1] - 1 - old_val)
                for d in range(1, max_inc + 1):
                    candidates.append(old_val + d)
            else:
                for d in range(1, 8):
                    candidates.append(old_val + d)
            
            if not candidates:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
            
            random.shuffle(candidates)
            
            static_pi = _build_static_diffs(current_rows[ri], k)[pi]
            
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
                
                conflict = False
                for d in new_diffs:
                    if d in all_used and d not in old_diffs:
                        conflict = True
                        break
                
                if conflict:
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
                    accept = random.random() < 0.15
                else:
                    if T > T_min:
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
            row_maxes = [max(r) for r in current_rows]
            T = 150.0
            no_improve = 0
    
    # 3. Row replacement phase: try to replace entire rows with better ones
    replace_deadline = min(deadline, start_time + 118.0)
    
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    no_improve_replace = 0
    max_no_improve_replace = 20000
    
    while time.time() < replace_deadline and no_improve_replace < max_no_improve_replace:
        # Pick a row to potentially replace (bias toward max rows)
        if random.random() < 0.6:
            max_idx = row_maxes.index(max(row_maxes))
            ri = max_idx
        else:
            ri = random.randrange(n_rows)
        
        target_max = row_maxes[ri] - 1  # Try to get a smaller max
        
        new_row = _row_replacement_greedy(
            n_rows, k, all_used, ri, current_rows, target_max)
        
        if new_row is not None:
            new_row_max = max(new_row)
            new_diffs = _row_diffs_set(new_row)
            
            # Check if this actually improves
            old_row_max = row_maxes[ri]
            if new_row_max < old_row_max:
                # Check conflicts with other rows
                other_diffs = set()
                for i in range(n_rows):
                    if i != ri:
                        other_diffs |= row_diffs[i]
                
                conflict = False
                for d in new_diffs:
                    if d in other_diffs:
                        conflict = True
                        break
                
                if not conflict:
                    # Accept
                    all_used -= row_diffs[ri]
                    all_used |= new_diffs
                    current_rows[ri] = new_row
                    row_diffs[ri] = new_diffs
                    row_maxes[ri] = new_row_max
                    current_val = max(row_maxes)
                    
                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve_replace = 0
                    else:
                        no_improve_replace += 1
                else:
                    no_improve_replace += 1
            else:
                no_improve_replace += 1
        else:
            no_improve_replace += 1
    
    # 4. Final polish: focused single-position decreases on max rows
    final_deadline = deadline
    
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    while time.time() < final_deadline:
        # Focus on rows with the global max
        max_val = max(row_maxes)
        max_rows = [i for i in range(n_rows) if row_maxes[i] == max_val]
        
        if not max_rows:
            break
            
        ri = random.choice(max_rows)
        # Focus on last position (the max)
        pi = k
        
        old_val = current_rows[ri][pi]
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = old_val - min_allowed
        
        if max_decrease <= 0:
            continue
        
        # Try decreasing the last element
        static_pi = _build_static_diffs(current_rows[ri], k)[pi]
        
        moved = False
        # Try from largest decrease
        for d in range(max_decrease, 0, -1):
            new_val = old_val - d
            if new_val <= min_allowed:
                break
            
            new_diffs, valid_internal = _compute_new_diffs_fast(
                current_rows[ri], pi, new_val, static_pi)
            
            if not valid_internal:
                continue
            
            old_diffs = row_diffs[ri]
            
            conflict = False
            for d2 in new_diffs:
                if d2 in all_used and d2 not in old_diffs:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # Valid move that reduces the max
            all_used -= old_diffs
            all_used |= new_diffs
            current_rows[ri][pi] = new_val
            row_diffs[ri] = new_diffs
            row_maxes[ri] = new_val
            current_val = max(row_maxes)
            
            if current_val < best_val:
                best_val = current_val
                best_rows = [r[:] for r in current_rows]
            
            moved = True
            break
    
    return best_rows

# EVOLVE-BLOCK-END
