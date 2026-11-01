import time
import random
import math

# EVOLVE-BLOCK-START

def _is_valid(rows):
    """Check if the set of rows is valid (all diffs distinct globally)."""
    used = set()
    for row in rows:
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
        if diffs & used:
            return False
        used |= diffs
    return True

def _greedy_init(n_rows, k):
    """
    Greedy initialization: each row takes the smallest next mark with no difference collision.
    """
    rows = []
    used = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if len(new_diffs) != len(row):
                continue
            if new_diffs & used:
                continue
            row.append(x)
            used |= new_diffs
        rows.append(row)
    return rows

def _row_diffs(row):
    """Compute the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _all_used(rows):
    """Compute the union of all differences across all rows."""
    all_used = set()
    for row in rows:
        all_used |= _row_diffs(row)
    return all_used

def _global_max(rows):
    """Compute the maximum entry across all rows."""
    return max(max(row) for row in rows)

def _validate_move_fast(cand_row, old_diffs, all_used, pi, old_val, new_val):
    """
    Fast validation for a single element change at index pi.
    Returns (new_diffs, is_valid).
    """
    removed = set()
    added = set()
    
    # Diffs with elements before pi
    for i in range(pi):
        removed.add(old_val - cand_row[i])
        added.add(new_val - cand_row[i])
        
    # Diffs with elements after pi
    for j in range(pi + 1, len(cand_row)):
        removed.add(cand_row[j] - old_val)
        added.add(cand_row[j] - new_val)
        
    # Check for duplicates in added (internal conflict)
    if len(added) < len(removed):
        return None, False
        
    # Check internal conflict: added vs remaining old diffs
    remaining_old = old_diffs - removed
    if added & remaining_old:
        return None, False
        
    # Check cross-row conflict: added vs diffs in other rows
    other_rows_diffs = all_used - old_diffs
    if added & other_rows_diffs:
        return None, False
        
    new_diffs = remaining_old | added
    return new_diffs, True

def _backtrack_construct(n_rows, k, max_entry, time_limit=5.0):
    """
    Backtracking search with forward checking to find a valid construction
    where all entries are <= max_entry. Returns (rows, found).
    """
    start = time.time()
    deadline = start + time_limit
    
    rows = []
    used = set()
    
    def backtrack(row_idx, row):
        nonlocal used
        if time.time() > deadline:
            return False
        
        if row_idx == n_rows:
            return True
        
        def build_row(pos, current_row, current_diffs):
            if time.time() > deadline:
                return None
            
            if pos == k + 1:
                return current_row[:]
            
            if pos == 0:
                return build_row(1, [0], set())
            
            prev = current_row[-1]
            remaining = k + 1 - pos
            min_next = prev + 1
            
            x = min_next
            while x <= max_entry:
                new_diffs = {x - a for a in current_row}
                if len(new_diffs) != len(current_row):
                    x += 1
                    continue
                
                if new_diffs & used:
                    x += 1
                    continue
                
                if x + (remaining - 1) > max_entry:
                    break
                
                result = build_row(pos + 1, current_row + [x], current_diffs | new_diffs)
                if result is not None:
                    return result
                
                x += 1
            
            return None
        
        new_row = build_row(0, [], set())
        if new_row is not None:
            rows.append(new_row)
            used_copy = set(used)
            used |= _row_diffs(new_row)
            
            if backtrack(row_idx + 1, new_row):
                return True
            
            used = used_copy
            rows.pop()
        
        return False
    
    found = backtrack(0, [])
    if found:
        return rows, True
    return None, False

def _simulated_annealing(rows, n_rows, k, time_budget, T_init=100.0, T_min=0.1, cool=0.99995):
    """
    Simulated annealing to minimize the global max entry.
    Returns improved rows and the best value found.
    """
    current_rows = [r[:] for r in rows]
    current_val = _global_max(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_val = current_val
    
    row_maxes = [max(r) for r in current_rows]
    all_used = _all_used(current_rows)
    row_diffs = [_row_diffs(r) for r in current_rows]
    
    start_time = time.time()
    deadline = start_time + time_budget
    T = T_init
    no_improve = 0
    max_no_improve = 15000
    restarts = 0
    max_restarts = 10
    
    while time.time() < deadline and no_improve < max_no_improve:
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        min_allowed = current_rows[ri][pi - 1] + 1
        
        # Build candidate list
        candidates = []
        
        # Try decreasing values (most important for minimization)
        max_decrease = min(50, old_val - min_allowed)
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)
        
        # Try some increasing values for exploration
        if pi < k:
            max_allowed = current_rows[ri][pi + 1] - 1
        else:
            max_allowed = current_val + 20
        max_increase = min(15, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            candidates.append(old_val + d)
        
        # Random jump
        if random.random() < 0.2:
            low = min_allowed
            high = max_allowed if pi < k else current_val + 20
            if low <= high:
                candidates.append(random.randint(low, high))
        
        if not candidates:
            no_improve += 1
            T = max(T_min, T * cool)
            continue
        
        random.shuffle(candidates)
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
            
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            
            new_diffs, valid = _validate_move_fast(cand_row, row_diffs[ri], all_used, pi, old_val, new_val)
            if not valid:
                continue
            
            cand_row_max = max(cand_row)
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
                    prob = math.exp(-delta / max(T, 0.1))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                all_used -= row_diffs[ri]
                all_used |= new_diffs
                current_rows[ri] = cand_row
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = max(row_maxes)
                
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
        
        if no_improve >= max_no_improve:
            restarts += 1
            if restarts >= max_restarts:
                break
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = _all_used(current_rows)
            row_diffs = [_row_diffs(r) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = T_init * 0.5
            no_improve = 0
    
    return best_rows, best_val

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # 1. Greedy initialization
    greedy_rows = _greedy_init(n_rows, k)
    best_rows = [r[:] for r in greedy_rows]
    best_val = _global_max(greedy_rows)
    
    # 2. Try backtracking with decreasing bounds
    bt_start = time.time()
    bt_budget = 20.0
    
    # Try a few backtracking attempts with tight bounds
    for max_entry in range(best_val - 1, max(100, best_val - 15), -1):
        if time.time() - bt_start > bt_budget:
            break
        time_remaining = bt_budget - (time.time() - bt_start)
        if time_remaining < 1.0:
            break
        time_per_attempt = min(2.0, time_remaining)
        
        rows, found = _backtrack_construct(n_rows, k, max_entry, time_limit=time_per_attempt)
        if found:
            val = _global_max(rows)
            if val < best_val:
                best_val = val
                best_rows = [r[:] for r in rows]
            break
    
    # 3. Simulated annealing - main optimization
    sa_budget = deadline - time.time() - 5.0  # Leave 5s for final checks
    if sa_budget > 10.0:
        # Run SA in multiple phases
        phase1_budget = min(sa_budget * 0.6, 60.0)
        phase2_budget = sa_budget - phase1_budget - 2.0
        
        if phase1_budget > 5.0:
            best_rows, best_val = _simulated_annealing(
                best_rows, n_rows, k, phase1_budget,
                T_init=200.0, T_min=1.0, cool=0.9999
            )
        
        if time.time() < deadline - 5.0:
            phase2_budget = min(deadline - time.time() - 5.0, phase2_budget)
            if phase2_budget > 5.0:
                best_rows, best_val = _simulated_annealing(
                    best_rows, n_rows, k, phase2_budget,
                    T_init=50.0, T_min=0.1, cool=0.99995
                )
    
    # 4. Final polish with very low temperature
    if time.time() < deadline - 2.0:
        polish_budget = min(deadline - time.time() - 2.0, 10.0)
        if polish_budget > 2.0:
            best_rows, best_val = _simulated_annealing(
                best_rows, n_rows, k, polish_budget,
                T_init=5.0, T_min=0.05, cool=0.99999
            )
    
    # Final validity check
    if not _is_valid(best_rows):
        if _is_valid(greedy_rows):
            return greedy_rows
        # Fallback
        fallback_row = [0, 1, 3, 7, 15, 31]
        return [fallback_row[:] for _ in range(n_rows)]
    
    return best_rows

# EVOLVE-BLOCK-END
