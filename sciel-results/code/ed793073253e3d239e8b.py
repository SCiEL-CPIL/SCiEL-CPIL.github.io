<<<<<<< SEARCH
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

def _is_move_valid(row, pi, new_val, static_pi, old_diffs, all_used):
    """
    Fast validity check without creating the new_diffs set.
    Returns True if the move is valid.
    """
    n = len(row)
    new_diffs_list = []
    
    # Compute new diffs involving pi
    for i in range(n):
        if i == pi:
            continue
        if i < pi:
            d = new_val - row[i]
        else:
            d = row[i] - new_val
        new_diffs_list.append(d)
        
    # Check internal validity
    seen_new = set()
    for d in new_diffs_list:
        # Must not be in static diffs (which are unchanged)
        if d in static_pi:
            return False
        # Must not be duplicate among new diffs
        if d in seen_new:
            return False
        seen_new.add(d)
        
    # Check global conflicts
    # A conflict exists if a new diff is in all_used but was NOT in old_diffs
    for d in new_diffs_list:
        if d in all_used and d not in old_diffs:
            return False
            
    return True

def _get_new_diffs_set(row, pi, new_val, static_pi):
    """
    Construct the new_diffs set. Only called if move is accepted.
    """
    n = len(row)
    new_diffs = set(static_pi)
    for i in range(n):
        if i == pi:
            continue
        if i < pi:
            d = new_val - row[i]
        else:
            d = row[i] - new_val
        new_diffs.add(d)
    return new_diffs

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 108.0  # Safety margin for 120s limit
    
    # 1. Initialization
    rows = _greedy_init(n_rows, k)
    all_used = _compute_all_used(rows)
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    best_sum_max = sum(max(r) for r in best_rows)
    
    # State
    current_rows = [r[:] for r in rows]
    current_val = best_val
    current_sum_max = best_sum_max
    
    row_maxes = [max(r) for r in current_rows]
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    
    # SA Parameters
    T = 200.0
    T_min = 0.5
    cool = 0.99995
    no_improve = 0
    max_no_improve = 30000
    
    # Phase 1: Main SA loop
    sa_deadline = deadline - 18.0  # Reserve 18s for polish phases
    
    while time.time() < sa_deadline and no_improve < max_no_improve:
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        # Generate candidates
        candidates = []
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(50, old_val - min_allowed)
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)
            
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 30
        max_increase = min(15, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            candidates.append(old_val + d)
            
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
        
        static_pi = static_diffs[ri][pi]
        old_diffs = row_diffs[ri]
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            # Fast validity check (no set construction)
            if not _is_move_valid(current_rows[ri], pi, new_val, static_pi, old_diffs, all_used):
                continue
                
            # Compute new metrics
            if pi == k:
                cand_row_max = new_val
            else:
                cand_row_max = current_rows[ri][k]
                
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            new_sum_max = current_sum_max - old_row_max + cand_row_max
            row_maxes[ri] = old_row_max
            
            # Acceptance with secondary objective
            accept = False
            if new_global_max < current_val:
                accept = True
            elif new_global_max == current_val:
                if new_sum_max < current_sum_max:
                    accept = True
                elif new_sum_max == current_sum_max:
                    accept = random.random() < 0.1
                else:
                    accept = random.random() < 0.01
            else:
                delta = new_global_max - current_val
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.1))
                    accept = random.random() < prob
            
            if accept:
                # Build the diff set only on acceptance
                new_diffs = _get_new_diffs_set(current_rows[ri], pi, new_val, static_pi)
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri][pi] = new_val
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = new_global_max
                current_sum_max = new_sum_max
                
                static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                
                if current_val < best_val or (current_val == best_val and current_sum_max < best_sum_max):
                    best_val = current_val
                    best_sum_max = current_sum_max
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
            current_sum_max = best_sum_max
            all_used = _compute_all_used(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            static_diffs = [_build_static_diffs(r, k) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 200.0
            no_improve = 0
    
    # Phase 2: Targeted shrink phase - focus on rows with maximum entry
    polish_deadline = min(deadline, start_time + 113.0)
    
    # Reset state to best
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    current_sum_max = best_sum_max
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    # Identify rows that achieve the global max
    max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
    
    no_improve = 0
    max_no_improve_polish = 20000
    
    while time.time() < polish_deadline and no_improve < max_no_improve_polish:
        # Bias toward rows with the maximum entry
        if max_row_indices and random.random() < 0.7:
            ri = random.choice(max_row_indices)
        else:
            ri = random.randrange(n_rows)
        
        # For the max row, prioritize the last position (which is the max)
        if ri in max_row_indices and random.random() < 0.5:
            pi = k
        else:
            pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        # Only try decreasing moves in polish phase
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(50, old_val - min_allowed)
        
        if max_decrease <= 0:
            no_improve += 1
            continue
        
        # Try from largest decrease to smallest
        candidates = [old_val - d for d in range(max_decrease, 0, -1)]
        
        static_pi = static_diffs[ri][pi]
        old_diffs = row_diffs[ri]
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            if not _is_move_valid(current_rows[ri], pi, new_val, static_pi, old_diffs, all_used):
                continue
                
            if pi == k:
                cand_row_max = new_val
            else:
                cand_row_max = current_rows[ri][k]
            
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            new_sum_max = current_sum_max - old_row_max + cand_row_max
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            # In polish phase, only accept improving or neutral moves
            if delta < 0:
                accept = True
            elif delta == 0:
                if new_sum_max < current_sum_max:
                    accept = True
                else:
                    accept = random.random() < 0.1
            else:
                accept = False
            
            if accept:
                new_diffs = _get_new_diffs_set(current_rows[ri], pi, new_val, static_pi)
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri][pi] = new_val
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = new_global_max
                current_sum_max = new_sum_max
                
                static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                
                if current_val < best_val or (current_val == best_val and current_sum_max < best_sum_max):
                    best_val = current_val
                    best_sum_max = current_sum_max
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                    max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
                else:
                    no_improve += 1
                    
                moved = True
                break
        
        if not moved:
            no_improve += 1
    
    # Phase 3: Final quick SA pass with very low temperature
    final_deadline = deadline
    T = 0.2
    T_min = 0.01
    cool = 0.99999
    
    # Reset state to best
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    current_sum_max = best_sum_max
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    while time.time() < final_deadline:
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(20, old_val - min_allowed)
        
        if max_decrease <= 0:
            continue
        
        # Try a random subset of decreasing moves
        n_try = min(max_decrease, random.randint(3, 8))
        candidates = random.sample(range(1, max_decrease + 1), n_try)
        candidates = [old_val - d for d in candidates]
        
        static_pi = static_diffs[ri][pi]
        old_diffs = row_diffs[ri]
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            if not _is_move_valid(current_rows[ri], pi, new_val, static_pi, old_diffs, all_used):
                continue
                
            if pi == k:
                cand_row_max = new_val
            else:
                cand_row_max = current_rows[ri][k]
            
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            new_sum_max = current_sum_max - old_row_max + cand_row_max
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.05
            else:
                accept = False
            
            if accept:
                new_diffs = _get_new_diffs_set(current_rows[ri], pi, new_val, static_pi)
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri][pi] = new_val
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = new_global_max
                current_sum_max = new_sum_max
                
                static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                
                if current_val < best_val or (current_val == best_val and current_sum_max < best_sum_max):
                    best_val = current_val
                    best_sum_max = current_sum_max
                    best_rows = [r[:] for r in current_rows]
                    
                moved = True
                break
    
    return best_rows

# EVOLVE-BLOCK-END
=======
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

def _resolve_row(row_idx, current_rows, all_used, k, max_val_hint=None, randomize=False):
    """
    Re-solve a single row given the global difference constraint from other rows.
    Uses backtracking with forward checking.
    Returns a valid row or None if no solution found within limits.
    """
    n = k + 1
    # Compute forbidden diffs (from all other rows)
    forbidden = set()
    for i in range(n_rows_global):
        if i == row_idx:
            continue
        forbidden |= _row_diffs_set(current_rows[i])
    
    # Also track which diffs are used by this row (to allow reuse within row)
    row_diffs = set()
    
    def backtrack(pos, current_row, used_in_row):
        if pos == n:
            return current_row[:]
        
        start = current_row[-1] + 1 if pos > 0 else 1
        end = max_val_hint if max_val_hint else 200
        
        # If randomize, shuffle the range slightly
        candidates = list(range(start, end + 1))
        if randomize:
            random.shuffle(candidates)
        
        for val in candidates:
            # Check internal validity
            new_diffs = set()
            valid = True
            for prev in current_row:
                d = val - prev
                if d in used_in_row:
                    valid = False
                    break
                new_diffs.add(d)
            if not valid:
                continue
            
            # Check global validity
            if new_diffs & forbidden:
                continue
            
            current_row.append(val)
            used_in_row |= new_diffs
            
            result = backtrack(pos + 1, current_row, used_in_row)
            if result is not None:
                return result
            
            current_row.pop()
            used_in_row -= new_diffs
        
        return None
    
    return backtrack(0, [0], set())

def _swap_rows(rows, i, j):
    """Swap two rows in the list."""
    rows[i], rows[j] = rows[j], rows[i]

def construct(n_rows, k):
    global n_rows_global
    n_rows_global = n_rows
    
    start_time = time.time()
    deadline = start_time + 108.0  # Safety margin for 120s limit
    
    # 1. Initialization
    rows = _greedy_init(n_rows, k)
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    best_sum_max = sum(max(r) for r in best_rows)
    
    # State
    current_rows = [r[:] for r in rows]
    current_val = best_val
    current_sum_max = best_sum_max
    
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    # Phase 1: Row-level reconstruction search
    # Mix of: row re-solve, row swap, and single-position moves
    T = 150.0
    T_min = 1.0
    cool = 0.99997
    no_improve = 0
    max_no_improve = 25000
    
    sa_deadline = deadline - 20.0  # Reserve 20s for polish
    
    while time.time() < sa_deadline and no_improve < max_no_improve:
        # Choose move type
        move_type = random.random()
        ri = random.randrange(n_rows)
        
        if move_type < 0.35:
            # Row re-solve: try to reconstruct row ri with a tighter bound
            old_row_max = row_maxes[ri]
            # Try with decreasing max hints
            max_hint = old_row_max
            new_row = None
            
            # Try a few different max hints
            for attempt in range(3):
                max_hint = old_row_max - attempt * 3
                if max_hint < old_row_max - 15:
                    break
                new_row = _resolve_row(ri, current_rows, all_used, k, max_val_hint=max_hint, randomize=True)
                if new_row is not None:
                    break
            
            if new_row is not None:
                new_row_max = max(new_row)
                new_global_max = max(row_maxes[:ri] + [new_row_max] + row_maxes[ri+1:])
                new_sum_max = current_sum_max - old_row_max + new_row_max
                
                delta = new_global_max - current_val
                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    if new_sum_max < current_sum_max:
                        accept = True
                    else:
                        accept = random.random() < 0.1
                else:
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.5))
                        accept = random.random() < prob
                
                if accept:
                    # Update state
                    all_used -= row_diffs[ri]
                    current_rows[ri] = new_row
                    row_diffs[ri] = _row_diffs_set(new_row)
                    all_used |= row_diffs[ri]
                    row_maxes[ri] = new_row_max
                    current_val = new_global_max
                    current_sum_max = new_sum_max
                    
                    if current_val < best_val or (current_val == best_val and current_sum_max < best_sum_max):
                        best_val = current_val
                        best_sum_max = current_sum_max
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
        elif move_type < 0.55:
            # Row swap
            rj = random.randrange(n_rows)
            if rj == ri:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
                
            # Swap rows ri and rj
            current_rows[ri], current_rows[rj] = current_rows[rj], current_rows[ri]
            row_maxes[ri], row_maxes[rj] = row_maxes[rj], row_maxes[ri]
            
            # Check if this improves (swap alone doesn't change global max or sum)
            # But it may help future row re-solves by changing which row is "hard"
            # Accept with small probability to explore
            if random.random() < 0.3:
                # Keep the swap
                pass
            else:
                # Undo swap
                current_rows[ri], current_rows[rj] = current_rows[rj], current_rows[ri]
                row_maxes[ri], row_maxes[rj] = row_maxes[rj], row_maxes[ri]
            
            no_improve += 1
            
        else:
            # Single-position move (original local search)
            pi = random.randrange(1, k + 1)
            old_val = current_rows[ri][pi]
            
            # Generate candidates
            candidates = []
            min_allowed = current_rows[ri][pi - 1] + 1
            max_decrease = min(40, old_val - min_allowed)
            for d in range(1, max_decrease + 1):
                candidates.append(old_val - d)
                
            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 25
            max_increase = min(12, max_allowed - old_val)
            for d in range(1, max_increase + 1):
                candidates.append(old_val + d)
                
            # Random jump
            if random.random() < 0.12:
                low = current_rows[ri][pi - 1] + 1
                high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 15
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
                    
                # Fast validity check
                new_diffs_list = []
                for i in range(k + 1):
                    if i == pi:
                        continue
                    if i < pi:
                        d = new_val - current_rows[ri][i]
                    else:
                        d = current_rows[ri][i] - new_val
                    new_diffs_list.append(d)
                    
                # Check internal validity
                seen_new = set()
                valid = True
                for d in new_diffs_list:
                    if d in row_diffs[ri] and d not in [current_rows[ri][pi] - current_rows[ri][i] if i < pi else current_rows[ri][i] - current_rows[ri][pi] for i in range(k+1) if i != pi]:
                        # This diff was already in the row and is not being removed
                        pass
                    if d in seen_new:
                        valid = False
                        break
                    seen_new.add(d)
                
                if not valid:
                    continue
                
                # Check global conflicts
                old_diffs_involving_pi = set()
                for i in range(k + 1):
                    if i == pi:
                        continue
                    if i < pi:
                        old_diffs_involving_pi.add(old_val - current_rows[ri][i])
                    else:
                        old_diffs_involving_pi.add(current_rows[ri][i] - old_val)
                
                valid = True
                for d in new_diffs_list:
                    if d in all_used and d not in old_diffs_involving_pi:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Compute new metrics
                if pi == k:
                    cand_row_max = new_val
                else:
                    cand_row_max = current_rows[ri][k]
                    
                old_row_max = row_maxes[ri]
                row_maxes[ri] = cand_row_max
                new_global_max = max(row_maxes)
                new_sum_max = current_sum_max - old_row_max + cand_row_max
                row_maxes[ri] = old_row_max
                
                # Acceptance
                accept = False
                if new_global_max < current_val:
                    accept = True
                elif new_global_max == current_val:
                    if new_sum_max < current_sum_max:
                        accept = True
                    elif new_sum_max == current_sum_max:
                        accept = random.random() < 0.1
                    else:
                        accept = random.random() < 0.01
                else:
                    delta = new_global_max - current_val
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.5))
                        accept = random.random() < prob
                
                if accept:
                    # Build new diff set
                    new_diffs = set()
                    for i in range(k + 1):
                        for j in range(i + 1, k + 1):
                            new_diffs.add(current_rows[ri][j] - current_rows[ri][i])
                    # Fix: compute with new_val at position pi
                    temp_row = current_rows[ri][:]
                    temp_row[pi] = new_val
                    new_diffs = _row_diffs_set(temp_row)
                    
                    all_used -= row_diffs[ri]
                    all_used |= new_diffs
                    current_rows[ri][pi] = new_val
                    row_diffs[ri] = new_diffs
                    row_maxes[ri] = cand_row_max
                    current_val = new_global_max
                    current_sum_max = new_sum_max
                    
                    if current_val < best_val or (current_val == best_val and current_sum_max < best_sum_max):
                        best_val = current_val
                        best_sum_max = current_sum_max
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
        if no_improve > 10000:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            current_sum_max = best_sum_max
            all_used = _compute_all_used(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 150.0
            no_improve = 0
    
    # Phase 2: Targeted row re-solve with tight bounds
    polish_deadline = deadline
    
    # Reset state to best
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    current_sum_max = best_sum_max
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    # Identify rows that achieve the global max
    max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
    
    no_improve = 0
    max_no_improve_polish = 15000
    
    while time.time() < polish_deadline and no_improve < max_no_improve_polish:
        # Bias toward rows with the maximum entry
        if max_row_indices and random.random() < 0.8:
            ri = random.choice(max_row_indices)
        else:
            ri = random.randrange(n_rows)
        
        old_row_max = row_maxes[ri]
        
        # Try to re-solve with progressively tighter bounds
        new_row = None
        for offset in range(0, 20):
            max_hint = old_row_max - offset
            if max_hint < old_row_max - 20:
                break
            new_row = _resolve_row(ri, current_rows, all_used, k, max_val_hint=max_hint, randomize=True)
            if new_row is not None:
                break
        
        if new_row is not None:
            new_row_max = max(new_row)
            new_global_max = max(row_maxes[:ri] + [new_row_max] + row_maxes[ri+1:])
            new_sum_max = current_sum_max - old_row_max + new_row_max
            
            delta = new_global_max - current_val
            
            if delta < 0:
                accept = True
            elif delta == 0:
                if new_sum_max < current_sum_max:
                    accept = True
                else:
                    accept = random.random() < 0.1
            else:
                accept = False
            
            if accept:
                all_used -= row_diffs[ri]
                current_rows[ri] = new_row
                row_diffs[ri] = _row_diffs_set(new_row)
                all_used |= row_diffs[ri]
                row_maxes[ri] = new_row_max
                current_val = new_global_max
                current_sum_max = new_sum_max
                
                if current_val < best_val or (current_val == best_val and current_sum_max < best_sum_max):
                    best_val = current_val
                    best_sum_max = current_sum_max
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                    max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
                else:
                    no_improve += 1
            else:
                no_improve += 1
        else:
            no_improve += 1
    
    return best_rows

# EVOLVE-BLOCK-END
>>>>>>> REPLACE
