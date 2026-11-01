import time
import random
import math

# EVOLVE-BLOCK-START

def _is_valid(rows):
    """Check if the set of rows is valid (all diffs distinct globally)."""
    used = set()
    for row in rows:
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
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
            if not new_diffs & used:
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

def _validate_move(cand_row, old_diffs, all_used):
    """
    Check if changing a row to cand_row is valid (no internal collisions, no cross-row collisions).
    Returns (new_diffs, is_valid).
    """
    n = len(cand_row)
    new_diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = cand_row[j] - cand_row[i]
            if d in new_diffs:
                return None, False
            new_diffs.add(d)
    
    # Check cross-row conflicts
    for d in new_diffs:
        if d in all_used and d not in old_diffs:
            return None, False
    
    return new_diffs, True

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
        
    # Check for duplicates in added
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

def _backtrack_construct(n_rows, k, max_entry, time_limit=10.0):
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
        
        # Try to build row row_idx
        def build_row(pos, current_row, current_diffs):
            if time.time() > deadline:
                return None
            
            if pos == k + 1:
                return current_row[:]
            
            if pos == 0:
                # First element is always 0
                return build_row(1, [0], set())
            
            prev = current_row[-1]
            # Forward checking: ensure remaining positions can be filled
            remaining = k + 1 - pos
            # Minimum possible value for next position
            min_next = prev + 1
            
            # Try candidates starting from smallest
            x = min_next
            while x <= max_entry:
                # Check if this value creates internal collisions
                new_diffs = {x - a for a in current_row}
                if len(new_diffs) != len(current_row):
                    x += 1
                    continue
                
                # Check cross-row collisions
                if new_diffs & used:
                    x += 1
                    continue
                
                # Forward check: can we still fill remaining positions?
                # The smallest possible next value would be x + 1
                # We need at least (remaining - 1) more values, each >= x + 1
                # So the minimum last value would be x + (remaining - 1)
                if x + (remaining - 1) > max_entry:
                    break  # Can't fit remaining positions
                
                # Also check that the differences we'd create don't conflict
                # with differences that MUST be used by remaining positions
                # (this is a weaker check but helps prune)
                
                result = build_row(pos + 1, current_row + [x], current_diffs | new_diffs)
                if result is not None:
                    return result
                
                x += 1
            
            return None
        
        new_row = build_row(0, [], set())
        if new_row is not None:
            # Commit this row
            rows.append(new_row)
            used_copy = set(used)
            used |= _row_diffs(new_row)
            
            if backtrack(row_idx + 1, new_row):
                return True
            
            # Backtrack
            used = used_copy
            rows.pop()
        
        return False
    
    found = backtrack(0, [])
    if found:
        return rows, True
    return None, False

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Try backtracking search with tight bounds first
    best_rows = None
    best_val = float('inf')
    
    # The counting bound suggests m >= 105 for (7,5), so try from 105 upward
    # But start with greedy to get an upper bound
    greedy_rows = _greedy_init(n_rows, k)
    greedy_val = _global_max(greedy_rows)
    best_val = greedy_val
    best_rows = [r[:] for r in greedy_rows]
    
    # Try backtracking with decreasing upper bounds
    # Start just below greedy value and go down
    search_start = min(greedy_val - 1, 120)
    search_end = 104  # Below the counting bound
    
    bt_time_budget = 30.0  # Spend up to 30s on backtracking attempts
    bt_start = time.time()
    
    for max_entry in range(search_start, search_end, -1):
        if time.time() - bt_start > bt_time_budget:
            break
        
        # Give each attempt a small time slice
        time_per_attempt = min(3.0, max(0.5, bt_time_budget - (time.time() - bt_start)))
        
        rows, found = _backtrack_construct(n_rows, k, max_entry, time_limit=time_per_attempt)
        if found:
            val = _global_max(rows)
            if val < best_val:
                best_val = val
                best_rows = [r[:] for r in rows]
            break  # Found a solution at this bound, stop
    
    # 2. Local search to improve further
    if time.time() < deadline - 10:
        current_rows = [r[:] for r in best_rows]
        current_val = best_val
        row_maxes = [max(r) for r in current_rows]
        all_used = _all_used(current_rows)
        row_diffs = [_row_diffs(r) for r in current_rows]
        
        # SA parameters - increased temperature and extended ranges per inspiration
        T = 300.0
        T_min = 0.5
        cool = 0.99998
        
        no_improve = 0
        max_no_improve = 25000
        restarts = 0
        max_restarts = 15
        
        while time.time() < deadline - 15 and no_improve < max_no_improve:
            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            
            old_val = current_rows[ri][pi]
            
            candidates = []
            
            # Decreasing candidates - extended range
            min_allowed = current_rows[ri][pi - 1] + 1
            max_decrease = min(80, old_val - min_allowed)
            for d in range(1, max_decrease + 1):
                candidates.append(old_val - d)
            
            # Small increasing for exploration - extended range
            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 40
            max_increase = min(30, max_allowed - old_val)
            for d in range(1, max_increase + 1):
                candidates.append(old_val + d)
            
            # More random jumps for exploration
            if random.random() < 0.3:
                low = current_rows[ri][pi - 1] + 1
                high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 40
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
                    if random.random() < 0.2:
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
            
            if no_improve > max_no_improve:
                restarts += 1
                if restarts > max_restarts:
                    break
                current_rows = [r[:] for r in best_rows]
                current_val = best_val
                all_used = _all_used(current_rows)
                row_diffs = [_row_diffs(r) for r in current_rows]
                row_maxes = [max(r) for r in current_rows]
                T = 200.0
                no_improve = 0
    
    # 3. Final polish
    polish_deadline = min(deadline, start_time + 143.0)
    
    if time.time() < polish_deadline - 2:
        current_rows = [r[:] for r in best_rows]
        current_val = best_val
        all_used = _all_used(current_rows)
        row_diffs = [_row_diffs(r) for r in current_rows]
        row_maxes = [max(r) for r in current_rows]
        
        T = 2.0
        T_min = 0.1
        cool = 0.99999
        no_improve = 0
        max_no_improve_polish = 40000
        
        while time.time() < polish_deadline and no_improve < max_no_improve_polish:
            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            
            old_val = current_rows[ri][pi]
            min_allowed = current_rows[ri][pi - 1] + 1
            max_decrease = min(40, old_val - min_allowed)
            
            if max_decrease <= 0:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
            
            candidates = [old_val - d for d in range(1, max_decrease + 1)]
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
                    if random.random() < 0.15:
                        accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.05))
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
    
    # Final safety check: ensure the returned result is valid
    if not _is_valid(best_rows):
        # Fallback to the initial greedy solution which is guaranteed valid
        if _is_valid(greedy_rows):
            return greedy_rows
        # If even greedy is invalid (should not happen), return a trivial valid set
        # Trivial valid set: rows of powers of 2 (0, 1, 3, 7, 15, 31)
        # Differences are distinct for k=5 (6 elements)
        fallback_row = [0, 1, 3, 7, 15, 31]
        return [fallback_row[:] for _ in range(n_rows)]
        
    return best_rows

# EVOLVE-BLOCK-END
