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

def _generate_valid_candidates(row, pi, all_used, static_pi, old_diffs, max_range=50):
    """
    Generate a list of valid candidates for position pi.
    Uses constraint-based filtering to skip obviously invalid values.
    """
    n = len(row)
    min_allowed = row[pi - 1] + 1
    max_allowed = (row[pi + 1] - 1) if pi < n - 1 else row[-1] + max_range
    
    # We want to find values in [min_allowed, max_allowed] that are valid.
    # To speed this up, we don't iterate every integer if the range is large,
    # but for local search, the range is usually small.
    # We prioritize decreasing moves to minimize max.
    
    candidates = []
    
    # Check a window around the current value and towards the lower bound
    current_val = row[pi]
    
    # 1. Try to decrease (primary goal)
    # We scan downwards from current_val - 1 to min_allowed
    # To avoid O(N) checks if N is large, we limit the scan or rely on the fact that
    # valid spots are sparse. However, in dense areas, we might need to check many.
    # Let's check a limited window first, then random samples if needed.
    
    # Window size for deterministic scan
    scan_depth = 30
    
    # Downwards
    for v in range(current_val - 1, max(min_allowed, current_val - scan_depth) - 1, -1):
        if _is_move_valid(row, pi, v, static_pi, old_diffs, all_used):
            candidates.append(v)
            
    # Upwards (less preferred, but allows escape)
    for v in range(current_val + 1, min(max_allowed, current_val + scan_depth) + 1):
        if _is_move_valid(row, pi, v, static_pi, old_diffs, all_used):
            candidates.append(v)
            
    # If no candidates found in the immediate window, try random samples in the wider range
    if not candidates:
        # Sample from the full valid range
        low = min_allowed
        high = max_allowed
        if low <= high:
            for _ in range(20):
                v = random.randint(low, high)
                if _is_move_valid(row, pi, v, static_pi, old_diffs, all_used):
                    candidates.append(v)
                    if len(candidates) > 5: break

    return candidates

def _local_search(rows, k, all_used, row_diffs, static_diffs, row_maxes, deadline, max_steps=10000):
    """
    Perform steepest-descent local search.
    Returns updated state and the best objective found.
    """
    n_rows = len(rows)
    current_rows = [r[:] for r in rows]
    current_all_used = set(all_used)
    current_row_diffs = [d.copy() for d in row_diffs]
    current_static_diffs = [s.copy() for s in static_diffs]
    current_row_maxes = row_maxes[:]
    
    current_global_max = max(current_row_maxes)
    current_sum_max = sum(current_row_maxes)
    
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a random row and position
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        static_pi = current_static_diffs[ri][pi]
        old_diffs = current_row_diffs[ri]
        
        # Generate valid candidates
        cands = _generate_valid_candidates(current_rows[ri], pi, current_all_used, static_pi, old_diffs)
        
        if not cands:
            no_improve += 1
            continue
            
        # Find the best candidate (minimize global max, then sum max)
        best_cand = None
        best_obj = None
        
        old_row_max = current_row_maxes[ri]
        
        for v in cands:
            if pi == k:
                cand_row_max = v
            else:
                cand_row_max = current_rows[ri][k]
                
            new_global_max = max(max(current_row_maxes[:ri] + current_row_maxes[ri+1:]), cand_row_max)
            new_sum_max = current_sum_max - old_row_max + cand_row_max
            
            # Objective: (global_max, sum_max)
            obj = (new_global_max, new_sum_max)
            
            if best_obj is None or obj < best_obj:
                best_obj = obj
                best_cand = v
                
        if best_cand is not None and best_obj[0] < current_global_max or (best_obj[0] == current_global_max and best_obj[1] < current_sum_max):
            # Accept the best move
            v = best_cand
            new_diffs = _get_new_diffs_set(current_rows[ri], pi, v, static_pi)
            current_all_used -= old_diffs
            current_all_used |= new_diffs
            current_rows[ri][pi] = v
            current_row_diffs[ri] = new_diffs
            current_static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
            
            if pi == k:
                current_row_maxes[ri] = v
            # else row max unchanged
            
            current_global_max = best_obj[0]
            current_sum_max = best_obj[1]
            no_improve = 0
        else:
            no_improve += 1
            
    return current_rows, current_all_used, current_row_diffs, current_static_diffs, current_row_maxes, current_global_max, current_sum_max

def _perturb(rows, k, all_used, row_diffs, static_diffs, row_maxes):
    """
    Apply a perturbation to escape local optimum.
    We pick a few random positions and make valid moves, even if they worsen the objective.
    """
    n_rows = len(rows)
    current_rows = [r[:] for r in rows]
    current_all_used = set(all_used)
    current_row_diffs = [d.copy() for d in row_diffs]
    current_static_diffs = [s.copy() for s in static_diffs]
    current_row_maxes = row_maxes[:]
    
    # Perform 3-5 random valid moves
    n_kicks = random.randint(3, 5)
    
    for _ in range(n_kicks):
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        static_pi = current_static_diffs[ri][pi]
        old_diffs = current_row_diffs[ri]
        
        cands = _generate_valid_candidates(current_rows[ri], pi, current_all_used, static_pi, old_diffs)
        
        if cands:
            # Pick a random candidate, not necessarily the best
            v = random.choice(cands)
            
            new_diffs = _get_new_diffs_set(current_rows[ri], pi, v, static_pi)
            current_all_used -= old_diffs
            current_all_used |= new_diffs
            current_rows[ri][pi] = v
            current_row_diffs[ri] = new_diffs
            current_static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
            
            if pi == k:
                current_row_maxes[ri] = v

    return current_rows, current_all_used, current_row_diffs, current_static_diffs, current_row_maxes

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Initialization
    rows = _greedy_init(n_rows, k)
    all_used = _compute_all_used(rows)
    
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    best_sum_max = sum(max(r) for r in best_rows)
    
    # State for current iteration
    current_rows = [r[:] for r in rows]
    current_all_used = set(all_used)
    current_row_diffs = [_row_diffs_set(r) for r in current_rows]
    current_static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    current_row_maxes = [max(r) for r in current_rows]
    current_val = best_val
    current_sum_max = best_sum_max
    
    # ILS Loop
    while time.time() < deadline:
        # 1. Local Search
        # Allocate most time to local search
        ls_deadline = min(deadline, time.time() + 10.0)
        
        current_rows, current_all_used, current_row_diffs, current_static_diffs, current_row_maxes, current_val, current_sum_max = _local_search(
            current_rows, k, current_all_used, current_row_diffs, current_static_diffs, current_row_maxes, ls_deadline
        )
        
        # Check for improvement
        if current_val < best_val or (current_val == best_val and current_sum_max < best_sum_max):
            best_val = current_val
            best_sum_max = current_sum_max
            best_rows = [r[:] for r in current_rows]
            
        # 2. Perturbation
        # Only perturb if we haven't run out of time
        if time.time() >= deadline - 5.0:
            break
            
        current_rows, current_all_used, current_row_diffs, current_static_diffs, current_row_maxes = _perturb(
            current_rows, k, current_all_used, current_row_diffs, current_static_diffs, current_row_maxes
        )
        
        # Update metrics for next local search
        current_val = max(current_row_maxes)
        current_sum_max = sum(current_row_maxes)

    return best_rows

# EVOLVE-BLOCK-END
