import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k, max_attempts=50):
    """
    Randomized greedy initialization: try multiple random orderings and keep the best.
    """
    best_rows = None
    best_val = float('inf')
    
    for _ in range(max_attempts):
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

def _randomized_greedy_row(forbidden_diffs, k, max_limit=None, max_candidates=5, seed=None):
    """
    Generate a random valid row of length k+1 starting at 0.
    The row's internal differences must be distinct and not in forbidden_diffs.
    If max_limit is set, the last element must be < max_limit.
    Returns the row or None if no such row is found.
    """
    if seed is not None:
        # We don't actually reseed global random here to avoid overhead, 
        # but we can use a local random instance if needed. 
        # For simplicity, we rely on global random state.
        pass
        
    row = [0]
    used_in_row = set()
    last_val = 0
    
    for _ in range(k):
        candidates = []
        # Look ahead to find valid candidates
        # We need x such that:
        # 1. x > last_val
        # 2. x - a is not in forbidden_diffs for all a in row
        # 3. x - a is not in used_in_row for all a in row
        # 4. x - a are distinct for all a in row (implied by 3 if we check set size? 
        #    No, we need to check that the set of new diffs doesn't intersect used_in_row 
        #    AND the new diffs are distinct among themselves. 
        #    Since row is sorted, x-a are distinct if x is fixed. 
        #    Wait, if a1 < a2, then x-a1 > x-a2. So new diffs are always distinct.
        #    So we just need: new_diffs & forbidden_diffs == empty AND new_diffs & used_in_row == empty.
        
        x = last_val + 1
        # Safety bound to avoid infinite loop
        search_bound = 500 
        if max_limit:
             # If we have a limit, we don't need to search way past it, 
             # but we might need to jump over gaps. 
             # However, if max_limit is tight, we should stop early.
             pass
        
        while len(candidates) < max_candidates and x < search_bound:
            if max_limit is not None and x >= max_limit:
                break
                
            new_diffs = {x - a for a in row}
            
            if not (new_diffs & forbidden_diffs) and not (new_diffs & used_in_row):
                candidates.append(x)
            
            x += 1
        
        if not candidates:
            return None
        
        # Pick randomly from candidates
        # Bias towards smaller values to keep the row compact? 
        # Or uniform? Uniform is better for exploration.
        chosen_x = random.choice(candidates)
        
        row.append(chosen_x)
        new_diffs = {chosen_x - a for a in row[:-1]}
        used_in_row |= new_diffs
        last_val = chosen_x
        
    return row

def _local_search_polish(rows, n_rows, k, time_limit):
    """
    Fast local search to polish the solution.
    Tries to decrease the maximum entry by moving individual positions.
    """
    start_time = time.time()
    current_rows = [r[:] for r in rows]
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    all_used = set()
    for d in row_diffs:
        all_used |= d
    
    current_max = max(max(r) for r in current_rows)
    best_rows = [r[:] for r in current_rows]
    best_max = current_max
    
    # We will do a simple hill-climbing with random restarts of the "move" selection
    # Focus on the rows with the largest max entries.
    
    # Precompute static diffs for fast checking? 
    # For polish, simple checks might be fast enough if we limit the scope.
    
    while time.time() < time_limit:
        # Pick a row to modify, bias towards the one with the max entry
        max_row_idx = max(range(n_rows), key=lambda i: current_rows[i][-1])
        if random.random() < 0.7:
            ri = max_row_idx
        else:
            ri = random.randrange(n_rows)
            
        # Pick a position, bias towards the last one
        if random.random() < 0.6:
            pi = k
        else:
            pi = random.randrange(1, k + 1)
            
        old_val = current_rows[ri][pi]
        
        # Try to decrease the value
        min_allowed = current_rows[ri][pi - 1] + 1
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else old_val - 1
        
        # We only care about decreasing the global max, so we only look at values < old_val
        # If pi < k, decreasing old_val doesn't change the row max (which is at k).
        # So if pi < k, we only accept if it allows a *future* decrease of the last element?
        # That's too complex for simple polish. 
        # Let's only try to decrease the last element (pi=k) for immediate max reduction,
        # or try to decrease other elements if they are "blocking" the last element?
        # Actually, if we decrease an earlier element, the last element might be movable down later.
        # But for a fast polish, let's just try to reduce the last element directly.
        
        if pi != k:
            # If not the last element, we only accept if it reduces the max of the row? 
            # No, the max of the row is at k. 
            # So changing pi < k doesn't change current_max unless current_max is determined by something else? 
            # It doesn't. 
            # So for a fast "greedy polish", we should ONLY try to reduce the last element.
            # Unless we are doing a deeper search. 
            # Let's stick to reducing the last element for speed.
            continue
        
        # So we are reducing the last element (pi=k)
        # Range to check: from old_val - 1 down to min_allowed
        # We don't need to check all the way down, just a few steps?
        # Or check all? With max ~100, checking 100 values is fast.
        
        found_improvement = False
        for new_val in range(old_val - 1, min_allowed - 1, -1):
            # Check validity
            new_diffs = set()
            valid = True
            for i in range(len(current_rows[ri])):
                if i == pi:
                    continue
                d = current_rows[ri][pi] - current_rows[ri][i] if i < pi else current_rows[ri][i] - current_rows[ri][pi]
                # Wait, new_val is at pi.
                # d = abs(new_val - current_rows[ri][i])
                d = new_val - current_rows[ri][i] if i < pi else current_rows[ri][i] - new_val
                if d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
                
            # Check against other rows
            # We need to remove old diffs of row ri from all_used, then check new_diffs
            # But we can just check new_diffs against (all_used - row_diffs[ri])
            other_diffs = all_used - row_diffs[ri]
            
            if new_diffs & other_diffs:
                continue
                
            # Valid move!
            # Update state
            all_used -= row_diffs[ri]
            all_used |= new_diffs
            current_rows[ri][pi] = new_val
            row_diffs[ri] = new_diffs
            current_max = new_val # Since we reduced the max of the max row
            
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in current_rows]
            
            found_improvement = True
            break # Accept first improvement (greedy) and try next step
        
        if not found_improvement and random.random() < 0.1:
            # Occasionally restart from best to avoid getting stuck in a local min of the polish phase
            # But for polish, we just keep trying different rows/positions.
            pass
            
    return best_rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Initialization
    rows = _greedy_init(n_rows, k, max_attempts=30)
    best_rows = [r[:] for r in rows]
    best_val = max(max(r) for r in best_rows)
    
    # Track diffs for fast forbidden calculation
    row_diffs = [_row_diffs_set(r) for r in rows]
    
    # Simulated Annealing parameters for Row Replacement
    T = 10.0
    T_min = 0.01
    cool_rate = 0.999
    
    steps = 0
    no_improve = 0
    
    while time.time() < deadline - 10.0: # Leave 10s for final polish
        steps += 1
        
        # Pick a row to replace
        # Bias towards rows with larger max entries
        if random.random() < 0.5:
            ri = max(range(n_rows), key=lambda i: rows[i][-1])
        else:
            ri = random.randrange(n_rows)
            
        # Calculate forbidden diffs (union of all other rows' diffs)
        forbidden = set()
        for j in range(n_rows):
            if j != ri:
                forbidden |= row_diffs[j]
                
        # Try to generate a new row
        # We want to try to keep the max below the current global max
        # Or slightly above to allow exploration
        current_global_max = max(max(r) for r in rows)
        
        # Strategy: 
        # 1. Try to find a row with max < current_global_max
        # 2. If fail, try to find a row with max <= current_global_max + small_epsilon
        
        new_row = None
        # Attempt 1: Strictly smaller max
        # We don't know the exact limit, so we pass None and check later? 
        # Or we pass current_global_max as max_limit?
        # If we pass current_global_max, the last element must be < current_global_max.
        # This is a good constraint.
        
        # But what if the current row is already the one setting the global max?
        # Then we are trying to reduce it.
        
        # Let's try a few randomizations
        for attempt in range(5):
            # Add some randomness to the limit to allow exploration
            # If we are stuck, we might need to allow slightly larger rows temporarily?
            # No, the objective is to minimize the max. 
            # Allowing larger rows only helps if they free up differences for OTHER rows to be smaller.
            
            # For now, let's try to find a row with max < current_global_max
            # If current_global_max is the max of row ri, we are trying to improve it.
            # If current_global_max is the max of another row, we are trying to fit row ri 
            # into the space below that max.
            
            limit = current_global_max
            # If we want to be safe, we can try limit-1, limit, limit+5
            # Let's just use current_global_max as the upper bound for the last element.
            
            candidate_row = _randomized_greedy_row(forbidden, k, max_limit=limit, max_candidates=10)
            
            if candidate_row is not None:
                new_row = candidate_row
                break
        
        if new_row is None:
            # If we couldn't find a row strictly below the current global max,
            # maybe try a bit higher? 
            # This is risky as it increases the objective. 
            # Only do this if T is high enough.
            if T > 1.0:
                # Try with a slightly relaxed limit
                relaxed_limit = current_global_max + 5
                candidate_row = _randomized_greedy_row(forbidden, k, max_limit=relaxed_limit, max_candidates=10)
                if candidate_row is not None:
                    new_row = candidate_row
            else:
                # No improvement found, cool down
                T = max(T_min, T * cool_rate)
                no_improve += 1
                if no_improve > 1000:
                    # Restart temperature
                    T = 10.0
                    no_improve = 0
                continue
        
        # We have a new_row
        # Calculate new global max
        old_row = rows[ri]
        rows[ri] = new_row
        new_row_diffs = _row_diffs_set(new_row)
        row_diffs[ri] = new_row_diffs
        
        new_global_max = max(max(r) for r in rows)
        delta = new_global_max - current_global_max
        
        accept = False
        if delta < 0:
            accept = True
        elif delta == 0:
            if random.random() < 0.2:
                accept = True
        else:
            if T > T_min:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    accept = True
        
        if accept:
            current_global_max = new_global_max
            if current_global_max < best_val:
                best_val = current_global_max
                best_rows = [r[:] for r in rows]
                no_improve = 0
            else:
                no_improve += 1
        else:
            # Reject
            rows[ri] = old_row
            row_diffs[ri] = _row_diffs_set(old_row)
            no_improve += 1
            
        T = max(T_min, T * cool_rate)
        
        # Periodic Local Search Polish
        # Every 50 steps, or if we just improved, do a quick polish
        if steps % 50 == 0 or (delta < 0 and random.random() < 0.5):
            polish_deadline = time.time() + 1.0 # 1 second polish
            polished = _local_search_polish(rows, n_rows, k, polish_deadline)
            polish_val = max(max(r) for r in polished)
            if polish_val < best_val:
                best_val = polish_val
                best_rows = [r[:] for r in polished]
                rows = [r[:] for r in polished]
                row_diffs = [_row_diffs_set(r) for r in rows]
                current_global_max = polish_val
            elif polish_val < current_global_max:
                # Even if not best, it's better than current, so accept
                rows = [r[:] for r in polished]
                row_diffs = [_row_diffs_set(r) for r in rows]
                current_global_max = polish_val

    # 2. Final Polish with remaining time
    final_deadline = deadline - 2.0
    if time.time() < final_deadline:
        # Run a more thorough local search on the best solution
        # We can use the same polish function but with a longer budget
        final_polished = _local_search_polish(best_rows, n_rows, k, final_deadline)
        final_val = max(max(r) for r in final_polished)
        if final_val < best_val:
            best_val = final_val
            best_rows = final_polished

    return best_rows

# EVOLVE-BLOCK-END
