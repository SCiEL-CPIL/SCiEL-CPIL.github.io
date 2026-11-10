"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256

# EVOLVE-BLOCK-START
import random
import time

def _sidon_and_maximal_check(selected, N):
    """
    Check if selected is a valid Sidon set and inclusion-maximal in [1,N].
    Returns (is_valid_sidon, is_maximal, addable_count, used_sums_mask, selected_mask)
    """
    if not selected:
        return False, False, 0, 0, 0
        
    selected_set = set(selected)
    selected_mask = 0
    for x in selected:
        selected_mask |= (1 << x)
        
    used_sums_mask = 0
    valid = True
    for i, a in enumerate(selected):
        for j in range(i):
            b = selected[j]
            s = a + b
            bit = 1 << s
            if used_sums_mask & bit:
                valid = False
                break
            used_sums_mask |= bit
        if not valid:
            break
        s2 = 2 * a
        bit2 = 1 << s2
        if used_sums_mask & bit2:
            valid = False
            break
        used_sums_mask |= bit2
        
    if not valid:
        return False, False, 0, used_sums_mask, selected_mask
        
    addable_count = 0
    for x in range(1, N + 1):
        if selected_mask & (1 << x):
            continue
        new_sums_mask = (1 << (2 * x)) | (selected_mask << x)
        if (new_sums_mask & used_sums_mask) == 0:
            addable_count += 1
            
    is_maximal = (addable_count == 0)
    return True, is_maximal, addable_count, used_sums_mask, selected_mask

def _sa_optimize(start_set, N, rng, deadline):
    """
    Simulated Annealing to minimize size of maximal Sidon set.
    """
    current_set = list(start_set)
    current_size = len(current_set)
    
    # Get initial state
    is_sidon, is_max, add_count, used_sums, sel_mask = _sidon_and_maximal_check(current_set, N)
    if not is_sidon:
        # Fallback if start is invalid (shouldn't happen)
        return start_set
        
    # Cost function: size + penalty * (addable_count)
    # We want to minimize size, but penalize non-maximality heavily to guide search
    # If addable_count > 0, it's not a valid solution, but we allow it in search space
    # to jump over local minima.
    
    best_set = list(current_set) if is_max else []
    best_cost = current_size + (1000 if not is_max else 0)
    
    T = 1.0
    T_min = 0.01
    alpha = 0.999
    
    steps = 0
    while time.time() < deadline and T > T_min:
        # Generate neighbor
        r = rng.random()
        
        if r < 0.3 and current_set:
            # Remove random element
            idx = rng.randrange(len(current_set))
            x = current_set[idx]
            new_set = current_set[:idx] + current_set[idx+1:]
            
        elif r < 0.7:
            # Add random element (if valid Sidon-wise)
            # Find all addable elements
            addable = []
            for y in range(1, N + 1):
                if sel_mask & (1 << y):
                    continue
                new_sums = (1 << (2 * y)) | (sel_mask << y)
                if (new_sums & used_sums) == 0:
                    addable.append(y)
            
            if addable:
                y = rng.choice(addable)
                new_set = current_set + [y]
            else:
                # No addable elements, try removing
                if current_set:
                    idx = rng.randrange(len(current_set))
                    x = current_set[idx]
                    new_set = current_set[:idx] + current_set[idx+1:]
                else:
                    T *= alpha
                    continue
                    
        else:
            # Swap: remove one, add one
            if current_set:
                idx = rng.randrange(len(current_set))
                x = current_set[idx]
                # Find addable elements for the set without x
                temp_set = current_set[:idx] + current_set[idx+1:]
                temp_mask = sel_mask & ~(1 << x)
                temp_used = 0
                # Recompute temp_used is expensive, let's just try random adds and check validity
                # Or just pick a random y and check if swap is valid Sidon
                
                y = rng.randint(1, N)
                if y == x or (sel_mask & (1 << y)):
                    # Pick another y
                    y = rng.randint(1, N)
                    if y == x or (sel_mask & (1 << y)):
                        T *= alpha
                        continue
                        
                new_set = temp_set + [y]
            else:
                T *= alpha
                continue

        # Check validity of new_set
        is_sidon, is_max, add_count, new_used_sums, new_sel_mask = _sidon_and_maximal_check(new_set, N)
        
        if not is_sidon:
            T *= alpha
            continue
            
        new_size = len(new_set)
        new_cost = new_size + (1000 if not is_max else 0)
        
        # Acceptance criterion
        delta_cost = new_cost - (current_size + (1000 if not is_max else 0))
        
        if delta_cost < 0 or rng.random() < pow(2.71828, -delta_cost / T):
            current_set = new_set
            current_size = new_size
            is_max = is_max
            used_sums = new_used_sums
            sel_mask = new_sel_mask
            
            if is_max and new_cost < best_cost:
                best_cost = new_cost
                best_set = list(current_set)
                
        T *= alpha
        steps += 1
        
    # Final check: if best_set is empty or not maximal, try to repair or return current if it's better
    if best_set and _sidon_and_maximal_check(best_set, N)[1]:
        return best_set
        
    # If SA didn't find a maximal set, return the best maximal set found during process
    # or fallback to start if start was maximal
    if _sidon_and_maximal_check(current_set, N)[1]:
        return current_set
        
    return start_set

def _greedy_max_blocking_bitmask(N, rng=None, tie_break='max', weight_random=0.0):
    """
    Max-blocking greedy using bitmasks for O(1) validity and blocking checks.
    """
    selected = []
    used_sums_mask = 0
    selected_mask = 0
    candidates = list(range(1, N + 1))
    
    while candidates:
        # Find all valid candidates
        valid = []
        for x in candidates:
            # new_sums_mask for x
            new_sums_mask = (1 << (2 * x)) | (selected_mask << x)
            if (new_sums_mask & used_sums_mask) == 0:
                valid.append(x)
        
        if not valid:
            break
        
        # For each valid candidate, count how many other valid candidates it blocks
        best_x = None
        best_blocks = -1
        best_ties = []
        
        for x in valid:
            # Mask of sums added by x
            added_sums_mask = (1 << (2 * x)) | (selected_mask << x)
            
            blocks = 0
            for y in valid:
                if y == x:
                    continue
                
                # Check if y is blocked by adding x.
                # y is currently valid, so its sums don't collide with used_sums_mask.
                # Adding x changes used_sums to used_sums_mask | added_sums_mask.
                # y's potential sums are: {2*y} U {y+s | s in selected} U {y+x}
                # Mask for y's potential sums:
                # (1 << (2*y)) | (selected_mask << y) | (1 << (y+x))
                
                y_potential_mask = (1 << (2 * y)) | (selected_mask << y) | (1 << (y + x))
                
                # y is blocked if any of its potential sums collide with added_sums_mask
                # Note: We don't need to check against used_sums_mask because y is valid, 
                # so y_potential_mask & used_sums_mask == 0.
                # We only need to check if y_potential_mask & added_sums_mask != 0.
                
                if (y_potential_mask & added_sums_mask) != 0:
                    blocks += 1
            
            # Apply random weighting if desired
            score = blocks
            if weight_random > 0:
                score += rng.random() * weight_random
            
            if score > best_blocks:
                best_blocks = score
                best_x = x
                best_ties = [x]
            elif score == best_blocks:
                best_ties.append(x)
        
        # Select the best candidate
        if tie_break == 'random' and len(best_ties) > 1:
            chosen = rng.choice(best_ties)
        else:
            chosen = best_x
        
        # Add chosen to selected
        selected.append(chosen)
        # Update masks
        new_sums_mask = (1 << (2 * chosen)) | (selected_mask << chosen)
        used_sums_mask |= new_sums_mask
        selected_mask |= (1 << chosen)
        candidates.remove(chosen)
        
    return selected

def construct():
    deadline = time.time() + 55.0  # 55s safety margin before 60s kill
    best_size = N + 1
    best_set = None
    rng = random.Random(42)
    
    # Strategy 1: Max-blocking greedy to get a good starting point
    candidate = _greedy_max_blocking_bitmask(N, rng, tie_break='max')
    if _sidon_and_maximal_check(candidate, N)[1]:
        best_size = len(candidate)
        best_set = candidate
        
    # Strategy 2: Simulated Annealing to improve
    # Run multiple SA restarts
    for i in range(50):
        if time.time() > deadline - 5:
            break
            
        # Use the best set found so far as start, or greedy if none
        start_set = best_set if best_set else candidate
        
        improved = _sa_optimize(start_set, N, rng, deadline - 5)
        
        if improved:
            is_sidon, is_max, _, _, _ = _sidon_and_maximal_check(improved, N)
            if is_sidon and is_max:
                if len(improved) < best_size:
                    best_size = len(improved)
                    best_set = improved
                    if best_size <= 16:
                        break

    if best_set is None:
        # Fallback
        best_set = _greedy_max_blocking_bitmask(N, rng, tie_break='max')
        
    return best_set
# EVOLVE-BLOCK-END
