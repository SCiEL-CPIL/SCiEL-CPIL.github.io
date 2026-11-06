"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _build_conflict_masks():
    """
    For each element x, precompute a list of bitmasks representing pairs (y, z)
    such that {x, y, z} is a bad triple.
    We use a bitmask where bit i represents element i.
    """
    # conflicts[x] will be a list of integers, where each integer is a bitmask of the pair (y, z)
    conflicts = {x: [] for x in range(1, 129)}
    
    nums = list(range(1, 129))
    for a, b, c in combinations(nums, 3):
        lab = _lcm(a, b)
        lac = _lcm(a, c)
        lbc = _lcm(b, c)
        if lab == lac == lbc:
            # Create bitmask for pair (b, c)
            mask_bc = (1 << b) | (1 << c)
            # Create bitmask for pair (a, c)
            mask_ac = (1 << a) | (1 << c)
            # Create bitmask for pair (a, b)
            mask_ab = (1 << a) | (1 << b)
            
            conflicts[a].append(mask_bc)
            conflicts[b].append(mask_ac)
            conflicts[c].append(mask_ab)
            
    return conflicts

def _can_add(x, selected_mask, conflicts):
    """
    Check if x can be added to the set represented by selected_mask.
    selected_mask is an integer bitmask.
    """
    for mask in conflicts[x]:
        # If the pair represented by mask is fully contained in selected_mask, we can't add x
        if (selected_mask & mask) == mask:
            return False
    return True

def _greedy_initial(conflicts, order):
    """
    Greedy: add elements in given order if they don't create a bad triple.
    Returns the selected set as a bitmask.
    """
    selected_mask = 0
    for x in order:
        if _can_add(x, selected_mask, conflicts):
            selected_mask |= (1 << x)
    return selected_mask

def _popcount(n):
    """Count number of set bits in n."""
    return bin(n).count("1")

def _local_improve(selected_mask, conflicts, time_limit=10):
    """
    Try to improve by removing one element and adding one or more others.
    Also tries 2-swap.
    """
    best_mask = selected_mask
    best_size = _popcount(selected_mask)
    start = time.time()
    
    # Precompute list of elements for faster iteration
    all_elements = list(range(1, 129))
    
    while time.time() - start < time_limit:
        improved = False
        
        # 1. Try 1-swap: remove one, add one
        # Iterate over elements in the current set
        current_elements = [i for i in all_elements if (selected_mask >> i) & 1]
        
        for rem in current_elements:
            candidate_mask = selected_mask & ~(1 << rem)
            
            # Try to add any element not in candidate_mask
            for x in all_elements:
                if (candidate_mask >> x) & 1:
                    continue
                if _can_add(x, candidate_mask, conflicts):
                    new_mask = candidate_mask | (1 << x)
                    new_size = _popcount(new_mask)
                    if new_size > best_size:
                        best_mask = new_mask
                        best_size = new_size
                        selected_mask = new_mask
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            # 2. Try 2-swap: remove two, add two
            # This is expensive, so we limit attempts or use a heuristic
            # We only do this if we haven't improved in a while or periodically
            if random.random() < 0.1: # 10% chance to try 2-swap
                improved2 = False
                # Pick two random elements to remove
                if len(current_elements) >= 2:
                    rem1, rem2 = random.sample(current_elements, 2)
                    candidate_mask = selected_mask & ~(1 << rem1) & ~(1 << rem2)
                    
                    # Try to add two elements
                    added = 0
                    new_mask = candidate_mask
                    for x in all_elements:
                        if (new_mask >> x) & 1:
                            continue
                        if _can_add(x, new_mask, conflicts):
                            new_mask |= (1 << x)
                            added += 1
                            if added == 2:
                                break
                    
                    if added == 2:
                        new_size = _popcount(new_mask)
                        if new_size > best_size:
                            best_mask = new_mask
                            best_size = new_size
                            selected_mask = new_mask
                            improved2 = True
        
        if not improved and not (random.random() < 0.1 and improved2):
             # If no 1-swap and no successful 2-swap (or 2-swap not tried/successful), break?
             # Actually, if 1-swap fails, we might still be stuck. 
             # If we just broke out of 1-swap loop with no improvement, we should probably break the main loop 
             # unless we want to try 2-swap more aggressively.
             # Let's just break if no 1-swap improvement to avoid wasting time, 
             # as 2-swap is rare and expensive.
             break
    
    return best_mask

def _simulated_annealing(selected_mask, conflicts, time_limit=10):
    """
    Simulated annealing to escape local optima.
    """
    current_mask = selected_mask
    best_mask = selected_mask
    best_size = _popcount(selected_mask)
    
    start = time.time()
    temp = 5.0
    cooling = 0.999
    
    all_elements = list(range(1, 129))
    
    while time.time() - start < time_limit:
        temp *= cooling
        if temp < 0.01:
            break
        
        # Generate a neighbor
        # Randomly remove 1 or 2 elements
        current_elements = [i for i in all_elements if (current_mask >> i) & 1]
        if not current_elements:
            break
            
        num_removals = random.choice([1, 1, 2]) # Bias towards 1
        if len(current_elements) < num_removals:
            num_removals = len(current_elements)
            
        to_remove = random.sample(current_elements, num_removals)
        new_mask = current_mask
        for r in to_remove:
            new_mask &= ~(1 << r)
            
        # Try to add back some elements greedily or randomly
        # We try to add up to num_removals + 1 elements to potentially improve
        attempts = 0
        max_attempts = 10
        while attempts < max_attempts:
            # Find candidates that can be added
            candidates = [x for x in all_elements if not ((new_mask >> x) & 1) and _can_add(x, new_mask, conflicts)]
            if not candidates:
                break
            x = random.choice(candidates)
            new_mask |= (1 << x)
            attempts += 1
            
        new_size = _popcount(new_mask)
        current_size = _popcount(current_mask)
        
        if new_size > current_size:
            current_mask = new_mask
            if new_size > best_size:
                best_mask = new_mask
                best_size = new_size
        else:
            # Accept with probability exp(-delta / temp)
            delta = current_size - new_size
            if delta == 0:
                # Same size, accept with some probability to wander
                if random.random() < 0.5:
                    current_mask = new_mask
            else:
                # Calculate probability safely
                # Avoid overflow/underflow
                prob = 2.718281828 ** (-delta / max(temp, 1e-6))
                if random.random() < prob:
                    current_mask = new_mask
    
    return best_mask

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s
    
    # Precompute conflicts
    conflicts = _build_conflict_masks()
    
    best_mask = 0
    best_size = 0
    
    # 1. Initial greedy with natural order
    sel_mask = _greedy_initial(conflicts, list(range(1, 129)))
    best_mask = sel_mask
    best_size = _popcount(sel_mask)
    
    # 2. Local improvement
    if time.time() < deadline - 15:
        improved_mask = _local_improve(best_mask, conflicts, time_limit=10)
        if _popcount(improved_mask) > best_size:
            best_mask = improved_mask
            best_size = _popcount(best_mask)
    
    # 3. Multiple restarts with different orderings
    random.seed(42)
    restarts = 0
    while time.time() < deadline - 20:
        restarts += 1
        order = list(range(1, 129))
        random.shuffle(order)
        
        sel_mask = _greedy_initial(conflicts, order)
        sel_size = _popcount(sel_mask)
        
        if sel_size > best_size:
            best_mask = sel_mask
            best_size = sel_size
        
        # Local improve if promising
        if sel_size >= best_size - 2 and time.time() < deadline - 25:
            improved_mask = _local_improve(sel_mask, conflicts, time_limit=5)
            imp_size = _popcount(improved_mask)
            if imp_size > best_size:
                best_mask = improved_mask
                best_size = imp_size
        
        # Simulated Annealing if promising and time permits
        if sel_size >= best_size - 1 and time.time() < deadline - 10:
            sa_mask = _simulated_annealing(sel_mask, conflicts, time_limit=5)
            sa_size = _popcount(sa_mask)
            if sa_size > best_size:
                best_mask = sa_mask
                best_size = sa_size
        
        # Also try SA on the current best if we have time
        if restarts % 5 == 0 and time.time() < deadline - 10:
             sa_best_mask = _simulated_annealing(best_mask, conflicts, time_limit=5)
             sa_best_size = _popcount(sa_best_mask)
             if sa_best_size > best_size:
                 best_mask = sa_best_mask
                 best_size = sa_best_size

    # 4. Final polish on the best solution found
    if time.time() < deadline - 5 and best_size > 0:
        improved_mask = _local_improve(best_mask, conflicts, time_limit=10)
        if _popcount(improved_mask) > best_size:
            best_mask = improved_mask
            best_size = _popcount(best_mask)
            
        # One more SA pass
        if time.time() < deadline - 2:
            sa_mask = _simulated_annealing(best_mask, conflicts, time_limit=5)
            if _popcount(sa_mask) > best_size:
                best_mask = sa_mask
                best_size = _popcount(sa_mask)

    # Convert mask to list
    best_list = [i for i in range(1, 129) if (best_mask >> i) & 1]
    
    if not best_list:
        return [1]
        
    # Final validation (just in case)
    # We trust the logic, but a quick check is cheap for small sets? 
    # No, validation is O(N^3) or O(BadTriples). 
    # Let's do a quick check using the conflicts structure.
    # If any conflict pair is fully in the set, it's invalid.
    for x in best_list:
        for mask in conflicts[x]:
            # Check if the pair in mask is in best_list
            # mask has two bits set. 
            # We can check if (best_mask & mask) == mask
            if (best_mask & mask) == mask:
                # Invalid! Fallback to simple greedy
                best_list = [i for i in range(1, 129) if ((1 << i) & _greedy_initial(conflicts, list(range(1, 129))))]
                break
        else:
            continue
        break

    return best_list
# EVOLVE-BLOCK-END
