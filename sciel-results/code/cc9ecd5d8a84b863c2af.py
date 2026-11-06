import time
import random

def _can_add(current_reach, v, target):
    """
    Check if adding v to the set with reachability bitmask 'current_reach' 
    results in 'target' being reachable.
    Returns (True, new_reach) if safe, (False, current_reach) if not.
    """
    # If v itself is target, it's unsafe (unless target is 0, but target is 200)
    if v == target:
        return False, current_reach
    
    # new_reach = current_reach | (current_reach << v)
    # We only care about bits up to target.
    # We can mask to keep the integer small, though Python handles big ints well.
    # Mask: (1 << (target + 1)) - 1
    mask = (1 << (target + 1)) - 1
    
    new_reach = current_reach | ((current_reach << v) & mask)
    
    # Check if target bit is set in new_reach
    if (new_reach >> target) & 1:
        return False, current_reach
    
    return True, new_reach

def _compute_reach(S, target):
    """Compute the reachability bitmask for a set S."""
    reach = 1
    mask = (1 << (target + 1)) - 1
    for v in S:
        reach = reach | ((reach << v) & mask)
    return reach

def _try_add_elements(S, reach, candidates, target, deadline):
    """Try to add elements from candidates to S, updating reach."""
    added = 0
    for v in candidates:
        if time.time() > deadline:
            break
        safe, new_reach = _can_add(reach, v, target)
        if safe:
            S.add(v)
            reach = new_reach
            added += 1
    return reach, added

def _local_search_swap(S, reach, universe, target, deadline, max_swaps=100):
    """
    Try to improve S by swapping: remove one element, try to add another.
    This can allow us to add elements that were previously blocked.
    """
    improvements = 0
    S_list = list(S)
    
    for _ in range(max_swaps):
        if time.time() > deadline:
            break
        
        # Try removing each element and see if we can add more
        for i in range(len(S_list)):
            if time.time() > deadline:
                break
            
            v_remove = S_list[i]
            
            # Temporarily remove v_remove
            S.remove(v_remove)
            # Recompute reach without v_remove
            new_reach = _compute_reach(S, target)
            
            # Try to add elements not in S
            added_any = False
            for v_add in universe:
                if time.time() > deadline:
                    break
                if v_add in S:
                    continue
                safe, candidate_reach = _can_add(new_reach, v_add, target)
                if safe:
                    S.add(v_add)
                    new_reach = candidate_reach
                    added_any = True
                    break
            
            if added_any:
                # We made an improvement
                reach = new_reach
                S_list = list(S)
                improvements += 1
            else:
                # Put v_remove back
                S.add(v_remove)
    
    return S, reach, improvements

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s safety margin
    
    target = 200
    universe = list(range(1, 101))
    mask = (1 << (target + 1)) - 1
    
    best_set = None
    best_size = 0
    
    # Strategy 1: Ascending
    if time.time() < deadline:
        order = sorted(universe)
        S = set()
        reach = 1
        
        for v in order:
            safe, new_reach = _can_add(reach, v, target)
            if safe:
                S.add(v)
                reach = new_reach
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()

    # Strategy 2: Descending
    if time.time() < deadline:
        order = sorted(universe, reverse=True)
        S = set()
        reach = 1
        
        for v in order:
            safe, new_reach = _can_add(reach, v, target)
            if safe:
                S.add(v)
                reach = new_reach
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()

    # Strategy 3: Multiple randomized greedy attempts with local search
    attempts = 0
    while time.time() < deadline - 10.0:
        attempts += 1
        # Randomized Greedy Construction
        order = universe.copy()
        random.shuffle(order)
        
        S = set()
        reach = 1
        
        for v in order:
            safe, new_reach = _can_add(reach, v, target)
            if safe:
                S.add(v)
                reach = new_reach
        
        # Second pass: try to add remaining elements in random order
        remaining = [x for x in universe if x not in S]
        random.shuffle(remaining)
        reach, _ = _try_add_elements(S, reach, remaining, target, deadline - 5.0)
        
        # Local search: try swaps to improve
        if time.time() < deadline - 5.0:
            S, reach, improvements = _local_search_swap(S, reach, universe, target, deadline - 3.0, max_swaps=50)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    # Strategy 4: Try to improve the best set found so far with more aggressive local search
    if best_set is not None and time.time() < deadline - 2.0:
        S = best_set.copy()
        reach = _compute_reach(S, target)
        
        # Try multiple rounds of swap-based improvement
        for round_num in range(5):
            if time.time() > deadline - 1.0:
                break
            S, reach, improvements = _local_search_swap(S, reach, universe, target, deadline - 1.0, max_swaps=30)
            if improvements == 0:
                break
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()

    if best_set is None:
        best_set = {100}
        best_size = 1

    return list(best_set)
