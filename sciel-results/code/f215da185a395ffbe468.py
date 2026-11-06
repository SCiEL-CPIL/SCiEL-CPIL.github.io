import random
import time
import math

# EVOLVE-BLOCK-START

_LCM = 232792560
_INV = {x: _LCM // x for x in range(1, 23)}
_ALL = list(range(1, 23))

def _build_sums(S):
    """
    Build the set of all subset sums for a given set S.
    Returns the set of sums.
    """
    sums = {0}
    for x in S:
        v = _INV[x]
        shifted = {s + v for s in sums}
        sums = sums | shifted
    return sums

def _get_collision_count(S):
    """
    Returns the number of collisions: 2^|S| - |Sums(S)|.
    """
    if not S:
        return 0
    sums = _build_sums(S)
    return (1 << len(S)) - len(sums)

def _greedy_find_valid(deadline):
    """
    Fast greedy search to find a valid set of decent size.
    """
    best = set()
    best_size = 0
    # Try a few random orders
    for _ in range(50):
        if time.time() > deadline:
            break
        order = _ALL[:]
        random.shuffle(order)
        sums = {0}
        chosen = set()
        for x in order:
            inv = _INV[x]
            shifted = {s + inv for s in sums}
            if sums.isdisjoint(shifted):
                sums = sums | shifted
                chosen.add(x)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best = set(chosen)
    return best, best_size

def _simulated_annealing(S_init, target_size, deadline):
    """
    Run SA to fix collisions in S_init, maintaining size target_size.
    Returns a valid set of size target_size if found, else None.
    """
    if len(S_init) != target_size:
        if len(S_init) > target_size:
            items = list(S_init)
            random.shuffle(items)
            S = set(items[:target_size])
        else:
            remaining = [x for x in _ALL if x not in S_init]
            random.shuffle(remaining)
            S = set(S_init)
            for x in remaining:
                if len(S) == target_size:
                    break
                S.add(x)
    else:
        S = set(S_init)
        
    current_cost = _get_collision_count(S)
    if current_cost == 0:
        return S

    T = 10.0
    cooling_rate = 0.99
    min_T = 0.01
    
    all_elements = _ALL
    list_S = list(S)

    while time.time() < deadline:
        if T < min_T:
            break
            
        # Pick a random move: swap i in S with j not in S
        i = random.choice(list_S)
        
        # Pick j not in S
        # Create list of outside elements
        outside = [x for x in all_elements if x not in S]
        if not outside:
            break
        j = random.choice(outside)
        
        S_new = set(S)
        S_new.remove(i)
        S_new.add(j)
        
        new_cost = _get_collision_count(S_new)
        
        delta = new_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / T):
            S = S_new
            list_S = list(S)
            current_cost = new_cost
            if current_cost == 0:
                return S
        
        T *= cooling_rate
        
    return None

def construct():
    start = time.time()
    deadline = start + 85 # Safety margin

    # Phase 1: Find a good valid set using greedy
    best_set, best_size = _greedy_find_valid(deadline - 10)
    
    if not best_set:
        best_set = {1}
        best_size = 1

    # Phase 2: Try to extend using SA
    current_best_size = best_size
    target_size = current_best_size + 1
    
    # We will run multiple SA attempts for increasing target sizes
    # until time runs out or we can't improve
    
    while target_size <= 22 and time.time() < deadline - 5:
        found = False
        attempts = 0
        max_attempts = 30 # Limit per size to avoid getting stuck
        
        while attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            
            # Initialize SA state
            # Start with best_set and add a random element not in it
            remaining = [x for x in _ALL if x not in best_set]
            if not remaining:
                break
            
            init_element = random.choice(remaining)
            S_init = set(best_set)
            S_init.add(init_element)
            
            # If we are targeting a size larger than current_best_size + 1
            # we need to add more elements
            if len(S_init) < target_size:
                rem2 = [x for x in _ALL if x not in S_init]
                random.shuffle(rem2)
                for x in rem2:
                    if len(S_init) == target_size:
                        break
                    S_init.add(x)
            elif len(S_init) > target_size:
                items = list(S_init)
                random.shuffle(items)
                S_init = set(items[:target_size])
            
            result = _simulated_annealing(S_init, target_size, deadline - 2)
            
            if result:
                # Found a valid set of target_size
                best_set = result
                current_best_size = target_size
                found = True
                break
        
        if not found:
            # If we failed to find a larger set, we stop trying to extend.
            # It's unlikely we can jump to target_size + 1 if we can't do target_size.
            break

    return sorted(best_set)

# EVOLVE-BLOCK-END
