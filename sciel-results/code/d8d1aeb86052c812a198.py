import time
import random

_LCM = 232792560
_ALL = list(range(1, 23))
_WEIGHTS = [_LCM // i for i in _ALL]

def _count_collisions(weights):
    """
    Returns the number of collisions in the subset sums of the given weights.
    Collisions = (1 << k) - len(set_of_sums)
    """
    k = len(weights)
    sums = {0}
    for w in weights:
        new_sums = {s + w for s in sums}
        sums |= new_sums
    return (1 << k) - len(sums)

def _is_dissociated(weights):
    return _count_collisions(weights) == 0

def _find_dissociated_set(target_size, time_limit, seed):
    """
    Uses Simulated Annealing to find a dissociated set of the target size.
    """
    if target_size > len(_ALL):
        return None
    
    start_time = time.time()
    rng = random.Random(seed)
    
    # Initialize with a random set
    current_set = set(rng.sample(_ALL, target_size))
    current_weights = [_WEIGHTS[i-1] for i in current_set]
    current_cost = _count_collisions(current_weights)
    
    best_set = None
    best_cost = current_cost
    
    if current_cost == 0:
        return list(current_set)
        
    T = 50.0
    T_min = 0.01
    cooling = 0.99 # Slower cooling
    
    while time.time() - start_time < time_limit:
        # Generate neighbor
        # Convert set to list for indexing
        current_list = list(current_set)
        i_idx = rng.randrange(target_size)
        i_val = current_list[i_idx]
        
        # Pick j not in set
        available = [x for x in _ALL if x not in current_set]
        if not available:
            break
        j_val = rng.choice(available)
        
        # Swap
        new_set = set(current_set)
        new_set.remove(i_val)
        new_set.add(j_val)
        
        new_weights = [_WEIGHTS[i-1] for i in new_set]
        new_cost = _count_collisions(new_weights)
        
        delta = new_cost - current_cost
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or (delta > 0 and rng.random() < pow(2.718, -delta / T)):
            current_set = new_set
            current_weights = new_weights
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_set = list(current_set)
                
            if best_cost == 0:
                return best_set
                
        T *= cooling
        if T < T_min:
            T = T_min
            
    if best_cost == 0:
        return best_set
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    # 1. Try to find size 18
    # Budget 5s
    res = _find_dissociated_set(18, 5.0, 123)
    if res:
        return res
        
    # 2. Try to find size 17
    # Budget 30s
    # Use multiple seeds/restarts
    best_17 = None
    # 10 restarts, 3s each
    for seed in range(10):
        if time.time() > deadline - 20: # Leave time for 16
            break
        res = _find_dissociated_set(17, 3.0, 1000 + seed)
        if res:
            best_17 = res
            break
            
    if best_17:
        return best_17
        
    # 3. Find size 16
    # Greedy first
    best_16 = []
    current = []
    for i in _ALL:
        test = current + [i]
        if _is_dissociated([_WEIGHTS[j-1] for j in test]):
            current = test
            if len(current) == 16:
                best_16 = current
                break
                
    if best_16:
        return best_16
        
    # If greedy failed, use SA for 16
    res = _find_dissociated_set(16, 10.0, 42)
    if res:
        return res
        
    # Fallback
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
