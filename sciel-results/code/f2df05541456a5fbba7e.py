import time
import random
from math import gcd
from functools import reduce

# Calculate LCM of 1..22
def lcm(a, b):
    return a * b // gcd(a, b)

_LCM = reduce(lcm, range(1, 23))

def _verify_recip_dissociated(subset):
    if not subset:
        return True
    values = [_LCM // i for i in subset]
    sums = {0}
    for v in values:
        new_sums = {s + v for s in sums}
        if sums & new_sums:
            return False
        sums |= new_sums
    return True

def _can_add(sums, v):
    for s in sums:
        if s + v in sums:
            return False
    return True

def _backtrack_search(K, start_idx, current_set, current_sums, deadline, order, values_map):
    if len(current_set) == K:
        return current_set[:]
    
    if time.time() > deadline:
        return None
        
    if len(current_set) + (len(order) - start_idx) < K:
        return None

    for i in range(start_idx, len(order)):
        idx = order[i]
        v = values_map[idx]
        
        if _can_add(current_sums, v):
            new_sums = set(current_sums)
            for s in current_sums:
                new_sums.add(s + v)
            
            result = _backtrack_search(K, i + 1, current_set + [idx], new_sums, deadline, order, values_map)
            if result:
                return result
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    candidates = list(range(1, 23))
    values_map = {i: _LCM // i for i in candidates}
    
    best_set = []
    rng = random.Random(42)
    
    def greedy(order):
        current_set = []
        current_sums = {0}
        for idx in order:
            v = values_map[idx]
            if _can_add(current_sums, v):
                current_set.append(idx)
                for s in list(current_sums):
                    current_sums.add(s + v)
        return current_set

    # Get a baseline size
    max_greedy_size = 0
    for _ in range(50):
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        s = greedy(shuffled)
        if len(s) > max_greedy_size:
            max_greedy_size = len(s)
            best_set = s
            
    # Try to find larger sets
    # We try K from max_greedy_size + 1 to 22
    # We allocate a portion of the remaining time to each K
    # If we find a set of size K, we return it.
    
    remaining_time = deadline - time.time()
    if remaining_time <= 1.0:
        return best_set
        
    # We'll try K in descending order? No, ascending is better for finding *some* larger set.
    # But we want the MAXIMUM.
    # So we should try high K first?
    # If we try K=22, it will likely fail quickly (pruning).
    # If we try K=17, it might take longer.
    # Let's try K from 22 down to max_greedy_size + 1.
    # If we find one, return it.
    
    for K in range(22, max_greedy_size, -1):
        if time.time() > deadline - 5.0:
            break
            
        # Try multiple random orderings for this K
        found = None
        attempts = 0
        # Limit attempts per K to avoid spending too much time on impossible K
        max_attempts = 200
        
        while time.time() < deadline - 5.0 and attempts < max_attempts:
            shuffled = candidates[:]
            rng.shuffle(shuffled)
            
            # Heuristic: Sort by value?
            # Let's try sorting by value descending, then shuffling slightly?
            # Or just random.
            
            res = _backtrack_search(K, 0, [], {0}, deadline - 5.0, shuffled, values_map)
            if res:
                found = res
                break
            attempts += 1
            
        if found:
            if _verify_recip_dissociated(found):
                return found
        
    return best_set
