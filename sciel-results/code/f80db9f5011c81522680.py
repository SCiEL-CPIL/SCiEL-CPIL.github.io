import random
import time
from math import lcm

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit
    
    candidates = list(range(1, 23))
    
    # Precompute LCM of 1..22 to convert reciprocals to integers
    L = 1
    for i in range(1, 23):
        L = lcm(L, i)
    
    # Map each x to its integer reciprocal value scaled by L
    val_map = {x: L // x for x in candidates}
    
    def is_dissociated(S):
        """Check if all subset sums of reciprocals of S are distinct using integer arithmetic."""
        if not S:
            return True
        sums = {0}
        for x in S:
            v = val_map[x]
            # Check collision before updating
            for s in sums:
                if (s + v) in sums:
                    return False
            new_sums = [s + v for s in sums]
            sums.update(new_sums)
        return True
    
    def try_add(S, x):
        """Check if S union {x} is dissociated."""
        return is_dissociated(S + [x])
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Greedy with different deterministic orders
    orders = [
        sorted(candidates, reverse=True),
        sorted(candidates),
        [n for n in candidates if n >= 12],  # Large numbers only
        [n for n in candidates if n % 2 == 0],  # Evens
        [n for n in candidates if n % 2 == 1],  # Odds
        [p for p in candidates if all(p % d != 0 for d in range(2, int(p**0.5)+1))],  # Primes
    ]
    
    for order in orders:
        if time.time() - start_time > timeout:
            break
        current_set = []
        for x in order:
            if try_add(current_set, x):
                current_set.append(x)
        
        # Try to expand with remaining elements
        remaining = [x for x in candidates if x not in current_set]
        added = True
        while added and time.time() - start_time < timeout:
            added = False
            for x in remaining[:]:
                if try_add(current_set, x):
                    current_set.append(x)
                    remaining.remove(x)
                    added = True
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    # Strategy 2: Randomized restarts with expansion
    random.seed(42)
    restart_count = 0
    while time.time() - start_time < timeout:
        restart_count += 1
        order = candidates[:]
        random.shuffle(order)
        
        current_set = []
        for x in order:
            if try_add(current_set, x):
                current_set.append(x)
        
        # Try to expand with remaining elements
        remaining = [x for x in candidates if x not in current_set]
        added = True
        while added and time.time() - start_time < timeout:
            added = False
            for x in remaining[:]:
                if try_add(current_set, x):
                    current_set.append(x)
                    remaining.remove(x)
                    added = True
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            # If we found a very large set, we can stop early
            if best_size >= 15:
                break
    
    # Fallback to a known valid set if search fails
    if not best_set:
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return best_set
# EVOLVE-BLOCK-END
