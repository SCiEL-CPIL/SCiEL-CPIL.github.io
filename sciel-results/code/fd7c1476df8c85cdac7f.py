import random
import time
from fractions import Fraction
from math import lcm

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit
    
    # The universe of candidates
    candidates = list(range(1, 23))
    
    best_set = []
    best_size = 0
    
    # Precompute LCM of 1..22 to convert reciprocals to integers
    # LCM(1..22) is 232792560
    L = 1
    for i in range(1, 23):
        L = lcm(L, i)
        
    # Map each x to its integer reciprocal value scaled by L
    # val[x] = L // x
    val_map = {x: L // x for x in candidates}
    
    def try_order(order):
        current_set = []
        sums = {0}
        
        for x in order:
            v = val_map[x]
            # Check for collision
            new_sums = set()
            collision = False
            for s in sums:
                ns = s + v
                if ns in sums:
                    collision = True
                    break
                new_sums.add(ns)
            
            if not collision:
                current_set.append(x)
                sums.update(new_sums)
                
        return current_set
    
    def try_add_to_set(current_set, sums, x):
        """Try to add x to the current set. Returns (success, new_sums)"""
        v = val_map[x]
        new_sums = set()
        for s in sums:
            ns = s + v
            if ns in sums:
                return False, None
            new_sums.add(ns)
        return True, new_sums
    
    def local_search(current_set, sums):
        """Try to add more elements to the set by trying all possible additions"""
        improved = True
        while improved:
            improved = False
            for x in candidates:
                if x in current_set:
                    continue
                success, new_sums = try_add_to_set(current_set, sums, x)
                if success:
                    current_set.append(x)
                    sums.update(new_sums)
                    improved = True
                    break  # Restart the loop with the new state
        return current_set, sums
    
    # Strategy 1: Greedy with different orderings
    orders = [
        sorted(candidates, reverse=True),
        sorted(candidates),
        [p for p in candidates if all(p % d != 0 for d in range(2, int(p**0.5)+1))], # Primes
        [n for n in candidates if n > 10], # Large numbers
        [n for n in candidates if n <= 10], # Small numbers
    ]
    
    # Add some random orders
    for _ in range(10):
        shuffled = candidates[:]
        random.shuffle(shuffled)
        orders.append(shuffled)
        
    # Run through the prepared orders
    for order in orders:
        s = try_order(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]
            
        if time.time() - start_time > timeout:
            break
    
    # Local search on the best set found so far
    if best_set:
        # Rebuild sums for the best set
        sums = {0}
        for x in best_set:
            v = val_map[x]
            new_sums = set()
            for s in sums:
                new_sums.add(s + v)
            sums.update(new_sums)
        
        improved_set, improved_sums = local_search(best_set, sums)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set
    
    # Strategy 2: Random restarts with local search
    while time.time() - start_time < timeout:
        shuffled = candidates[:]
        random.shuffle(shuffled)
        s = try_order(shuffled)
        
        if len(s) > 0:
            # Rebuild sums
            sums = {0}
            for x in s:
                v = val_map[x]
                new_sums = set()
                for ss in sums:
                    new_sums.add(ss + v)
                sums.update(new_sums)
            
            # Local search
            improved_set, improved_sums = local_search(s, sums)
            
            if len(improved_set) > best_size:
                best_size = len(improved_set)
                best_set = improved_set[:]
                if best_size >= 16:
                    pass  # Keep searching
    
    if not best_set:
        # Fallback to a known valid small set
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]

    return best_set
# EVOLVE-BLOCK-END
