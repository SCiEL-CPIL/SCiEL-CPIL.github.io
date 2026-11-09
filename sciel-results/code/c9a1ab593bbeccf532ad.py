import time
import random
from math import gcd
from functools import reduce

def construct():
    deadline = time.time() + 80.0
    
    def is_dissociated(subset):
        if not subset:
            return True
        n = len(subset)
        
        # Calculate LCM
        def lcm(a, b):
            return a * b // gcd(a, b)
            
        try:
            L = reduce(lcm, subset)
        except ZeroDivisionError:
            return False
            
        # Scale reciprocals to integers
        scaled = [L // s for s in subset]
        
        # Check if all 2^n subset sums are distinct
        sums_set = {0}
        for x in scaled:
            prev_size = len(sums_set)
            # Generate new sums
            # Using a list comprehension is faster than a loop for small sets
            new_sums = [s + x for s in sums_set]
            sums_set.update(new_sums)
            
            # If size didn't double, there was a collision
            if len(sums_set) != prev_size * 2:
                return False
                
        return True

    best_set = []
    best_size = 0
    
    # 1. Deterministic Greedy (Ascending, Descending)
    base_orders = [
        list(range(1, 23)),
        list(range(22, 0, -1)),
    ]
    
    for order in base_orders:
        S = []
        for x in order:
            trial = S + [x]
            if is_dissociated(trial):
                S = trial
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    # 2. Randomized Greedy
    # We can run many iterations
    # Estimate: 10000 iterations * 22 checks * ~100 ops/check = 22M ops. Safe.
    
    iterations = 20000
    for i in range(iterations):
        if time.time() > deadline:
            break
            
        perm = list(range(1, 23))
        random.shuffle(perm)
        
        S = []
        for x in perm:
            trial = S + [x]
            if is_dissociated(trial):
                S = trial
                
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
            # If we reach a very large size, we might stop early?
            # Max possible is 22, but likely < 15.
            # If we find 14, we can continue trying for 15.
            
    # 3. Local Improvement on best_set
    # Try to add any missing element
    current_best = list(best_set)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        missing = [x for x in range(1, 23) if x not in current_best]
        
        # Try to add
        for x in missing:
            trial = current_best + [x]
            if is_dissociated(trial):
                current_best = trial
                improved = True
                break
        
        if improved:
            continue
            
        # Try to swap
        # If we can't add, try swapping one element to allow adding another?
        # This is heuristic.
        for i in range(len(current_best)):
            if time.time() > deadline:
                break
            removed = current_best[i]
            reduced = current_best[:i] + current_best[i+1:]
            
            # Try to add a different element
            for x in missing:
                if x == removed:
                    continue
                trial = reduced + [x]
                if is_dissociated(trial):
                    # Accept the swap if it allows us to add more later?
                    # For now, accept if size is same, hoping it opens up.
                    # But we need to be careful not to loop forever.
                    # Only accept if the new set is "better" in some way?
                    # Let's just accept and continue the outer loop.
                    current_best = trial
                    improved = True
                    break
            if improved:
                break

    return current_best
