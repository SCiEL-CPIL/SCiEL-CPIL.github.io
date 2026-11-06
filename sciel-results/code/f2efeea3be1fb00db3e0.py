import time
import numpy as np
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _is_dissociated_subset(S, recips=None):
    """
    Checks if a set of integers S is dissociated in the reciprocals.
    i.e., all subset sums of {1/s for s in S} are distinct.
    """
    if not S:
        return True
    
    n = len(S)
    
    if recips is None:
        reciprocals = [1.0 / s for s in S]
    else:
        reciprocals = [recips[s] for s in S]
    
    # Generate all subset sums
    sums = [0.0]
    for r in reciprocals:
        # Add r to all existing sums
        new_sums = [s + r for s in sums]
        sums.extend(new_sums)
        
    # Check for duplicates
    sorted_sums = sorted(sums)
    for i in range(len(sorted_sums) - 1):
        if sorted_sums[i] == sorted_sums[i+1]:
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 60s limit (conservative)
    
    # Precompute reciprocals
    recips = {i: 1.0/i for i in range(1, 23)}
    universe = list(range(1, 23))
    
    # Known good seed
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_set)
    
    # Strategy 1: Greedy with various orders
    # This is fast and often finds good local optima
    
    orders = [
        list(range(1, 23)),
        list(range(22, 0, -1)),
    ]
    
    # Add some random permutations
    random.seed(42)
    np.random.seed(42)
    
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        cand_order = list(range(1, 23))
        random.shuffle(cand_order)
        orders.append(cand_order)
    
    for order in orders:
        if time.time() > deadline - 10:
            break
            
        current_set = []
        used = set()
        
        for x in order:
            if x in used:
                continue
            # Try adding x
            new_set = current_set + [x]
            if _is_dissociated_subset(new_set, recips):
                current_set = new_set
                used.add(x)
                
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)
            
        # Early exit if we reached a very high number
        if best_size >= 15:
            break

    # Strategy 2: Random sampling of specific sizes
    # Try to find sets of size > best_size by random sampling
    # We try sizes from min(22, best_size + 3) down to best_size + 1
    
    max_target = min(22, best_size + 3)
    
    for target_size in range(max_target, best_size, -1):
        if time.time() > deadline - 5:
            break
        
        # Number of attempts to try
        attempts = 0
        # Adjust attempts based on how hard it is
        # Larger sizes are harder, so we might need more attempts or give up faster
        max_attempts = 50000 if target_size <= 12 else 10000
        
        while attempts < max_attempts:
            if time.time() > deadline - 2:
                break
            
            # Random subset of target_size
            candidate = random.sample(universe, target_size)
            if _is_dissociated_subset(candidate, recips):
                best_set = candidate
                best_size = target_size
                break  # Found one, move to next (larger) size? No, we are going down.
                # Actually, if we found size 12, we should try 13?
                # The loop is `for target_size in range(max_target, best_size, -1)`
                # So if best_size was 9, max_target is 12.
                # Loop: 12, 11, 10.
                # If we find 12, best_size becomes 12.
                # Next iteration: range(12, 12, -1) is empty. So we stop.
                # This is correct: we found the max possible in this search space.
                # But wait, if we found 11, best_size becomes 11.
                # Next iteration: range(12, 11, -1) -> 12. We try 12 again?
                # No, `max_target` was fixed at start.
                # If we found 11, we should still try 12?
                # The logic `range(max_target, best_size, -1)` updates `best_size` inside.
                # Python `range` is evaluated once.
                # So if we start with best_size=9, max_target=12.
                # Range is [12, 11, 10].
                # If we find 11 at step 11, best_size becomes 11.
                # But we already passed 12.
                # So we might miss a 12 if we only found an 11 first.
                # To be safe, we should iterate upwards? Or just accept this limitation.
                # Given the rarity of size 12+ sets, finding a 11 is a big win.
                # Finding a 12 is even better.
                # Let's just break on first success for a target size.
                attempts = max_attempts
                break
            attempts += 1

    # Fallback
    if not best_set:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set
# EVOLVE-BLOCK-END
