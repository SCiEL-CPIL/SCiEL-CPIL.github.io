import math
import random

def _check_valid(S):
    """Check if all reciprocal subset sums are distinct using integer arithmetic."""
    if not S:
        return True
    L = 1
    for x in S:
        L = L * x // math.gcd(L, x)
    
    coeffs = [L // x for x in S]
    
    sums = {0}
    for c in coeffs:
        new_sums = set()
        for s in sums:
            new_sums.add(s + c)
        if sums & new_sums:
            return False
        sums = sums | new_sums
    return True

def construct():
    # Fallback to seed if search fails
    seed = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_set = seed
    best_size = len(seed)
    
    all_elements = list(range(1, 23))
    
    # Strategy 1: Greedy with various orders
    orders = [
        sorted(all_elements),
        sorted(all_elements, reverse=True),
        [x for x in all_elements if x in [2,3,5,7,11,13,17,19,23]], # Primes (23 not in range, so up to 19)
        list(reversed(sorted(all_elements))),
    ]
    
    # Add some random orders
    for _ in range(10):
        rand_order = all_elements.copy()
        random.shuffle(rand_order)
        orders.append(rand_order)
        
    for order in orders:
        current_set = []
        for x in order:
            if x in current_set:
                continue
            new_set = current_set + [x]
            if _check_valid(new_set):
                current_set = new_set
        
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)
            
    # Strategy 2: Refinement. Try to add any missing element to best_set
    # We might be able to add more elements by trying all combinations of missing elements?
    # Just try adding each missing element one by one to the best_set found so far.
    # But best_set might not be the best base.
    # Let's run a local search on the best_set found.
    
    improved = True
    while improved:
        improved = False
        current_best = best_set.copy()
        for x in all_elements:
            if x in current_best:
                continue
            new_set = current_best + [x]
            if _check_valid(new_set):
                current_best = new_set
                best_set = current_best
                best_size = len(current_best)
                improved = True
                break # Restart loop with new best
        
    return best_set
