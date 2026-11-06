import time
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def _is_dissociated_subset(S):
    """
    Checks if a set of integers S is dissociated in the reciprocals.
    i.e., all subset sums of {1/s for s in S} are distinct.
    """
    if not S:
        return True
    
    n = len(S)
    # We work with the reciprocals
    reciprocals = [1.0 / s for s in S]
    
    # To check distinctness of all 2^n subset sums, we can generate them.
    # For n <= 15, 2^15 = 32768, which is very fast.
    # If n > 15, it might be slower, but we stop early if collisions are found.
    
    # Optimization: Use integer arithmetic if possible? 
    # 1/a = 1/b implies a=b. 
    # Sum(1/a_i) = Sum(1/b_j) is hard to check with integers without huge LCMs.
    # Floating point with high precision or careful sorting is usually sufficient for n~10-12.
    
    # Generate all subset sums
    sums = [0.0]
    for r in reciprocals:
        # Add r to all existing sums
        new_sums = [s + r for s in sums]
        sums.extend(new_sums)
        
    # Check for duplicates
    # Sorting is O(N log N). N = 2^n.
    # If n=12, N=4096. Very fast.
    # If n=15, N=32768. Still fast.
    
    sorted_sums = sorted(sums)
    for i in range(len(sorted_sums) - 1):
        # Use a small epsilon for float comparison? 
        # Since these are sums of distinct unit fractions, exact equality in float 
        # is rare unless they are mathematically identical. 
        # However, 1/2 + 1/3 = 5/6 vs 1/6 + 1/2 + 1/3? No, distinct subsets.
        # 1/2 + 1/3 = 0.8333...
        # 1/6 + 1/2 + 1/3 = 1.0
        # The risk is floating point precision causing 0.1+0.2 == 0.3 to be False when it should be True, 
        # or True when it should be False.
        # Given the small numbers (1..22), precision is not an issue for distinguishing distinct sums 
        # which differ by at least 1/(22*21*...) which is very small but representable.
        # Actually, the minimum difference between distinct sums of unit fractions from [1..22]
        # is bounded below by 1/LCM(1..22) which is huge, so 1/LCM is tiny.
        # Wait, 1/a - 1/b = (b-a)/(ab). The smallest difference is between 1/21 and 1/22? No.
        # It's between sums.
        # Standard double precision has ~15-17 decimal digits.
        # LCM(1..22) is approx 2.3e9. 1/LCM is approx 4e-10.
        # This is well within double precision relative to 1.0.
        # So exact float comparison after sorting should be safe, or a very tight epsilon.
        if sorted_sums[i] == sorted_sums[i+1]:
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin for 60s limit
    
    # Greedy approach:
    # Start with empty set.
    # Try to add elements from 1 to 22 in increasing order?
    # Or try to maximize cardinality.
    # A known good set is [1, 2, 3, 5, 7, 11, 13, 17, 19] (size 9).
    # Can we get size 10, 11, 12?
    
    # Strategy:
    # 1. Try to build a set of size K.
    # 2. Start with K=12 and go down? Or start small and grow?
    # Growing is safer for validity.
    
    # Let's try to grow from a base set that is known to work or just start from scratch.
    # Starting from scratch with greedy "add smallest possible" might get stuck in local minima.
    # Better: Start with the known size 9 set, then try to add more?
    # Or use a randomized local search / hill climbing.
    
    # Let's use a simple heuristic:
    # 1. Initialize with a small valid set, e.g., [1]
    # 2. Iterate candidate additions from 2 to 22.
    # 3. If adding candidate keeps it dissociated, add it.
    # 4. If stuck, try backtracking or random restarts.
    
    # Actually, the "smallest possible" greedy often works well for dissociated sets because
    # larger denominators contribute less to the sum, making collisions less likely?
    # No, smaller denominators contribute more.
    # Let's try adding from 22 down to 1? Or 1 up to 22?
    
    # Let's try a few different greedy orders and keep the best.
    
    best_set = []
    
    # Order 1: Increasing
    # Order 2: Decreasing
    # Order 3: Randomized (multiple tries)
    
    orders = [
        list(range(1, 23)),
        list(range(22, 0, -1)),
        # Add some random permutations if time permits
    ]
    
    for _ in range(20):
        if time.time() > deadline:
            break
        # Random order
        cand_order = list(range(1, 23))
        np.random.shuffle(cand_order)
        orders.append(cand_order)
    
    for order in orders:
        if time.time() > deadline:
            break
            
        current_set = []
        used = set()
        
        for x in order:
            if x in used:
                continue
            # Try adding x
            new_set = current_set + [x]
            if _is_dissociated_subset(new_set):
                current_set = new_set
                used.add(x)
                
        if len(current_set) > len(best_set):
            best_set = current_set
            
        # Early exit if we reached max possible (22) or a very high number (e.g. 15)
        if len(best_set) >= 15:
            break

    # If best_set is empty (shouldn't happen), fallback to seed
    if not best_set:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set
# EVOLVE-BLOCK-END
