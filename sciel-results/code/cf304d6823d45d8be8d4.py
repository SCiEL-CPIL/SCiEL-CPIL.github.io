import time
from math import gcd
from itertools import combinations

def _is_dissociated(S):
    """Check if the set of reciprocals of S has all distinct subset sums using integer arithmetic."""
    if not S:
        return True
    n = len(S)
    if n > 15: # Heuristic cutoff for speed, though 22 is small enough if optimized
        return False 
    
    # Scale by LCM of all elements to convert to integers
    # LCM of 1..22 is 232792560, which fits in 64-bit int easily.
    # We can compute LCM of S specifically to keep numbers smaller.
    from functools import reduce
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, S, 1)
    
    # Convert reciprocals to integers: val_i = L // a_i
    vals = [L // a for a in S]
    
    # Generate all subset sums
    # Use a set to detect duplicates
    sums = {0}
    current_sums = [0]
    
    for v in vals:
        new_sums = [s + v for s in current_sums]
        for ns in new_sums:
            if ns in sums:
                return False
            sums.add(ns)
        current_sums.extend(new_sums)
        
    return True

def construct():
    deadline = time.time() + 85  # 85s safety margin (limit 90s)
    
    # Universe: 1..22
    universe = list(range(1, 23))
    
    best = []
    
    # Pre-compute pairwise conflicts? 
    # Actually, let's just use the fast checker in a greedy + local search + B&B.
    
    # 1. Greedy with multiple orderings
    import random
    random.seed(42)
    
    orderings = [universe, universe[::-1]]
    for _ in range(10):
        shuffled = universe[:]
        random.shuffle(shuffled)
        orderings.append(shuffled)
        
    def greedy_add(order):
        current = []
        for a in order:
            if time.time() > deadline:
                break
            test = current + [a]
            if _is_dissociated(test):
                current = test
        return current

    for order in orderings:
        if time.time() > deadline:
            break
        result = greedy_add(order)
        if len(result) > len(best):
            best = result[:]

    # 2. Branch and Bound
    # Sort universe by "promisingness". 
    # Larger numbers have smaller reciprocals, might be less likely to cause small-sum collisions?
    # Or smaller numbers are more "dense". 
    # Let's try sorting descending (22, 21, ...) as they are "safer" individually.
    
    def bb_search(available, current, idx=0):
        nonlocal best
        if time.time() > deadline:
            return
        
        # Pruning: if even adding all remaining can't beat best
        if len(current) + len(available) <= len(best):
            return
            
        if not available:
            return
            
        # Heuristic: try to include available[0] first? Or exclude?
        # Usually, including "safe" elements first helps fill the set.
        
        a = available[0]
        rest = available[1:]
        
        # Branch 1: Include a
        if _is_dissociated(current + [a]):
            if len(current + [a]) > len(best):
                best = current + [a]
            bb_search(rest, current + [a], idx + 1)
            
        # Branch 2: Exclude a
        bb_search(rest, current, idx + 1)
        
    # Run B&B
    # Start with a good ordering. 
    # Numbers that are "hard" to fit (like 1, 2, 3) should be considered later? 
    # Or earlier to prune? 
    # If we include 1, 1/1=1. 1/2+1/2 is not allowed (distinct elements). 
    # 1 = 1/2 + 1/2 (no, elements distinct).
    # 1/1 = 1/2 + 1/2 (invalid, same element).
    # 1/1 = 1/2 + 1/3 + 1/6. So {1,2,3,6} is bad.
    # 1/2 = 1/3 + 1/6. So {2,3,6} is bad.
    # 1/3 = 1/4 + 1/12. So {3,4,12} is bad.
    # 1/4 = 1/5 + 1/20. So {4,5,20} is bad.
    # 1/5 = 1/6 + 1/30 (30 out of range).
    # 1/6 = 1/7 + 1/42 (42 out).
    
    # So 1, 2, 3, 4 are "dangerous". 5, 6, 7... are safer.
    # Let's order by descending value to pick safe ones first?
    bb_order = sorted(universe, reverse=True)
    
    # To speed up B&B, we can use a "conflict" check.
    # If we have a set C, and we add a, we only need to check sums involving a.
    # But _is_dissociated is fast enough for n<12.
    
    bb_search(bb_order, [])
    
    # 3. Local Search / Improvement
    # Try to swap elements in best to allow more elements?
    # Or just try to add one more element to best by removing one.
    
    # Since B&B explores the space, it should find the global max if time permits.
    # Given n=22, B&B with pruning should be very fast if the constraint is tight.
    
    return best
