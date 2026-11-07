import time
import random
from itertools import combinations

def is_valid(s):
    """
    Check if the set s satisfies the cross-cardinality disjoint sum constraint.
    """
    k = len(s)
    # sums_by_card[c] will hold the set of all sums of subsets of size c
    # Using sets to automatically handle duplicates within same cardinality
    sums_by_card = [set() for _ in range(k + 1)]
    sums_by_card[0].add(0)
    
    # Optimization: If we find an intersection, return False early
    # We generate sums incrementally
    # For each cardinality c, we generate all sums.
    # k is small (max 10-12), so combinations(s, c) is fast.
    
    for c in range(1, k + 1):
        for comb in combinations(s, c):
            sums_by_card[c].add(sum(comb))
        
        # Check intersection with all previous cardinalities
        for prev_c in range(c):
            if sums_by_card[c] & sums_by_card[prev_c]:
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50 # 50s safety margin for 60s limit
    
    # Known valid set of size 9
    base = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    best_set = base[:]
    
    # Try to find a set of size 10
    # Strategy: Randomized local search or random restarts
    # Given the tightness, random might be slow.
    # Let's try a structured search first.
    
    # Try replacing 256 with smaller numbers and adding one more
    # Or trying different small bases
    
    # Attempt 1: Search for size 10 using random sampling with constraints
    # We'll try to generate sets close to superincreasing but with more elements
    
    attempts = 0
    max_attempts = 10000 # Limit to avoid timeout
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        # Generate a random candidate set of size 10
        # Bias towards small numbers for the first few, large for the rest?
        # Or just uniform?
        # Uniform random is likely to fail validation quickly.
        
        # Better heuristic: Start with a base and mutate
        # Let's try to construct a "tight" sequence
        # x1=1, x2=2, x3=4 ...
        # Randomize the "growth factor" slightly?
        
        # Alternative: Random sample from 1..256
        cand = random.sample(range(1, 257), 10)
        cand.sort()
        
        if is_valid(cand):
            return cand
    
    # If random didn't find one, try a more directed search
    # Try to extend the size 9 set by replacing 256 with two smaller numbers?
    # No, size must be 10.
    # Try: [1, 2, 4, 8, 16, 32, 64, x8, x9, x10]
    # We need x8, x9, x10 to be in 1..256, distinct, and > 64.
    # And valid.
    
    # Let's try a few specific patterns
    patterns = [
        [1, 2, 4, 8, 16, 32, 64, 100, 150, 200],
        [1, 2, 4, 8, 16, 32, 64, 90, 160, 230],
        [1, 2, 4, 8, 16, 32, 64, 80, 170, 240],
        [1, 2, 4, 8, 16, 32, 64, 110, 140, 210],
        [1, 2, 4, 8, 16, 32, 64, 95, 145, 225],
    ]
    
    for p in patterns:
        if all(1 <= x <= 256 for x in p) and len(set(p)) == 10:
            if is_valid(p):
                return p
    
    # If nothing found, return the best known (size 9)
    return best_set
