import time
import random
from itertools import combinations

# Global deadline
START_TIME = time.time()
TIME_LIMIT = 50.0 # Safety margin for 60s limit

def can_add(S, x):
    """
    Check if x can be added to valid set S.
    S is a set of integers.
    Condition:
    1. x does not divide sum of any 3 distinct elements in S.
    2. No a in S divides sum of x and any 2 distinct elements in S \ {a}.
    """
    S_list = list(S)
    n = len(S_list)
    
    # Check 1: x divides sum of 3 distinct in S
    # Sum of 3 distinct b,c,d from S. x | (b+c+d)
    # If n < 3, no such triple exists, so this is vacuously true.
    if n >= 3:
        for b, c, d in combinations(S_list, 3):
            if (b + c + d) % x == 0:
                return False

    # Check 2: a in S divides sum of x and 2 distinct in S \ {a}
    # For each a in S:
    #   Check if a | (x + b + c) for b,c in S \ {a}, b != c
    for a in S_list:
        # We need pairs b, c from S \ {a}
        # S \ {a} has n-1 elements.
        # We need to check all pairs (b,c) from S \ {a}
        # If n-1 < 2, no such pair.
        if n < 3:
            continue
            
        # Optimization: Iterate through S_list, skip a
        # Generate pairs
        # To avoid creating a new list every time, we can just loop
        # But combinations on a filtered list is clean.
        # For speed, manual loops might be better if n is large, but n<=192.
        # Let's use combinations on the filtered list.
        others = [v for v in S_list if v != a]
        for b, c in combinations(others, 2):
            if (x + b + c) % a == 0:
                return False

    return True

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_set = set()
    best_size = 0
    
    numbers = list(range(1, 193))
    
    # Strategy: Start with a dense set of large numbers (upper half)
    # Large numbers are less likely to divide sums of smaller numbers.
    # Then repair by removing violators, and backfill.
    
    while time.time() - START_TIME < TIME_LIMIT:
        # 1. Initialize with a random subset of larger numbers
        # Let's pick random numbers from 96..192 first, then fill rest
        # Or just random shuffle but bias towards larger numbers?
        # The "Knowledge" suggests: "start with dense upper-half sets"
        
        # Let's try a hybrid:
        # Start with all numbers > 100 (92 numbers). Check validity.
        # If invalid, remove violators.
        # Then try to add remaining numbers.
        
        current_set = set()
        
        # Phase 1: Add large numbers greedily (random order among large)
        large_numbers = [x for x in numbers if x > 96]
        random.shuffle(large_numbers)
        for x in large_numbers:
            if can_add(current_set, x):
                current_set.add(x)
                
        # Phase 2: Add smaller numbers greedily (random order)
        small_numbers = [x for x in numbers if x <= 96 and x not in current_set]
        random.shuffle(small_numbers)
        for x in small_numbers:
            if can_add(current_set, x):
                current_set.add(x)
                
        # Phase 3: Local Search / Repair
        # If we are stuck, try swapping elements to allow more additions.
        # Simple repair: if a new element x can't be added, maybe removing one existing element y allows x to be added?
        # This is expensive. Let's just do a few random restarts within the time limit.
        # The previous failure was due to timeout in the complex swap loop.
        # A simpler approach: just run multiple greedy passes with different random seeds/orders.
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
    return list(best_set)
