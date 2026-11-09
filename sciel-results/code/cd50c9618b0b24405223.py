"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 80.0  # Safety margin
    
    # Start with the safe tail [171, 256]
    start_threshold = 171
    current_set = set(range(start_threshold, n + 1))
    best_set = set(current_set)
    best_size = len(current_set)
    
    # Precompute membership bitset for fast lookup
    def get_bitset(S):
        """Return a boolean array where index i is True if i+1 in S"""
        bs = np.zeros(n + 1, dtype=bool)
        for x in S:
            if 1 <= x <= n:
                bs[x] = True
        return bs
    
    def is_valid_addition_bitset(S, x, S_bitset):
        """Check if adding x to S is valid using bitset operations"""
        # We need to check three cases:
        # 1. x is smallest (a): for all b,c in S with x<b<=c, x does NOT divide (b+c)
        # 2. x is middle (b): for all a,c in S with a<x<=c, a does NOT divide (x+c)  
        # 3. x is largest (c): for all a,b in S with a<b<=x, a does NOT divide (b+x)
        
        # Case 1: x is a (smallest)
        # Need: no pair (b,c) in S with x<b<=c such that (b+c) % x == 0
        # b+c ranges from x+1+x+2 to n+n = 2n
        # For each residue r mod x, we need to check if there exist b,c in S with 
        # b,c > x, b<=c, and b+c ≡ 0 (mod x)
        
        # Efficient approach: for each possible sum s = b+c where b,c in S and b,c > x,
        # check if s % x == 0
        
        # Build list of elements > x in S
        greater_than_x = [y for y in S if y > x]
        
        if len(greater_than_x) >= 2:
            # Check all pairs
            for i in range(len(greater_than_x)):
                for j in range(i, len(greater_than_x)):
                    b = greater_than_x[i]
                    c = greater_than_x[j]
                    if b <= c and (b + c) % x == 0:
                        return False
        
        # Case 2: x is b (middle)
        # Need: no pair (a,c) in S with a<x<=c such that (x+c) % a == 0
        less_than_x = [y for y in S if y < x]
        greater_or_equal_x = [y for y in S if y >= x]
        
        for a in less_than_x:
            for c in greater_or_equal_x:
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is c (largest)
        # Need: no pair (a,b) in S with a<b<=x such that (b+x) % a == 0
        less_or_equal_x = [y for y in S if y <= x]
        
        for i in range(len(less_or_equal_x)):
            for j in range(i + 1, len(less_or_equal_x)):
                a = less_or_equal_x[i]
                b = less_or_equal_x[j]
                if a < b and (b + x) % a == 0:
                    return False
        
        return True
    
    # Candidate pool: numbers not in current set
    candidates = list(range(1, start_threshold))
    random.shuffle(candidates)
    
    # Local search: try to add elements from candidates
    attempts = 0
    consecutive_failures = 0
    max_consecutive_failures = 200
    
    while time.time() < deadline:
        if not candidates:
            break
        
        x = candidates.pop()
        
        if is_valid_addition_bitset(current_set, x, None):
            current_set.add(x)
            if len(current_set) > best_size:
                best_set = set(current_set)
                best_size = len(current_set)
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures > max_consecutive_failures:
                # Reshuffle remaining candidates to try different order
                random.shuffle(candidates)
                consecutive_failures = 0
        
        # Check time periodically
        if time.time() > deadline - 5:
            break
    
    # Final verification and return
    return sorted(list(best_set))
# EVOLVE-BLOCK-END
