import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    best_set = list(range(64, 129))
    best_size = len(best_set)
    
    def is_valid_addition(S_set, x):
        # Case 1: x is a base element
        # Need a, b in S_set such that a+x in S_set, b+x in S_set, and a+b in S_set or a+b == x
        candidates = [a for a in S_set if (a + x) in S_set]
        if len(candidates) >= 2:
            for i in range(len(candidates)):
                a = candidates[i]
                for j in range(i + 1, len(candidates)):
                    b = candidates[j]
                    s = a + b
                    if s == x or s in S_set:
                        return False
        
        # Case 2: x is a sum a+b
        # Need a, b in S_set such that a+b == x
        # And c in S_set \ {a,b} such that a+c in S_set and b+c in S_set
        # We can find pairs (a,b) summing to x
        # Since S_set is a set, we can iterate a in S_set, check if x-a in S_set
        # To avoid duplicates, we can just check all a.
        # Optimization: Only check a < x-a? Or just check all and handle symmetry?
        # If we find a violation, we return False.
        
        # Let's iterate a in S_set
        # If a >= x, x-a <= 0, not in S_set (assuming positive).
        for a in S_set:
            b = x - a
            if b in S_set and a != b:
                # Found pair a, b summing to x
                # Now check if there is c in S_set \ {a,b} such that a+c in S_set and b+c in S_set
                # We can iterate c in S_set
                # This is O(|S|) for each pair.
                # Number of pairs summing to x is small (at most |S|/2).
                # So total O(|S|^2).
                
                # To speed up, we can precompute?
                # Just iterate c.
                for c in S_set:
                    if c == a or c == b:
                        continue
                    if (a + c) in S_set and (b + c) in S_set:
                        return False
        return True

    # Randomized Greedy
    while time.time() < deadline:
        nums = list(range(1, 129))
        random.shuffle(nums)
        S = []
        S_set = set()
        
        for x in nums:
            if is_valid_addition(S_set, x):
                S.append(x)
                S_set.add(x)
                
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
    return best_set
