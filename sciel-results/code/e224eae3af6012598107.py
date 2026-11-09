import math
import random

def construct():
    N = 256
    
    # Helper to check if a set is valid (no 3-AP)
    # We will use a fast check for adding a single element to a set represented as a set of ints.
    
    def is_valid_add(s, x):
        """Check if adding x to set s creates a 3-term AP."""
        # x, a, b => b = 2x - a. Need b in s.
        # a, x, b => b = 2x - a. Same as above? 
        # Let's check all pairs (a, b) in s? No, that's O(N^2).
        # Check for each a in s:
        # 1. Is there b in s such that a, x, b is AP? b = 2x - a.
        # 2. Is there b in s such that a, b, x is AP? b = (a+x)/2.
        # 3. Is there b in s such that x, a, b is AP? b = 2a - x.
        
        for a in s:
            if a == x:
                continue
            
            # Case 1: a, x, b. b = 2x - a
            b = 2 * x - a
            if 1 <= b <= N and b != x and b in s:
                return False
            
            # Case 2: a, b, x. b = (a + x) / 2
            if (a + x) % 2 == 0:
                b = (a + x) // 2
                if b != a and b != x and b in s:
                    return False
            
            # Case 3: x, a, b. b = 2a - x
            b = 2 * a - x
            if 1 <= b <= N and b != x and b != a and b in s:
                return False
                
        return True

    # 1. Greedy Initialization
    S = set()
    for i in range(1, N + 1):
        if is_valid_add(S, i):
            S.add(i)
            
    current_score = sum(1.0 / i for i in S)
    
    # 2. Local Search: Remove-Refill
    # We iterate until no improvement is found.
    # To avoid infinite loops or getting stuck in local minima due to order,
    # we can randomize the order of removals or use a counter.
    
    improved = True
    # Limit iterations to prevent timeouts, though it should converge fast.
    max_iterations = 1000 
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # We try removing each element in S.
        # To make it deterministic but robust, we iterate over a sorted list.
        # If we find an improvement, we apply it and restart the scan (or continue?).
        # Restarting is safer to ensure we don't miss interactions.
        
        elements_to_try = list(S)
        # Shuffle to avoid bias? Or sort by 1/x (descending) to try removing "expensive" items first?
        # Removing small items (high weight) is risky but might allow many medium items.
        # Let's try sorting by value (1/x) descending, i.e., smallest x first.
        elements_to_try.sort(key=lambda x: x) 
        
        for x in elements_to_try:
            if x not in S:
                continue
                
            S_trial = S.copy()
            S_trial.remove(x)
            
            # Greedily refill S_trial with elements not in S_trial
            # Candidates are all y in [1, N] not in S_trial.
            # Sort candidates by 1/y descending (i.e., y ascending) to maximize gain.
            candidates = [y for y in range(1, N + 1) if y not in S_trial]
            candidates.sort() # Ascending y means descending 1/y
            
            gain = 0.0
            new_elements = []
            
            for y in candidates:
                if is_valid_add(S_trial, y):
                    S_trial.add(y)
                    gain += 1.0 / y
                    new_elements.append(y)
            
            # Compare gain with cost of removing x
            cost = 1.0 / x
            
            if gain > cost:
                # Accept the move
                S = S_trial
                current_score += gain - cost
                improved = True
                # Break out of the for loop to restart the while loop with new S
                break
                
    return list(S)
