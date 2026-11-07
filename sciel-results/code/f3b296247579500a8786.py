import time
import random
from math import gcd

def construct():
    deadline = time.time() + 55.0
    
    # Precompute multiples for fast checking
    # multiples_of[k] contains all m in [2, 256] such that m % k == 0
    multiples_of = [[] for _ in range(257)]
    for k in range(1, 257):
        for m in range(k, 257, k):
            if m >= 2:
                multiples_of[k].append(m)
                
    # Candidates are 2..256
    candidates = list(range(2, 257))
    
    def can_add(a, in_set):
        """
        Check if a can be added to the set represented by in_set (boolean array/list).
        in_set[i] is True if i is in the set.
        """
        # We need to check for all b in S (where in_set[b] is True):
        # 1. a | b*c for some c in S, c != b
        #    This is equivalent to: (a / gcd(a,b)) | c
        # 2. b | a*c for some c in S, c != b
        #    This is equivalent to: (b / gcd(a,b)) | c
        
        for b in range(2, 257):
            if not in_set[b]:
                continue
            
            g = gcd(a, b)
            
            # Check 1: a | b*c => (a/g) | c
            k1 = a // g
            # We need c in S, c != b, such that c is a multiple of k1
            # Note: b is a multiple of k1? 
            # b = k1 * (b/g) * (g/g)? No.
            # b = (a/g) * (b/a) * g? 
            # Let's just check the list of multiples.
            for c in multiples_of[k1]:
                if c == b:
                    continue
                if in_set[c]:
                    return False
            
            # Check 2: b | a*c => (b/g) | c
            k2 = b // g
            for c in multiples_of[k2]:
                if c == b:
                    continue
                if in_set[c]:
                    return False
                    
        return True

    def greedy_construction(order):
        in_set = [False] * 257
        current_set = []
        for a in order:
            if can_add(a, in_set):
                in_set[a] = True
                current_set.append(a)
        return current_set

    def local_search(S):
        """
        Try to improve set S by swapping elements.
        Returns a new set if improved, else S.
        """
        in_set = [False] * 257
        for x in S:
            in_set[x] = True
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Try removing each element and adding a better one
            for i in range(len(S)):
                x = S[i]
                # Remove x
                in_set[x] = False
                S_temp = S[:i] + S[i+1:]
                
                # Try to add any y not in S
                # We want to increase size, so we just need to add one y
                # To maximize future potential, we might want to add the "least restrictive" y
                # But for now, just try all y
                added = False
                for y in range(2, 257):
                    if in_set[y]:
                        continue
                    if can_add(y, in_set):
                        # Add y
                        in_set[y] = True
                        S_temp.append(y)
                        # Since we removed 1 and added 1, size is same.
                        # This doesn't increase size. 
                        # The goal is to maximize cardinality.
                        # A swap only helps if it allows adding MORE elements later.
                        # Simple hill climbing on size won't work with swaps if size doesn't increase immediately.
                        # We need to check if the new set allows adding more elements than the old set did.
                        # This is expensive.
                        # Alternative: Just try to add elements to the original set first.
                        # If greedy is stuck, maybe a different order works.
                        # Let's stick to multiple greedy restarts.
                        pass
                        
                # Revert
                in_set[x] = True
                S_temp = None
                
        return S

    best_set = []
    
    # Strategy 1: Multiple Greedy Restarts with different orders
    orders = [
        list(range(256, 1, -1)), # Descending
        list(range(2, 257)),     # Ascending
    ]
    
    # Add some random orders
    for _ in range(50):
        if time.time() > deadline - 5.0:
            break
        random.shuffle(candidates)
        orders.append(list(candidates))
        
    for order in orders:
        if time.time() > deadline - 2.0:
            break
        S = greedy_construction(order)
        if len(S) > len(best_set):
            best_set = S
            
    # Strategy 2: Local Search / Improvement on best_set
    # Try to add any remaining elements to best_set? 
    # Greedy already tried to add all in that order. 
    # But maybe an element rejected early could be added now? No, constraints are monotonic.
    # If a was rejected because of b,c in S, and S only grows, a will remain rejected.
    # So greedy is complete for that order.
    
    # To improve, we need to find a better order or a different set structure.
    # Let's try a "conflict-aware" greedy.
    # At each step, pick the candidate that has the fewest conflicts with the current set?
    # Or the one that blocks the fewest other candidates?
    
    # Given the time limit, multiple random restarts is the most robust simple heuristic.
    # Let's increase the number of random restarts if time permits.
    
    start_time = time.time()
    num_restarts = 0
    while time.time() < deadline - 5.0:
        random.shuffle(candidates)
        S = greedy_construction(list(candidates))
        num_restarts += 1
        if len(S) > len(best_set):
            best_set = S
            # If we found a very good set, we might want to polish it
            # But polishing is hard.
            
    # Final check: Try to add any element from 2..256 not in best_set
    # This is redundant if greedy was exhaustive for that order, but just in case.
    in_set = [False] * 257
    for x in best_set:
        in_set[x] = True
        
    # Try adding in descending order (large numbers are safer)
    for a in range(256, 1, -1):
        if not in_set[a]:
            if can_add(a, in_set):
                in_set[a] = True
                best_set.append(a)
                
    # Try adding in ascending order
    in_set = [False] * 257
    for x in best_set:
        in_set[x] = True
    for a in range(2, 257):
        if not in_set[a]:
            if can_add(a, in_set):
                in_set[a] = True
                best_set.append(a)

    # Ensure return type is list of ints
    result = [int(x) for x in best_set]
    return result
