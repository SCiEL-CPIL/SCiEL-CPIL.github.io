import time
import random
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def can_add(x, S_set):
        """Check if x can be added to S_set while maintaining sum-free property."""
        # 1. x + a in S or a + x in S (same)
        # 2. 2x in S
        # 3. a + b = x for a, b in S
        
        # Optimization: 
        # For 1: if a in S and a+x in S.
        # For 3: if x-a in S for any a in S.
        
        # We can combine 1 and 3?
        # If a in S:
        #   if a+x in S: return False
        #   if x-a in S and a != x-a (i.e., 2a != x): return False 
        #   if 2a == x (i.e., a = x/2) and a in S: return False (this is a+b=x with a=b)
        
        # Actually, simpler:
        # For a in S:
        #   if a + x in S: return False
        #   if x - a in S: return False  # This covers a+b=x. Note if x-a == a, then 2a=x.
        
        # Wait, if x-a in S, let b = x-a. Then a+b=x.
        # This single loop covers both "x is sum of two" and "x+a is in S"?
        # No. 
        # Case 1: a+b=x. b = x-a. If b in S, conflict.
        # Case 2: a+x=c. c in S. If c in S, conflict.
        
        # So:
        for a in S_set:
            if a + x in S_set:
                return False
            if x - a in S_set:
                return False
            # Note: if x-a == a, i.e., 2a=x, then x-a is a, which is in S. 
            # So the second condition covers the a=b case for sum.
            # What about 2x in S?
            # If we add x, and 2x is in S, then x+x=2x.
            # This is covered by a=x? No, x is not in S_set yet.
            # So we need to check 2x in S_set separately?
            # If we check `if x - a in S_set`, and a=x, x is not in S_set.
            # So we MUST check `if 2*x in S_set` separately.
            
        if 2 * x in S_set:
            return False
            
        return True

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def solve_instance(S):
        S_set = set(S)
        odds = [x for x in S if x % 2 != 0]
        evens = [x for x in S if x % 2 == 0]
        
        best_set = set(odds)
        best_size = len(best_set)
        
        # We will run multiple iterations with different random orders for evens
        # and local search.
        
        num_restarts = 20 # Adjust based on time
        rng = Random(42)
        
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            current_set = set(odds)
            
            # Shuffle evens for this restart
            shuffled_evens = evens[:]
            rng.shuffle(shuffled_evens)
            
            # Greedy Add
            for e in shuffled_evens:
                if time.time() > deadline - 5:
                    break
                if can_add(e, current_set):
                    current_set.add(e)
            
            # Local Search
            # Try to improve by swapping out elements
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                # Try removing each element in current_set
                # To avoid O(N^2) complexity blowup, we limit the number of swaps or use random sampling if N is large
                # Here N is small (<= 240), so full pass is ok.
                
                # Create a list of elements to try removing
                # We prioritize removing elements that might be "blocking" many others.
                # Heuristic: Remove elements with high "degree" in conflict graph?
                # Simpler: Just iterate through a random permutation of current_set
                elements_to_try = list(current_set)
                rng.shuffle(elements_to_try)
                
                for u in elements_to_try:
                    if time.time() > deadline - 10:
                        break
                        
                    test_set = set(current_set)
                    test_set.remove(u)
                    
                    # Try to add back any evens (or odds? No, odds are fixed base)
                    # We should try to add any element from (S \ test_set) that is even?
                    # Or any element?
                    # If we removed an odd, we can't add it back (other odds are there, no conflict, but size same).
                    # If we removed an even, we can add other evens.
                    # So we iterate through all evens not in test_set.
                    
                    # To be more aggressive, we can also try adding odds if we removed an odd?
                    # No, all other odds are in. Adding the removed odd back is trivial.
                    # So we only look for evens to add.
                    
                    added = False
                    for e in evens:
                        if e in test_set:
                            continue
                        if can_add(e, test_set):
                            test_set.add(e)
                            added = True
                            
                    if len(test_set) > len(current_set):
                        current_set = test_set
                        improved = True
                        # Restart the while loop? Or continue?
                        # Restarting is safer to find new improvements.
                        break 
                        
            if len(current_set) > best_size:
                best_set = current_set
                best_size = len(best_set)
                
        return list(best_set)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        
        # Fallback if time is tight
        if time.time() > deadline - 10:
             results.append([x for x in S if x % 2])
             continue
             
        res = solve_instance(S)
        
        # Verify
        if not is_sum_free(res):
            # Fallback to parity
            res = [x for x in S if x % 2]
            
        results.append(res)
        
    return results
