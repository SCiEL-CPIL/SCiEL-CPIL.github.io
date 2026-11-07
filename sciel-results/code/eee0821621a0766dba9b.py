import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # The seed returns 64 elements (65..128).
    # We aim to beat this.
    best_set = set(range(65, 129))
    best_size = len(best_set)
    
    def is_valid(s):
        """
        Check if set s violates the condition:
        No three distinct members a, b, c in s such that a+b, a+c, b+c are all in s.
        """
        # To check efficiently, we can iterate over pairs (a, b) in s
        # and check if there exists a c in s such that a+b, a+c, b+c are in s.
        # Actually, the condition is: 
        # For any distinct a, b, c in S, it is NOT the case that (a+b in S AND a+c in S AND b+c in S).
        
        # A faster way:
        # For each pair a, b in S (a < b), let sum_ab = a + b.
        # If sum_ab is in S, then for any c in S (c != a, c != b), 
        # if a+c in S and b+c in S, then we have a violation.
        # So for a fixed pair (a, b) with a+b in S, we must ensure that there is NO c in S 
        # such that c is distinct from a, b, and (a+c) in S and (b+c) in S.
        
        # Let's precompute membership for speed.
        s_list = list(s)
        n = len(s_list)
        if n < 3:
            return True
        
        # We can iterate over all pairs a, b in s.
        # If a+b is in s, then we look for c.
        # c must be in s.
        # a+c must be in s.
        # b+c must be in s.
        
        # Optimization:
        # If a+b is not in s, this pair (a,b) cannot be part of a violation as the 'base' pair.
        # But wait, the condition is symmetric. The triple is {a,b,c}.
        # The sums are a+b, a+c, b+c.
        # So we just need to check all triples? That's O(N^3).
        # N is around 64-100. 100^3 = 1,000,000. This is fast enough in Python if done carefully, 
        # but we will do many iterations. 
        # 
        # Better check:
        # For each c in s:
        #   For each a in s, a < c:
        #     For each b in s, a < b < c:
        #        Check if a+b, a+c, b+c in s.
        # This is still O(N^3).
        
        # Alternative:
        # For each pair a, b in s with a < b:
        #   s_ab = a + b
        #   if s_ab in s:
        #       # Now we need to find if there is a c in s, c != a, c != b, 
        #       # such that a+c in s and b+c in s.
        #       # Let's iterate c in s.
        #       # This is O(N^2) per valid sum pair.
        #       # Worst case O(N^3).
        
        # Let's try to optimize the inner loop.
        # For fixed a, b, we need c such that c in s, a+c in s, b+c in s.
        # This means c in (s & (s-a) & (s-b)).
        # Where s-a = {x-a | x in s}.
        
        # We can use set operations.
        # s_set = s
        # candidates_c = s_set
        # for a in s_set:
        #     ...
        
        # Let's just use the triple loop but with early breaks and sets.
        s_set = s
        # Iterate a, b, c with a < b < c
        # This covers each distinct triple exactly once.
        # For each triple, check the 3 sums.
        # Number of triples is C(N,3). For N=80, C(80,3) = 80*79*78/6 = 82160.
        # 82k checks per validation.
        # If we do 10000 iterations, that's 800 million checks. Might be too slow in Python.
        
        # We need a faster validity check or a smarter search.
        
        # Let's use the property:
        # A violation exists if there are a, b, c in S such that a+b, a+c, b+c in S.
        # Note that a+b, a+c, b+c are distinct if a,b,c are distinct?
        # a+b = a+c => b=c (no)
        # a+b = b+c => a=c (no)
        # a+c = b+c => a=b (no)
        # So the three sums are distinct.
        
        # Let's try to optimize the check by iterating over sums.
        # For each sum S_sum in S:
        #   It can be a+b.
        #   We need to know if there are a, b in S such that a+b = S_sum.
        #   And then check if there is a c...
        
        # Actually, let's just use the O(N^3) check but with NumPy or optimized loops?
        # No, standard library only for logic, but we can use sets.
        
        # Let's try a different angle for the check:
        # For each c in s:
        #   For each a in s, a < c:
        #     if a+c in s:
        #        # Now we need b in s, b != a, b != c, such that b+c in s and a+b in s.
        #        # Let b be in s.
        #        # b+c in s => b in s-c.
        #        # a+b in s => b in s-a.
        #        # So b must be in s & (s-c) & (s-a) & {x | x != a, x != c}.
        #        # We can compute the intersection.
        
        # This is getting complex. Let's stick to a simple O(N^3) check first and see if it's fast enough for a few hundred iterations.
        # If N=100, C(100,3) = 161700.
        # 161k iterations. In Python, ~0.01-0.05s per check?
        # 1000 checks = 10-50s. Too slow for 50s budget if we want many restarts.
        
        # We need a faster check.
        
        # How about:
        # For each a in s:
        #   For each b in s, b > a:
        #     s_ab = a + b
        #     if s_ab in s:
        #        # Check if there is c in s, c != a, c != b, such that a+c in s and b+c in s.
        #        # Iterate c in s:
        #        #   if c == a or c == b: continue
        #        #   if a+c in s and b+c in s: return False
        #        
        #        # Optimization: 
        #        # The set of valid c must be in s.
        #        # Also a+c in s => c in s-a.
        #        # Also b+c in s => c in s-b.
        #        # So c in s & (s-a) & (s-b).
        #        # We can compute this intersection.
        #        # If the intersection contains any element other than a or b, it's invalid.
        
        # Let's implement this.
        
        for a in s_set:
            for b in s_set:
                if a >= b:
                    continue
                s_ab = a + b
                if s_ab in s_set:
                    # Find c
                    # c must be in s_set
                    # c must be in s_set - a (i.e., c+a in s_set)
                    # c must be in s_set - b (i.e., c+b in s_set)
                    # c != a, c != b
                    
                    # We can just iterate c in s_set.
                    # But we can filter.
                    # Let's just iterate, it's O(N) per pair. Total O(N^3) worst case, but with early exit.
                    # Actually, if s_ab is in s, it's likely we find a violation quickly if one exists?
                    # Or we can compute the intersection.
                    
                    # Intersection approach:
                    # candidates = s_set
                    # for x in s_set:
                    #    if x != a and x != b:
                    #        if x + a in s_set and x + b in s_set:
                    #            return False
                    # return True
                    
                    # This is O(N) per pair (a,b) where a+b in s.
                    # How many such pairs? Up to N^2/2.
                    # So O(N^3).
                    
                    # Is there a faster way?
                    # If N is small, O(N^3) is fine.
                    # Let's assume N <= 100.
                    
                    for c in s_set:
                        if c == a or c == b:
                            continue
                        if (a + c) in s_set and (b + c) in s_set:
                            return False
        return True

    # Greedy construction
    def greedy_construct():
        s = set()
        # Randomize order to get different results
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        
        for x in candidates:
            trial = s.union([x])
            if is_valid(trial):
                s = trial
        return s

    # Local search
    def local_search(s):
        s = set(s)
        improved = True
        while improved:
            improved = False
            # Try adding
            for x in range(1, 129):
                if x not in s:
                    trial = s.union([x])
                    if is_valid(trial):
                        s = trial
                        improved = True
                        break # Restart loop to check new additions
            # Try removing and adding (swap)
            if not improved:
                for x in list(s):
                    for y in range(1, 129):
                        if y not in s:
                            trial = s.difference([x]).union([y])
                            if is_valid(trial) and len(trial) > len(s):
                                s = trial
                                improved = True
                                break
                    if improved:
                        break
        return s

    # Strategy:
    # 1. Start with greedy.
    # 2. Local search.
    # 3. Repeat with different random seeds.
    
    # Since is_valid is O(N^3) and N~80, it might be slow.
    # Let's optimize is_valid with a simple bitset or just efficient sets.
    # The current is_valid is okay for N=64.
    # C(64,3) = 41664.
    # 41k checks. Very fast.
    
    best_set = greedy_construct()
    best_size = len(best_set)
    
    # Local search on the greedy result
    improved_set = local_search(best_set)
    if len(improved_set) > best_size:
        best_set = improved_set
        best_size = len(best_set)
    
    # Try a few random restarts
    restarts = 0
    while time.time() < deadline - 5:
        restarts += 1
        s = greedy_construct()
        s = local_search(s)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
            
        # If we found a large set, we can try to improve it further
        if best_size > 80:
             # Maybe try removing one element and adding two?
             # This is complex. Stick to single add/swap.
             pass

    # Convert to list of ints
    return list(best_set)
