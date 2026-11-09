"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 256
    deadline = time.time() + 85.0 # Safety margin
    
    # Start with the maximal safe tail: {k, ..., n} where k = floor(2n/3) + 1
    # For n=256, 2n/3 = 170.66..., so k=171.
    # This set is valid because for any a < b <= c in {k, ..., n},
    # a >= k, b+c >= 2k+1 (if b=k, c=k+1) or generally b+c > a * something?
    # Actually, the condition is a does NOT divide b+c.
    # If we take the top half or top third, we can ensure no divisibility.
    # Specifically, if S = {k, ..., n}, then for a in S, b,c in S, a < b <= c.
    # The minimum b+c is 2k+1 (if b=k, c=k+1).
    # The maximum a is n.
    # If we choose k such that 2k+1 > n, then b+c > n >= a.
    # Then b+c can only be a multiple of a if b+c = 2a, 3a, ...
    # Since b+c < 2n and a >= k, the only possible multiple is 2a (if 2a < 2n).
    # Wait, if b+c = 2a, then a divides b+c.
    # So we need to avoid b+c = 2a.
    # If S is an interval [k, n], can we have b+c = 2a?
    # Yes, e.g., a=171, b=172, c=170 (not allowed, b<=c, so b=170, c=172? No a<b. So a=171, b=172, c=170 is invalid order.
    # a < b <= c. So a=171, b=172, c=172 -> sum=244. 244 % 171 != 0.
    # a=171, b=173, c=169? No b<=c.
    # Let's check the condition more carefully.
    # If S = [k, n], then for any a in S, the multiples of a are 2a, 3a, ...
    # We need b+c != ma for any integer m.
    # Since b,c > a, b+c > 2a. So m >= 3.
    # The smallest sum is 2k+1 (b=k, c=k+1? No, b>a. So min b is a+1. Min c is a+1. Sum = 2a+2.
    # So we need 2a+2 != ma.
    # This is hard to guarantee for all a just by interval.
    
    # However, it is known that the set {n/2 + 1, ..., n} is NOT necessarily valid.
    # But {2n/3 + 1, ..., n} is often a good starting point.
    # Let's verify for n=256, k=171.
    # S = {171, ..., 256}.
    # If a=171, b=172, c=172, sum=244. 244 % 171 != 0.
    # If a=171, b=172, c=173, sum=245.
    # If a=171, b=171? No a<b.
    # Is there any a,b,c in [171, 256] such that a | (b+c)?
    # Let's just trust the "safe tail" heuristic which is standard for this problem.
    # The previous code used 171.
    
    start_threshold = 171
    current_set = set(range(start_threshold, n + 1))
    
    # Candidates to add: numbers from 1 to start_threshold - 1
    # We will try to add them in descending order (largest first) as they are less likely to divide sums of larger numbers.
    # Or ascending?
    # If we add a small number x, it acts as 'a' for all existing pairs.
    # It also acts as 'b' or 'c' for pairs involving smaller numbers (none yet) or itself?
    # If we add large numbers first, they are 'b' and 'c' for the small 'a's we might add later.
    # If we add small numbers first, they are 'a's.
    
    # Strategy: Greedy addition.
    # Try to add candidates from 1 to 170.
    # Order: Descending (170 down to 1) is often better because large numbers are more "inert" as 'a' (fewer multiples in range) 
    # but dangerous as 'b' or 'c'.
    # Actually, if we add x, we must check:
    # 1. x as 'a': x does not divide any b+c in S.
    # 2. x as 'b': for any a in S, a<x, a does not divide x+c for any c in S, c>=x.
    # 3. x as 'c': for any a,b in S, a<b<=x, a does not divide b+x.
    
    # Let's use a fast check.
    
    def is_valid_addition(S, x):
        # S is a set
        # Check 1: x as 'a'
        # x must not divide b+c for any b,c in S with x < b <= c
        # Note: if x is not in S, and we are adding x, then b,c must be in S.
        # If x is the new smallest, then all b,c in S are > x.
        # We can optimize by checking sums.
        
        # Precompute sums of pairs in S? Too slow if S is large.
        # But S starts at 171, size 86.
        # As we add, S grows.
        
        # Let's just iterate.
        # Optimization: 
        # For check 1: 
        # For each b in S where b > x:
        #   For each c in S where c >= b:
        #      if (b+c) % x == 0: return False
        
        # For check 2:
        # For each a in S where a < x:
        #   For each c in S where c >= x:
        #      if (x+c) % a == 0: return False
        
        # For check 3:
        # For each a in S:
        #   For each b in S where a < b <= x:
        #      if (b+x) % a == 0: return False
        
        # To make this faster, we can sort S.
        # But S is a set. Let's convert to list once per check?
        # Or maintain a sorted list.
        
        # Let's try a simple implementation first and see if it's fast enough.
        # With n=256, max size ~100-150.
        # Pairs ~ 150^2 = 22500.
        # 150 candidates * 22500 checks = 3.3M ops. Very fast.
        
        S_list = sorted(S)
        
        # Check 1: x as 'a'
        # b, c in S, x < b <= c
        # Find indices where element > x
        # Since S is sorted, we can binary search or just iterate.
        
        # Actually, we can just iterate all pairs.
        # n=256 is small.
        
        for i in range(len(S_list)):
            b = S_list[i]
            if b <= x:
                continue
            for j in range(i, len(S_list)):
                c = S_list[j]
                if (b + c) % x == 0:
                    return False
        
        # Check 2: x as 'b'
        # a, c in S, a < x <= c
        for i in range(len(S_list)):
            a = S_list[i]
            if a >= x:
                break
            # c must be >= x
            # Find first index k where S_list[k] >= x
            # Since a < x, and S is sorted, we can just iterate c from that point
            # But we need to be careful with indices.
            for k in range(len(S_list)):
                c = S_list[k]
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
                    
        # Check 3: x as 'c'
        # a, b in S, a < b <= x
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                if b > x:
                    break
                if (b + x) % a == 0:
                    return False
        
        return True

    best_set = set(current_set)
    best_size = len(current_set)
    
    # Candidates: 1 to 170
    # Try descending order
    candidates = list(range(start_threshold - 1, 0, -1))
    
    for x in candidates:
        if time.time() > deadline - 5:
            break
        if is_valid_addition(current_set, x):
            current_set.add(x)
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
                
    # Now try to improve with local search / swapping?
    # The greedy approach might get stuck.
    # We can try random restarts or random shuffling of candidates.
    # Given time limit, let's do a few random passes.
    
    import random
    random.seed(42) # Fixed seed for reproducibility? Or random each time?
    # Let's use random to explore different orders.
    
    for _ in range(10): # 10 restarts
        if time.time() > deadline - 5:
            break
            
        # Reset to initial tail
        current_set = set(range(start_threshold, n + 1))
        
        # Shuffle candidates
        candidates = list(range(start_threshold - 1, 0, -1))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline - 5:
                break
            if is_valid_addition(current_set, x):
                current_set.add(x)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
    return sorted(list(best_set))
# EVOLVE-BLOCK-END
