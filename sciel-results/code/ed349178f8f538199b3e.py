import numpy as np
from collections import defaultdict

def construct():
    n = 512
    max_size = 48
    
    # Precompute pair conflicts: x is the mean of {a, b} iff a+b = 2x
    # For a given x, the conflicting pairs (a,b) are those with a+b = 2x.
    # We'll precompute for each x, the list of pairs that conflict with it.
    # Actually, we need: for each x, and for each pair (a,b) in the set, 
    # check if a+b == 2x. But we also need higher-order subsets.
    
    # The condition is: no element x in S is the arithmetic mean of any subset T of S\{x} with |T| >= 2.
    # x = (sum(T))/|T|  <=>  sum(T) = x * |T|
    
    # We use a greedy approach with a conflict graph built incrementally,
    # but since the conflict depends on subset sums, we can't precompute all edges simply.
    # Instead, we use a smarter ordering and fast validation.
    
    # Key insight: if we pick numbers that are "far apart" or "extreme", 
    # the subset sums are less likely to hit x*k.
    
    # Strategy: 
    # 1. Try multiple heuristic orderings.
    # 2. Use fast bitset DP for validation.
    # 3. Add a local search / swap improvement step.
    
    def is_valid_set(selected, candidate):
        """Check if adding candidate to selected keeps the set valid."""
        m = len(selected)
        if m < 2:
            # Check if any existing element is mean of subset including candidate
            # For m=0 or 1, the only possible subset of OTHER elements including candidate
            # would need at least 2 elements total.
            # If m=0: no other elements, so no subset of size >=2.
            # If m=1: subset of size >=2 from {candidate} U {other} requires both, 
            #          but then the mean is (c+o)/2. We need to check if c or o is the mean.
            #          But the subset must be of OTHER elements relative to the candidate being checked.
            #          For candidate c: subset of selected (size 1) of size >=2? No.
            #          For existing o: subset of (selected\{o}) U {c} = {c} of size >=2? No.
            # So m<2 is always valid.
            return True
        
        # 1. Check if candidate is the mean of any subset of selected (size >= 2)
        # dp[j] is a bitset where bit s is 1 if sum s is achievable with j elements
        dp = [0] * (m + 1)
        dp[0] = 1
        
        for x in selected:
            for j in range(m - 1, -1, -1):
                if dp[j]:
                    dp[j + 1] |= (dp[j] << x)
        
        for k in range(2, m + 1):
            target = k * candidate
            if (dp[k] >> target) & 1:
                return False
        
        # 2. Check if any existing element 'a' in selected becomes the mean of a subset
        #    of (selected \ {a}) U {candidate} that includes candidate.
        for a in selected:
            others = [x for x in selected if x != a]
            mo = len(others)
            if mo < 1:
                continue
                
            dp2 = [0] * (mo + 1)
            dp2[0] = 1
            
            for x in others:
                for j in range(mo - 1, -1, -1):
                    if dp2[j]:
                        dp2[j + 1] |= (dp2[j] << x)
            
            # We need subset T' of others such that:
            # c + sum(T') = (1 + |T'|) * a
            # sum(T') = (1 + k) * a - c  where k = |T'|
            # k ranges from 1 to mo
            for k in range(1, mo + 1):
                target_sum = (k + 1) * a - candidate
                if target_sum < 0:
                    continue
                if target_sum > 512 * k:  # max possible sum
                    continue
                if (dp2[k] >> target_sum) & 1:
                    return False
        
        return True
    
    # Generate diverse orderings
    orders = []
    
    # 1. Descending (extremes first)
    orders.append(list(range(n, 0, -1)))
    
    # 2. Ascending
    orders.append(list(range(1, n + 1)))
    
    # 3. Extremes-in: 1, n, 2, n-1, 3, n-2, ...
    ext_in = []
    lo, hi = 1, n
    while lo <= hi:
        ext_in.append(lo)
        if lo != hi:
            ext_in.append(hi)
        lo += 1
        hi -= 1
    orders.append(ext_in)
    
    # 4. Descending odds, then descending evens
    odds_desc = list(range(n if n % 2 == 1 else n - 1, 0, -2))
    evens_desc = list(range(n if n % 2 == 0 else n - 1, 0, -2))
    orders.append(odds_desc + evens_desc)
    
    # 5. Ascending odds, then ascending evens
    odds_asc = list(range(1, n + 1, 2))
    evens_asc = list(range(2, n + 1, 2))
    orders.append(odds_asc + evens_asc)
    
    # 6. High half then low half
    orders.append(list(range(n // 2 + 1, n + 1)) + list(range(1, n // 2 + 1)))
    
    # 7. Low half then high half
    orders.append(list(range(1, n // 2 + 1)) + list(range(n // 2 + 1, n + 1)))
    
    # 8. Spaced: pick every k-th number
    for k in [3, 5, 7]:
        spaced = list(range(1, n + 1, k))
        if len(spaced) < max_size:
            # Fill in gaps from the end
            remaining = [x for x in range(n, 0, -1) if x not in set(spaced)]
            spaced = spaced + remaining[:max_size - len(spaced)]
        orders.append(spaced)
    
    best_set = []
    
    for order in orders:
        selected = []
        for x in order:
            if len(selected) >= max_size:
                break
            if is_valid_set(selected, x):
                selected.append(x)
        
        if len(selected) > len(best_set):
            best_set = selected[:]
            if len(best_set) >= max_size:
                break
    
    # Local search: try to improve by swapping
    if len(best_set) < max_size:
        improved = True
        while improved:
            improved = False
            best_set_copy = best_set[:]
            for i in range(len(best_set_copy)):
                for j in range(i + 1, len(best_set_copy)):
                    # Try removing best_set_copy[i] and best_set_copy[j]
                    # and see if we can add two new elements
                    remaining = [x for x in best_set_copy if x != best_set_copy[i] and x != best_set_copy[j]]
                    candidates = [x for x in range(1, n + 1) if x not in set(remaining) and x not in (best_set_copy[i], best_set_copy[j])]
                    
                    # Try to add two candidates
                    found = False
                    for c1 in candidates:
                        if len(remaining) + 1 >= max_size:
                            break
                        if is_valid_set(remaining, c1):
                            new_remaining = remaining + [c1]
                            for c2 in candidates:
                                if c2 == c1:
                                    continue
                                if len(new_remaining) + 1 >= max_size:
                                    break
                                if is_valid_set(new_remaining, c2):
                                    new_set = new_remaining + [c2]
                                    if len(new_set) > len(best_set_copy):
                                        best_set_copy = new_set
                                        found = True
                                        break
                        if found:
                            break
                    if found:
                        best_set = best_set_copy
                        improved = True
                        break
                if improved:
                    break
    
    return best_set[:max_size]
