import math
import time
from itertools import product

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin for 90s limit
    
    # 1. Calculate 12!
    target = math.factorial(12)  # 479001600
    
    # 2. Generate all divisors of target efficiently using prime factorization
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    divisors = set()
    for a in range(11):      # 2^a
        for b in range(6):   # 3^b
            for c in range(3): # 5^c
                for d in range(2): # 7^d
                    for e in range(2): # 11^e
                        val = (2**a) * (3**b) * (5**c) * (7**d) * (11**e)
                        divisors.add(val)
    
    # Filter: must be > 12
    valid_divs = sorted([d for d in divisors if d > 12])
    
    # Initial best solution: single factor [target]
    best_max = target
    best_factors = [target]
    
    # Precompute a mapping from divisor value to its index in valid_divs for quick lookup
    div_index = {d: i for i, d in enumerate(valid_divs)}
    
    # DFS to find a decomposition with max factor < current best_max
    def dfs(remaining, start_idx, current_factors, current_max):
        nonlocal best_max, best_factors
        
        if time.time() > deadline:
            return
        
        if remaining == 1:
            if len(current_factors) >= 1:
                if current_max < best_max:
                    best_max = current_max
                    best_factors = current_factors[:]
            return
        
        # We can have at most 12 factors
        if len(current_factors) >= 12:
            return
        
        # Pruning: if current_max is already >= best_max, no point continuing
        # Since we add factors in increasing order (due to sorted valid_divs and start_idx),
        # current_max is non-decreasing along the path.
        if current_max >= best_max:
            return
        
        # Pruning: if remaining is a valid divisor and we can use it as the last factor
        # and it's <= current_max, we could potentially get a better solution.
        # But more importantly, we can prune if the smallest possible product of 
        # remaining factors (at least 13 each, distinct) exceeds remaining.
        # Actually, a simpler and effective prune: 
        # If remaining itself is a divisor > 12 and remaining <= current_max, 
        # then we can just add it as the last factor. But we need to ensure distinctness.
        # Since we're picking in increasing order, if remaining is in valid_divs and 
        # its index >= start_idx, we can try it. But the loop below handles this.
        
        # Better prune: if the number of factors we still need to place (at least 1) 
        # times the smallest possible factor (13) exceeds remaining, we can't finish.
        # Actually, since factors are distinct and > 12, the smallest possible next factor 
        # is at least max(13, valid_divs[start_idx]) if start_idx < len(valid_divs).
        # But this is complex. Let's use a different approach.
        
        # Pruning based on the fact that all future factors must be > current last factor
        # and distinct. The minimum possible product of k distinct factors all > M is 
        # (M+1)(M+2)...(M+k). But this is hard to compute quickly.
        
        # Instead, let's just do the loop with good pruning.
        
        for i in range(start_idx, len(valid_divs)):
            d = valid_divs[i]
            
            # Since valid_divs is sorted, if d > remaining, no further d can divide remaining
            # (except if remaining is 1, which is handled above).
            if d > remaining:
                break
            
            if remaining % d != 0:
                continue
            
            new_max = max(current_max, d)
            
            # If new_max >= best_max, we can prune.
            # Since d is increasing, new_max will only stay same or increase.
            # So if new_max >= best_max, all subsequent d will also lead to new_max >= best_max.
            if new_max >= best_max:
                break
            
            current_factors.append(d)
            dfs(remaining // d, i + 1, current_factors, new_max)
            current_factors.pop()
            
            # If we found a better solution, best_max might have decreased.
            # The loop condition `if new_max >= best_max: break` will handle pruning for next iterations.
            
    dfs(target, 0, [], 0)
    
    return best_factors
