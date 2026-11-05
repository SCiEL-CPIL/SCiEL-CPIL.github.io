import time
import math

def construct():
    target = math.factorial(12)
    best_max = target
    best_factors = [target]
    
    # Deadline guard
    start_time = time.time()
    timeout = 50.0 # Safety margin
    
    def dfs(rem, min_f, count, factors):
        nonlocal best_max, best_factors
        
        if time.time() - start_time > timeout:
            return
            
        if rem == 1:
            if 1 <= count <= 12:
                current_max = max(factors)
                if current_max < best_max:
                    best_max = current_max
                    best_factors = factors.copy()
            return
            
        if count == 12:
            return
            
        # Pruning: if the smallest possible next factor is > best_max, 
        # we can't improve the max (since factors are increasing, 
        # the last one is the max). 
        # Actually, if min_f > best_max, any new factor will be > best_max.
        # So the max of the final set will be > best_max.
        if min_f > best_max:
            return

        # We need to find a factor f of rem such that f >= min_f and f <= best_max
        # and f divides rem.
        # Also, if we pick f, the remaining product is rem // f.
        
        # Optimization: 
        # If we pick f, and we want to minimize the max, 
        # we should probably pick smaller f first? 
        # But we want to find ANY solution with small max.
        # The max of the set is the largest element.
        # Since we pick in increasing order, the last element is the max.
        # So we want to keep the last element small.
        
        # Iterate f from min_f to min(rem, best_max)
        # But we can bound it tighter.
        # If we pick f, we need to decompose rem//f into distinct factors > f.
        # The smallest such factor is f+1.
        # So if we pick only one more factor, f * (f+1) <= rem is not required, 
        # but f * g = rem with g > f implies f^2 < rem.
        # So if f * f >= rem, we can only pick f if f == rem (and no more factors).
        # If f * f > rem, we cannot pick f unless f == rem.
        
        # Let's just iterate divisors.
        # Finding divisors of rem is fast.
        divisors = []
        i = 1
        while i * i <= rem:
            if rem % i == 0:
                if i >= min_f:
                    divisors.append(i)
                if i != rem // i and rem // i >= min_f:
                    divisors.append(rem // i)
            i += 1
        
        # Sort divisors to try smaller ones first? 
        # Or larger?
        # If we try smaller f, the remaining product is larger, 
        # allowing for more factors, potentially keeping the max lower?
        # Example: 12! = 13 * 14 * ...
        # If we pick a small f, we have more "room" for other factors.
        # If we pick a large f, the remaining product is small, 
        # forcing the next factors to be small (but > f), which might be impossible 
        # or force a large max later? 
        # Actually, if we pick a large f, the max is at least f.
        # Since we want to minimize the max, picking a large f is bad 
        # unless it leads to a very small remaining product that can be 
        # decomposed into factors < f? No, factors must be > f.
        # So if we pick f, all subsequent factors are > f.
        # So the max will be >= last factor > f.
        # So picking a large f immediately sets a high floor for the max.
        # Thus, we should try smaller f first to keep the floor low.
        
        divisors.sort()
        
        for f in divisors:
            if f > best_max:
                break
                
            # If we pick f, the remaining product is rem // f.
            # We need to decompose rem // f into distinct factors > f.
            # If rem // f == 1, we are done.
            # If rem // f > 1, we continue.
            
            # Pruning: 
            # If we pick f, the next factor must be > f.
            # The product of the remaining factors is rem // f.
            # If rem // f is not 1, we need at least one more factor.
            # The smallest next factor is f + 1.
            # So we must have f + 1 <= rem // f ?
            # i.e., f(f+1) <= rem ?
            # If f(f+1) > rem, then we cannot pick any factor > f 
            # that divides rem//f (since the smallest such factor is f+1, 
            # and f+1 > rem//f implies no divisor exists).
            # Exception: if rem//f == 1, we don't need more factors.
            
            if rem // f != 1:
                if f * (f + 1) > rem:
                    # No valid next factor can exist
                    continue
            
            dfs(rem // f, f + 1, count + 1, factors + [f])
            
    dfs(target, 13, 0, [])
    
    return best_factors
