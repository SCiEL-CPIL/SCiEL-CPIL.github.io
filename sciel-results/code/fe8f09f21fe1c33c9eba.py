def construct():
    import math
    
    limit = 10**18
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    best_n = 1
    best_divs = 1
    
    def search(idx, current_n, current_divs, last_exp):
        nonlocal best_n, best_divs
        
        if current_divs > best_divs:
            best_divs = current_divs
            best_n = current_n
            
        if idx >= len(primes):
            return
            
        p = primes[idx]
        # Try exponents from last_exp down to 1
        # We must ensure current_n * p^e <= limit
        # Also, to keep the number of divisors high, we generally want
        # exponents to be non-increasing.
        
        e = 0
        val = 1
        while True:
            next_n = current_n * val
            if next_n > limit:
                break
            
            next_divs = current_divs * (e + 1)
            
            # Pruning: if even with max possible divisors from remaining primes
            # we can't beat best_divs, we can stop?
            # Simple pruning: if next_divs * (some upper bound) < best_divs
            
            search(idx + 1, next_n, next_divs, e)
            
            e += 1
            val *= p
            
            # Optimization: if e > last_exp, we should stop?
            # No, we want to explore all valid e <= last_exp.
            # But if e > last_exp, the sequence of exponents is not non-increasing.
            # Standard HCN generation requires non-increasing exponents.
            if e > last_exp:
                break
                
    # Initial call
    # We need to set an initial last_exp high enough
    search(0, 1, 1, 100)
    
    return best_n
