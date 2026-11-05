import time

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

def construct():
        start_time = time.time()
        deadline = start_time + 85.0
        
        # Range for n is [1000, 1000000]
        # We need factors for n+1 to n+24
        # Max n = 1000000 -> max val = 1000024
        # Min n = 1000 -> min val = 1001
        
        max_val = 1000000 + 24
        min_val = 1000 + 1
        
        # Precompute extracted factors
        # Array size max_val + 1
        # f[m] = extracted factor of m
        
        # Optimization: 
        # We can compute f[m] iteratively? 
        # f[m] is not simply related to f[m-1].
        # But we can just loop.
        
        # To speed up, use a list of zeros/ones?
        # Just a list of ints.
        
        extracted = [1] * (max_val + 1)
        
        # For each prime, we could update?
        # No, simpler to just factor each number.
        # But 10^6 numbers * 9 primes is fine.
        
        # Actually, we can be smarter.
        # For each prime p, we know which numbers are divisible by p.
        # But the exponent matters.
        
        # Let's just do the direct calculation.
        # It's O(N * log N) roughly.
        
        # Wait, 10^6 * 9 * (avg divs) is small.
        
        for m in range(min_val, max_val + 1):
            res = 1
            temp = m
            for p in PRIMES:
                while temp % p == 0:
                    res *= p
                    temp //= p
            extracted[m] = res
            
        # Now scan windows
        best_n = 1000
        best_count = 100
        
        for n in range(1000, 1000001):
            if time.time() > deadline:
                break
                
            # Get factors for n+1 ... n+24
            # We can use a set
            # To avoid creating a new set object every time, we can reuse?
            # No, set is cheap.
            
            # Optimization: 
            # Since window slides by 1, we can maintain a frequency map.
            # But 24 is small. Set creation is O(24).
            # 10^6 * 24 = 24 * 10^6 ops.
            
            window_factors = extracted[n+1 : n+25]
            distinct_count = len(set(window_factors))
            
            if distinct_count < best_count:
                best_count = distinct_count
                best_n = n
                
                if best_count == 2:
                    # Found theoretical minimum
                    return best_n
        
        return best_n
