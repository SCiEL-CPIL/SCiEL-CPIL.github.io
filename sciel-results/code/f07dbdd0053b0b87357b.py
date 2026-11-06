import numpy as np

def construct():
    M = 1000000 + 25
    S = np.ones(M, dtype=np.int64)
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    for p in primes:
        # Get indices of multiples of p
        idx = np.arange(p, M, p)
        if len(idx) == 0:
            continue
        
        vals = np.arange(M)[idx]
        temp = vals.copy()
        power_part = np.ones_like(vals)
        
        # Extract p-power from each val
        while True:
            divisible = temp % p == 0
            if not np.any(divisible):
                break
            temp[divisible] = temp[divisible] // p
            power_part[divisible] = power_part[divisible] * p
            
        S[idx] = S[idx] * power_part

    best_n = 1000
    best_count = 25
    
    # We know min distinct is at least 2 because S(x) and S(x+1) have different parity
    # (one is even, one is odd, assuming x > 1).
    # Actually, if S(x)=1 (x has no small factors), it is odd.
    # If S(x+1)=1, it is odd.
    # Can S(x)=S(x+1)=1?
    # x and x+1 are consecutive. One is even.
    # If x is even, x is divisible by 2. So S(x) has factor 2. S(x) != 1.
    # If x+1 is even, x+1 is divisible by 2. So S(x+1) != 1.
    # So at least one of S(x), S(x+1) is even.
    # Thus they cannot be equal (one even, one odd).
    # So distinct count >= 2 for any window of size >= 2.
    
    for n in range(1000, 1000001):
        # Window is n+1 to n+24
        window = S[n+1:n+25]
        # Count distinct
        # Using set is fast enough for size 24
        count = len(set(window))
        if count < best_count:
            best_count = count
            best_n = n
            if best_count == 2:
                # Can't get better than 2
                break
                
    return best_n
