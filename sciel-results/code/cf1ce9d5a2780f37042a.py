def construct():
    # We want to find n in [10000, 1000000] that minimizes the largest prime factor of n(n+1).
    # The largest prime factor of a product is the maximum of the largest prime factors of the factors.
    # So we need to minimize max(lpf(n), lpf(n+1)).
    
    max_n = 1000000
    min_n = 10000
    
    # Sieve for largest prime factor (lpf) for all numbers up to max_n.
    # lpf[i] will store the largest prime factor of i.
    # Initialize with 0.
    lpf = [0] * (max_n + 1)
    
    # Iterate through numbers to populate lpf.
    # Standard sieve variant: for each prime p, set lpf[multiple] = p.
    # Since we iterate p in increasing order, the last prime that divides a number is its largest prime factor.
    # Wait, if we just overwrite, we need to ensure we process primes in increasing order.
    # Yes, the outer loop i goes from 2 to max_n. If lpf[i] == 0, i is prime.
    # Then for all multiples j of i, we set lpf[j] = i.
    # Since i is increasing, the final value assigned to lpf[j] will be the largest prime factor of j.
    
    for i in range(2, max_n + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, max_n + 1, i):
                lpf[j] = i
    
    # Now find n in [min_n, max_n - 1] that minimizes max(lpf[n], lpf[n+1])
    best_n = min_n
    best_score = float('inf')
    
    for n in range(min_n, max_n):
        score = max(lpf[n], lpf[n+1])
        if score < best_score:
            best_score = score
            best_n = n
    
    return best_n
