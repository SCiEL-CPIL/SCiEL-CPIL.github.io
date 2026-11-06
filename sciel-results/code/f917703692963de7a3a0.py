import time

def construct():
    N = 123457
    
    # Precompute largest prime factor for all numbers up to N
    # lpf[i] will store the largest prime factor of i
    lpf = list(range(N + 1))
    
    # Sieve of Eratosthenes variant to find largest prime factor
    # We iterate through potential primes p
    # If lpf[p] == p, then p is prime
    for p in range(2, int(N**0.5) + 1):
        if lpf[p] == p:
            # p is prime, update all multiples
            for j in range(p, N + 1, p):
                lpf[j] = p
                
    # Note: For primes greater than sqrt(N), lpf[p] remains p, which is correct.
    # For composite numbers, they will have been updated by their prime factors.
    # Does this correctly capture the LARGEST prime factor?
    # Example: 12.
    # p=2: lpf[12]=2
    # p=3: lpf[12]=3
    # p=5..: no change.
    # lpf[12]=3. Correct.
    
    # Example: 10.
    # p=2: lpf[10]=2
    # p=3..: no change.
    # lpf[10]=2. Correct.
    
    # Example: 6.
    # p=2: lpf[6]=2
    # p=3: lpf[6]=3
    # lpf[6]=3. Correct.
    
    # Example: 30.
    # p=2: lpf[30]=2
    # p=3: lpf[30]=3
    # p=5: lpf[30]=5
    # lpf[30]=5. Correct.
    
    # The loop goes up to sqrt(N). 
    # If a number n has a prime factor p > sqrt(N), then n/p < sqrt(N).
    # So n is a multiple of some q <= sqrt(N).
    # The largest prime factor is either <= sqrt(N) or is n itself (if n is prime) 
    # or a large prime factor paired with a small one.
    # Wait. If n = p * q with p < q and p <= sqrt(n) < q.
    # q > sqrt(N) is possible.
    # In the sieve above, we only iterate p up to sqrt(N).
    # If q > sqrt(N), we never use q to update lpf[q] (it stays q).
    # But for n = p*q, do we update lpf[n] to q?
    # No, because the loop `for p in range(2, int(N**0.5) + 1)` only runs for small p.
    # So lpf[n] will be updated to p (the small factor).
    # This is WRONG. We need the LARGEST prime factor.
    
    # Correction:
    # We need to ensure that if n has a large prime factor, we capture it.
    # Actually, if n is composite and not a prime, its largest prime factor is either
    # 1. <= sqrt(n) (if n is a product of small primes)
    # 2. > sqrt(n) (if n = p * q where p < q and q > sqrt(n), then p < sqrt(n))
    # In case 2, the large factor q is a prime > sqrt(N).
    # The sieve above only updates with p <= sqrt(N).
    # So lpf[n] will end up being the largest prime factor <= sqrt(N).
    # But the true LPF is q.
    
    # How to fix?
    # We can just compute LPF for each number individually? O(N sqrt(N)) might be too slow?
    # N=123457. sqrt(N) ~ 350.
    # 123457 * 350 ~ 4.3 * 10^7. In Python, this might be close to the limit but likely okay with optimization or PyPy.
    # However, we can do better.
    
    # Better Sieve Approach for LARGEST Prime Factor:
    # Initialize lpf = [0] * (N + 1)
    # For i from 2 to N:
    #   if lpf[i] == 0: # i is prime
    #      for j in range(i, N + 1, i):
    #          lpf[j] = i
    # This iterates through ALL primes up to N.
    # Complexity: N * (1/2 + 1/3 + 1/5 + ...) = N log log N.
    # This is efficient and correct. The last prime that divides j will be the largest.
    
    lpf = [0] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    # Now lpf[k] contains the largest prime factor of k.
    # lpf[1] is 0, but P(1)=1. We should handle 1 separately or set lpf[1]=1.
    lpf[1] = 1
    
    best_a = 1
    best_b = N - 1
    best_max_lpf = max(lpf[best_a], lpf[best_b])
    
    # Search for the pair that minimizes max(P(a), P(b))
    # We can iterate a from 1 to N//2
    for a in range(1, N // 2 + 1):
        b = N - a
        current_max = max(lpf[a], lpf[b])
        if current_max < best_max_lpf:
            best_max_lpf = current_max
            best_a = a
            best_b = b
            
    return [best_a, best_b]
