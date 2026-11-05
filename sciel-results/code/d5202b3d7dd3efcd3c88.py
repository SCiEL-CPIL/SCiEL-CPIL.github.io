import time

def construct():
    N = 123457
    deadline = time.time() + 80  # safety margin before 90s
    
    # Precompute largest prime factor for all numbers up to N-1
    # largest_prime_factor[i] = largest prime factor of i, or 1 if i=1
    lpf = [1] * N  # indices 0 to N-1, we use 1 to N-1
    
    # Sieve: for each prime p, mark all multiples
    # We need primes up to N-1
    is_prime = [True] * N
    is_prime[0] = False
    is_prime[1] = False
    
    primes = []
    for i in range(2, N):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, N, i):
                is_prime[j] = False
    
    # For each prime, update lpf for its multiples
    for p in primes:
        for m in range(p, N, p):
            if p > lpf[m]:
                lpf[m] = p
    
    # Now scan for best a
    best_score = float('inf')
    best_a = 1
    for a in range(1, N):
        b = N - a
        score = max(lpf[a], lpf[b])
        if score < best_score:
            best_score = score
            best_a = a
            if best_score <= 2:
                # Can't do better than 2 (both powers of 2), but sum is odd so impossible
                # Actually min possible is 2 if both are powers of 2, but sum must be odd
                # So one must have an odd prime factor. Best is 3.
                pass
    
    if time.time() > deadline:
        return [best_a, N - best_a]
    
    return [best_a, N - best_a]
