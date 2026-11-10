"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    N = 123457
    # Sieve to compute largest prime factor for all integers up to N
    lpf = [0] * (N + 1)
    lpf[1] = 1
    
    # For each prime, mark its multiples with the prime as a candidate for largest prime factor
    # We need to be careful: we want the LARGEST prime factor.
    # Approach: initialize lpf[i] = 0. For each prime p, for each multiple m of p, set lpf[m] = p.
    # Since we iterate primes in increasing order, the last prime that divides m will be the largest.
    # This works if we iterate primes in increasing order.
    
    # Generate primes up to N using a simple sieve
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    
    # Now for each prime, update lpf for its multiples
    for p in range(2, N + 1):
        if is_prime[p]:
            for m in range(p, N + 1, p):
                lpf[m] = p
    
    # Now lpf[i] holds the largest prime factor of i for i >= 2, and lpf[1] = 1
    # Verify: lpf[123457] should be 123457 if it's prime, or its largest prime factor
    
    # Iterate through all splits a + b = N, where a and b are positive integers
    # a ranges from 1 to N-1, b = N - a
    best_max_p = float('inf')
    best_pair = (1, N - 1)
    
    for a in range(1, N):
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        current_max = max(pa, pb)
        if current_max < best_max_p:
            best_max_p = current_max
            best_pair = (a, b)
            # Early termination: if best_max_p is 1, that's the minimum possible
            if best_max_p == 1:
                break
            # If best_max_p is 2, the only way to get lower is 1, which only happens for 1
            # But since a >= 1 and b >= 1, and N is large, it's unlikely to get max=1
            # Actually P(1)=1, so if a=1 or b=1, one of them has P=1
            # But the other would be N-1, which has a large prime factor
            # So we can't early terminate on 2 easily
    
    return [best_pair[0], best_pair[1]]
# EVOLVE-BLOCK-END
