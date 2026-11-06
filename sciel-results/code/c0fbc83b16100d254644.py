"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    """
    Find n in [10000, 1000000] that minimizes the largest prime factor of n(n+1).
    
    Strategy: Use a segmented sieve to identify P-smooth numbers (composed of
    primes below threshold P), then verify candidate windows where consecutive
    integers are both smooth. This shifts cost from dense linear scan to sparse
    verification of high-smoothness clusters.
    """
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline with 5s safety margin
    
    N_MIN = 10000
    N_MAX = 1000000
    N_MAX_PLUS_1 = N_MAX + 1
    
    # Generate all primes up to sqrt(N_MAX_PLUS_1) for trial division
    limit = int(math.isqrt(N_MAX_PLUS_1)) + 1
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    def largest_prime_factor(num):
        """Compute largest prime factor via trial division."""
        if num <= 1:
            return 1
        lpf = 1
        for p in primes:
            if p * p > num:
                break
            while num % p == 0:
                lpf = p
                num //= p
        if num > 1:
            lpf = num
        return lpf
    
    # Phase 1: Use segmented sieve to find P-smooth numbers
    # A number is P-smooth if all its prime factors are <= P
    # We'll try different thresholds P and find clusters of consecutive smooth numbers
    
    best_n = N_MIN
    best_val = float('inf')
    
    # Try multiple smoothness thresholds
    for P in [50, 100, 200, 500, 1000, 2000, 5000]:
        if time.time() > deadline - 10:
            break
        
        # Generate all primes <= P
        if P <= limit:
            small_primes = [p for p in primes if p <= P]
        else:
            # Need primes up to P
            is_prime_p = [True] * (P + 1)
            is_prime_p[0] = is_prime_p[1] = False
            for i in range(2, int(math.isqrt(P)) + 1):
                if is_prime_p[i]:
                    for j in range(i*i, P + 1, i):
                        is_prime_p[j] = False
            small_primes = [i for i in range(2, P + 1) if is_prime_p[i]]
        
        # Segmented sieve: mark P-smooth numbers in [N_MIN, N_MAX_PLUS_1]
        # A number is P-smooth if after removing all small prime factors, remainder is 1
        smooth = [False] * (N_MAX_PLUS_1 - N_MIN + 2)  # indices 0..(N_MAX_PLUS_1-N_MIN+1)
        
        # For each position, check if it's P-smooth
        # Optimization: only need to check numbers that aren't obviously not smooth
        # Use a faster approach: for each small prime, mark multiples as "divisible"
        # Then a number is P-smooth iff it can be fully factored by small primes
        
        # Better approach: compute the "rough part" for each number
        # rough[i] = N_MIN + i after removing all factors of primes <= P
        # If rough[i] == 1, then N_MIN+i is P-smooth
        
        rough = list(range(N_MIN, N_MAX_PLUS_1 + 1))
        
        for p in small_primes:
            if time.time() > deadline - 5:
                break
            # Remove factor p from all multiples
            start_idx = (N_MIN + p - 1) // p * p - N_MIN  # first multiple of p >= N_MIN
            for idx in range(start_idx, len(rough), p):
                while rough[idx] % p == 0:
                    rough[idx] //= p
        
        # Now smooth[i] = (rough[i] == 1)
        # Find consecutive pairs where both are smooth
        for i in range(1, len(rough)):
            if time.time() > deadline - 2:
                break
            if rough[i] == 1 and rough[i-1] == 1:
                n = N_MIN + i - 1
                if N_MIN <= n <= N_MAX:
                    # Verify with exact trial division
                    val = max(largest_prime_factor(n), largest_prime_factor(n+1))
                    if val < best_val:
                        best_val = val
                        best_n = n
    
    # Phase 2: Also do a direct scan for any missed candidates
    # Focus on numbers that might have small LPF
    if time.time() < deadline - 5:
        # Quick scan: check candidates where n or n+1 is divisible by small powers
        for n in range(N_MIN, N_MAX + 1, 1):
            if time.time() > deadline - 2:
                break
            # Quick check: if n is even, skip odd n+1 check partially
            val = max(largest_prime_factor(n), largest_prime_factor(n+1))
            if val < best_val:
                best_val = val
                best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
