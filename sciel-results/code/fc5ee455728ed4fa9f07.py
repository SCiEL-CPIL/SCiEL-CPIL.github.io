"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math

# EVOLVE-BLOCK-START
def construct():
    """
    Find n in [10000, 1000000] minimizing the largest prime factor of n*(n+1).
    
    Strategy:
    1. Generate small primes up to a threshold P.
    2. Use a segmented sieve to identify numbers in [10000, 1000000] that are P-smooth
       (all prime factors <= P).
    3. Find consecutive pairs (n, n+1) where both are P-smooth.
    4. Among these, compute the exact largest prime factor of n*(n+1) via trial division
       and return the n with the minimum such value.
    """
    LO = 10000
    HI = 1000000
    
    # Step 1: Generate small primes up to threshold P
    # We want P large enough to catch smooth numbers but small enough for efficiency
    P = 500
    
    # Sieve of Eratosthenes for primes up to P
    def sieve_primes(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i*i, limit+1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit+1) if is_prime[i]]
    
    small_primes = sieve_primes(P)
    
    # Step 2: Segmented sieve to find P-smooth numbers in [LO, HI]
    # A number is P-smooth if it has no prime factor > P
    # We use a segmented approach: for each segment, we maintain a count of how many
    # times each number can be divided by primes <= P. If after all divisions the
    # remainder is 1, the number is P-smooth.
    
    # For efficiency, we'll work in segments
    segment_size = 50000
    
    # We need to check if a number has any prime factor > P
    # Equivalent: after dividing out all prime factors <= P, if remainder is 1, it's P-smooth
    
    # Precompute for each prime, its multiples in each segment
    # For a segment [seg_start, seg_end], for each prime p <= P:
    #   Mark all multiples of p in this segment as "divisible by p"
    #   But we need to track the maximum power of p dividing each number
    
    # Better approach: for each number in the segment, divide out all primes <= P
    # and check if remainder is 1. But do this efficiently using the fact that
    # most numbers will have a large prime factor early.
    
    # Use a different strategy: 
    # For each prime p <= P, precompute which numbers in [LO, HI] are NOT divisible by p
    # A number is P-smooth iff for every prime p > P, the number is not divisible by p
    # But that's equivalent to: after removing all prime factors <= P, remainder is 1
    
    # Efficient segmented approach:
    # For each segment, create an array of remainders initialized to the number itself
    # For each prime p <= P, divide out all factors of p from each number in the segment
    # If any number still has a factor > P, its remainder will be > 1
    
    # But this is still O(segment_size * num_primes) which might be slow
    
    # Alternative: Use the property that if n is P-smooth, then n has no prime factor > P
    # We can check: for each prime q > P up to sqrt(HI), if q divides n, then n is not P-smooth
    # But there are many such primes
    
    # Best approach for this problem: 
    # 1. For each number n in [LO, HI], compute the largest prime factor by trial division
    # 2. But do it smartly: precompute smallest prime factor (SPF) up to HI using a sieve
    # 3. Then for each n, we can quickly factorize using SPF
    # 4. Find consecutive pairs where max(lpf(n), lpf(n+1)) is minimized
    
    # Let's use SPF sieve approach
    max_val = HI + 1
    spf = list(range(max_val))  # smallest prime factor
    
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, max_val, i):
                if spf[j] == j:
                    spf[j] = i
    
    def largest_prime_factor_spf(x, spf_arr):
        """Compute largest prime factor using precomputed SPF."""
        if x <= 1:
            return 1
        lpf = 1
        while x > 1:
            p = spf_arr[x]
            lpf = p
            while x % p == 0:
                x //= p
        return lpf
    
    # Find all n in [LO, HI] such that max(lpf(n), lpf(n+1)) is minimized
    # We can do a single pass
    best_lpf = float('inf')
    best_n = LO
    
    for n in range(LO, HI):
        lpf_n = largest_prime_factor_spf(n, spf)
        if lpf_n >= best_lpf:
            # Even if lpf(n+1) is 1, max would be >= lpf_n >= best_lpf
            # But we need to check lpf(n+1) too since it could be smaller
            pass
        
        lpf_n1 = largest_prime_factor_spf(n + 1, spf)
        max_lpf = max(lpf_n, lpf_n1)
        
        if max_lpf < best_lpf:
            best_lpf = max_lpf
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
