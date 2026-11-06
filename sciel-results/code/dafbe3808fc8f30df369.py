"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math

# EVOLVE-BLOCK-START
def construct():
    """
    Find n in [10000, 1000000] minimizing the largest prime factor of n*(n+1).
    
    Strategy:
    1. Generate small primes up to a threshold P.
    2. Use a segmented sieve to identify numbers in [10000, 1000000] that are P-smooth
       (all prime factors <= P) by tracking remaining factors after dividing out small primes.
    3. Find consecutive pairs (n, n+1) where both are P-smooth.
    4. Among these, compute the exact largest prime factor of n*(n+1) via trial division
       and return the n with the minimum such value.
    """
    LO = 10000
    HI = 1000000
    
    # Step 1: Generate small primes up to threshold P
    P = 200
    
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
    # For each number, we track the remaining factor after dividing out all primes <= P.
    # If remaining == 1, the number is P-smooth.
    
    # We'll process the entire range [LO, HI+1] (need HI+1 for n+1 when n=HI)
    # Use an array to store the remaining factor for each number
    # To save memory, process in segments
    
    segment_size = 50000
    # Store smooth flags for the current segment
    # smooth_flags[i] = True if (seg_start + i) is P-smooth
    
    candidates = []
    
    # Helper to compute largest prime factor via trial division
    def largest_prime_factor(x, primes_up_to):
        if x <= 1:
            return 1
        lpf = 1
        d = 2
        while d * d <= x:
            if x % d == 0:
                lpf = d
                while x % d == 0:
                    x //= d
            d += 1 if d == 2 else 2
        if x > 1:
            lpf = x
        return lpf
    
    for seg_start in range(LO, HI + 1, segment_size):
        seg_end = min(seg_start + segment_size - 1, HI)
        seg_len = seg_end - seg_start + 1
        
        # remaining[i] stores the remaining factor for (seg_start + i)
        # Initialize to the number itself
        remaining = [0] * seg_len
        for i in range(seg_len):
            remaining[i] = seg_start + i
        
        # For each small prime p, divide it out of all multiples in the segment
        for p in small_primes:
            # Find the first multiple of p in [seg_start, seg_end]
            first_mult = ((seg_start + p - 1) // p) * p
            if first_mult < seg_start:
                first_mult += p
            # Iterate through multiples of p in the segment
            for mult in range(first_mult, seg_end + 1, p):
                idx = mult - seg_start
                # Divide out all factors of p
                while remaining[idx] % p == 0:
                    remaining[idx] //= p
        
        # Now check which numbers are P-smooth (remaining == 1)
        # and find consecutive pairs
        for i in range(seg_len - 1):  # i goes from 0 to seg_len-2, so n = seg_start+i
            n = seg_start + i
            n1 = n + 1
            if remaining[i] == 1 and remaining[i+1] == 1:
                # Both n and n+1 are P-smooth
                lpf_n = largest_prime_factor(n, small_primes)
                lpf_n1 = largest_prime_factor(n1, small_primes)
                lpf_product = max(lpf_n, lpf_n1)
                candidates.append((lpf_product, n))
        
        # Also check if seg_end and seg_end+1 are both smooth (cross-segment boundary)
        if seg_end < HI:
            n = seg_end
            n1 = seg_end + 1
            # n is at index seg_len-1 in current segment
            if remaining[seg_len-1] == 1:
                # Check if n1 is P-smooth - need to verify separately
                # Since n1 is in next segment, we'll handle it when that segment is processed
                # But we need to avoid double counting, so let's just check here
                temp = n1
                for p in small_primes:
                    if p * p > temp:
                        break
                    while temp % p == 0:
                        temp //= p
                if temp == 1:
                    lpf_n = largest_prime_factor(n, small_primes)
                    lpf_n1 = largest_prime_factor(n1, small_primes)
                    lpf_product = max(lpf_n, lpf_n1)
                    candidates.append((lpf_product, n))
    
    if not candidates:
        return 10000
    
    candidates.sort()
    return candidates[0][1]
# EVOLVE-BLOCK-END
