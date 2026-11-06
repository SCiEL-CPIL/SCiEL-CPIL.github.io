"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    """
    Find n in [10000, 1000000] minimizing the largest prime factor of n*(n+1).
    
    Strategy:
    1. Use a segmented sieve to track, for each number in [LO, HI], the largest
       prime factor that divides it (from primes up to P).
    2. A number is P-smooth if after removing all prime factors <= P, remainder is 1.
    3. Find consecutive pairs (n, n+1) where both are P-smooth.
    4. Among these, compute the exact largest prime factor of n*(n+1) via trial division
       and return the n with the minimum such value.
    """
    LO = 10000
    HI = 1000000
    
    # Sieve of Eratosthenes for primes up to a threshold
    def sieve_primes(limit):
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    # We'll use a segmented approach to find P-smooth numbers
    # For each segment, we track the "remaining" value after dividing out small primes
    # If remaining == 1, the number is P-smooth
    
    P = 500  # Threshold for "small" primes
    
    small_primes = sieve_primes(P)
    
    # We'll process in segments. For each number in a segment, we'll compute
    # the result of dividing out all prime factors <= P.
    # If the result is 1, the number is P-smooth.
    
    # To do this efficiently, for each prime p <= P, we'll mark all multiples
    # in the segment and divide them by p as many times as possible.
    
    segment_size = 200000
    candidates = []  # (lpf_product, n)
    
    def largest_prime_factor(x, primes):
        """Compute largest prime factor of x using trial division with primes list."""
        if x <= 1:
            return 1
        lpf = 1
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                lpf = p
                while x % p == 0:
                    x //= p
        if x > 1:
            lpf = max(lpf, x)
        return lpf
    
    for seg_start in range(LO, HI + 1, segment_size):
        seg_end = min(seg_start + segment_size - 1, HI)
        seg_len = seg_end - seg_start + 1
        
        # remaining[i] = (seg_start + i) with all prime factors <= P removed
        remaining = list(range(seg_start, seg_end + 1))
        
        for p in small_primes:
            # Find the first multiple of p in [seg_start, seg_end]
            first_mult = ((seg_start + p - 1) // p) * p
            if first_mult < seg_start:
                first_mult += p
            
            for m in range(first_mult, seg_end + 1, p):
                idx = m - seg_start
                while remaining[idx] % p == 0:
                    remaining[idx] //= p
        
        # Now find consecutive pairs where both are P-smooth
        for i in range(seg_len - 1):
            if remaining[i] == 1 and remaining[i + 1] == 1:
                n = seg_start + i
                # Compute exact largest prime factor of n and n+1
                lpf_n = largest_prime_factor(n, small_primes)
                lpf_n1 = largest_prime_factor(n + 1, small_primes)
                lpf_product = max(lpf_n, lpf_n1)
                candidates.append((lpf_product, n))
        
        # Also check the last element of the segment paired with the first of next
        # (handled by next segment's first element check)
    
    # Handle the boundary: if seg_end < HI, check if seg_end and seg_end+1 are both smooth
    # This is already handled since we check i in range(seg_len-1), so the last pair
    # in each segment is (seg_end-1, seg_end). The pair (seg_end, seg_end+1) would be
    # checked in the next segment as (first, second).
    
    if not candidates:
        return 10000
    
    candidates.sort()
    return candidates[0][1]
# EVOLVE-BLOCK-END
