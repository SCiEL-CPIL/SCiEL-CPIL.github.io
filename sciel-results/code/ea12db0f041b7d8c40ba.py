import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin for 120s limit
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return n == 2 or n == 3
        # Deterministic Miller-Rabin for 64-bit
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        # Witnesses sufficient for n < 3.4 * 10^14
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(2, n - 1)
            c = random.randint(1, n - 1)
            m = random.randint(1, n - 1)
            g, r, q = 1, 1, 1
            while g == 1:
                x = y
                for _ in range(r):
                    y = (y * y + c) % n
                k = 0
                while k < r and g == 1:
                    ys = y
                    for _ in range(min(m, r - k)):
                        y = (y * y + c) % n
                        q = (q * abs(x - y)) % n
                    g = math.gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                while True:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
                    if g > 1:
                        break
            if g != n:
                return g

    def largest_prime_factor(n):
        if n <= 1:
            return 1
        if is_prime(n):
            return n
        factors = []
        while n > 1:
            if is_prime(n):
                factors.append(n)
                break
            d = pollard_brent(n)
            if d == n:
                continue
            factors.append(d)
            n //= d
        return max(factors)

    # Pre-compute small primes up to 100000 for trial division
    limit = 100000
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    small_primes = [i for i, is_p in enumerate(sieve) if is_p]
    
    # Segmented sieve to find primes in [low, high]
    def primes_in_range(low, high):
        if high < 2:
            return []
        if low < 2:
            low = 2
        size = high - low + 1
        is_prime_seg = [True] * size
        for p in small_primes:
            if p * p > high:
                break
            start = max(p * p, ((low + p - 1) // p) * p)
            for j in range(start, high + 1, p):
                is_prime_seg[j - low] = False
        result = []
        for i in range(size):
            if is_prime_seg[i]:
                result.append(low + i)
        return result

    # Check if a number is p-smooth using segmented sieve approach
    # First, try to factor using small primes
    # If remainder is 1, it's smooth
    # If remainder is prime <= p, it's smooth
    # If remainder is composite, use Pollard-Brent to verify
    
    def is_p_smooth(n, p):
        """Check if n has all prime factors <= p"""
        if n <= 1:
            return True
        if p < 2:
            return n <= 1
        
        # Trial divide by small primes up to min(p, 100000)
        rem = n
        for sp in small_primes:
            if sp > p:
                break
            if rem % sp == 0:
                while rem % sp == 0:
                    rem //= sp
                if rem == 1:
                    return True
        
        if rem == 1:
            return True
        
        # rem is now composed of primes > 100000 (or > p if p < 100000)
        # If rem is prime, check if rem <= p
        if is_prime(rem):
            return rem <= p
        
        # rem is composite. Find its largest prime factor.
        # If largest prime factor <= p, then smooth.
        return largest_prime_factor(rem) <= p

    best_k = 0
    best_p = 2
    
    # Strategy: Use segmented sieve to rapidly generate candidate primes
    # For each candidate prime p, check the run length using is_p_smooth
    
    # We'll search in segments to be efficient
    # Start with medium-sized primes, then expand
    
    # Phase 1: Search primes in [10^5, 10^7] - good density
    # Phase 2: Search primes in [10^7, 10^9] - sparser but larger p
    
    # For each prime, check k from 1 upward until failure
    # The key insight: most primes will fail at k=1 or k=2
    # So we can quickly eliminate them
    
    # Use a time budget per phase
    phase1_deadline = start_time + 40.0
    phase2_deadline = DEADLINE
    
    # Phase 1: [10^5, 10^7]
    while time.time() < phase1_deadline:
        # Pick a random segment in [10^5, 10^7]
        seg_low = random.randint(10**5, 10**7 - 1000)
        seg_high = min(seg_low + 1000, 10**7)
        
        primes = primes_in_range(seg_low, seg_high)
        
        for p in primes:
            if time.time() >= phase1_deadline:
                break
            
            p2 = p * p
            k = 0
            while k < 200:
                val = p2 + k + 1
                if not is_p_smooth(val, p):
                    break
                k += 1
            
            if k > best_k:
                best_k = k
                best_p = p
    
    # Phase 2: [10^7, 10^9]
    while time.time() < phase2_deadline:
        # Pick a random segment in [10^7, 10^9]
        # Larger segments for larger range
        seg_size = random.randint(5000, 20000)
        seg_low = random.randint(10**7, 10**9 - seg_size)
        seg_high = min(seg_low + seg_size, 10**9)
        
        primes = primes_in_range(seg_low, seg_high)
        
        for p in primes:
            if time.time() >= phase2_deadline:
                break
            
            p2 = p * p
            k = 0
            while k < 200:
                val = p2 + k + 1
                if not is_p_smooth(val, p):
                    break
                k += 1
            
            if k > best_k:
                best_k = k
                best_p = p
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
