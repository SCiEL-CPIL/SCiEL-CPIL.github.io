"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    deadline = time.time() + 110.0  # safety margin before 150s kill
    
    # Precompute small primes up to 10^6 for trial division and sieve
    def sieve_primes(n):
        if n < 2:
            return []
        bs = bytearray(b'\x01') * (n + 1)
        bs[0] = bs[1] = 0
        for i in range(2, int(math.isqrt(n)) + 1):
            if bs[i]:
                bs[i*i::i] = bytearray(len(bs[i*i::i]))
        return [i for i in range(2, n + 1) if bs[i]]
    
    small_primes = sieve_primes(10**6)
    small_primes_set = set(small_primes)
    
    # Deterministic Miller-Rabin for n < 3.2e18
    def is_prime(n):
        if n < 2:
            return False
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return n == p
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = x * x % n
                if x == n - 1:
                    break
            else:
                return False
        return True
    
    # Pollard's rho with Brent's improvement
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        while True:
            c = random.randrange(1, n - 1)
            y = random.randrange(0, n - 1)
            m = 128
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
                        q = q * abs(x - y) % n
                    g = math.gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
            if g != n:
                return g
    
    def is_p_smooth(n, p):
        """Check if all prime factors of n are <= p using trial division + Pollard-Brent"""
        if n <= 1:
            return True
        if is_prime(n):
            return n <= p
        
        # Trial division by small primes
        for sp in small_primes:
            if sp * sp > n:
                break
            if n % sp == 0:
                if sp > p:
                    return False
                while n % sp == 0:
                    n //= sp
                if n == 1:
                    return True
                if not is_p_smooth(n, p):
                    return False
                return True
        
        if n == 1:
            return True
            
        if is_prime(n):
            return n <= p
            
        # Use Pollard's Rho to find a factor
        d = pollard_brent(n)
        if d == n:
            return n <= p
            
        if not is_p_smooth(d, p):
            return False
        if not is_p_smooth(n // d, p):
            return False
        return True
    
    def find_smooth_run(p, max_k=200):
        """Find the longest run of p-smooth numbers starting at p^2+1"""
        base = p * p
        k = 0
        for i in range(1, max_k + 1):
            n = base + i
            if not is_p_smooth(n, p):
                break
            k = i
        return k
    
    def sieve_screen_run(p, max_k=200):
        """
        Use segmented sieve over [p^2+1, p^2+max_k] to quickly identify
        which numbers might be p-smooth. Returns list of (offset, number) 
        that survived the small-prime sieve AND have no prime factor > p
        from the sieve. Numbers that survive need only Pollard-Brent verification
        for large prime factors.
        
        Returns: list of offsets i (1-indexed) where p^2+i is potentially p-smooth,
                 or None if any number in the range has a prime factor > p
                 (meaning the run breaks at that point).
        """
        base = p * p
        low = base + 1
        high = base + max_k
        size = max_k
        
        # Create sieve array: 1 means "still potentially smooth"
        sieve_arr = bytearray(b'\x01') * size
        
        # For each prime q <= p, mark multiples of q in [low, high] as "covered"
        # For each prime q > p, if any multiple exists in [low, high], the run breaks
        # 
        # Strategy: For primes q > p, check if any multiple of q falls in [low, high].
        # If so, that number is NOT p-smooth, so the run breaks at the smallest such offset.
        
        # First, handle primes q > p: find the smallest offset i in [1, max_k] 
        # such that base + i is divisible by some prime q > p.
        
        # For efficiency, only check primes q where q <= high (obviously)
        # and q > p. The smallest such prime that divides any number in [low, high]
        # determines where the run breaks.
        
        # For each prime q > p, the multiples in [low, high] are:
        #   ceil(low/q) * q, ceil(low/q)*q + q, ... <= high
        # If any exist, the smallest offset is (ceil(low/q)*q - base)
        
        # We want the minimum over all primes q > p of the smallest offset.
        
        # But checking all primes q > p is expensive. Instead:
        # - For small primes q (up to some limit), we can check directly.
        # - For large primes q > sqrt(high), each q can divide at most one number in [low, high].
        #   So we can check: for each number n in [low, high], is n prime and n > p?
        #   Or does n have a prime factor > p?
        
        # Better approach: 
        # 1. Use trial division by primes <= p to factor out small factors.
        # 2. For each number, if the remaining cofactor is > 1 and > p, it's not p-smooth.
        # 3. If the remaining cofactor is a prime <= p, it's fine.
        # 4. If the remaining cofactor is composite, use Pollard-Brent.
        
        # Actually, the key insight: after dividing by all primes <= p, 
        # the remaining number must be 1 or a prime > p (since any composite 
        # remaining would have a prime factor > p... wait, no).
        
        # Let me think more carefully. After removing all prime factors <= p,
        # the remaining number m satisfies:
        # - m = 1: p-smooth ✓
        # - m is prime and m <= p: p-smooth ✓ (but we already removed all primes <= p)
        # - m is prime and m > p: NOT p-smooth ✗
        # - m is composite: all prime factors of m are > p, so NOT p-smooth ✗
        
        # So after trial division by all primes <= p, we just need to check if m == 1.
        # If m > 1, it's NOT p-smooth.
        
        # But trial division by ALL primes <= p is expensive for large p.
        
        # Alternative: use the segmented sieve differently.
        # For each prime q in [2, p], mark multiples of q in [low, high].
        # After processing all primes q <= p, any number still unmarked has no prime factor <= p.
        # Such a number is either 1 (impossible since n >= p^2+1 > 1) or has all prime factors > p.
        # So it's NOT p-smooth.
        
        # But we also need to handle: a number might have some prime factors <= p and some > p.
        # The sieve only tells us if it has NO prime factors <= p.
        
        # Correct approach:
        # For each prime q <= p, divide out q from all multiples in [low, high].
        # After processing all primes q <= p, if any number still has value > 1, 
        # it has a prime factor > p, so it's not p-smooth.
        
        # But this requires tracking the actual value, not just a boolean.
        
        # Let me use a different strategy:
        # For each number n = base + i, i in [1, max_k]:
        #   Divide by all primes q <= min(p, sqrt(n)) 
        #   If remaining > 1, check if it's prime and <= p, or use Pollard-Brent.
        
        # This is essentially what is_p_smooth does, but we can batch it.
        
        # For rapid screening, let's use this approach:
        # 1. For each prime q > p up to high, check if any multiple falls in [low, high].
        #    The smallest such multiple determines the break point.
        # 2. For primes q <= p, they don't cause breaks (they're allowed factors).
        
        # So the run breaks at the smallest i such that base+i has a prime factor > p.
        
        # To find this efficiently:
        # - For primes q > p where q <= high: find smallest multiple in [low, high].
        # - The answer is the minimum of these smallest multiples (as offsets).
        
        # But there are many primes q > p. However, for large q > high, no multiples exist.
        # For q in (p, high], we need to check.
        
        # Key optimization: if q > high, no multiples. If q > high/2, at most one multiple.
        # For q in (high/2, high], the multiple is either q itself (if q in [low, high]) or 2q (if 2q in [low, high]).
        
        # Let me just implement a direct check for the break point.
        
        # For each offset i from 1 to max_k:
        #   n = base + i
        #   Check if n is p-smooth using a fast method.
        
        # Fast check: 
        #   - If n is prime and n > p: not p-smooth.
        #   - If n has any prime factor > p: not p-smooth.
        #   - Otherwise: p-smooth.
        
        # To check if n has a prime factor > p:
        #   - Divide n by all primes <= p (or up to sqrt(n)).
        #   - If remaining > 1, it has a prime factor > p.
        
        # But dividing by all primes <= p is expensive. Instead:
        #   - Divide by primes up to min(p, sqrt(n)).
        #   - If remaining > 1, check if remaining is prime.
        #     - If prime and <= p: OK (but we divided up to sqrt(n), so if remaining > sqrt(n), 
        #       and remaining is prime, then remaining > p only if remaining > p).
        #     - Actually, after dividing by all primes <= min(p, sqrt(n)):
        #       - If remaining == 1: p-smooth ✓
        #       - If remaining is prime:
        #         - If remaining <= p: p-smooth ✓
        #         - If remaining > p: NOT p-smooth ✗
        #       - If remaining is composite:
        #         - All prime factors of remaining are > min(p, sqrt(n))
        #         - If min(p, sqrt(n)) = sqrt(n) (i.e., p >= sqrt(n)):
        #           - Prime factors of remaining are > sqrt(n), so remaining has at most one prime factor > sqrt(n)
        #           - Wait, this is getting complicated.
        
        # Let me just use a simpler approach for the sieve screen:
        # For each prime q > p, find if it divides any number in [low, high].
        # The smallest such q (or smallest multiple) determines the break.
        
        # But iterating over all primes q > p is expensive. Instead:
        # For each number n in [low, high], check if n has a prime factor > p.
        # This is equivalent to checking if n / (product of all prime factors <= p) > 1.
        
        # Let me just implement a fast p-smooth check that uses trial division 
        # up to a limit, then Pollard-Brent for the rest.
        
        # For the sieve screen, I'll use this heuristic:
        # 1. For each n in [low, high], divide by small primes (up to 10^6 or so).
        # 2. If remaining > 1, use is_prime to check if it's prime.
        #    - If prime and <= p: OK
        #    - If prime and > p: NOT p-smooth
        #    - If composite: use Pollard-Brent to find a factor, then recurse.
        
        # This is basically is_p_smooth but we can stop early if we find a non-smooth number.
        
        # Let me just implement find_smooth_run with early termination and 
        # a faster is_p_smooth that prioritizes small-prime trial division.
        
        # Actually, let me implement a dedicated fast check for the screening phase.
        
        # For each n in [low, high]:
        #   m = n
        #   for q in small_primes where q <= min(p, isqrt(m)):
        #       if m % q == 0:
        #           while m % q == 0: m //= q
        #   if m > 1:
        #       if is_prime(m):
        #           if m > p: return False
        #       else:
        #           # m is composite, all factors > min(p, isqrt(n))
        #           # Check if m has a factor > p
        #           d = pollard_brent(m)
        #           if d > p or m // d > p:
        #               return False
        #           # Recursively check d and m//d
        #           if not is_p_smooth(d, p) or not is_p_smooth(m // d, p):
        #               return False
        #   return True
        
        # This is essentially is_p_smooth. Let me just use it with early termination.
        
        # The key improvement from the directive: use a segmented sieve to quickly
        # identify numbers that CANNOT be p-smooth without full factorization.
        
        # Insight: if n has a prime factor q > p, then n is not p-smooth.
        # For the sieve, we can mark numbers that have a prime factor > p.
        
        # For each prime q > p:
        #   Mark all multiples of q in [low, high] as "not p-smooth".
        
        # But we can't iterate over all primes q > p. However:
        # - For q > high: no multiples in range.
        # - For q in (high/2, high]: at most one multiple (q itself if q in range).
        # - For q in (high/3, high/2]: at most two multiples.
        # - etc.
        
        # Total work: sum over q > p of (high/low ratio) ≈ (high/p) * (number of primes in (p, high])
        # This is still large.
        
        # Alternative: for each number n in [low, high], check if n has a prime factor > p
        # by checking if n / gcd(n, M) > 1 where M is the product of all primes <= p.
        # But M is astronomically large.
        
        # I think the practical approach is:
        # 1. Use trial division by primes up to min(p, 10^6) for each n.
        # 2. If remaining m > 1, check if m is prime using Miller-Rabin.
        #    - If prime and m > p: not p-smooth.
        #    - If prime and m <= p: p-smooth.
        #    - If composite: use Pollard-Brent.
        
        # This is what is_p_smooth already does. The "sieve screen" idea is to
        # batch the trial division across all n in the range.
        
        # Let me implement a batched version:
        # For each small prime q <= min(p, limit):
        #   For each multiple of q in [low, high]:
        #     Divide out q from that number.
        # After processing all small primes, for each n, if remaining > 1, 
        # do the prime/Pollard check.
        
        # This batches the trial division, which can be faster.
        
        # Let me implement this.
        
        limit = min(p, 10**6)  # Trial division limit
        # Initialize remaining values
        remaining = [base + i for i in range(1, max_k + 1)]
        
        # For each prime q <= limit, divide out q from multiples
        for q in small_primes:
            if q > limit:
                break
            if q > p:
                break
            # Find first multiple of q in [low, high]
            first = ((low + q - 1) // q) * q
            for n in range(first, high + 1, q):
                idx = n - low  # 0-indexed
                while remaining[idx] % q == 0:
                    remaining[idx] //= q
        
        # Now check each remaining value
        for i in range(1, max_k + 1):
            m = remaining[i - 1]
            if m > 1:
                if is_prime(m):
                    if m > p:
                        return i - 1  # Run breaks at offset i, so k = i-1
                else:
                    # m is composite, check if all factors <= p
                    if not is_p_smooth(m, p):
                        return i - 1
            # m == 1 or m is p-smooth, continue
        
        return max_k  # All numbers are p-smooth
    
    def find_smooth_run_optimized(p, max_k=200):
        """Optimized smooth run finder using batched trial division."""
        base = p * p
        limit = min(p, 10**6)
        
        # Initialize remaining values
        remaining = [base + i for i in range(1, max_k + 1)]
        
        # For each prime q <= limit and q <= p, divide out q from multiples
        for q in small_primes:
            if q > limit or q > p:
                break
            # Find first multiple of q in [base+1, base+max_k]
            first = ((base + 1 + q - 1) // q) * q
            if first > base + max_k:
                continue
            for n in range(first, base + max_k + 1, q):
                idx = n - base - 1  # 0-indexed
                while remaining[idx] % q == 0:
                    remaining[idx] //= q
        
        # Now check each remaining value
        for i in range(1, max_k + 1):
            m = remaining[i - 1]
            if m > 1:
                if is_prime(m):
                    if m > p:
                        return i - 1
                else:
                    if not is_p_smooth(m, p):
                        return i - 1
        
        return max_k
    
    # Generate candidate primes and test them
    best_k = 0
    best_p = 2
    
    # 1. Test small primes first
    for p in small_primes:
        if time.time() > deadline - 5:
            break
        k = find_smooth_run_optimized(p)
        if k > best_k:
            best_k = k
            best_p = p
    
    # 2. Search larger primes using segmented sieve
    def segmented_sieve(low, high):
        """Generate primes in [low, high]"""
        if high < 2:
            return []
        limit = int(math.isqrt(high)) + 1
        if limit <= len(small_primes) and small_primes[-1] >= limit:
            small = [p for p in small_primes if p <= limit]
        else:
            small = sieve_primes(limit)
            
        sieve = bytearray(b'\x01') * (high - low + 1)
        for p in small:
            if p * p > high:
                break
            start = max(p * p, ((low + p - 1) // p) * p)
            for j in range(start, high + 1, p):
                sieve[j - low] = 0
        primes = []
        for i in range(low, high + 1):
            if sieve[i - low] and i >= 2:
                primes.append(i)
        return primes
    
    # Search ranges with adaptive step
    current_low = 10**6
    current_high = 10**7
    
    while current_high <= 10**9 and time.time() < deadline - 10:
        candidates = segmented_sieve(current_low, current_high)
        
        for p in candidates:
            if time.time() > deadline - 5:
                break
            
            # Quick pre-check: test first 3 numbers with fast method
            base = p * p
            # Fast check for base+1
            m1 = base + 1
            trial_limit = min(p, 10**5)
            for q in small_primes:
                if q > trial_limit:
                    break
                while m1 % q == 0:
                    m1 //= q
            if m1 > 1 and is_prime(m1) and m1 > p:
                continue
            
            # Fast check for base+2
            m2 = base + 2
            for q in small_primes:
                if q > trial_limit:
                    break
                while m2 % q == 0:
                    m2 //= q
            if m2 > 1 and is_prime(m2) and m2 > p:
                continue
            
            # Fast check for base+3
            m3 = base + 3
            for q in small_primes:
                if q > trial_limit:
                    break
                while m3 % q == 0:
                    m3 //= q
            if m3 > 1 and is_prime(m3) and m3 > p:
                continue
            
            # If pre-check passes, do full check
            k = find_smooth_run_optimized(p)
            if k > best_k:
                best_k = k
                best_p = p
        
        # Increase range
        current_low = current_high
        current_high = min(current_high * 10, 10**9)
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
