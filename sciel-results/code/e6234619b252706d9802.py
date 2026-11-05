import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Sieve for small primes up to 10^6 for fast trial division and primality testing
    limit_small = 10**6
    sieve = [True] * (limit_small + 1)
    sieve[0:2] = [False, False]
    for i in range(2, int(math.isqrt(limit_small)) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    small_primes = [i for i, is_p in enumerate(sieve) if is_p]
    small_prime_set = set(small_primes)

    # 2. Deterministic Miller-Rabin for n < 2^64
    def is_prime_mr(n):
        if n < 2:
            return False
        # Check small primes first for speed
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n == p:
                return True
            if n % p == 0:
                return False
        
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        
        # Bases sufficient for n < 2^64
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

    # 3. Pollard's Rho / Brent's Algorithm for factorization
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        
        y, c, m = random.randint(1, n - 1), random.randint(1, n - 1), random.randint(1, n - 1)
        g, q, r, x, ys = 1, 1, 1, 0, y
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
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g > 1:
                    break
        return g

    def factor(n, factors=None):
        if factors is None:
            factors = []
        if n == 1:
            return factors
        if is_prime_mr(n):
            factors.append(n)
            return factors
        # Try small primes first
        for p in small_primes[:100]: # First 100 primes
            if n % p == 0:
                factors.append(p)
                return factor(n // p, factors)
        
        d = pollard_brent(n)
        if d == n:
            # Retry with different random seed if it failed
            return factor(n, factors)
        factor(d, factors)
        factor(n // d, factors)
        return factors

    def is_p_smooth(n, p):
        """Check if all prime factors of n are <= p."""
        if n <= p:
            return True
        # We need to check if the largest prime factor is <= p.
        # Factor n.
        # Optimization: If n is not too big, trial division might be faster for very small factors.
        # But Pollard Rho is generally good.
        
        # Quick check: divide by small primes up to p if p is small?
        # Actually, just factor and check max.
        
        # To avoid deep recursion or overhead, we can do a loop.
        temp = n
        # Divide by small primes first to reduce size for Pollard
        for q in small_primes:
            if q > p:
                break
            if q * q > temp:
                break
            while temp % q == 0:
                temp //= q
        
        if temp == 1:
            return True
        
        # If temp > 1, it has no prime factors <= min(p, sqrt(original_temp)) from the loop.
        # If p is large, temp might still have factors <= p.
        # We need to factor temp.
        # If temp is prime:
        if is_prime_mr(temp):
            return temp <= p
        
        # If temp is composite, it must have a factor.
        # If any factor > p, return False.
        # We can try to find one factor.
        d = pollard_brent(temp)
        if d == temp:
            # If Pollard fails, assume it's prime (unlikely for composite) or try again
            # For safety, if we can't factor, and temp > p, it's risky.
            # But if temp is composite and > p, it likely has a factor > p?
            # Not necessarily. e.g. p=100, temp=101*103. Both > p.
            # But if temp = 97*99 (97<=p, 99 composite).
            # Let's just fully factor temp to be sure.
            pass
        
        # Full factorization of remaining temp
        # Since temp has no small factors (from the loop above), it's likely a product of larger primes.
        # We just need to know if MAX factor <= p.
        
        def check_max_factor(m, limit):
            if m == 1:
                return True
            if is_prime_mr(m):
                return m <= limit
            # Find a factor
            d = pollard_brent(m)
            if d == m:
                # Retry or assume prime? If it's composite, Pollard should find a factor.
                # If it fails repeatedly, it's a pseudoprime or issue.
                # For now, let's assume if we can't split, we check primality again.
                return m <= limit
            return check_max_factor(d, limit) and check_max_factor(m // d, limit)

        return check_max_factor(temp, p)

    best_k = 0
    best_p = 2

    # Search Strategy:
    # We want to find a prime p such that p^2+1 ... p^2+k are all p-smooth.
    # Larger p generally allows for longer runs because the "density" of p-smooth numbers
    # increases as p grows (for a fixed range size relative to p^2? No, the condition is absolute).
    # Actually, for a random number N ~ p^2, the probability that all factors are <= p
    # is related to the Dickman function. As p increases, the condition becomes less restrictive
    # relative to the magnitude of the numbers?
    # Let's just search primes.
    
    # To maximize chances within time, we should check primes in a smart order.
    # Checking every prime is too slow.
    # We can sample primes.
    
    # Generate candidate primes.
    # We have small_primes up to 10^6.
    # We can also generate primes up to 10^7 or 10^8?
    # Sieve up to 10^7 takes ~1s and 10MB.
    # Let's extend sieve to 10^7 to have more candidates.
    
    limit_mid = 10**7
    if limit_mid > limit_small:
        sieve_mid = [True] * (limit_mid + 1)
        # Copy small sieve
        sieve_mid[:limit_small+1] = sieve[:limit_small+1]
        for i in range(int(math.isqrt(limit_small)) + 1, int(math.isqrt(limit_mid)) + 1):
            if sieve_mid[i]:
                sieve_mid[i*i::i] = [False] * len(sieve_mid[i*i::i])
        mid_primes = [i for i, is_p in enumerate(sieve_mid) if is_p]
    else:
        mid_primes = small_primes

    # We will check primes from mid_primes.
    # To save time, we will not check ALL primes.
    # We will check a subset.
    # Strategy: Check primes in descending order? Or random?
    # Long runs are rare.
    # Let's try checking primes from the upper end of our list first, then step down.
    # Or just iterate through them, but with a very fast "fail" check.
    
    # Optimization for is_p_smooth:
    # If n is not p-smooth, it has a prime factor > p.
    # This means n = k * q where q > p is prime.
    # Then k < n/p ~ p.
    # So if we can find a divisor k < p, we can check if n/k is prime and > p.
    # But finding k is the hard part.
    
    # Let's stick to the factorization approach but optimize the loop.
    # We will check primes p.
    # For each p, we check k=1, 2, ... until fail.
    
    # To speed up, we can pre-compute something? No.
    
    # Let's iterate through mid_primes.
    # If we have time, we check more.
    
    # Sort primes? They are already sorted.
    # Let's check from larger p to smaller p?
    # Larger p -> larger numbers -> harder to factor?
    # But larger p -> more "room" for factors?
    # Actually, the condition "all factors <= p" is easier to satisfy for large p
    # if the numbers are not too large.
    # For p=10^6, n~10^12. Factors must be <= 10^6.
    # For p=10, n~100. Factors must be <= 10.
    
    # Let's try iterating from the end of the list (largest p) downwards.
    # Why? Because for small p, the condition is very tight.
    # For p=3, n=10,11... 10=2*5 (5>3) fail.
    # For p=5, n=26,27... 26=2*13 (13>5) fail.
    # For p=100, n=10001...
    
    # We will check primes in reverse order.
    # If we find a good k, we can stop or continue to see if we can beat it.
    # Since we want to MAXIMIZE k, we should check all candidates?
    # No, time limit.
    # We will check candidates and keep the best.
    
    # To ensure we don't miss small p with huge k (unlikely but possible),
    # we can check a mix.
    # But typically, larger p allows longer runs.
    
    # Let's iterate through mid_primes.
    # We will skip some to save time if we find a good k early?
    # No, let's just run through as many as possible.
    
    # Optimization: 
    # In is_p_smooth, the trial division loop `for q in small_primes` breaks if q > p.
    # For large p, this loop is long.
    # But `q*q > temp` breaks it even earlier.
    # For n ~ p^2, sqrt(n) ~ p.
    # So we might iterate all primes up to p.
    # For p=10^7, pi(p) ~ 664k.
    # 664k divisions per number.
    # If k is small (e.g., 1-5), this is ~3M divisions.
    # In Python, this is ~0.1-0.5s per p.
    # Too slow for many p's.
    
    # We need a faster check for "is n p-smooth".
    # If n is NOT p-smooth, it has a factor > p.
    # Let's use the property: if n has a factor > p, then n = k * q with q > p.
    # k < p.
    # If we can find ANY divisor k of n with k < p, we can check if n/k > p and is prime.
    # But finding divisors is hard.
    
    # Alternative:
    # Use Pollard Rho to find a factor.
    # If the factor found is > p, and n/factor < p? No.
    # If factor > p, then n is NOT p-smooth ONLY IF the other factor is 1? No.
    # If n = a * b, and a > p.
    # If b >= 1, then n has a factor a > p.
    # So if we find ANY prime factor > p, return False.
    
    # So the algorithm is:
    # 1. Find a prime factor f of n.
    # 2. If f > p, return False.
    # 3. If f <= p, divide n by f. Repeat.
    # 4. If n becomes 1, return True.
    
    # This is essentially trial division but using Pollard Rho to find the NEXT factor.
    # Pollard Rho finds a factor in O(n^{1/4}).
    # For n ~ 10^14 (p=10^7), n^{1/4} ~ 1000.
    # This is much faster than 664k trial divisions.
    
    # Revised is_p_smooth:
    def is_p_smooth_fast(n, p):
        if n <= p:
            return True
        
        temp = n
        # First, strip small factors using trial division up to a small limit (e.g. 1000)
        # This reduces the size for Pollard Rho.
        for q in small_primes:
            if q > 1000: # Limit trial division to small primes
                break
            if q * q > temp:
                break
            while temp % q == 0:
                if q > p: # Should not happen if q <= 1000 and p >= 1000
                    return False
                temp //= q
        
        if temp == 1:
            return True
        
        # Now temp has no factors <= 1000.
        # We need to check if temp has any prime factor > p.
        # If temp is prime:
        if is_prime_mr(temp):
            return temp <= p
        
        # If temp is composite:
        # We need to find a factor.
        # If we find a factor f:
        #   If f > p, return False.
        #   If f <= p, check n/f recursively?
        #   Actually, if f <= p, we can divide it out and continue.
        
        # Let's loop:
        while temp > 1:
            if is_prime_mr(temp):
                if temp > p:
                    return False
                else:
                    return True # Since all remaining factors are <= p (if we divide them out)
                # Wait, if temp is prime and <= p, it's smooth.
                # But what if there are more factors?
                # If temp is prime, it's the last factor.
            
            # Find a factor
            # If temp is small, trial division might be faster?
            # But we already stripped small factors.
            
            d = pollard_brent(temp)
            if d == temp:
                # Pollard failed, assume prime?
                return temp <= p
            
            # Check if d or temp//d is > p
            # If d > p, return False?
            # Not necessarily. d might be composite? No, pollard_brent returns a non-trivial factor.
            # It might not be prime.
            # So we need to factor d?
            # Or just check: if d > p, it doesn't mean d is a prime factor > p.
            # d could be p1 * p2 where p1, p2 <= p.
            # Example: p=100, temp=101*103. d=101*103. d > 100.
            # But 101 > 100 and 103 > 100.
            # If d is composite, we need to split it.
            
            # To be safe, we can recursively check factors.
            # But to keep it simple and fast:
            # If d > p, we don't immediately fail.
            # We divide temp by d? No, d might not divide temp evenly if it's not a factor?
            # pollard_brent returns a factor, so it divides.
            
            # Let's just split temp into d and temp//d.
            # We need to ensure ALL prime factors are <= p.
            # This is equivalent to:
            #   is_p_smooth(d, p) AND is_p_smooth(temp//d, p)
            
            # Recursion depth might be an issue, but numbers get small fast.
            
            # Optimization:
            # If d <= p and temp//d <= p, and both are prime?
            # If d is composite, it has factors.
            
            # Let's just call is_p_smooth_fast on the factors.
            # To avoid infinite recursion, we know temp decreases.
            
            if not is_p_smooth_fast(d, p):
                return False
            if not is_p_smooth_fast(temp // d, p):
                return False
            
            # If we get here, both parts are p-smooth.
            # So temp is p-smooth.
            return True

    # Wait, the recursive call is expensive.
    # Let's optimize the main loop.
    
    # We will check primes from mid_primes.
    # We will iterate from largest to smallest?
    # Or just iterate and hope.
    
    # Let's iterate from the end of mid_primes.
    # We will check up to 1000 candidates or until time runs out.
    
    candidates = mid_primes[::-1] # Largest to smallest
    
    for p in candidates:
        if time.time() > deadline:
            break
        
        base = p * p
        k = 0
        
        # Check consecutive integers
        # We can break early if k is small and we've checked many p's?
        # No, we want max k.
        
        # Limit k to 200 as per task.
        for offset in range(1, 201):
            if time.time() > deadline:
                break
                
            n = base + offset
            if is_p_smooth_fast(n, p):
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
            
            # If we found a very good k, we can continue to see if we can beat it.
            # But if k is close to 200, it's a strong result.
            
    return {'p': best_p, 'k': best_k}

# Test
if __name__ == "__main__":
    res = construct()
    print(res)
