import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 1. Generate primes
    # We need primes up to some limit for trial division and as candidates p.
    # Let's generate primes up to 10^7.
    limit = 10**7
    # Simple sieve
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    
    primes = [i for i, is_prime in enumerate(sieve) if is_prime]
    
    # 2. Miller-Rabin for primality testing
    # Deterministic for 64-bit integers
    def is_prime(n):
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        # Bases for 64-bit
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    # 3. Pollard-Brent
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(1, n - 1)
            c = random.randint(1, n - 1)
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
                while True:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
                    if g > 1:
                        break
            if g != n:
                return g

    # 4. Factorization and p-smooth check
    # Returns True if n is p-smooth
    def is_p_smooth(n, p):
        if n <= 1:
            return True
        # Trial division by small primes
        # We can use the primes list generated earlier, but only up to p
        # However, iterating over all primes <= p for every n is slow.
        # We only need to divide by primes <= min(p, sqrt(n))
        # Since n ~ p^2, sqrt(n) ~ p.
        # So we effectively divide by all primes <= p.
        
        # Optimization: Divide by small primes first
        for prime in primes:
            if prime > p:
                break
            if prime * prime > n:
                break
            while n % prime == 0:
                n //= prime
                if n == 1:
                    return True
            if time.time() > deadline:
                return False # Abort, assume not smooth to be safe? Or handle timeout globally.
        
        if n == 1:
            return True
        
        # If n > 1, it has no prime factors <= min(p, sqrt(original_n))
        # If n is prime, check if n <= p
        if is_prime(n):
            return n <= p
        
        # If n is composite, it must have a factor <= sqrt(n)
        # Since we divided by all primes <= p, and p >= sqrt(n) is NOT necessarily true?
        # Wait. n is derived from N ~ p^2.
        # If we divided by all primes <= p, and n > 1:
        # If n is composite, its smallest prime factor q must be > p.
        # Then n >= q^2 > p^2.
        # But original N < p^2 + 200.
        # So n < p^2 + 200.
        # If q > p, then n > p^2.
        # This is possible if N is close to p^2 + 200 and p is small?
        # No, N = p^2 + i.
        # If we divide N by all primes <= p, and remainder is R.
        # If R is composite, let q be smallest prime factor of R.
        # q > p.
        # R >= q^2 > p^2.
        # But R divides N. So R <= N.
        # So p^2 < R <= p^2 + 200.
        # This implies R is a prime or product of primes > p.
        # If R is composite, R = q1 * q2 ... with qi > p.
        # R > p^2.
        # So R can be $q_1 q_2$ where $q_1, q_2 > p$.
        # In this case, R has a prime factor > p, so it is NOT p-smooth.
        # So if R > 1 and we have divided by all primes <= p:
        # If R is prime, check R <= p. (But R > p because we divided by all primes <= p? No, if R is prime and R <= p, we would have divided it. So R > p).
        # So if R is prime, R > p, so NOT p-smooth.
        # If R is composite, it has factors > p, so NOT p-smooth.
        
        # Wait, this logic implies:
        # If we divide n by all primes <= p, and n != 1, then n is NOT p-smooth.
        # Is this true?
        # Yes. If n is p-smooth, all its prime factors are <= p.
        # So dividing by all primes <= p should reduce it to 1.
        # So the check is simply:
        # 1. Divide n by all primes <= p.
        # 2. If result is 1, True. Else False.
        
        # But doing this for every n is expensive if p is large.
        # Number of primes <= 10^7 is ~ 660,000.
        # Dividing 200 numbers by 660,000 primes is 132,000,000 divisions.
        # Too slow.
        
        # We need to be smarter.
        # We only need to divide by primes <= sqrt(n) to find factors.
        # If after dividing by primes <= sqrt(n), the remainder is 1, it's smooth.
        # If remainder is R > 1:
        # If R is prime, check R <= p.
        # If R is composite, it has a factor <= sqrt(R).
        # Since we divided by all primes <= sqrt(n), and R < n, sqrt(R) < sqrt(n).
        # So we have already divided by all primes <= sqrt(R).
        # So R must be prime?
        # No. If R is composite, it has a prime factor q <= sqrt(R).
        # Since q <= sqrt(R) < sqrt(n), we would have divided by q.
        # So R cannot have any prime factor <= sqrt(n).
        # So R is either 1 or prime?
        # If R is composite, it has a factor q. q <= sqrt(R).
        # If q > sqrt(n), then R > n, impossible.
        # So q <= sqrt(n).
        # So we would have found q.
        # So R must be prime.
        
        # Conclusion:
        # Divide n by all primes <= sqrt(n).
        # If remainder is 1, True.
        # If remainder is R > 1, R is prime.
        # Check if R <= p.
        
        # This reduces the number of divisions to primes <= sqrt(n) ~ p.
        # Still 660,000 divisions for p=10^7.
        
        # We can stop early if we find a factor > p?
        # No, we divide by primes in increasing order.
        # If prime > p, we stop.
        # But we need to check if the REMAINDER is <= p.
        
        # Revised Algorithm for is_p_smooth(n, p):
        # 1. For each prime q in primes:
        #    if q > p: break
        #    if q * q > n: break
        #    while n % q == 0: n //= q
        # 2. if n == 1: return True
        # 3. if is_prime(n): return n <= p
        # 4. else: return False (as argued above, n must be prime if we divided by all q <= sqrt(n))
        
        # Wait, step 4 logic:
        # We divided by all q <= min(p, sqrt(n)).
        # Case A: p >= sqrt(n). We divided by all q <= sqrt(n).
        #        Remainder n' is 1 or prime.
        #        If n' is prime, check n' <= p. (Since n' <= n <= p^2, and n' is prime, if n' > p, it's not smooth).
        # Case B: p < sqrt(n). We divided by all q <= p.
        #        Remainder n'.
        #        If n' is 1, True.
        #        If n' > 1, it has no prime factors <= p.
        #        So it is NOT p-smooth.
        
        # So the logic holds.
        
        # The cost is iterating through primes up to min(p, sqrt(n)).
        # For p=10^7, sqrt(n) ~ 10^7.
        # This is too slow for many p.
        
        # We need to optimize the trial division.
        # We can precompute primes.
        # But 10^7 primes is a lot to iterate over for each n.
        
        # Alternative:
        # Only use trial division for small primes (e.g., < 1000).
        # Then use Pollard-Brent for the rest?
        # Pollard-Brent is fast.
        # Let's try:
        # 1. Divide by primes < 1000.
        # 2. If n == 1, True.
        # 3. If is_prime(n), return n <= p.
        # 4. Else, use Pollard-Brent to find a factor f.
        #    If f > p, return False.
        #    If f <= p, recurse on f and n//f.
        
        # This should be much faster.
        # Pollard-Brent is O(n^1/4).
        # For n ~ 10^12, n^1/4 ~ 1000.
        # So it's fast.
        
        return True # Placeholder, will implement in recursive helper

    # Recursive helper for p-smooth check using Pollard-Brent
    def check_smooth(n, p):
        if n == 1:
            return True
        if n <= p:
            # If n is prime, it's smooth. If composite, it has factors <= n <= p.
            # So if n <= p, it is p-smooth?
            # Yes, all factors of n are <= n <= p.
            return True
        
        # Divide by small primes
        for q in primes:
            if q > 1000: # Small prime limit
                break
            if q * q > n:
                break
            if n % q == 0:
                # If q > p, fail (but q <= 1000, p is usually larger)
                # If q <= p, divide
                if q > p:
                    return False
                while n % q == 0:
                    n //= q
                if n == 1:
                    return True
                # Recurse or continue?
                # We can just continue with the reduced n.
                # But we need to restart the small prime loop?
                # Or just break and use Pollard-Brent on the rest?
                # Let's break and use Pollard-Brent on n.
                break
        
        if n == 1:
            return True
            
        if is_prime(n):
            return n <= p
        
        # Use Pollard-Brent
        f = pollard_brent(n)
        if f > p:
            return False
        # Check f and n//f
        return check_smooth(f, p) and check_smooth(n // f, p)

    # Main search loop
    best_k = 0
    best_p = 0
    
    # We iterate through primes p
    # We can skip very small primes as they are unlikely to have long runs
    # But we should check them just in case.
    
    for p in primes:
        if time.time() > deadline:
            break
            
        if p < 10:
            continue
            
        k = 0
        # Check p^2 + 1, p^2 + 2, ...
        # Max k is 200
        for i in range(1, 201):
            n = p * p + i
            if not check_smooth(n, p):
                break
            k += 1
        
        if k > best_k:
            best_k = k
            best_p = p
            # Update best
            # print(f"New best: p={p}, k={k}")
            
    return {'p': best_p, 'k': best_k}
