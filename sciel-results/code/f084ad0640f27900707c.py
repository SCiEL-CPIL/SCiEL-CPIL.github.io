import time
import random
import math
import sys

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    # 1. Sieve of Eratosthenes / Smallest Prime Factor (SPF)
    # Limit chosen to balance memory and speed. 10^7 is ~40MB for int array, might be tight.
    # Let's use 10^6 for SPF to be safe, but extend trial division.
    # Actually, for phi(n), just having primes is enough for trial division.
    # Let's stick to primes up to 10^6 to save memory, but optimize the loop.
    
    SIEVE_LIMIT = 10**6
    is_prime_arr = bytearray(b'\x01') * (SIEVE_LIMIT + 1)
    is_prime_arr[0] = 0
    is_prime_arr[1] = 0
    for i in range(2, int(SIEVE_LIMIT**0.5) + 1):
        if is_prime_arr[i]:
            # Mark multiples
            start = i * i
            is_prime_arr[start:SIEVE_LIMIT + 1:i] = bytearray(len(is_prime_arr[start:SIEVE_LIMIT + 1:i]))
    
    small_primes = [i for i, p in enumerate(is_prime_arr) if p]

    # 2. Miller-Rabin for large primes
    def is_prime_large(n):
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        # Small checks
        if n < 10**6:
            return is_prime_arr[n] == 1 if n <= SIEVE_LIMIT else False
        
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        # Deterministic for n < 3,317,044,064,679,887,385,961,981
        # For our range (10^12), bases [2, 3, 5, 7, 11, 13] are sufficient?
        # Actually, for < 3.4 * 10^14, bases 2,3,5,7,11,13,17 are enough.
        for a in [2, 3, 5, 7, 11, 13, 17]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    # 3. Pollard Rho
    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        
        while True:
            c = random.randint(1, n - 1)
            x = random.randint(2, n - 1)
            y = x
            d = 1
            # Batch gcd optimization? Standard is fine for 10^12 factors usually.
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d

    # 4. Factorization
    def factor_large(n, factors):
        if n == 1:
            return factors
        if n <= SIEVE_LIMIT:
            if is_prime_arr[n]:
                factors[n] = factors.get(n, 0) + 1
                return factors
            # If not prime and <= limit, it should have been caught by trial division?
            # No, phi_fast does trial division. This is only called for remainders.
            # If remaining is composite and <= limit, it's a product of primes > sqrt(remaining)?
            # No, if we trial divided up to sqrt(remaining), remaining must be prime.
            # So this branch is rare unless we change strategy.
            pass
        
        if is_prime_large(n):
            factors[n] = factors.get(n, 0) + 1
            return factors
            
        d = pollard_rho(n)
        factor_large(d, factors)
        factor_large(n // d, factors)
        return factors

    # 5. Phi calculation with Caching
    phi_cache = {}
    
    def phi_fast(n):
        if n in phi_cache:
            return phi_cache[n]
        
        if n == 1:
            phi_cache[1] = 1
            return 1
        
        result = n
        remaining = n
        
        # Trial division by small primes
        # Optimization: Only iterate while p*p <= remaining
        for p in small_primes:
            if p * p > remaining:
                break
            if remaining % p == 0:
                result -= result // p
                while remaining % p == 0:
                    remaining //= p
        
        if remaining > 1:
            if is_prime_large(remaining):
                result -= result // remaining
            else:
                # Factor the remaining composite part
                facs = {}
                factor_large(remaining, facs)
                for p in facs:
                    result -= result // p
        
        phi_cache[n] = result
        return result

    best_k = 1
    best_s = 1
    max_n = 10**12

    def check_window(s, max_len=200):
        seen = set()
        k = 0
        # We only need to check up to max_len or until collision
        # Note: phi(n) for n>=3 is even. If we have two odd numbers in the window,
        # their phis are even, so no collision from parity.
        while k < max_len:
            n_val = s + k
            if n_val > max_n:
                break
            v = phi_fast(n_val)
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k

    # Search Strategy
    random.seed(42)
    
    # 1. Quick scan of small numbers (deterministic, fast due to small factors)
    # We can afford to scan a bit more here
    scan_limit = 200000
    for s in range(1, scan_limit):
        if time.time() - start_time > DEADLINE - 10:
            break
        k = check_window(s)
        if k > best_k:
            best_k = k
            best_s = s

    # 2. Random search in large numbers
    # Use a larger step or different distribution?
    # Uniform is fine, but let's try to jump around to avoid correlation.
    attempts = 0
    while time.time() - start_time < DEADLINE:
        # Random start
        s = random.randint(10**6, max_n - 200)
        
        k = check_window(s)
        
        if k > best_k:
            best_k = k
            best_s = s
        
        # If we found a decent run, refine locally
        # Local refinement is where the cache helps significantly
        if k > 5:
            # Check neighborhood
            # The window length is k. Let's check +/- 100 around s
            # We want to find the start of the *longest* run in this area.
            # Since runs are contiguous, if we have a run of length k starting at s,
            # the run might extend backwards or forwards.
            
            # Check backwards
            start_check = max(1, s - 100)
            # We already know s..s+k-1 is good.
            # We need to check s-1, s-2...
            # And s+k, s+k+1...
            
            # Efficient local search:
            # Expand the current window if possible?
            # No, the current window broke because of a collision or limit.
            # If it broke due to collision, extending is useless for that specific start.
            # But a neighbor start might have a longer run.
            
            for offset in range(-50, 51):
                if time.time() - start_time > DEADLINE:
                    break
                ns = s + offset
                if 1 <= ns <= max_n - 200:
                    nk = check_window(ns)
                    if nk > best_k:
                        best_k = nk
                        best_s = ns

        attempts += 1
        # Optional: Adjust search strategy if stuck?
        # If we haven't improved in a while, try a different range?
        # For now, pure random is robust.

    return {'start': best_s, 'length': best_k}

# Ensure the function is available at module level if needed, 
# though the task says "Keep the function names... as in the seed program".
# The seed program defines construct() globally.
