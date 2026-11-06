import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin

    def now():
        return time.time()

    # 1. Precompute phi up to 10^6
    LIMIT = 10**6
    phi_small = list(range(LIMIT + 1))
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = is_prime[1] = False
    
    # Sieve of Eratosthenes and Phi calculation
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if is_prime[i]:
            for j in range(i*i, LIMIT + 1, i):
                is_prime[j] = False
            # Update phi for multiples of i
            for j in range(i, LIMIT + 1, i):
                phi_small[j] -= phi_small[j] // i
                
    # Ensure phi[1] = 1 (already is)
    # For primes > sqrt(LIMIT), phi[p] = p-1. 
    # The loop above only updated multiples for i <= sqrt(LIMIT).
    # We need to finalize phi for all n.
    # The standard way: phi[n] = n * prod(1 - 1/p).
    # My loop above: phi[j] -= phi[j] // i. This works if we iterate i over primes.
    # Let's re-verify. 
    # If i is prime, for all j multiple of i, phi[j] = phi[j] / i * (i-1).
    # phi[j] -= phi[j] // i is equivalent to phi[j] = phi[j] * (i-1) / i if phi[j] is divisible by i.
    # Since we start with phi[j]=j, and we do this for all prime factors, it works.
    
    # 2. Find max distinct run in small range
    best_k = 1
    best_s = 1
    
    # Sliding window on phi_small
    # We want max k such that phi_small[s...s+k-1] are distinct.
    # Since max k is likely small, we can just check each start s.
    # But O(N*K) is fine for N=10^6, K~20.
    
    # Optimization: Use a set for the current window.
    # Actually, just checking each s up to LIMIT is fast enough in Python if optimized.
    
    # Let's do a more efficient scan.
    # We know k is small. Let's just check all s.
    # To speed up, we can break early if phi_small[s] == phi_small[s+1] etc.
    
    # A faster method: 
    # For each s, expand window as long as values are distinct.
    # This is O(N * avg_k).
    
    for s in range(1, LIMIT):
        if now() > DEADLINE - 5.0:
            break
        seen = set()
        k = 0
        # Max k is 200, but practically much smaller
        while s + k < LIMIT and k < 200:
            v = phi_small[s + k]
            if v in seen:
                break
            seen.add(v)
            k += 1
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == 200:
                return {'start': best_s, 'length': best_k}

    # 3. Search large range using random sampling and local refinement
    # We need a fast phi(n) for n > LIMIT.
    # Precompute primes up to 10^5 for trial division.
    PRIME_LIMIT = 10**5
    is_prime_large = bytearray(b'\x01') * (PRIME_LIMIT + 1)
    is_prime_large[0] = is_prime_large[1] = 0
    for i in range(2, int(math.isqrt(PRIME_LIMIT)) + 1):
        if is_prime_large[i]:
            is_prime_large[i*i::i] = bytearray(len(is_prime_large[i*i::i]))
    primes = [i for i in range(2, PRIME_LIMIT + 1) if is_prime_large[i]]

    def is_prime_mr(n):
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0:
            return False
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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

    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            c = random.randint(1, n - 1)
            x = random.randint(2, n - 1)
            y = x
            d = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d

    def factor(n, factors=None):
        if factors is None:
            factors = set()
        if n == 1:
            return factors
        if n <= PRIME_LIMIT:
            if is_prime_large[n]:
                factors.add(n)
            else:
                # Should not happen if we trial divided, but safe
                for p in primes:
                    if p * p > n:
                        break
                    while n % p == 0:
                        factors.add(p)
                        n //= p
                if n > 1:
                    factors.add(n)
            return factors
        
        # Trial division by small primes
        temp_n = n
        for p in primes:
            if p * p > temp_n:
                break
            if temp_n % p == 0:
                factors.add(p)
                while temp_n % p == 0:
                    temp_n //= p
        if temp_n > 1:
            if is_prime_mr(temp_n):
                factors.add(temp_n)
            else:
                # Pollard Rho for the remaining composite
                d = pollard_rho(temp_n)
                factor(d, factors)
                factor(temp_n // d, factors)
        return factors

    def phi_large(n):
        if n <= LIMIT:
            return phi_small[n]
        facs = factor(n)
        result = n
        for p in facs:
            result -= result // p
        return result

    def check_window(s, max_len=200):
        seen = set()
        k = 0
        while k < max_len and s + k <= 10**12:
            v = phi_large(s + k)
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k

    random.seed(42)
    
    # Strategy: 
    # 1. Randomly pick starts.
    # 2. Check window.
    # 3. If promising, do local search.
    
    max_n = 10**12
    
    while now() < DEADLINE:
        # Pick a random start
        # Bias towards larger numbers? Or uniform?
        # Uniform is safer.
        s = random.randint(LIMIT, max_n - 200)
        
        # Quick check: first 5 values
        # If they are distinct, proceed to full check.
        # This saves time on "bad" starts where collisions happen early.
        initial_vals = []
        abort = False
        for i in range(5):
            if s + i > max_n:
                break
            v = phi_large(s + i)
            if v in initial_vals:
                abort = True
                break
            initial_vals.append(v)
        
        if abort:
            continue
            
        k = check_window(s)
        
        if k > best_k:
            best_k = k
            best_s = s
            # Local search around the best found
            # Expand search radius
            for offset in range(-50, 51, 5):
                ns = s + offset
                if 1 <= ns <= max_n - 200:
                    nk = check_window(ns)
                    if nk > best_k:
                        best_k = nk
                        best_s = ns
        
        # If we found a very good run, we might want to search more densely around it
        # But for now, random sampling should be sufficient to find the global max
        # given the short time limit.

    return {'start': best_s, 'length': best_k}
