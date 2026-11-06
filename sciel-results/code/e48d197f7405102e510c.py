import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    # Pre-compute primes up to 10^6 for fast division
    LIMIT = 10**6
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(LIMIT**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, LIMIT + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    # Miller-Rabin for large numbers
    def is_prime_mr(n):
        if n < 2: return False
        # Check small primes first
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n == p: return True
            if n % p == 0: return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= n: continue
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

    # Pollard's rho for factoring composites
    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        while True:
            x = random.randint(2, n - 1)
            y = x
            c = random.randint(1, n - 1)
            d = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d

    def factorize(n):
        """Full factorization using trial division + Pollard's rho"""
        factors = []
        for p in primes:
            if p * p > n:
                break
            while n % p == 0:
                factors.append(p)
                n //= p
        if n > 1:
            if is_prime_mr(n):
                factors.append(n)
            else:
                # Use Pollard's rho
                stack = [n]
                while stack:
                    m = stack.pop()
                    if m == 1:
                        continue
                    if is_prime_mr(m):
                        factors.append(m)
                    else:
                        d = pollard_rho(m)
                        if d == m:
                            # Retry
                            stack.append(m)
                        else:
                            stack.append(d)
                            stack.append(m // d)
        return factors

    def phi(n):
        if n == 1:
            return 1
        result = n
        temp = n
        # Divide by small primes
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                result -= result // p
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            if is_prime_mr(temp):
                result -= result // temp
            else:
                # Use Pollard's rho to factor the composite remainder
                facs = factorize(temp)
                for p in set(facs):
                    result -= result // p
        return result

    def check_window(s, max_len=200, max_n=10**12):
        seen = set()
        k = 0
        for i in range(max_len):
            v = s + i
            if v > max_n:
                break
            pv = phi(v)
            if pv in seen:
                break
            seen.add(pv)
            k += 1
        return k

    best_k = 1
    best_s = 1
    
    max_n = 10**12

    # Strategy 1: Check small starts with wider range
    for s in range(1, 2000):
        if time.time() - start_time > DEADLINE:
            break
        k = check_window(s, 200)
        if k > best_k:
            best_k = k
            best_s = s

    # Strategy 2: Structured search around promising regions
    # Try starts at multiples of various values
    step = 100
    for s in range(100, 100000, step):
        if time.time() - start_time > DEADLINE - 10:
            break
        k = check_window(s, 200)
        if k > best_k:
            best_k = k
            best_s = s

    # Strategy 3: Random search with better distribution
    random.seed(42)
    for _ in range(500):
        if time.time() - start_time > DEADLINE - 5:
            break
        # Focus on regions where long runs might exist
        # Based on number theory, runs tend to occur near products of small primes
        s = random.randint(1, max_n - 200)
        k = check_window(s, 200)
        if k > best_k:
            best_k = k
            best_s = s

    return {'start': best_s, 'length': best_k}
