import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    
    # Pre-compute small primes for trial division
    SMALL_PRIME_LIMIT = 1000000
    small_primes = []
    is_sm = bytearray(SMALL_PRIME_LIMIT + 1)
    is_sm[0] = is_sm[1] = 1
    for i in range(2, SMALL_PRIME_LIMIT + 1):
        if not is_sm[i]:
            small_primes.append(i)
            if i * i <= SMALL_PRIME_LIMIT:
                is_sm[i*i::i] = bytearray(len(is_sm[i*i::i]))
    
    phi_cache = {}
    
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
        if n % 5 == 0:
            return 5
        if n % 7 == 0:
            return 7
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
    
    def factor(n):
        if n in phi_cache:
            return phi_cache[n]
        result = []
        for p in small_primes:
            if p * p > n:
                break
            while n % p == 0:
                result.append(p)
                n //= p
        if n > 1:
            if is_prime_mr(n):
                result.append(n)
            else:
                d = pollard_rho(n)
                result.extend(factor(d))
                result.extend(factor(n // d))
        phi_cache[n] = result
        return result
    
    def phi(n):
        if n in phi_cache:
            return phi_cache[n]
        if n == 1:
            phi_cache[n] = 1
            return 1
        facs = factor(n)
        unique = set(facs)
        result = n
        for p in unique:
            result -= result // p
        phi_cache[n] = result
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
    
    # Strategy 1: Search in dense regions near small numbers
    for s in [1, 2, 3, 4, 5, 6, 10, 20, 50, 100, 500, 1000, 5000, 10000, 100000, 1000000]:
        if time.time() - start_time > DEADLINE:
            break
        k = check_window(s, 200)
        if k > best_k:
            best_k = k
            best_s = s
        if k > 3:
            for offset in range(-50, 51):
                if time.time() - start_time > DEADLINE:
                    break
                ns = s + offset
                if ns >= 1:
                    nk = check_window(ns, 200)
                    if nk > best_k:
                        best_k = nk
                        best_s = ns
    
    # Strategy 2: Search near primorials and smooth numbers
    primorial_values = [1]
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
        if primorial_values[-1] * p > 10**12:
            break
        primorial_values.append(primorial_values[-1] * p)
    
    max_n = 10**12
    
    for pm in primorial_values:
        if time.time() - start_time > DEADLINE:
            break
        for mult in [1, 2, 3, 5, 7, 10, 13, 17, 20, 25, 30, 40, 50]:
            s = pm * mult
            if s > max_n - 200:
                continue
            k = check_window(s, 200)
            if k > best_k:
                best_k = k
                best_s = s
            for delta in range(-10, 11):
                if time.time() - start_time > DEADLINE:
                    break
                ns = s + delta
                if 1 <= ns <= max_n - 200:
                    nk = check_window(ns, 200)
                    if nk > best_k:
                        best_k = nk
                        best_s = ns
    
    # Strategy 3: Targeted search around numbers with many distinct prime factors
    # Numbers that are products of small primes tend to have diverse phi values nearby
    random.seed(42)
    
    # Precompute some useful prime products
    prime_products = []
    base_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for subset_size in range(1, 7):
        for i in range(len(base_primes)):
            prod = 1
            for j in range(i, min(i + subset_size, len(base_primes))):
                prod *= base_primes[j]
                if prod > 10**9:
                    break
            if prod > 1 and prod <= 10**9:
                prime_products.append(prod)
    
    prime_products = list(set(prime_products))
    random.shuffle(prime_products)
    
    # Strategy 4: Random search with bias toward regions with diverse factorizations
    for iteration in range(3000):
        if time.time() - start_time > DEADLINE:
            break
        if time.time() - start_time > DEADLINE - 15:
            break
        
        r = random.random()
        if r < 0.2:
            # Near a prime product
            if prime_products:
                pp = random.choice(prime_products)
                s = pp * random.randint(1, 1000)
            else:
                s = random.randint(1000, 10**9)
        elif r < 0.4:
            # Small range
            s = random.randint(100, 1000000)
        elif r < 0.6:
            # Medium range
            s = random.randint(1000000, 10**9)
        elif r < 0.8:
            # Large range
            s = random.randint(10**9, max_n - 200)
        else:
            # Very large range
            s = random.randint(10**10, max_n - 200)
        
        s = max(1, min(s, max_n - 200))
        
        # Check a window starting at s
        k = check_window(s, min(200, best_k + 10))
        if k > best_k:
            best_k = k
            best_s = s
            
            # Try to extend the window left
            for back in range(1, 100):
                if time.time() - start_time > DEADLINE:
                    break
                ns = best_s - back
                if ns < 1:
                    break
                # Check if we can prepend
                test_start = ns
                test_len = back + best_k
                if test_start + test_len - 1 > max_n:
                    break
                nk = check_window(test_start, test_len)
                if nk >= test_len:
                    best_k = nk
                    best_s = test_start
                    break
                elif nk > 0:
                    # Partial extension
                    new_k = nk + (best_s - ns)
                    if new_k > best_k and nk + (best_s - ns - (best_k - nk)) > best_k:
                        # Complex, just try direct
                        pass
        elif k > 5:
            # Good but not best, try nearby
            for delta in [-20, -10, -5, -3, -2, -1, 1, 2, 3, 5, 10, 20]:
                if time.time() - start_time > DEADLINE:
                    break
                ns = s + delta
                if 1 <= ns <= max_n - 200:
                    nk = check_window(ns, min(200, best_k + 10))
                    if nk > best_k:
                        best_k = nk
                        best_s = ns
    
    # Strategy 5: Final targeted search around best found
    if best_k > 1:
        # Search in a larger neighborhood of the best start
        for delta in range(-200, 201, 5):
            if time.time() - start_time > DEADLINE:
                break
            ns = best_s + delta
            if 1 <= ns <= max_n - 200:
                nk = check_window(ns, 200)
                if nk > best_k:
                    best_k = nk
                    best_s = ns
    
    return {'start': best_s, 'length': best_k}
