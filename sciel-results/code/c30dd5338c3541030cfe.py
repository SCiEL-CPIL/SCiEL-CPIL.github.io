import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin

    def now():
        return time.time()

    # 1. Precompute primes up to 10^6 for trial division
    LIMIT = 10**6
    is_prime = bytearray(b'\x01') * (LIMIT + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, LIMIT + 1) if is_prime[i]]

    # 2. Efficient phi calculation using trial division first, then Miller-Rabin/Pollard's Rho
    def is_prime_mr(n):
        if n < 2:
            return False
        if n < LIMIT:
            return bool(is_prime[n])
        # Small primes check
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return n == p
        # Miller-Rabin
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = (x * x) % n
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
        if n == 1:
            return []
        if is_prime_mr(n):
            return [n]
        d = pollard_rho(n)
        return factorize(d) + factorize(n // d)

    def phi(n):
        if n == 1:
            return 1
        # Try trial division with precomputed primes first
        temp = n
        result = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
        if temp > 1:
            # temp is either prime or a product of large primes
            # Use factorize to be safe, or just treat as prime if it's large enough?
            # Actually, if temp > 1, it has no prime factors <= sqrt(n_original) roughly.
            # But wait, the loop breaks when p*p > temp. So temp is prime?
            # No, if n = p1 * p2 where p1, p2 > 10^6, then p1*p2 > 10^12.
            # Since n <= 10^12, temp can be composite only if it's product of two primes > 10^6?
            # 10^6 * 10^6 = 10^12. So yes, temp could be p1*p2.
            # But our primes list only goes to 10^6.
            # So we need to factorize temp.
            factors = factorize(temp)
            unique_factors = set(factors)
            for p in unique_factors:
                result -= result // p
        return result

    # 3. Helper to find max distinct run in a list of phi values
    def max_distinct_run(vals):
        best_k = 0
        best_start_idx = 0
        n = len(vals)
        
        for i in range(n):
            if n - i < best_k:
                break
            seen = set()
            for j in range(i, min(i + 200, n)):
                v = vals[j]
                if v in seen:
                    break
                seen.add(v)
                curr_k = j - i + 1
                if curr_k > best_k:
                    best_k = curr_k
                    best_start_idx = i
                    if best_k == 200:
                        return best_k, best_start_idx
        return best_k, best_start_idx

    best_k = 1
    best_s = 1

    # 4. Search Strategy
    # Phase 1: Sieve-like computation for small range (1 to 200,000)
    # This is very fast and likely to find the global max for small k.
    # We compute phi for all numbers in range, then slide window.
    
    N_SMALL = 200000
    phi_small = [0] * (N_SMALL + 1)
    for i in range(1, N_SMALL + 1):
        phi_small[i] = phi(i)
    
    # Find best run in phi_small[1..N_SMALL]
    k, idx = max_distinct_run(phi_small[1:])
    if k > best_k:
        best_k = k
        best_s = idx + 1

    if best_k == 200:
        return {'start': best_s, 'length': best_k}

    # Phase 2: Random search in larger range
    # We sample random starting points and check runs.
    # We bias towards larger numbers where distinctness might be more common?
    # Or just uniform.
    
    # Use a set of visited ranges to avoid redundant work?
    # For now, just random.
    
    random.seed(12345) # Fixed seed for reproducibility in dev, but maybe random is better for exploration?
    # Let's use random.seed() for true randomness in eval?
    # The prompt says "random" is available.
    # Let's stick with deterministic if possible, or just use random.
    # Since we want to maximize score, and score is raw k, finding a larger k is key.
    # Random search might hit a lucky spot.
    
    # Let's do a mix:
    # 1. Check around 10^6, 10^7, ..., 10^12
    # 2. Random samples
    
    candidates = []
    for b in [10**6, 10**7, 10**8, 10**9, 10**10, 10**11, 10**12]:
        # Check a block of 1000 around b
        start = max(1, b - 500)
        end = min(10**12, b + 500)
        candidates.append((start, end))
        
        # Also check b itself and neighbors
        candidates.append((b, b))
        
    # Add random ranges
    for _ in range(100):
        start = random.randint(10**6, 10**12 - 1000)
        candidates.append((start, start + 1000))
        
    # Process candidates
    for start, end in candidates:
        if now() > DEADLINE - 2.0:
            break
        
        if start < 1:
            start = 1
        if end > 10**12:
            end = 10**12
        if start > end:
            continue
            
        # Compute phis for this range
        # If range is large, we might run out of time.
        # Limit block size to 2000 to be safe?
        # If end - start > 2000, we just take a subset?
        # No, we want to find runs within this range.
        # Let's just compute up to 2000 numbers.
        limit = min(end, start + 2000)
        
        try:
            vals = [phi(start + i) for i in range(limit - start + 1)]
        except Exception:
            continue
            
        k, local_idx = max_distinct_run(vals)
        
        if k > best_k:
            best_k = k
            best_s = start + local_idx
            
            if best_k == 200:
                break
                
    return {'start': best_s, 'length': best_k}
