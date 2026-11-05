import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    min_n = 1000
    max_n = 1000000
    limit = max_n + 24
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    prime_set = set(primes)
    
    # Build Smallest Prime Factor (SPF) table
    spf = np.arange(limit + 1, dtype=np.int32)
    for i in range(2, int(np.sqrt(limit)) + 1):
        if spf[i] == i:  # i is prime
            spf[i*i::i] = np.where(spf[i*i::i] == spf[i*i::i], i, spf[i*i::i])
            # Actually, standard sieve for SPF:
            # for j in range(i*i, limit+1, i):
            #   if spf[j] == j: spf[j] = i
    
    # Corrected SPF construction
    spf = np.arange(limit + 1, dtype=np.int32)
    for i in range(2, int(np.sqrt(limit)) + 1):
        if spf[i] == i:
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_smooth_part(m):
        """Calculate the product of prime factors < 24 for m."""
        res = 1
        while m > 1:
            p = spf[m]
            if p >= 24:
                break
            while m % p == 0:
                res *= p
                m //= p
        return res

    def compute_score(n):
        factors = set()
        for i in range(1, 25):
            m = n + i
            factors.add(get_smooth_part(m))
        return len(factors)

    best_n = min_n
    best_score = float('inf')
    
    # Initial scan: check a few thousand random points
    rng = np.random.default_rng(42)
    
    # Try structured candidates first: multiples of large smooth numbers
    # e.g. n such that n+i is divisible by 2, 3, 5...
    
    # Random search
    num_samples = 200000
    for _ in range(num_samples):
        if time.time() > deadline:
            break
        n = int(rng.integers(min_n, max_n))
        s = compute_score(n)
        if s < best_score:
            best_score = s
            best_n = n
            if best_score <= 1:
                return best_n
        
    # Local search
    if best_score > 1 and time.time() < deadline - 5:
        # Explore neighborhood
        for _ in range(100000):
            if time.time() > deadline - 2:
                break
            # Jump to a new random location or step from best
            if _ % 10 == 0:
                n = int(rng.integers(min_n, max_n))
            else:
                n = best_n + int(rng.integers(-100, 100))
            
            if n < min_n or n > max_n:
                continue
                
            s = compute_score(n)
            if s < best_score:
                best_score = s
                best_n = n
                if best_score <= 1:
                    return best_n

    return best_n
