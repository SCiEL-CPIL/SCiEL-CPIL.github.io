import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    min_n = 1000
    max_n = 1000000
    
    # For each n+i (i=1..24), extract the factor composed of primes strictly below 24
    # This means: for m = n+i, remove all factors >= 24, keep only the 2^a * 3^b * 5^c * ... part
    
    # Strategy: 
    # For a given n, for each i in 1..24, compute g(n+i) = the part of (n+i) composed of primes < 24
    # Then count distinct values of g(n+i)
    # We want to minimize this count
    
    # Key insight: if n+i is divisible by a large prime >= 24, that prime is stripped out.
    # We want n+i values to share small-prime factors.
    
    # A powerful approach: make all n+i divisible by 2 (so all are even).
    # That requires n+1 even, n+3 even, n+5 even, ... impossible for consecutive.
    
    # Better: make many n+i share the same small-prime factor part.
    # If n+i is a multiple of 2*3*5*7*11*13*17*19*23 = 223092870, that's too big.
    
    # Let's think differently. The "extracted factor" for m is:
    # f(m) = product over p in primes of p^(v_p(m))
    # where v_p(m) is the exponent of p in m.
    
    # We want the set {f(n+1), f(n+2), ..., f(n+24)} to have as few distinct values as possible.
    
    # If f(n+i) = 1 for all i, that means n+i has no prime factor < 24, i.e., n+i is composed only of primes >= 24.
    # That's hard but let's check: can we find n where all 24 consecutive numbers are "rough" (no small prime factors)?
    
    # Actually, let's just do a direct search with a good heuristic.
    
    # Precompute: for a range of n, compute the score efficiently.
    
    # For each n in range, score(n) = number of distinct f(n+i) for i=1..24
    
    # Let's try a batch approach. For efficiency, we can precompute for each prime p < 24,
    # the set of positions where p divides n+i.
    
    # Actually, let's just do a smart random + local search approach.
    
    # First, try some structured candidates:
    # n such that n ≡ -1 (mod 2), n ≡ -2 (mod 3), etc. -- CRT-based
    
    # Let's try: make n+i all even for odd i, all divisible by 3 for i ≡ 2 mod 3, etc.
    
    # Actually, the simplest strong candidate: 
    # If we can make all 24 numbers share a common small prime factor in their extracted part,
    # that reduces distinct values.
    
    # Let's just do a random search with local improvement.
    
    def compute_score(n):
        """Compute the number of distinct extracted factors for n+1..n+24"""
        factors = set()
        for i in range(1, 25):
            m = n + i
            f = 1
            for p in primes:
                while m % p == 0:
                    f *= p
                    m //= p
            factors.add(f)
        return len(factors)
    
    best_n = min_n
    best_score = float('inf')
    
    # Quick scan of some structured candidates
    # Try multiples of large products of small primes
    # n such that n+i is divisible by 2 for many i
    
    # Let's try n = k * (product of some primes) - i for various i
    # Actually, let's just do random search + local search
    
    rng = np.random.default_rng(42)
    
    # Initial random samples
    num_samples = 50000
    for _ in range(num_samples):
        if time.time() > deadline:
            break
        n = rng.integers(min_n, max_n)
        s = compute_score(int(n))
        if s < best_score:
            best_score = s
            best_n = int(n)
            if best_score <= 1:
                break
    
    # Local search around best
    if best_score > 1 and time.time() < deadline - 5:
        # Try neighbors
        for delta in range(-1000, 1001):
            if time.time() > deadline - 2:
                break
            n = best_n + delta
            if n < min_n or n > max_n:
                continue
            s = compute_score(n)
            if s < best_score:
                best_score = s
                best_n = n
                if best_score <= 1:
                    return best_n
    
    # If we found a good score, try to refine
    if best_score > 1 and time.time() < deadline - 10:
        # More extensive local search
        for _ in range(200000):
            if time.time() > deadline - 2:
                break
            n = best_n + rng.integers(-10000, 10000)
            if n < min_n or n > max_n:
                continue
            s = compute_score(int(n))
            if s < best_score:
                best_score = s
                best_n = int(n)
                if best_score <= 1:
                    return best_n
    
    return best_n
