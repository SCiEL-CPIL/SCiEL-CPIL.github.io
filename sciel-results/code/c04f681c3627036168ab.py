import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110.0  # Safety margin
    
    def is_prime_mr(n):
        if n < 2:
            return False
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in MR_BASES:
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

    def phi_single(n):
        if n == 1:
            return 1
        factors = factorize(n)
        unique_factors = set(factors)
        result = n
        for p in unique_factors:
            result -= result // p
        return result

    best_k = 1
    best_s = 1

    def check_start(s, max_len=200):
        nonlocal best_k, best_s
        seen = set()
        k = 0
        for i in range(max_len):
            val = phi_single(s + i)
            if val in seen:
                break
            seen.add(val)
            k += 1
        if k > best_k:
            best_k = k
            best_s = s
            return True
        return False

    def segmented_phi_numpy(lo, W):
        """
        Compute phi(n) for n in [lo, lo+W) using segmented sieve with NumPy.
        Returns numpy array of length W where result[i] = phi(lo+i).
        """
        hi = lo + W - 1
        # Initialize phi array
        phi_arr = np.arange(lo, hi + 1, dtype=np.int64)
        
        # Get all primes up to sqrt(hi)
        limit = int(math.isqrt(hi)) + 1
        
        # Sieve of Eratosthenes up to limit
        is_prime = np.ones(limit + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = False
        primes = np.where(is_prime)[0]
        
        # For each prime p, apply phi[n] *= (1 - 1/p) for multiples of p in window
        for p in primes:
            # Find first multiple of p in [lo, hi]
            start = ((lo + p - 1) // p) * p
            if start < lo:
                start += p
            if start > hi:
                continue
            # Indices in phi_arr: start-lo, start-lo+p, ...
            indices = np.arange(start - lo, W, p)
            # Apply phi -= phi // p
            phi_arr[indices] -= phi_arr[indices] // p
        
        # Handle numbers that are prime themselves
        # If phi_arr[i] == lo+i and (lo+i) > 1, then lo+i is prime, so phi = n-1
        n_vals = np.arange(lo, hi + 1, dtype=np.int64)
        prime_mask = (phi_arr == n_vals) & (n_vals > 1)
        phi_arr[prime_mask] = n_vals[prime_mask] - 1
        
        # Handle n=1
        if lo == 1:
            phi_arr[0] = 1
        
        return phi_arr

    def find_longest_run(phi_arr, lo):
        """
        Given phi_arr for [lo, lo+W), find the longest run of consecutive
        integers with pairwise distinct phi values.
        Returns (best_k, best_start) within this window.
        """
        W = len(phi_arr)
        # Use sliding window with a set
        left = 0
        seen = set()
        max_len = 0
        max_start = lo
        
        for right in range(W):
            val = int(phi_arr[right])
            # Shrink window from left until no duplicate
            while val in seen:
                seen.discard(int(phi_arr[left]))
                left += 1
            seen.add(val)
            cur_len = right - left + 1
            if cur_len > max_len:
                max_len = cur_len
                max_start = lo + left
        
        return max_len, max_start

    # Phase 1: Scan small ranges with segmented sieve
    W = 200000  # Window size
    
    # Scan from 1 to some limit using segmented sieve
    scan_lo = 1
    scan_hi = 5000000  # Scan up to 5 million
    
    while scan_lo <= scan_hi:
        if time.time() > DEADLINE:
            break
        
        W_cur = min(W, scan_hi - scan_lo + 1)
        phi_arr = segmented_phi_numpy(scan_lo, W_cur)
        
        k, s = find_longest_run(phi_arr, scan_lo)
        if k > best_k:
            best_k = k
            best_s = s
        
        if best_k == 200:
            break
        
        # Move to next window with some overlap
        overlap = 1000
        scan_lo += W_cur - overlap
    
    # Phase 2: Random sampling with segmented sieve in larger ranges
    random.seed(42)
    
    W_large = 1000000  # Larger window for random sampling
    
    while time.time() < DEADLINE - 10.0:
        # Pick a random start point
        s = random.randint(1000000, 10**12 - W_large - 200)
        
        phi_arr = segmented_phi_numpy(s, W_large)
        k, s_found = find_longest_run(phi_arr, s)
        if k > best_k:
            best_k = k
            best_s = s_found
        
        if best_k == 200:
            break
    
    return {'start': best_s, 'length': best_k}
