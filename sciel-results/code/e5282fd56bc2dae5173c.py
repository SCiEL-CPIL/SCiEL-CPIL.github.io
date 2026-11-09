import time
import math
import random
import numpy as np

# ---------- Deterministic Miller-Rabin for 64-bit ----------
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def _is_prime_mr(n):
    if n < 2:
        return False
    for p in _MR_BASES:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_BASES:
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

# ---------- Pollard-Brent ----------
def _pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randrange(1, n)
        c = random.randrange(1, n)
        m = random.randrange(1, n)
        g = r = q = 1
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
                if g != 1:
                    break
        if g != n:
            return g

def _factor(n, result):
    if n <= 1:
        return
    if _is_prime_mr(n):
        result.append(n)
        return
    d = _pollard_brent(n)
    _factor(d, result)
    _factor(n // d, result)

# ---------- Sieve ----------
def _sieve(limit):
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(limit + 1) if is_prime[i]]

# ---------- Batched segment sieve for a single prime p ----------
# Given p, check if all numbers p^2+1 .. p^2+K have all prime factors <= p.
# Returns the maximum k (1..K) such that p^2+1..p^2+k are all p-smooth.
def _check_prime_batched(p, primes_small, K, deadline):
    """
    primes_small: numpy array of all primes up to some limit (>= sqrt(p^2+K) for small p,
                  or just up to 10^6 for larger p).
    Returns k in [0, K].
    """
    p_sq = p * p
    # The segment is [p_sq+1, p_sq+K]
    # We want to know for each offset i in 1..K whether p_sq+i has a prime factor > p.
    # Strategy: for each small prime q <= p, divide out all powers of q from the segment.
    # After processing all q <= p, any remaining value > 1 has a prime factor > p.
    
    # Use a numpy array to track the remaining value for each position.
    # But values can be up to ~10^18, so use int64.
    # Actually, we just need to know if the remaining value is 1 or > 1.
    # Better: track a boolean "bad" array. If any prime factor > p remains, mark bad.
    
    # More efficient: for each prime q <= p, check if q divides any number in the segment
    # that hasn't been fully divided. But we need to divide out ALL powers.
    
    # Alternative approach: for each prime q <= p, find positions in [1, K] where
    # (p_sq + i) % q == 0, and divide out q. After all such q, check remaining.
    
    # Let's use the direct approach with numpy:
    # seg = np.arange(p_sq + 1, p_sq + K + 1, dtype=np.int64)
    # For each q in primes_small where q <= p:
    #   mask = (seg % q == 0)
    #   while any(mask):
    #       seg[mask] //= q
    #       mask = (seg % q == 0)
    # Then bad = seg > 1
    
    # But this is still O(K * pi(p)) in the worst case. For p up to 10^6, pi(p) ~ 78000,
    # K=200, so ~15.6M operations per prime. Too slow in Python/NumPy for many primes.
    
    # Better: only iterate over primes q <= p AND q <= sqrt(p_sq + K) ~ p.
    # For small p (say p < 10000), pi(p) is small. For large p, we can't afford this.
    
    # Let's use a hybrid: for p <= 10^6, use the numpy approach.
    # For p > 10^6, use trial division with the pre-sieved primes up to 10^6,
    # then check the remainder.
    
    if p <= 10**6:
        # Use numpy batched division
        seg = np.arange(p_sq + 1, p_sq + K + 1, dtype=np.int64)
        
        for q in primes_small:
            if q > p:
                break
            if time.time() > deadline:
                break
            # Find positions divisible by q
            mask = (seg % q == 0)
            if not np.any(mask):
                continue
            # Divide out all powers of q
            while np.any(mask):
                seg[mask] //= q
                mask = (seg % q == 0)
        
        if time.time() > deadline:
            return 0
        
        # Now seg[i-1] is the remaining part after dividing all primes <= p.
        # If seg[i-1] > 1, then p_sq+i has a prime factor > p.
        bad = seg > 1
        # Find the first bad position
        bad_indices = np.where(bad)[0]
        if len(bad_indices) == 0:
            return K
        else:
            return int(bad_indices[0])  # This is the offset of the first bad number
    else:
        # For p > 10^6, use trial division with primes up to 10^6
        # Then check the remainder
        k = 0
        for i in range(1, K + 1):
            if time.time() > deadline:
                break
            val = p_sq + i
            t = val
            for q in primes_small:
                if q * q > t:
                    break
                while t % q == 0:
                    t //= q
            if t == 1:
                k = i
            else:
                # t > 1: need to check if all prime factors of t are <= p
                if _is_prime_mr(t):
                    if t <= p:
                        k = i
                    else:
                        break
                else:
                    factors = []
                    _factor(t, factors)
                    if all(f <= p for f in factors):
                        k = i
                    else:
                        break
        return k

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Precompute primes up to 10^6
    small_primes_list = _sieve(1000000)
    small_primes_np = np.array(small_primes_list, dtype=np.int64)
    
    best_k = 0
    best_p = 3
    K = 200
    
    # Phase 1: Check primes up to 10^6 with batched numpy approach
    for p in small_primes_list:
        if time.time() > deadline:
            break
        k = _check_prime_batched(p, small_primes_np, K, deadline)
        if k > best_k:
            best_k = k
            best_p = p
            if best_k == K:
                break
    
    # Phase 2: If not found k=200, try larger primes with random sampling
    if best_k < K:
        # Use a smaller set of primes for trial division (up to 10^6 is fine)
        for _ in range(3000):
            if time.time() > deadline:
                break
            # Sample primes in [10^6, 10^9]
            log_val = 6 + random.random() * 3
            p = int(10 ** log_val)
            if p % 2 == 0:
                p += 1
            while not _is_prime_mr(p):
                p += 2
                if p > 10**9:
                    break
            if p > 10**9 or not _is_prime_mr(p):
                continue
            k = _check_prime_batched(p, small_primes_np, K, deadline)
            if k > best_k:
                best_k = k
                best_p = p
                if best_k == K:
                    break
    
    return {'p': best_p, 'k': best_k}
