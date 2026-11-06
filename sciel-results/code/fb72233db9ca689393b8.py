import time
import math
import random

# ── Deterministic Miller-Rabin for n < 2^64 ──────────────────────────────────
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
                 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
                 101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
                 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]

def is_prime(n):
    """Deterministic Miller-Rabin. Exact for n < 2^64 with 12 base primes."""
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
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

# ── Pollard's Rho / Brent factorisation ──────────────────────────────────────

def _pollard_brent(n):
    """Return a non-trivial factor of composite n."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    c = random.randrange(1, n - 1)
    y = random.randrange(0, n - 1)
    m = 128
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
                q = (q * abs(x - y)) % n
            g = math.gcd(q, n)
            k += m
        r *= 2
    if g == n:
        g = 1
        while g == 1:
            ys = (ys * ys + c) % n
            g = math.gcd(abs(x - ys), n)
    if g == n:
        # retry with different parameters
        return _pollard_brent(n)
    return g

# Precomputed primes for trial division in smoothness check
_SMOOTH_PRIMES = None

def _get_smooth_primes(limit):
    """Get primes up to limit for trial division."""
    global _SMOOTH_PRIMES
    if _SMOOTH_PRIMES is not None and _SMOOTH_PRIMES[-1] >= limit:
        return _SMOOTH_PRIMES
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            start = i * i
            step = i
            sieve[start::step] = bytearray(len(sieve[start::step]))
    primes = [i for i, is_p in enumerate(sieve) if is_p]
    _SMOOTH_PRIMES = primes
    return primes

def is_p_smooth(n, p, trial_primes):
    """
    Check if all prime factors of n are <= p.
    trial_primes: list of primes up to some limit for quick division.
    """
    if n <= 1:
        return True
    if n <= p:
        return True
    
    for q in trial_primes:
        if q > p:
            break
        if n % q == 0:
            while n % q == 0:
                n //= q
            if n == 1:
                return True
            if n <= p:
                return True
    
    if n == 1:
        return True
    if n <= p:
        return True
    
    # n > p after trial division
    if is_prime(n):
        return False
    
    # n is composite and > p. Use Pollard-Brent.
    d = _pollard_brent(n)
    return is_p_smooth(d, p, trial_primes) and is_p_smooth(n // d, p, trial_primes)

def _check_run(p, max_k, trial_primes):
    """Check how many consecutive integers p^2+1, ..., p^2+k are p-smooth."""
    base = p * p
    k = 0
    for i in range(1, max_k + 1):
        val = base + i
        if not is_p_smooth(val, p, trial_primes):
            break
        k += 1
    return k

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_k = 0
    best_p = 2
    
    # Precompute primes for smoothness checking (up to 100000 for fast trial division)
    trial_primes = _get_smooth_primes(100000)
    
    # First pass: sieve up to 5M to find primes for segmented sieve base
    SIEVE_LIMIT = 5_000_000
    small_sieve = bytearray(b'\x01') * (SIEVE_LIMIT + 1)
    small_sieve[0] = small_sieve[1] = 0
    for i in range(2, int(math.isqrt(SIEVE_LIMIT)) + 1):
        if small_sieve[i]:
            small_sieve[i*i::i] = bytearray(len(small_sieve[i*i::i]))
    small_primes = [i for i in range(2, SIEVE_LIMIT + 1) if small_sieve[i]]
    
    # Check all primes up to SIEVE_LIMIT first (they're readily available)
    for p in small_primes:
        if time.time() > deadline:
            break
        
        # Fast pre-filter: check p^2 + 1
        val = p * p + 1
        rem = val
        for q in trial_primes:
            if q > p:
                break
            if rem % q == 0:
                while rem % q == 0:
                    rem //= q
                if rem == 1:
                    break
                if rem <= p:
                    break
        
        if rem > p and rem != 1:
            if is_prime(rem):
                continue
            if not is_p_smooth(rem, p, trial_primes):
                continue
        
        k = _check_run(p, 200, trial_primes)
        if k > best_k:
            best_k = k
            best_p = p
    
    # Second pass: segmented sieve from SIEVE_LIMIT to 10^9
    if time.time() < deadline - 5.0:
        SEG_SIZE = 1_000_000
        start = SIEVE_LIMIT + 1
        end = min(1_000_000_000, SIEVE_LIMIT + 500_000_000)  # Try up to 500M first
        
        while start <= end and time.time() < deadline - 5.0:
            seg_end = min(start + SEG_SIZE - 1, end)
            seg = bytearray(b'\x01') * (seg_end - start + 1)
            
            for q in small_primes:
                if q * q > seg_end:
                    break
                # Find first multiple of q in [start, seg_end]
                first = ((start + q - 1) // q) * q
                if first == q:
                    first += q
                for j in range(first - start, seg_end - start + 1, q):
                    seg[j] = 0
            
            # Check candidates in this segment
            for offset in range(seg_end - start + 1):
                if seg[offset]:
                    p = start + offset
                    if p > 10**9:
                        break
                    
                    # Fast pre-filter: check p^2 + 1
                    val = p * p + 1
                    rem = val
                    for q in trial_primes:
                        if q > p:
                            break
                        if rem % q == 0:
                            while rem % q == 0:
                                rem //= q
                            if rem == 1:
                                break
                            if rem <= p:
                                break
                    
                    if rem > p and rem != 1:
                        if is_prime(rem):
                            continue
                        if not is_p_smooth(rem, p, trial_primes):
                            continue
                    
                    k = _check_run(p, 200, trial_primes)
                    if k > best_k:
                        best_k = k
                        best_p = p
            
            start = seg_end + 1
    
    return {'p': best_p, 'k': best_k}
