"""Program for: Long runs after p^2 with no prime factor exceeding p (p <= 10^9)."""

import time
import random
import math

# ---------- Deterministic Miller-Rabin for n < 3.3e24 ----------
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def is_prime(n):
    if n < 2:
        return False
    for p in _MR_BASES:
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
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
        x = random.randrange(2, n)
        y = x
        c = random.randrange(1, n)
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
        factors = []
    if n == 1:
        return factors
    if is_prime(n):
        factors.append(n)
        return factors
    d = pollard_rho(n)
    factor(d, factors)
    factor(n // d, factors)
    return factors

def sieve_primes(limit):
    """Generate list of all primes up to limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return [i for i in range(2, limit + 1) if is_p[i]]

def is_p_smooth(n, p, small_primes):
    """Check if all prime factors of n are <= p."""
    if n <= 1:
        return True
    if is_prime(n):
        return n <= p
    
    v = n
    sqrt_n = math.isqrt(n)
    limit = min(p, sqrt_n)
    
    for sp in small_primes:
        if sp > limit:
            break
        while v % sp == 0:
            if sp > p:
                return False
            v //= sp
    
    if v == 1:
        return True
    
    if is_prime(v):
        return v <= p
    
    return False

def construct():
    deadline = time.time() + 110  # safety margin before 150s kill
    
    best_k = 0
    best_p = 2
    
    # Precompute small primes for trial division
    small_primes = sieve_primes(200000)
    
    # Strategy: For each prime p, we need p^2+1, ..., p^2+k all to be p-smooth.
    # Key insight: For a fixed offset j, p^2 + j has a prime factor > p iff
    # there exists a prime q > p such that q | (p^2 + j).
    # Equivalently, p^2 ≡ -j (mod q) has a solution, i.e., -j is a quadratic residue mod q.
    # 
    # Structural approach: Precompute for each small prime q the set of offsets j in [1,200]
    # where p^2 + j could have q as a factor (i.e., -j is QR mod q).
    # For a given p, if q > p and q divides p^2+j for some j in [1,k], then that j is "bad".
    #
    # More practically: For each p, compute which offsets j in [1,200] are "potentially bad"
    # (have a prime factor > p), then verify the remaining candidates.
    #
    # Even faster: For small p, most offsets will be p-smooth. For large p, fewer will be.
    # The run length is limited by the first offset where a prime factor > p appears.
    
    # Phase 1: Search primes up to 5M with efficient per-prime checking
    search_limit = 5000000
    primes_to_search = sieve_primes(search_limit)
    
    # For efficiency, precompute p^2 mod small_primes for quick rejection
    # If p^2 + j ≡ 0 (mod q) for some prime q > p and q <= 200000, then j is bad.
    # We can precompute for each q the residues r where r^2 ≡ -j (mod q) for j=1..200.
    
    # Build a table: for each prime q (up to some limit), for each j in 1..200,
    # check if -j is a quadratic residue mod q. If so, then for any p with p^2 ≡ -j (mod q),
    # q divides p^2 + j.
    
    # But we need q > p. So for a given p, we only care about q in (p, q_max].
    # This means for small p, many q's are relevant.
    
    # Alternative structural approach: bitset sieve over p values.
    # For each prime q up to Q_max, and each j in 1..200 where -j is QR mod q,
    # mark all p such that p^2 ≡ -j (mod q) as having offset j "bad" (if q > p).
    # Then for each p, the maximum k is the largest k such that offsets 1..k are all "good".
    
    # Let's implement this:
    # 1. For each prime q up to Q_max (say 10^6), compute the set of j in [1,200] 
    #    where -j is a QR mod q.
    # 2. For each such (q, j), find the residues r mod q where r^2 ≡ -j (mod q).
    # 3. For each prime p < q with p ≡ r (mod q) for some such r, mark offset j as bad for p.
    # 4. For each p, compute max consecutive good offsets starting from 1.
    
    # This is complex. Let me use a hybrid: fast pre-filter + verification.
    
    # Pre-filter: For each p, quickly check if p^2+j has any small prime factor > p.
    # For j in 1..200, compute p^2+j mod q for small primes q. If q > p and q | (p^2+j), bad.
    
    # Even faster: precompute for each j in 1..200 and each small prime q, 
    # whether -j is a QR mod q. Store as a bitmask per (j, q).
    
    # Let's precompute QR tables for primes up to 50000
    qr_limit = 50000
    qr_primes = [q for q in small_primes if q <= qr_limit]
    
    # qr_table[j][q_idx] = True if -j is a QR mod qr_primes[q_idx]
    # And qr_residues[j][q_idx] = list of residues r where r^2 ≡ -j (mod q)
    
    # This might be too memory-intensive. Let's use a simpler approach.
    
    # Simple fast approach: For each prime p, check offsets 1..200 using modular arithmetic
    # with precomputed small primes.
    
    for p in primes_to_search:
        if time.time() > deadline:
            break
        
        base = p * p
        k = 0
        
        for j in range(1, 201):
            n = base + j
            
            # Quick check: is n p-smooth?
            # Trial divide by primes up to min(p, sqrt(n))
            v = n
            smooth = True
            sqrt_v = math.isqrt(v)
            
            for sp in small_primes:
                if sp > p or sp > sqrt_v:
                    break
                if v % sp == 0:
                    while v % sp == 0:
                        v //= sp
                    sqrt_v = math.isqrt(v)
            
            if v > 1:
                # Remaining factor v > 1
                # Check if v is prime
                if is_prime(v):
                    if v > p:
                        smooth = False
                else:
                    # v is composite, has a prime factor > sqrt(v) > p (since we divided all <= min(p,sqrt_v))
                    # Actually need to be more careful
                    smooth = False
            
            if smooth:
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
    
    # Phase 2: If time remains, try larger primes with segmented sieve
    if best_k < 200 and time.time() < deadline - 30:
        # For larger p, the condition is easier to satisfy since p is the bound
        # Try primes in [5M, 50M]
        for start in range(5000000, 50000000, 5000000):
            if time.time() > deadline - 20:
                break
            end = min(start + 5000000, 50000000)
            
            # Segmented sieve
            window = end - start
            sieve = [True] * window
            for sp in small_primes:
                if sp * sp > end:
                    break
                first = ((start + sp - 1) // sp) * sp
                for j in range(first - start, window, sp):
                    sieve[j] = False
            
            for i in range(window):
                if time.time() > deadline - 20:
                    break
                cand = start + i
                if cand < 2:
                    continue
                if sieve[i] and is_prime(cand):
                    base = cand * cand
                    k = 0
                    for j in range(1, 201):
                        n = base + j
                        v = n
                        smooth = True
                        sqrt_v = math.isqrt(v)
                        
                        for sp in small_primes:
                            if sp > cand or sp > sqrt_v:
                                break
                            if v % sp == 0:
                                while v % sp == 0:
                                    v //= sp
                                sqrt_v = math.isqrt(v)
                        
                        if v > 1:
                            if is_prime(v):
                                if v > cand:
                                    smooth = False
                            else:
                                smooth = False
                        
                        if smooth:
                            k += 1
                        else:
                            break
                    
                    if k > best_k:
                        best_k = k
                        best_p = cand
    
    if best_p == 0:
        best_p = 2
        best_k = 1
    
    return {'p': best_p, 'k': best_k}
