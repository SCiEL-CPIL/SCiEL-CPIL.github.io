import time
import math
import random
import numpy as np

# ---------- Deterministic Miller-Rabin for 64-bit ----------
_MILLER_RABIN_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def _is_prime_mr(n):
    if n < 2:
        return False
    for p in _MILLER_RABIN_BASES:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MILLER_RABIN_BASES:
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
    if n % 5 == 0:
        return 5
    while True:
        y = random.randrange(1, n)
        c = random.randrange(1, n)
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

def _sieve(limit):
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # ---------- Precompute ----------
    # We'll search p up to ~50,000,000. Sieve primes up to that for candidate generation
    # and also for the "danger zone" primes used in the congruence sieve.
    SIEVE_LIMIT = 50_000_000
    small_primes = _sieve(SIEVE_LIMIT)
    
    # Primes for quick trial-division stripping (up to 10,000)
    strip_primes = [p for p in small_primes if p <= 10_000]
    
    # "Danger zone" primes for the congruence sieve: primes q where q > 10,000
    # For a candidate p, if p^2 + j is divisible by q and q > p, then p is NOT a valid
    # candidate for run length >= j. We use q in range (10000, SIEVE_LIMIT] for sieving.
    # But we only need q where q could divide p^2+j for p in our search range.
    # For p up to 50M, p^2+200 up to 2.5*10^15. So q up to 2.5*10^15 could be a factor,
    # but we can only sieve with q up to SIEVE_LIMIT (50M) efficiently.
    # Actually, the key insight: if q > p and q | (p^2+j), then p^2+j = q * r where r <= p+200/p.
    # For the congruence sieve, we mark p as "bad for run >= j" if there exists q > 10000
    # (our strip limit) such that q | (p^2+j). We can only handle q up to our sieve limit.
    # But for q > SIEVE_LIMIT, we can't sieve. However, for p <= 50M, if q > 50M and q | (p^2+j),
    # then r = (p^2+j)/q < p^2/50M = p/1000 (for p=50M). So r is small, and we'd catch it
    # in verification. The congruence sieve is a *filter*, not a complete verifier.
    
    # ---------- Phase 1: Exhaustive search for small p (p <= 100,000) ----------
    # For small p, we can do full verification quickly.
    best_k = 0
    best_p = 3
    offsets = np.arange(1, 201, dtype=np.int64)
    
    phase1_primes = [p for p in small_primes if p <= 100_000]
    BATCH = 50
    for batch_start in range(0, len(phase1_primes), BATCH):
        if time.time() > deadline:
            break
        batch = phase1_primes[batch_start:batch_start+BATCH]
        B = len(batch)
        p_arr = np.array(batch, dtype=np.int64)
        vals = p_arr[:, None] * p_arr[:, None] + offsets[None, :]
        
        # Strip small primes
        for q in strip_primes:
            mask_p = p_arr >= q
            if not np.any(mask_p):
                continue
            sub_vals = vals[mask_p]
            for _ in range(10):
                div_mask = (sub_vals % q == 0)
                if not np.any(div_mask):
                    break
                sub_vals[div_mask] //= q
            vals[mask_p] = sub_vals
        
        for i in range(B):
            p_val = batch[i]
            row = vals[i]
            k = 0
            for j in range(200):
                v = row[j]
                if v == 1:
                    k = j + 1
                    continue
                if v <= p_val:
                    # After stripping primes <= 10000, if v <= p_val and p_val <= 100000,
                    # v is either 1 (handled) or prime (since any composite v with factors > 10000
                    # would have product > 10^8 > p_val for p_val <= 10000). So v is prime <= p_val.
                    k = j + 1
                else:
                    # v > p_val: check if prime
                    if _is_prime_mr(int(v)):
                        break
                    else:
                        factors = []
                        _factor(int(v), factors)
                        if all(f <= p_val for f in factors):
                            k = j + 1
                        else:
                            break
            if k > best_k:
                best_k = k
                best_p = p_val
                if best_k == 200:
                    return {'p': best_p, 'k': best_k}
    
    # ---------- Phase 2: Congruence-based segmented sieve for p in (100,000, 50,000,000] ----------
    # For each target run length k_target (from best_k+1 up to 200), we want to find primes p
    # such that p^2+1, ..., p^2+k_target are all p-smooth.
    # The congruence sieve: for a given k_target, mark p as "survivor" if for all j in 1..k_target,
    # p^2+j has no prime factor q in (10000, SIEVE_LIMIT] with q > p.
    # 
    # Implementation: For each q in (10000, SIEVE_LIMIT], for each j in 1..k_target,
    # solve p^2 ≡ -j (mod q). The solutions p mod q are "bad residues" for run length >= j.
    # We mark a bitarray for p in [LO, HI] as bad if p mod q is in the bad residues.
    #
    # But doing this for ALL q up to 50M is too expensive. Instead, we use a *probabilistic* filter:
    # only use a subset of q values (e.g., the first 10,000 primes above 10,000) to get a strong
    # filter, then verify survivors with full factorization.
    
    # We'll search in segments of p. For each segment, build a survivor bitarray.
    SEG_SIZE = 1_000_000  # 1M candidates per segment
    LO = 100_000
    HI = min(50_000_000, SIEVE_LIMIT)
    
    # Use a subset of danger-zone primes for the sieve
    danger_primes = [q for q in small_primes if 10_000 < q <= 500_000]  # ~30,000 primes
    
    # For efficiency, we'll only use the first 5000 danger primes for the sieve
    # This gives a strong filter without being too slow.
    sieve_primes = danger_primes[:5000]
    
    for seg_start in range(LO, HI, SEG_SIZE):
        if time.time() > deadline:
            break
        seg_end = min(seg_start + SEG_SIZE, HI)
        seg_len = seg_end - seg_start
        if seg_len <= 0:
            break
        
        # Survivor bitarray: 1 = potential survivor
        survivors = bytearray(b'\x01') * seg_len
        
        # For each danger prime q, mark bad residues
        for q in sieve_primes:
            # For each j in 1..200, solve p^2 ≡ -j (mod q)
            # We only care about j up to 200 (max run length)
            for j in range(1, 201):
                target = (-j) % q
                # Solve p^2 ≡ target (mod q) using Tonelli-Shanks or brute force for small q
                # For q up to 500,000, we can use a simple approach
                # Check if target is a quadratic residue mod q
                # Use Euler's criterion: target^((q-1)/2) ≡ 1 (mod q)
                if pow(target, (q - 1) // 2, q) != 1:
                    continue  # No solution
                
                # Find square roots of target mod q
                # For prime q, use Tonelli-Shanks
                roots = _tonelli_shanks(target, q)
                for root in roots:
                    # Mark all p in [seg_start, seg_end] with p ≡ root (mod q) as bad
                    # First p >= seg_start with p ≡ root (mod q)
                    first = seg_start + ((root - seg_start) % q)
                    if first > seg_end:
                        continue
                    # Mark first, first+q, first+2q, ... < seg_end
                    for idx in range(first - seg_start, seg_len, q):
                        survivors[idx] = 0
        
        # Now find primes in the segment that are survivors
        # We need a primality sieve for this segment
        # Simple segmented sieve
        seg_primes = bytearray(b'\x01') * seg_len
        # Mark composites using small primes up to sqrt(seg_end)
        sqrt_seg_end = int(math.isqrt(seg_end)) + 1
        for sp in small_primes:
            if sp > sqrt_seg_end:
                break
            # First multiple of sp in [seg_start, seg_end]
            first_mult = ((seg_start + sp - 1) // sp) * sp
            for idx in range(first_mult - seg_start, seg_len, sp):
                seg_primes[idx] = 0
        
        # Combine: candidate p is prime AND survivor
        candidates = []
        for idx in range(seg_len):
            if seg_primes[idx] and survivors[idx]:
                p_val = seg_start + idx
                if p_val >= 2:
                    candidates.append(p_val)
        
        # Verify each candidate with full factorization
        for p_val in candidates:
            if time.time() > deadline:
                break
            p_sq = p_val * p_val
            k = 0
            for j in range(1, 201):
                val = p_sq + j
                t = val
                # Strip small primes
                for q in strip_primes:
                    if q > p_val:
                        break
                    while t % q == 0:
                        t //= q
                if t == 1:
                    k = j
                    continue
                if t <= p_val:
                    k = j
                else:
                    if _is_prime_mr(t):
                        break
                    else:
                        factors = []
                        _factor(t, factors)
                        if all(f <= p_val for f in factors):
                            k = j
                        else:
                            break
            if k > best_k:
                best_k = k
                best_p = p_val
                if best_k == 200:
                    return {'p': best_p, 'k': best_k}
        
        # Early exit if we've found a very good k
        if best_k >= 150:
            break
    
    # ---------- Phase 3: Random sampling for large p (50M to 10^9) ----------
    if best_k < 200 and time.time() < deadline - 30:
        for _ in range(500):
            if time.time() > deadline:
                break
            log_val = 7.7 + random.random() * 1.3  # 5*10^7 to 10^9
            p = int(10 ** log_val)
            if p % 2 == 0:
                p += 1
            if p > 10**9:
                continue
            while not _is_prime_mr(p):
                p += 2
                if p > 10**9:
                    break
            if p > 10**9 or not _is_prime_mr(p):
                continue
            
            p_sq = p * p
            k = 0
            for j in range(1, 201):
                if time.time() > deadline:
                    break
                val = p_sq + j
                t = val
                for q in strip_primes:
                    if q > p:
                        break
                    while t % q == 0:
                        t //= q
                if t == 1:
                    k = j
                    continue
                if t <= p:
                    k = j
                else:
                    if _is_prime_mr(t):
                        break
                    else:
                        factors = []
                        _factor(t, factors)
                        if all(f <= p for f in factors):
                            k = j
                        else:
                            break
            if k > best_k:
                best_k = k
                best_p = p
                if best_k == 200:
                    break
    
    return {'p': best_p, 'k': best_k}

def _tonelli_shanks(n, p):
    """Solve x^2 ≡ n (mod p) for prime p using Tonelli-Shanks."""
    if n == 0:
        return [0]
    if p == 2:
        return [n % 2]
    if pow(n, (p - 1) // 2, p) != 1:
        return []  # No solution
    
    # Factor p-1 as q * 2^s
    q = p - 1
    s = 0
    while q % 2 == 0:
        q //= 2
        s += 1
    
    if s == 1:
        x = pow(n, (p + 1) // 4, p)
        return [x, p - x] if x != p - x else [x]
    
    # Find a quadratic non-residue z
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    
    m = s
    c = pow(z, q, p)
    t = pow(n, q, p)
    r = pow(n, (q + 1) // 2, p)
    
    while t != 1:
        # Find least i such that t^(2^i) ≡ 1
        i = 0
        t2 = t
        while t2 != 1:
            t2 = (t2 * t2) % p
            i += 1
        
        b = pow(c, 1 << (m - i - 1), p)
        m = i
        c = (b * b) % p
        t = (t * c) % p
        r = (r * b) % p
    
    return [r, p - r] if r != p - r else [r]
