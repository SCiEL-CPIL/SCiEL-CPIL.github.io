import time
import math

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

def _sieve(limit):
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i * i::i] = bytearray(len(is_prime[i * i::i]))
    return [i for i in range(limit + 1) if is_prime[i]]

def _next_prime_after(a):
    if a <= 2:
        return 2
    if a % 2 == 0:
        a += 1
    while not _is_prime_mr(a):
        a += 2
    return a

def _bulk_psmooth(p, primes, deadline):
    """Return the max k in [1,200] such that p^2+1..p^2+k are all p-smooth."""
    p2 = p * p
    # residue[i] = (p^2 + i) after removing all prime factors <= p
    residue = [p2 + i for i in range(201)]
    for q in primes:
        if q > p:
            break
        # find one multiple of q in (p^2, p^2+200]
        rem = p2 % q
        start = q - rem if rem else 0
        if start > 200:
            continue
        for off in range(start, 201, q):
            t = residue[off]
            if t % q == 0:
                while t % q == 0:
                    t //= q
                residue[off] = t
    k = 0
    for i in range(1, 201):
        t = residue[i]
        if t == 1:
            k = i
        elif t <= p:
            # t is prime (no factor <= p remains, and t <= p) -> p-smooth
            k = i
        else:
            break
        if time.time() > deadline:
            break
    return k

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # primes up to ~400000 for bulk sieve; primes up to 100000 for pre-filters
    primes = _sieve(400000)
    small_primes = _sieve(100000)

    best_k = 0
    best_p = 3

    # ---- Phase 1: exhaustive over primes <= 400000 (cheap bulk sieve) ----
    for p in primes:
        if time.time() > deadline:
            break
        k = _bulk_psmooth(p, primes, deadline)
        if k > best_k:
            best_k = k
            best_p = p
            if best_k == 200:
                break

    # ---- Phase 2: deterministic stride over larger primes ----
    # fast pre-filters: first 200 numbers must be 2^a3^b5^c6^d-smooth
    for start in range(400000, 1000000000, 6):
        if time.time() > deadline:
            break
        p = _next_prime_after(start)
        if p > 1000000000:
            break
        p2 = p * p
        # Pre-filter: check 2^a3^b5^c6^d smoothness of the first few offsets.
        # If even the first 5 offsets fail, skip this p entirely (cheap reject).
        ok = True
        for i in (1, 2, 3, 4, 5):
            t = p2 + i
            for q in (2, 3, 5, 6):
                while t % q == 0:
                    t //= q
            if t > p:
                ok = False
                break
        if not ok:
            continue
        # Full bulk verification for this p
        k = _bulk_psmooth(p, primes, deadline)
        if k > best_k:
            best_k = k
            best_p = p
            if best_k == 200:
                break

    return {'p': best_p, 'k': best_k}
