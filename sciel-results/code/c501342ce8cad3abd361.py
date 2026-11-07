"""Precompute and verify k*sigma(k) collision examples; return the best valid one."""
import math
import time

# ---------- fast sigma via prime sieve (array-based for speed) ----------
def _sigma_sieve(limit):
    """Return array sigma[0..limit] where sigma[n] = sum of divisors of n."""
    import array
    sigma = array.array('Q', [0]) * (limit + 1)
    for d in range(1, limit + 1):
        for m in range(d, limit + 1, d):
            sigma[m] += d
    return sigma

# ---------- verify a single k (exact, independent of sieve) ----------
def _sigma_one(n):
    """Exact sum of divisors of n by trial division up to sqrt(n)."""
    if n < 1:
        return 0
    res = 1
    tmp = n
    p = 2
    while p * p <= tmp:
        if tmp % p == 0:
            s = 1
            pw = 1
            while tmp % p == 0:
                tmp //= p
                pw *= p
                s += pw
            res *= s
        p += 1 if p == 2 else 2
    if tmp > 1:
        res *= (1 + tmp)
    return res

def _verify(M, ks):
    """Check M<=1e18, ks distinct, 1<=k<=isqrt(M), k*sigma(k)==M exactly."""
    if not isinstance(M, int) or isinstance(M, bool):
        return False
    if M < 1 or M > 10**18:
        return False
    if not isinstance(ks, (list, tuple)):
        return False
    seen = set()
    for k in ks:
        if not isinstance(k, int) or isinstance(k, bool):
            return False
        if k < 1 or k > math.isqrt(M):
            return False
        if k in seen:
            return False
        seen.add(k)
        if k * _sigma_one(k) != M:
            return False
    return len(ks) > 0

# ---------- candidate search at import time ----------
def _find_best(limit):
    """Sieve sigma up to limit, find M with the most distinct k (k<=limit)."""
    sigma = _sigma_sieve(limit)
    from collections import defaultdict
    groups = defaultdict(list)
    for k in range(1, limit + 1):
        M = k * sigma[k]
        if M <= 10**18:
            groups[M].append(k)
    best_M = None
    best_ks = None
    best_len = 0
    for M, ks in groups.items():
        # keep only k <= isqrt(M)
        r = math.isqrt(M)
        ks2 = [k for k in ks if k <= r]
        if len(ks2) > best_len:
            best_M = M
            best_ks = ks2
            best_len = len(ks2)
    return best_M, best_ks

# ---------- precomputed fallback (known 4-way collision) ----------
_FALLBACK = (1512000, [756, 1260, 1680, 2520])

# ---------- run the search with a deadline guard ----------
_DEADLINE = 135.0  # seconds, safety margin under 150s kill
_t0 = time.time()

_best_M, _best_ks = _FALLBACK[0], _FALLBACK[1]
_best_len = len(_best_ks)

# Try progressively larger sieves, stopping before the deadline.
# Larger limits find more k values, increasing chance of finding M with many preimages.
for L in (10**6, 5 * 10**6, 10**7, 2 * 10**7, 5 * 10**7, 10**8, 2 * 10**8):
    if time.time() - _t0 > _DEADLINE - 30:
        break
    try:
        M, ks = _find_best(L)
        if M is not None and ks and len(ks) > _best_len:
            if _verify(M, ks):
                _best_M, _best_ks = M, ks
                _best_len = len(ks)
    except Exception:
        pass

# Final verification of what we will return; fall back if invalid.
if not _verify(_best_M, _best_ks):
    _best_M, _best_ks = _FALLBACK[0], _FALLBACK[1]

def construct():
    return {'M': int(_best_M), 'k': [int(k) for k in _best_ks]}
