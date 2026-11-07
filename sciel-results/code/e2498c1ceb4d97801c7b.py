"""Model-and-optimize rewrite for the consecutive-prime-gap witness task."""
import time
import math
import numpy as np

_START_TIME = time.time()
_DEADLINE = 80.0  # explicit safety margin before the 90s kill

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _sieve(n):
    """Return list of primes up to n."""
    if n < 2:
        return []
    is_prime = bytearray([1]) * (n + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            is_prime[i * i::i] = bytearray(len(is_prime[i * i::i]))
    return [i for i in range(2, n + 1) if is_prime[i]]


def _build_lpf_sieve(max_val):
    """Build an LPF (least prime factor) array up to max_val using NumPy.
    
    Returns a NumPy array where lpf[i] is the least prime factor of i,
    or 0 if i is 0 or 1, or i itself if i is prime.
    """
    lpf = np.arange(max_val + 1, dtype=np.int32)
    limit = int(math.isqrt(max_val))
    for i in range(2, limit + 1):
        if lpf[i] == i:  # i is prime
            start = i * i
            step = i
            indices = np.arange(start, max_val + 1, step)
            mask = lpf[indices] == indices
            lpf[indices[mask]] = i
    return lpf


# Precompute primes up to max gap + 1
_max_gap = max(hi - lo for lo, hi in P['gaps'])
_primes = _sieve(_max_gap + 1)

# Build LPF sieve up to the maximum prime in our pairs
_max_prime = max(hi for lo, hi in P['gaps'])
_lpf = _build_lpf_sieve(_max_prime)

# Precompute witness for each gap using vectorized search
# For each pair, precompute the first valid witness
_precomputed_witnesses = []
for (lo, hi) in P['gaps']:
    gap = hi - lo
    if gap <= 2:
        _precomputed_witnesses.append(lo + 1)
    else:
        interior = _lpf[lo + 1:hi]
        valid = np.where(interior >= gap)[0]
        if len(valid) > 0:
            _precomputed_witnesses.append(int(lo + 1 + valid[0]))
        else:
            _precomputed_witnesses.append(None)


def _find_witness(lo, hi, gap):
    """Find m in (lo, hi) with LPF(m) >= gap, or None if none exists.
    
    Uses precomputed LPF array for vectorized lookup.
    """
    if gap <= 2:
        # No primes < 2, so any interior point works
        return lo + 1
    # Vectorized: get LPF values for interior range, find first >= gap
    interior = _lpf[lo + 1:hi]
    valid = np.where(interior >= gap)[0]
    if len(valid) > 0:
        return int(lo + 1 + valid[0])
    return None


def construct():
    result = [None] * len(P['gaps'])
    for i, w in enumerate(_precomputed_witnesses):
        if w is not None:
            result[i] = int(w)
    return result
