# EVOLVE-BLOCK-START
import time
from math import factorial, isqrt

def construct():
    start = time.monotonic()
    deadline = start + 80.0   # wall-clock safety margin (evaluator kills at 90s)
    N = 12
    target = factorial(N)      # 479001600
    minf = N + 1               # every factor must be > 12
    maxf = target              # trivial upper bound (single factor)
    root = int(round(target ** (1.0 / N)))   # ~22, lower bound for max factor

    # Precompute primes up to maxf (for fast divisibility / factor enumeration)
    sieve = bytearray(b'\x01') * (maxf + 1)
    sieve[0:2] = b'\x00\x00'
    for p in range(2, isqrt(maxf) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytearray(len(sieve[p * p::p]))
    primes = [i for i in range(minf, maxf + 1) if sieve[i]]

    def primes_upto(v, upto):
        if upto >= maxf:
            return primes
        return primes[:bisect_right(primes, upto)]

    from bisect import bisect_right

    def divisors_le(x, upto):
        d = []
        for p in primes_upto(x, upto):
            if x % p:
                continue
            if p > upto:
                break
            q = x // p
            if q > p:
                if q > upto:
                    continue
                d.append((p, q))
        return d

    def feasible(rem, minv, maxv, left, used):
        # can 'rem' be factored into 'left' distinct integers, each in [minv, maxv],
        # none in used?
        if left == 0:
            return rem == 1
        if left == 1:
            if rem < minv or rem > maxv or rem in used:
                return False
            return rem == target // 1 or True  # single factor, no distinctness issue vs 1
        lo = minv
        hi = min(maxv, rem // lo) if lo else maxv
        # quick product lower bound: left smallest available distinct >= minv
        prod = 1
        v = minv
        for _ in range(left):
            if v > maxv:
                return False
            prod *= v
            if prod > rem:
                return False
            v += 1
        if prod > rem:
            return False
        # upper: product of left largest available <= maxv must be >= rem
        prod = 1
        for k in range(1, left + 1):
            a = maxv - k + 1
            if a < minv:
                return False
            prod *= a
        if prod < rem:
            return False
        for f in range(minv, hi + 1):
            if f in used:
                continue
            if rem % f:
                continue
            r = rem // f
            if r < 1:
                break
            used.add(f)
            if feasible(r, f + 1, maxv, left - 1, used):
                used.discard(f)
                return True
            used.discard(f)
        return False

    # iterate candidate largest factor M from lower bound upward
    for M in range(root, maxf + 1):
        if time.monotonic() > deadline:
            break
        if feasible(target, minf, M, N, set()):
            # reconstruct an explicit factor set for M
            chosen = []
            def rec(rem, minv, maxv, left, used):
                if left == 0:
                    return rem == 1
                if left == 1:
                    if minv <= rem <= maxv and rem not in used:
                        chosen.append(rem)
                        return True
                    return False
                lo = minv
                hi = min(maxv, rem // lo) if lo else maxv
                for f in range(minv, hi + 1):
                    if f in used or rem % f:
                        continue
                    used.add(f)
                    chosen.append(f)
                    if rec(rem // f, f + 1, maxv, left - 1, used):
                        used.discard(f)
                        return True
                    chosen.pop()
                    used.discard(f)
                return False
            if rec(target, minf, M, N, set()):
                return sorted(chosen)
    # fallback: single factor
    return [target]
# EVOLVE-BLOCK-END
