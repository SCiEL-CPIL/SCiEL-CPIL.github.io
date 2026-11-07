"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from fractions import Fraction
from math import isqrt

MAX_N = 10000000
TARGET = Fraction(3, 2)


def _sieve_spf(limit):
    """Return smallest prime factor for each number up to limit."""
    spf = list(range(limit + 1))
    if limit >= 0:
        spf[0] = 0
    if limit >= 1:
        spf[1] = 1
    for i in range(2, isqrt(limit) + 1):
        if spf[i] == i:  # i is prime
            step = i
            start = i * i
            for j in range(start, limit + 1, step):
                if spf[j] == j:
                    spf[j] = i
    return spf


def _sigma_from_sf(sp, n):
    """Compute sigma(n) using smallest prime factor table."""
    result = 1
    while n > 1:
        p = sp[n]
        power = p
        s = 1 + p
        n //= p
        while n > 1 and sp[n] == p:
            n //= p
            power *= p
            s += power
        result *= s
    return result


def _ratio_err_frac(a, b):
    """Return |a/b - 3/2| as a Fraction (exact)."""
    return abs(Fraction(a, b) - TARGET)


def _ratio_err_float(a, b):
    """Return |a/b - 3/2| as float (fast, exact-zero -> inf)."""
    if 2 * a == 3 * b:
        return float('inf')
    return abs(2 * a - 3 * b) / (2.0 * b)


def _sigma_from_factor_pairs(factors):
    """Compute sigma from list of (p, e) factor pairs."""
    s = 1
    for p, e in factors:
        s *= (p ** (e + 1) - 1) // (p - 1)
    return s


def construct():
    start_time = time.time()
    deadline = start_time + 78.0  # safety margin before 90s kill

    # ---------- Phase 1: Build sigma table and search sigma-fibers ----------
    sp = _sieve_spf(MAX_N)

    # Compute sigma for all n via spf
    sigma = [0] * (MAX_N + 1)
    sigma[1] = 1
    for i in range(2, MAX_N + 1):
        sigma[i] = _sigma_from_sf(sp, i)

    if time.time() > deadline:
        return [11, 6]

    # Group numbers by sigma value
    sigma_map = {}
    for i in range(1, MAX_N + 1):
        s = sigma[i]
        if s not in sigma_map:
            sigma_map[s] = []
        sigma_map[s].append(i)

    # Search for best pair within sigma-fibers
    best_a, best_b = 11, 6
    best_err = _ratio_err_frac(11, 6)

    for s, nums in sigma_map.items():
        if time.time() > deadline - 2.0:
            break
        if len(nums) < 2:
            continue
        nums.sort()
        m = len(nums)
        # Two-pointer: find pair (a, b) with a/b close to 3/2
        # We want 2*a ≈ 3*b, so a ≈ 1.5*b
        i = 0
        for j in range(m):
            b = nums[j]
            target_a = 1.5 * b
            # Advance i so that nums[i] >= target_a
            while i < m and nums[i] < target_a - 1:
                i += 1
            # Check candidates around i
            for ii in (i - 1, i):
                if 0 <= ii < m and ii != j:
                    a = nums[ii]
                    if a != b:
                        err = _ratio_err_frac(a, b)
                        if err < best_err:
                            best_err = err
                            best_a, best_b = a, b

    if time.time() > deadline - 3.0:
        return [int(best_a), int(best_b)]

    # ---------- Phase 2: Parametric family refinement ----------
    # Families: a = A0*p + A1, b = B0*p + B1, with sigma(a) = sigma(b)
    # For specific prime p, sigma(A0*p+A1) = sigma(A1) * sigma(p) if p ∤ A1
    # and similar for b. We search primes near the optimal ratio point.

    def _is_prime_fast(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    # (A0, A1, A_sigma_factors, B0, B1, B_sigma_factors)
    # sigma(A0*p + A1) must equal sigma(B0*p + B1) for prime p
    # Typically A0*p+A1 = A1 * p (when A0 = A1, i.e. a = A1*(p+1))
    # or more general forms. We use known valid families.
    FAMILIES = [
        # (A0, A1, sigma_A_base, B0, B1, sigma_B_base)
        # where sigma(A0*p+A1) = sigma_A_base * (p+1) if p ∤ A1
        # and sigma(B0*p+B1) = sigma_B_base * (p+1) if p ∤ B1
        # Need sigma_A_base == sigma_B_base
        (3, 1, 11, 1, 2, 11),    # a=3p+1, b=p+2; sigma(3p+1)=11*(p+1) if 3p+1=p*3+1... 
        # Actually let's use verified families from inspiration 2:
    ]

    # Use the families from inspiration 2 which are verified to have matching sigma
    # Format: (A0, A1, A_sf, B0, B1, B_sf)
    FAMILIES2 = [
        ((3, 1, [(11, 1)]), (1, 2, [(6, 1)])),
        ((6, 5, [(14, 1)]), (3, 2, [(6, 1)])),
        ((5, 2, [(14, 1)]), (3, 1, [(6, 1)])),
        ((5, 1, [(11, 1)]), (3, 1, [(6, 1)])),
        ((5, 2, [(14, 1)]), (2, 1, [(6, 1)])),
        ((6, 5, [(14, 1)]), (2, 1, [(6, 1)])),
        ((3, 1, [(11, 1)]), (2, 1, [(6, 1)])),
        ((4, 1, [(14, 1)]), (3, 1, [(6, 1)])),
        ((11, 3, [(2, 2), (5, 1)]), (1, 2, [(11, 1)])),
        ((22, 6, [(2, 2), (5, 1)]), (2, 4, [(11, 1)])),
        ((2, 5, [(6, 1)]), (1, 1, [(6, 1)])),
        ((4, 10, [(6, 1)]), (2, 2, [(6, 1)])),
        ((2, 3, [(10, 1)]), (1, 1, [(6, 1)])),
        ((2, 3, [(10, 1)]), (1, 2, [(6, 1)])),
        ((2, 3, [(10, 1)]), (1, 3, [(12, 1)])),
        ((2, 3, [(10, 1)]), (1, 4, [(15, 1)])),
        ((3, 1, [(11, 1)]), (1, 1, [(11, 1)])),
        ((5, 2, [(14, 1)]), (1, 2, [(14, 1)])),
        ((5, 2, [(14, 1)]), (1, 3, [(15, 1)])),
        ((5, 1, [(11, 1)]), (1, 1, [(11, 1)])),
        ((5, 1, [(11, 1)]), (1, 2, [(13, 1)])),
        ((11, 3, [(2, 2), (5, 1)]), (1, 1, [(11, 1)])),
        ((11, 3, [(2, 2), (5, 1)]), (1, 3, [(15, 1)])),
        ((11, 3, [(2, 2), (5, 1)]), (1, 5, [(21, 1)])),
        ((11, 3, [(2, 2), (5, 1)]), (1, 6, [(22, 1)])),
    ]

    for A, B in FAMILIES2:
        if time.time() > deadline - 2.0:
            break
        A0, A1, Asf = A
        B0, B1, Bsf = B
        # Verify base sigmas match
        if _sigma_from_factor_pairs(Asf) != _sigma_from_factor_pairs(Bsf):
            continue

        # Find optimal p where a/b ≈ 3/2
        # a = A0*p + A1, b = B0*p + B1
        # a/b = 3/2 => 2*A0*p + 2*A1 = 3*B0*p + 3*B1
        # p*(2*A0 - 3*B0) = 3*B1 - 2*A1
        num = 3 * B1 - 2 * A1
        den = 2 * A0 - 3 * B0
        if den != 0:
            pm = num / den
        elif B0 != 0:
            pm = (A1 - 1.5 * B1) / (1.5 * B0)
        else:
            pm = 0.0

        base = int(pm)
        # Search primes in a window around base
        window = 50000
        lo = max(2, base - window)
        hi = min(MAX_N // max(A0, B0, 1) - 1, base + window)

        for p in range(lo, hi + 1):
            if not _is_prime_fast(p):
                continue
            a = A0 * p + A1
            b = B0 * p + B1
            if a < 2 or b < 2 or a > MAX_N or b > MAX_N:
                continue
            if a == b:
                continue
            # Verify sigma equality
            if sigma[a] != sigma[b]:
                continue
            err = _ratio_err_frac(a, b)
            if err < best_err:
                best_err = err
                best_a, best_b = a, b

    # Final verification
    if sigma[best_a] != sigma[best_b]:
        best_a, best_b = 11, 6

    return [int(best_a), int(best_b)]
# EVOLVE-BLOCK-END
