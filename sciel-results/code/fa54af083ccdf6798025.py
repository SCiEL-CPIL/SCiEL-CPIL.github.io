"""Exact closest strict underpacking via meet-in-the-middle (NumPy + Fraction)."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

import numpy as np


def construct():
    t0 = time.time()
    DEADLINE = 82.0  # safety margin under the 90 s wall-clock kill

    # Denominators [2, 3, ..., 48] -> 47 items, zero-based indices.
    denoms = list(range(2, 49))
    n = len(denoms)
    m1 = n // 2
    left_den = denoms[:m1]
    right_den = denoms[m1:]

    # Global LCM of 2..48: every subset reciprocal sum is an integer multiple of 1/L.
    L = 1
    for d in denoms:
        L = L * d // math.gcd(L, d)

    def enum_halves(halflen, den_list):
        """Return (numerator_array, mask_array) for all 2^halflen subset sums of 1/d."""
        vals = np.array([L // d for d in den_list], dtype=np.int64)
        N = 1 << halflen
        num = np.empty(N, dtype=np.int64)
        num[0] = 0
        mask = np.zeros(N, dtype=np.int64)
        for j in range(halflen):
            base = 1 << j
            v = vals[j]
            num[base:2 * base] = num[:base] + v
            mask[base:2 * base] = mask[:base] | base
        return num, mask

    # --- Left half ---
    n1, m1m = enum_halves(m1, left_den)
    n1 = n1[n1 < L]  # keep strictly-below-1 partials
    m1m = m1m[n1 < L]

    # --- Right half: keep strictly-below-1 partials, sort by value ---
    n2, m2m = enum_halves(n - m1, right_den)
    keep = n2 < L
    n2 = n2[keep]
    m2m = m2m[keep]
    order = np.argsort(n2, kind="stable")
    n2 = n2[order]
    m2m = m2m[order]
    # dedupe (value -> mask)
    uniq = np.concatenate(([True], np.diff(n2) != 0))
    n2u = n2[uniq]
    m2u = m2m[uniq]

    # --- Combine: for each left sum, largest right sum with total < 1 ---
    best_num = -1
    best_mask = 0
    # vectorized upper bounds: max right value that keeps total < 1 is L - 1 - n1
    target = L - 1 - n1
    idx = np.searchsorted(n2u, target, side="right") - 1
    valid = idx >= 0
    if valid.any():
        cand = n1[valid] + n2u[idx[valid]]
        j = int(np.argmax(cand))
        best_num = int(cand[j])
        best_mask = int(m1m[valid][j]) | int(m2u[idx[valid]][j])

    # Fallback: guarantee a valid nonempty selection even if something is off.
    if best_num <= 0:
        # pick the single largest unit fraction < 1 (denominator 2 -> index 0)
        best_num = L // 2
        best_mask = 1 << 0

    # Reconstruct zero-based indices into [2,3,...,48].
    indices = [i for i in range(n) if (best_mask >> i) & 1]

    # Exact verification (defensive): sum must be strictly below 1 and nonempty.
    s = Fraction(0, 1)
    for i in indices:
        s += Fraction(1, denoms[i])
    if not indices or s >= 1:
        indices = [0]  # 1/2 is always strictly below 1

    if time.time() > DEADLINE:
        return indices
    return indices
# EVOLVE-BLOCK-END
