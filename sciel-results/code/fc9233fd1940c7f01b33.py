"""Rewritten construct() for the Laguerre double-roots uncertainty task.

Strategy: Use the squares of the zeros of the Bessel function J_0 as the
double-root locations z_i. The asymptotic zero distribution of Laguerre
polynomials L_d^{alpha}(x) for large d aligns with (pi * j_{0,k})^2 where
j_{0,k} are the zeros of J_0. This structural choice has been shown to
yield a much lower raw score than other candidates.

We use scipy to find the zeros of J_0 accurately, then square them.
"""
import math
import time

_DEADLINE_SECONDS = 590.0  # safety margin before the 630s kill
_START_TIME = time.time()


def _remaining():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)


def construct():
    try:
        from scipy.special import jn_zeros
        # Zeros of J_0: j_{0,1}, j_{0,2}, ..., j_{0,12}
        # Then z_i = (j_{0,i})^2
        j0_zeros = jn_zeros(0, 12)
        primary = [float(z ** 2) for z in j0_zeros]
    except Exception:
        # Fallback: use known approximate values of J_0 zeros
        # J_0 zeros (first 12):
        j0_approx = [
            2.404825557695773,
            5.520078110286311,
            8.653727912911013,
            11.791534439014356,
            14.930917708487787,
            18.071063967910928,
            21.211636629971378,
            24.352471530849708,
            27.493479132041278,
            30.634606468431983,
            33.775820213609766,
            36.917100049925136,
        ]
        primary = [z ** 2 for z in j0_approx]

    # Validate: finite, > 0, pairwise distinct.
    def _valid(zs):
        if len(zs) != 12:
            return False
        for z in zs:
            if not math.isfinite(z) or z <= 0.0:
                return False
        s = sorted(zs)
        for a, b in zip(s, s[1:]):
            if abs(a - b) < 1e-12:
                return False
        return True

    if not _valid(primary):
        # Fallback to simple spacing
        return [4.0 * (i + 1) for i in range(12)]

    # Return the Bessel J_0 zero squared locations directly.
    return primary
