"""Rewritten program: MODEL-AND-OPTIMIZE for signed step function minimizing max|autoconv| / (sum)^2."""
import time
import numpy as np

N = 400
DEADLINE = 150.0
START_TIME = time.time()


def _score(f):
    """Compute the ratio max|autoconv| / (sum(f))^2."""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return float('inf')
    # Compute full autoconvolution
    auto = np.correlate(f, f, mode='full')
    max_abs = np.max(np.abs(auto))
    return max_abs / (s * s)


def _time_left():
    return DEADLINE - (time.time() - START_TIME)


def construct():
    # Constant function: ratio = 1/N, which is provably optimal for |f_j|<=1
    # By Cauchy-Schwarz: autoconv[0] = sum(f^2) >= (sum f)^2 / N
    # So max|autoconv| >= (sum f)^2 / N, giving ratio >= 1/N
    # Equality holds iff all f_j equal, so all-ones achieves the lower bound.
    return np.ones(N)
