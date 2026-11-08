import time
import math
import numpy as np

def compute_score(U):
    if len(U) < 2:
        return 0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0
    n = len(U)
    U_arr = np.array(U, dtype=np.int64)
    # Compute all sums and differences efficiently
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    sum_set_size = len(np.unique(sums))
    diff_set_size = len(np.unique(diffs))
    ratio = diff_set_size / sum_set_size
    if ratio <= 0:
        return 0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return 0
    return 1 + log_ratio / log_denom

def construct():
    # The optimal strategy is to use a set of size 3: {0, 1, N} with very large N.
    # This minimizes |U+U| (constant 6) while maximizing |U-U| (linear in N).
    # The score approaches 2 as N -> infinity.
    # We choose N = 10**18 to be well within int64 limits but large enough to maximize the score.
    return [0, 1, 10**18]
