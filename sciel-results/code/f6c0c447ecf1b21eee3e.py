# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Uses LP to find optimal values for a given domain, maximizing -sum f(k)*log(k)/k
    subject to M(x) <= 1 for all x in [1, 10K] and sum f(k)/k = 0.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600s wall-clock limit

    # Try increasing K values
    best_score = -np.inf
    best_result = None

    for K in [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]:
        if time.time() > deadline - 30:
            break

        try:
            domain = np.arange(1, K + 1, dtype=np.int64)
            m = K

            # Objective: maximize -sum f(k)*log(k)/k
            # In linprog form: minimize sum f(k)*log(k)/k
            c = np.log(np.arange(1, K + 1, dtype=np.float64)) / np.arange(1, K + 1, dtype=np.float64)

            # Equality constraint: sum f(k)/k = 0
            A_eq = (1.0 / np.arange(1, K + 1, dtype=np.float64)).reshape(1, -1)
            b_eq = np.array([0.0])

            # Inequality constraints: M(x) = sum_k f(k) * floor(x/k) <= 1 for x in [1, 10K]
            # We only need to check x values where floor(x/k) changes, i.e., x = 1, 2, ..., 10K
            # But that's too many. Instead, note that M(x) is a step function that only changes
            # at x = k*j for k in domain, j >= 1. So we need to check x at those points.
            # Actually, M(x) is non-increasing between these points? No, it's a sum of steps.
            # Let's just check all integers x in [1, 10K] but that's 10K rows which is fine for small K.

            X_max = 10 * K
            # For each x in [1, X_max], M(x) = sum_k f(k) * floor(x/k)
            # We need M(x) <= 1 for all x

            # Build A_ub matrix: row x has A_ub[x, k] = floor((x+1)/k) for k=1..K
            # Actually floor(x/k) for x in 1..X_max
            # This is a X_max x K matrix
            x_vals = np.arange(1, X_max + 1, dtype=np.int64)
            k_vals = np.arange(1, K + 1, dtype=np.int64)

            # Compute floor(x/k) efficiently using broadcasting
            # A_ub[i, j] = floor(x_vals[i] / k_vals[j])
            A_ub = np.floor(x_vals[:, None].astype(np.float64) / k_vals[None, :].astype(np.float64))
            b_ub = np.ones(X_max, dtype=np.float64)

            # Bounds: f(k) can be any real (clipped to [-10, 10] by evaluator, so we can bound)
            bounds = [(-10.0, 10.0)] * K

            # Solve LP
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method='highs')

            if res.status != 0:
                continue

            f_vals = res.x

            # Compute the actual objective after clipping and adjustment
            f_clipped = np.clip(f_vals, -10.0, 10.0)

            # Adjust f(1) so that sum f(k)/k = 0
            S = np.sum(f_clipped / np.arange(1, K + 1, dtype=np.float64))
            f_adjusted = f_clipped.copy()
            f_adjusted[0] -= S  # f(1) is at index 0

            # Verify M(x) <= 1.0001 for all x in [1, 10K]
            M_vals = A_ub @ f_adjusted
            if np.any(M_vals > 1.0001):
                continue

            # Compute score: a = -sum f(k) * log(k) / k
            a = -np.sum(f_adjusted * np.log(np.arange(1, K + 1, dtype=np.float64)) /
                        np.arange(1, K + 1, dtype=np.float64))

            if a > best_score:
                best_score = a
                best_result = (domain, f_adjusted)

        except Exception:
            continue

    # Return best result found, or fallback to seed
    if best_result is not None:
        return best_result
    else:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values


# EVOLVE-BLOCK-END
