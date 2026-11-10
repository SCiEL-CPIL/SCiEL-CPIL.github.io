# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f
    """
    start_time = time.time()
    DEADLINE = 580  # seconds, safety margin before 630s kill

    def mobius_up_to(n):
        """Compute Möbius function mu(k) for k = 1..n using sieve."""
        mu = np.ones(n + 1, dtype=np.int8)
        is_prime = np.ones(n + 1, dtype=bool)
        primes = []
        for i in range(2, n + 1):
            if is_prime[i]:
                primes.append(i)
                mu[i] = -1
                for j in range(i, n + 1, i):
                    is_prime[j] = False
                    mu[j] *= -1
            for p in primes:
                if i * p > n:
                    break
                is_prime[i * p] = False
                if i % p == 0:
                    mu[i * p] = 0
                    break
        mu[0] = 0
        return mu

    # Compute Möbius function
    N_max = 200000
    mu = mobius_up_to(N_max)

    # Try different subset sizes and find the best feasible solution
    # Strategy: use mu(k) for k where mu(k) != 0, up to some limit
    # Then solve LP to maximize a = -sum f(k)*log(k)/k subject to M(x) <= 1

    # For efficiency, work with a subset of k values
    # The constraint M(x) = sum_k f(k) * floor(x/k) <= 1 for x in [1, 10K]
    # where K = max domain value

    # Let's try increasing N and see what works
    best_score = 0
    best_domain = None
    best_values = None

    # Use mu(k) != 0 values as candidates
    nonzero_k = np.where(mu[1:] != 0)[0] + 1  # k values where mu(k) != 0

    # Try different maximum K values
    for K in [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]:
        if time.time() - start_time > DEADLINE:
            break

        # Get k values up to K where mu(k) != 0
        k_vals = nonzero_k[nonzero_k <= K]
        m = len(k_vals)

        if m == 0:
            continue

        # f(k) = mu(k) is the natural choice
        f_vals = mu[k_vals].astype(np.float64)

        # Check constraint: M(x) = sum f(k) * floor(x/k) <= 1 for x in [1, 10K]
        # For efficiency, check at critical points
        # M(x) is a step function that changes at x = k for each k in domain
        # So we need to check at x = 1, 2, ..., 10K but can be smarter

        # Actually, M(x) = sum_k f(k) * floor(x/k)
        # This is piecewise constant, changing only when x/k crosses an integer
        # i.e., when x = j*k for some integer j
        # So critical points are x = j*k for k in domain, j >= 1, x <= 10K

        # For now, let's just check at all integers up to 10K if feasible
        max_x = 10 * K

        if max_x <= 200000:  # Can check all points
            # Compute M(x) for x = 1..max_x
            # M(x) = sum_k f(k) * floor(x/k)
            # Vectorized: for each k, floor(x/k) for x in 1..max_x
            x_arr = np.arange(1, max_x + 1, dtype=np.int64)
            M = np.zeros(max_x, dtype=np.float64)
            for i in range(m):
                k = k_vals[i]
                fk = f_vals[i]
                if fk != 0:
                    M += fk * (x_arr // k)

            if np.max(M) <= 1.0001:
                # Compute score
                # First, the evaluator clips values to [-10, 10]
                f_clipped = np.clip(f_vals, -10, 10)
                # S = sum_k f(k)/k
                S = np.sum(f_clipped / k_vals)
                # f(1) adjustment: if 1 is in domain, f(1) -= S
                # Then sum f(k)/k = 0
                # a = -sum f(k)*log(k)/k after adjustment
                # If 1 is in domain: f(1)_new = f(1) - S
                # a = -[f(1)_new*log(1)/1 + sum_{k>1} f(k)*log(k)/k]
                # = -[0 + sum_{k>1} f(k)*log(k)/k] since log(1)=0
                # = -sum_{k>1} f(k)*log(k)/k
                # If 1 not in domain: f(1) = 0 - S = -S
                # a = -[f(1)_new*log(1)/1 + sum f(k)*log(k)/k]
                # = -sum f(k)*log(k)/k (since log(1)=0)

                # Actually, after adjustment, sum f(k)/k = 0
                # a = -sum f(k)*log(k)/k
                # The f(1) term vanishes because log(1) = 0
                # So a = -sum_{k in domain, k>1} f(k)*log(k)/k
                # (using clipped values, and the adjusted f)

                # But wait, the adjustment changes f(1), not other values
                # So for k > 1, f(k) is unchanged (clipped)
                # a = -sum_{k>1} f(k)*log(k)/k

                if 1 in k_vals:
                    idx_1 = np.where(k_vals == 1)[0][0]
                    # f(1) is adjusted but log(1) = 0, so it doesn't contribute
                    # a = -sum_{k>1} f_clipped(k)*log(k)/k
                    mask = k_vals > 1
                    a = -np.sum(f_clipped[mask] * np.log(k_vals[mask]) / k_vals[mask])
                else:
                    # f(1) = -S, but log(1) = 0, so still doesn't contribute
                    a = -np.sum(f_clipped * np.log(k_vals) / k_vals)

                if a > best_score:
                    best_score = a
                    best_domain = k_vals.copy()
                    best_values = f_clipped.copy()

        else:
            # Too large to check all points, use a subset of critical points
            # Check at x = 1, 2, ..., min(10000, max_x) and at x = j*k for small j
            check_points = set()
            for i in range(m):
                k = k_vals[i]
                for j in range(1, min(100, max_x // k + 1)):
                    check_points.add(j * k)
            check_points = sorted(check_points[:50000])

            if check_points:
                x_check = np.array(check_points, dtype=np.int64)
                M_check = np.zeros(len(x_check), dtype=np.float64)
                for i in range(m):
                    k = k_vals[i]
                    fk = f_vals[i]
                    if fk != 0:
                        M_check += fk * (x_check // k)

                if np.max(M_check) <= 1.0001:
                    f_clipped = np.clip(f_vals, -10, 10)
                    a = -np.sum(f_clipped * np.log(k_vals) / k_vals)
                    if a > best_score:
                        best_score = a
                        best_domain = k_vals.copy()
                        best_values = f_clipped.copy()

    # If no good solution found, use a simple feasible one
    if best_domain is None or len(best_domain) == 0:
        # Use a small feasible set
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    else:
        domain = best_domain
        values = best_values

    return domain, values

# EVOLVE-BLOCK-END
