# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = 600

    def mobius_sieve(n):
        """Compute Mobius function mu(k) for k=1..n using a linear sieve."""
        mu = np.zeros(n + 1, dtype=np.int8)
        mu[1] = 1
        is_prime = np.ones(n + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        primes = []
        for i in range(2, n + 1):
            if is_prime[i]:
                primes.append(i)
                mu[i] = -1
            for p in primes:
                if i * p > n:
                    break
                is_prime[i * p] = False
                if i % p == 0:
                    mu[i * p] = 0
                    break
                else:
                    mu[i * p] = -mu[i]
        return mu

    def check_feasibility_divisor_count(domain, values, K, max_x):
        """
        Check M(x) = sum_k f(k)*floor(x/k) <= 1.0001 for all x in [1, max_x].
        Uses divisor counting: M(x) = sum_d f(d) * count_of_x_where_d_divides_x.
        For each d, the x in [1, max_x] divisible by d are d, 2d, 3d, ..., floor(max_x/d)*d.
        So M(x) for x = q*d is sum_{d|x} f(d).
        We compute M(x) for all x by: for each d in domain, add f(d) to M[q*d] for q=1..floor(max_x/d).
        """
        m = len(domain)
        # M array indexed by x (1-indexed conceptually, 0-indexed in array)
        M = np.zeros(max_x + 1, dtype=np.float64)
        domain_float = domain.astype(np.float64)
        values_float = values.astype(np.float64)
        
        # For each d in domain, add f(d) to M[d], M[2d], M[3d], ...
        # Vectorized: for each d, the indices are d, 2d, 3d, ..., floor(max_x/d)*d
        for i in range(m):
            d = domain[i]
            f_d = values_float[i]
            if f_d == 0.0:
                continue
            # Indices: d, 2d, 3d, ..., floor(max_x/d)*d
            n_multiples = max_x // d
            if n_multiples == 0:
                continue
            # Use numpy to add f_d to M[d::d][:n_multiples]
            M[d::d][:n_multiples] += f_d
        
        # Check all x in [1, max_x]
        max_M = np.max(M[1:max_x+1])
        return max_M <= 1.0001, max_M

    def compute_metric(domain, values):
        """Compute a = -sum_k f(k) * log(k) / k"""
        domain_float = domain.astype(np.float64)
        log_k = np.log(domain_float)
        return -np.sum(values * log_k / domain_float)

    best_domain = None
    best_values = None
    best_score = -np.inf

    # Try increasing K values
    K_candidates = [5000, 10000, 20000, 50000, 100000, 200000]
    
    for K in K_candidates:
        if time.time() - start_time > deadline - 30:
            break

        mu = mobius_sieve(K)
        nonzero_idx = np.where(mu[1:] != 0)[0] + 1
        domain = nonzero_idx.astype(np.int64)
        values = mu[nonzero_idx].astype(np.float64)

        m = len(domain)
        if m < 1:
            continue

        max_x = 10 * K
        est_ops = m * 10 * K
        if est_ops > 2e9:
            continue

        # Check feasibility using divisor counting
        feasible, max_M = check_feasibility_divisor_count(domain, values, K, max_x)

        if not feasible:
            continue

        a = compute_metric(domain, values)

        if a > best_score:
            best_score = a
            best_domain = domain.copy()
            best_values = values.copy()

    if best_domain is None:
        # Fallback: small valid function
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    else:
        domain = best_domain
        values = best_values

    return domain, values


# EVOLVE-BLOCK-END
