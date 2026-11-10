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
        """Compute Mobius function mu(k) for k=1..n using a sieve."""
        mu = np.ones(n + 1, dtype=np.int8)
        mu[0] = 0
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

    def check_feasibility(domain, values, K, batch_size=10000):
        """Check M(x) <= 1.0001 for all x in [1, 10K] using vectorized ops."""
        max_x = 10 * K
        m = len(domain)
        
        # For x <= K, M(x) = 1 exactly (since floor(x/k)=0 for k>x)
        # So we only need to check x in [K+1, 10K]
        if K >= max_x:
            return True
        
        check_start = K + 1
        check_end = max_x
        n_checks = check_end - check_start + 1
        
        feasible = True
        for batch_start in range(check_start, check_end + 1, batch_size):
            if time.time() - start_time > deadline - 60:
                return False  # timeout, assume infeasible to be safe
            
            batch_end = min(batch_start + batch_size - 1, check_end)
            xs = np.arange(batch_start, batch_end + 1, dtype=np.int64)
            
            # Compute M(x) for each x in batch
            # M(x) = sum_k values[k] * floor(x / domain[k])
            # Process in sub-batches to avoid memory issues
            sub_batch = 1000
            for sb_start in range(len(xs), 0, -sub_batch):
                if time.time() - start_time > deadline - 60:
                    return False
                sb_end = min(sb_start + sub_batch, len(xs))
                xs_sub = xs[sb_start:sb_end]
                # shape: (n_x, m)
                floor_mat = (xs_sub[:, None] // domain[None, :])
                M_batch = np.sum(values[None, :] * floor_mat, axis=1)
                if np.any(M_batch > 1.0001):
                    return False
        
        return True

    best_domain = None
    best_values = None
    best_score = -np.inf

    for K in [1000, 5000, 10000, 20000, 50000, 100000, 200000]:
        if time.time() - start_time > deadline - 30:
            break

        mu = mobius_sieve(K)
        nonzero_idx = np.where(mu[1:] != 0)[0] + 1
        domain = nonzero_idx.astype(np.int64)
        values = mu[nonzero_idx].astype(np.float64)

        m = len(domain)
        if m < 1:
            continue

        est_ops = m * 10 * K
        if est_ops > 2e9:
            continue

        log_k = np.log(domain.astype(np.float64))
        a = -np.sum(values * log_k / domain.astype(np.float64))

        # Check feasibility
        feasible = check_feasibility(domain, values, K)

        if not feasible:
            continue

        if a > best_score:
            best_score = a
            best_domain = domain.copy()
            best_values = values.copy()

    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    else:
        domain = best_domain
        values = best_values

    return domain, values


# EVOLVE-BLOCK-END
