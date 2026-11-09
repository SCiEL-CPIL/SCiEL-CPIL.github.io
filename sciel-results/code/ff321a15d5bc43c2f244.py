# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(K):
    """Compute the Möbius function mu(n) for n = 1..K using sieve."""
    mu = np.ones(K + 1, dtype=np.int8)
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    primes = []
    
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu


def _compute_M_max(domain, values, K_max):
    """Compute max M(x) for x in [1, 10*K_max] efficiently using divisor sums."""
    x_max = 10 * K_max
    m = len(domain)
    if m == 0 or x_max <= 0:
        return 0.0
    
    # h(j) = sum_{k|j, k in domain} f(k)
    # M(x) = sum_{j=1}^{x} h(j)
    # So M(x) = cumsum(h[1..x])
    
    h = np.zeros(x_max + 1, dtype=np.float64)
    for i in range(m):
        k = domain[i]
        v = values[i]
        if v != 0:
            h[k::k] += v
    
    M_vals = np.cumsum(h[1:])
    return float(np.max(M_vals))


def _check_feasibility(domain, values, K_max, tolerance=1.0001):
    """Check if M(x) <= tolerance for all x in [1, 10*K_max]."""
    m_max = _compute_M_max(domain, values, K_max)
    return m_max <= tolerance


def _compute_metric(domain, values):
    """Compute a = -sum_k f(k) * log(k) / k."""
    if len(domain) == 0:
        return 0.0
    return float(-np.sum(values * np.log(domain) / domain))


def _apply_clipping_and_normalization(domain, values):
    """Apply clipping to [-10, 10] and normalize so sum f(k)/k = 0."""
    values = np.clip(values, -10.0, 10.0)
    
    # Compute S = sum_k f(k)/k
    S = np.sum(values / domain)
    
    # f(1) is replaced by f(1) - S
    # If 1 is not in domain, f(1) = 0 before this
    idx_1 = np.where(domain == 1)[0]
    if len(idx_1) > 0:
        values[idx_1[0]] -= S
    else:
        # Need to add 1 to domain with value -S
        domain = np.concatenate([[1], domain])
        values = np.concatenate([[-S], values])
        # Sort by domain
        sort_idx = np.argsort(domain)
        domain = domain[sort_idx]
        values = values[sort_idx]
    
    return domain, values


def _evaluate_final(domain, values, K_max, tolerance=1.0001):
    """
    Apply the full evaluation pipeline: clip, normalize, check feasibility, compute metric.
    Returns (is_feasible, metric, final_domain, final_values).
    """
    # Step 1: Clip
    clipped = np.clip(values, -10.0, 10.0)
    
    # Step 2: Normalize
    S = np.sum(clipped / domain)
    idx_1 = np.where(domain == 1)[0]
    if len(idx_1) > 0:
        final_values = clipped.copy()
        final_values[idx_1[0]] -= S
        final_domain = domain.copy()
    else:
        final_domain = np.concatenate([[1], domain])
        final_values = np.concatenate([[-S], clipped])
        sort_idx = np.argsort(final_domain)
        final_domain = final_domain[sort_idx]
        final_values = final_values[sort_idx]
    
    # Step 3: Check feasibility
    if not _check_feasibility(final_domain, final_values, K_max, tolerance):
        return False, -np.inf, final_domain, final_values
    
    # Step 4: Compute metric
    metric = _compute_metric(final_domain, final_values)
    return True, metric, final_domain, final_values


def _local_optimize(domain, values, K_max, deadline, max_iterations=200):
    """
    Local optimization: try small perturbations to improve the metric while maintaining feasibility.
    Uses a coordinate descent approach on the values.
    """
    start_time = time.time()
    best_metric, _, best_domain, best_values = _evaluate_final(domain, values, K_max)
    
    if not np.isfinite(best_metric):
        return best_domain, best_values, best_metric
    
    m = len(domain)
    current_values = best_values.copy()
    
    # Precompute log(k)/k for metric computation
    log_over_k = np.log(domain) / domain
    
    for iteration in range(max_iterations):
        if time.time() > deadline - 30:
            break
        
        improved = False
        # Try adjusting each value slightly
        for i in range(m):
            if time.time() > deadline - 30:
                break
            
            k = domain[i]
            if k == 1:
                continue  # Skip f(1) as it's determined by normalization
            
            current_val = current_values[i]
            # Try small perturbations
            for delta in [0.01, -0.01, 0.1, -0.1, 0.5, -0.5]:
                new_val = current_val + delta
                if new_val < -10 or new_val > 10:
                    continue
                
                test_values = current_values.copy()
                test_values[i] = new_val
                
                feasible, metric, _, _ = _evaluate_final(domain, test_values, K_max)
                if feasible and metric > best_metric + 1e-9:
                    best_metric = metric
                    best_values = test_values.copy()
                    current_values = test_values.copy()
                    improved = True
                    break
        
        if not improved:
            # Try larger steps
            for i in range(m):
                if time.time() > deadline - 30:
                    break
                
                k = domain[i]
                if k == 1:
                    continue
                
                current_val = current_values[i]
                for delta in [1.0, -1.0, 2.0, -2.0]:
                    new_val = current_val + delta
                    if new_val < -10 or new_val > 10:
                        continue
                    
                    test_values = current_values.copy()
                    test_values[i] = new_val
                    
                    feasible, metric, _, _ = _evaluate_final(domain, test_values, K_max)
                    if feasible and metric > best_metric + 1e-9:
                        best_metric = metric
                        best_values = test_values.copy()
                        current_values = test_values.copy()
                        improved = True
                        break
                
                if improved:
                    break
        
        if not improved:
            break
    
    return best_domain, best_values, best_metric


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    
    Uses the Möbius function as the base and applies local optimization to improve the metric.
    """
    start_time = time.time()
    deadline = start_time + 580  # Leave 50s safety margin
    
    # Compute Möbius function up to K
    K = 200000
    mu = _compute_mobius(K)
    
    # Use all k where mu(k) != 0
    domain = np.where(mu[1:] != 0)[0] + 1
    values = mu[domain].astype(np.float64)
    
    # Check feasibility with the full evaluation pipeline
    feasible, metric, final_domain, final_values = _evaluate_final(domain, values, K)
    
    if not feasible:
        # If infeasible, try smaller K
        for K_try in [100000, 50000, 20000, 10000, 5000, 2000, 1000]:
            if time.time() > deadline:
                break
            mu_try = _compute_mobius(K_try)
            domain_try = np.where(mu_try[1:] != 0)[0] + 1
            values_try = mu_try[domain_try].astype(np.float64)
            feasible, metric, final_domain_try, final_values_try = _evaluate_final(domain_try, values_try, K_try)
            if feasible:
                domain, values, K = domain_try, values_try, K_try
                final_domain, final_values = final_domain_try, final_values_try
                break
        else:
            # Fallback to a small feasible solution
            domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            return domain, values
    
    # Apply local optimization if time permits
    if time.time() < deadline - 60:
        final_domain, final_values, final_metric = _local_optimize(domain, values, K, deadline)
    
    return final_domain, final_values


# EVOLVE-BLOCK-END
