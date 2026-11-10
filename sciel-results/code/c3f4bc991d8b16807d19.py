import numpy as np
import time

def _compute_mobius(N):
    """Compute Moebius function mu(k) for k = 1..N using linear sieve."""
    mu = np.zeros(N + 1, dtype=np.int8)
    mu[1] = 1
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > N:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu

def _verify_feasibility(domain, values, max_x):
    """
    Check M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for x in [1, max_x].
    Uses vectorized NumPy for speed.
    """
    m = len(domain)
    if m == 0:
        return True

    g = np.zeros(max_x + 1, dtype=np.float64)
    for i in range(m):
        k = int(domain[i])
        v = float(values[i])
        if v == 0:
            continue
        g[k::k] += v

    M = np.cumsum(g)
    if np.any(M[1:max_x + 1] > 1.0001):
        return False
    return True

def _compute_objective(domain, values):
    """
    Compute a = -sum_k f(k) * log(k) / k using the evaluator's
    clipping and S-adjustment rules.
    """
    m = len(domain)
    if m == 0:
        return 0.0

    clipped = np.clip(values, -10.0, 10.0)
    k_float = domain.astype(np.float64)

    # S = sum_k f(k)/k over clipped values
    S = np.sum(clipped / k_float)

    # f(1) is replaced by f(1) - S; if 1 not in domain, f(1)=0 before
    # so effectively we subtract S/k from every term where k=1, i.e. subtract S
    # More precisely: adjusted[1] = clipped[1] - S (if 1 in domain), else add 0 - S? No.
    # The task says: "f(1) is replaced by f(1) - S with S = sum_k f(k)/k over the
    # clipped values (f(1) = 0 before this if 1 is not in the domain)"
    # So if 1 is in domain: f_adj(1) = clipped[1] - S
    # If 1 is not in domain: f_adj(1) = 0 - S = -S, but 1 is not in domain so
    # we don't include it. The sum sum_k f(k)/k = 0 after adjustment.
    # The adjustment only changes the value at k=1.
    # So for the objective, we use clipped values except at k=1 where we use
    # clipped[1] - S if 1 is in domain.

    f_adj = clipped.copy()
    idx_1 = np.where(domain == 1)[0]
    if len(idx_1) > 0:
        f_adj[idx_1[0]] = clipped[idx_1[0]] - S

    log_k = np.log(k_float)
    a = -np.sum(f_adj * log_k / k_float)
    return a

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = 630.0

    # Max K allowed is 200,000
    K_max = 200000

    # Compute mu for the maximum possible K
    mu = _compute_mobius(K_max)

    best_domain = None
    best_values = None
    best_a = -np.inf

    # Binary search for the largest feasible K
    # Feasibility is likely monotone: if K is feasible, smaller K should be too
    # (since we're summing fewer terms, M(x) should be smaller in magnitude)
    # Actually not strictly monotone, but let's try a coarse-to-fine approach.

    # First, try the maximum K
    def try_K(K):
        nonlocal best_domain, best_values, best_a
        if time.time() > deadline - 30:
            return False
        max_x = 10 * K
        non_zero = np.where(mu[1:K + 1] != 0)[0] + 1
        if len(non_zero) == 0:
            return False
        domain = non_zero.astype(np.int64)
        values = mu[domain].astype(np.float64)
        if not _verify_feasibility(domain, values, max_x):
            return False
        a = _compute_objective(domain, values)
        if a > best_a:
            best_a = a
            best_domain = domain.copy()
            best_values = values.copy()
        return True

    # Try K_max first
    if try_K(K_max):
        # Found feasible at max, done
        pass
    else:
        # Binary search between 1 and K_max
        lo, hi = 1, K_max
        # First find any feasible K by halving
        while hi > lo and time.time() < deadline - 30:
            mid = (lo + hi) // 2
            if try_K(mid):
                lo = mid + 1
            else:
                hi = mid - 1
        # lo is now the largest feasible K (or 1 if none found above 1)
        # But try_K already updates best, so we just need to make sure
        # we've tried the best one. The binary search may have skipped some.
        # Let's try a few around lo-1
        for K_try in range(max(1, lo - 5), lo + 1):
            if time.time() > deadline - 30:
                break
            try_K(K_try)

    # Fallback
    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    else:
        domain = best_domain
        values = best_values

    return domain, values
