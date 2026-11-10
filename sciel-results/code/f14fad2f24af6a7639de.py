# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(K):
    """Compute the Möbius function μ(n) for n = 1..K using a sieve."""
    mu = np.ones(K + 1, dtype=np.int64)
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
    if K >= 1:
        is_prime[1] = False
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


def _check_feasibility(domain, values, K):
    """Check if M(x) <= 1.0001 for all x in [1, 10K]."""
    max_x = 10 * K
    # M(x) = sum_k f(k) * floor(x/k)
    # We can compute this efficiently using the fact that floor(x/k) is piecewise constant
    # For a given x, floor(x/k) changes at k = floor(x/j) for j = 1, 2, ...
    # But for checking all x, we can use a different approach:
    # M(x) = sum_{k=1}^{min(x,K)} f(k) * floor(x/k)
    # This can be rewritten as: M(x) = sum_{j=1}^{x} (sum_{k: floor(x/k)=j} f(k))
    # But the simplest is to just compute it directly for each x using vectorization
    
    # Actually, let's use the identity:
    # M(x) = sum_{k=1}^{K} f(k) * floor(x/k)
    # For small K this is fast enough. For larger K we need to be careful.
    
    # Build a lookup array for f(k)
    f = np.zeros(K + 1, dtype=np.float64)
    for k, v in zip(domain, values):
        if k <= K:
            f[k] = v
    
    # Check M(x) for x = 1, 2, ..., 10K
    # M(x) = sum_{k=1}^{min(x,K)} f(k) * floor(x/k)
    # We can vectorize over x using the fact that for fixed k, floor(x/k) is a step function
    
    # For efficiency, process in batches
    batch_size = 10000
    max_x = 10 * K
    
    for start in range(1, max_x + 1, batch_size):
        end = min(start + batch_size, max_x + 1)
        xs = np.arange(start, end)
        
        # For each x, M(x) = sum_{k=1}^{min(x,K)} f(k) * floor(x/k)
        # We can compute this as: for each k, add f(k) * floor(x/k) to M(x)
        # But this is O(K * batch_size) which might be too slow for large K
        
        # Alternative: use the fact that floor(x/k) = j for x in [j*k, (j+1)*k - 1]
        # So for each k, we can add f(k) to M(x) for x in ranges where floor(x/k) = j
        
        # Actually, let's just do the direct computation but be smart about it
        # For x in [start, end), the relevant k are 1 to min(end-1, K)
        k_max = min(end - 1, K)
        
        # Build matrix: rows = x values, cols = k values
        # M(x) = sum_k f(k) * floor(x/k)
        
        # For memory efficiency, process in chunks of k
        M_vals = np.zeros(len(xs), dtype=np.float64)
        
        k_chunk_size = 5000
        for k_start in range(1, k_max + 1, k_chunk_size):
            k_end = min(k_start + k_chunk_size, k_max + 1)
            ks = np.arange(k_start, k_end)
            fk = f[ks]
            
            # floor(x/k) for x in xs, k in ks
            # This is a (len(xs), len(ks)) matrix
            # floor(x/k) = x // k
            floor_vals = np.outer(xs, np.ones(len(ks), dtype=np.float64)) // np.outer(np.ones(len(xs), dtype=np.float64), ks)
            
            # M_vals += floor_vals @ fk
            M_vals += floor_vals @ fk
        
        if np.any(M_vals > 1.0001):
            return False
    
    return True


def _compute_max_M(f, K):
    """Compute the maximum of M(x) = sum_k f(k)*floor(x/k) for x in [1, 10K]."""
    max_x = 10 * K
    best = -np.inf
    
    # For x <= K: M(x) = sum_{k=1}^{x} f(k)*floor(x/k) which should be ~1
    # For x > K: M(x) = sum_{k=1}^{K} f(k)*floor(x/k) which is the truncated sum
    # The maximum likely occurs at x = 10K or near it
    
    # Check x = 10K first (likely the worst case)
    x = max_x
    k_max = min(x, K)
    ks = np.arange(1, k_max + 1)
    fk = f[ks]
    floor_vals = x // ks
    M_x = np.sum(fk * floor_vals)
    best = max(best, M_x)
    
    # Check a few other critical points
    for x in [K, 2*K, 5*K, 7*K, 9*K, 10*K]:
        if x > max_x:
            break
        k_max = min(x, K)
        ks = np.arange(1, k_max + 1)
        fk = f[ks]
        floor_vals = x // ks
        M_x = np.sum(fk * floor_vals)
        best = max(best, M_x)
    
    # Also check some smaller x values where the sum might overshoot
    for x in [1, 2, 3, 5, 10, 100, 1000]:
        if x > max_x:
            break
        k_max = min(x, K)
        ks = np.arange(1, k_max + 1)
        fk = f[ks]
        floor_vals = x // ks
        M_x = np.sum(fk * floor_vals)
        best = max(best, M_x)
    
    return best


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    
    Uses the Möbius function as the natural solution, which satisfies the key identity
    sum_{k<=x} mu(k) * floor(x/k) = 1 for all x >= 1.
    Scales values to exploit the full slack in the feasibility constraint.
    """
    start_time = time.time()
    deadline = 600.0  # seconds
    
    # Start with a reasonable K and increase if time permits
    # The Möbius function gives a good approximation
    # Let's use K up to 200000 as allowed
    
    # First, try a moderate K to get a baseline
    K = 5000
    
    mu = _compute_mobius(K)
    
    # Filter to non-zero values (μ(k) = 0 for k with repeated prime factors)
    domain = np.where(mu != 0)[0]
    values = mu[domain].astype(np.float64)
    
    # Check feasibility
    if not _check_feasibility(domain, values, K):
        # If infeasible, we need to adjust
        # The Möbius function should satisfy M(x) = 1 for all x, so this shouldn't happen
        # But due to floating point or other issues, let's check
        # If infeasible, reduce the magnitude
        values = values * 0.5
        if not _check_feasibility(domain, values, K):
            # Fall back to a simple feasible solution
            domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            return domain, values
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return domain, values
    
    # Try to increase K if we have time
    for K_try in [10000, 20000, 50000, 100000, 200000]:
        if K_try <= K:
            continue
        elapsed = time.time() - start_time
        if elapsed > deadline - 60:
            break
        
        mu = _compute_mobius(K_try)
        domain = np.where(mu != 0)[0]
        values = mu[domain].astype(np.float64)
        
        # Quick feasibility check on a sample of x values
        # M(x) should be exactly 1 for all x if using full Möbius
        # Let's check a few points
        K_check = K_try
        f = np.zeros(K_check + 1, dtype=np.float64)
        for k, v in zip(domain, values):
            if k <= K_check:
                f[k] = v
        
        # Check M(x) for x = K, 2K, 5K, 10K
        feasible = True
        for x in [K_check, 2*K_check, 5*K_check, 10*K_check]:
            # M(x) = sum_{k=1}^{min(x,K)} f(k) * floor(x/k)
            k_max = min(x, K_check)
            ks = np.arange(1, k_max + 1)
            fk = f[ks]
            floor_vals = x // ks
            M_x = np.sum(fk * floor_vals)
            if M_x > 1.0001:
                feasible = False
                break
        
        if feasible:
            K = K_try
            elapsed = time.time() - start_time
            if elapsed > deadline - 30:
                break
        else:
            break
    
    # Now scale the values to exploit the full slack
    # Compute the maximum M(x) for the current solution
    f_full = np.zeros(K + 1, dtype=np.float64)
    for k, v in zip(domain, values):
        if k <= K:
            f_full[k] = v
    
    max_M = _compute_max_M(f_full, K)
    
    # The constraint is M(x) <= 1.0001
    # If max_M < 1.0001, we can scale up by factor 1.0001 / max_M
    if max_M > 1e-10 and max_M < 1.0001:
        scale = 1.0001 / max_M
        # Also ensure we don't exceed the clipping bounds [-10, 10]
        max_abs_val = np.max(np.abs(values))
        if max_abs_val * scale <= 10.0:
            values = values * scale
        else:
            # Scale to hit the clipping bound
            scale = 10.0 / max_abs_val
            values = values * scale
    
    # Final feasibility check
    if not _check_feasibility(domain, values, K):
        # Revert to unscaled if scaling caused issues
        f_full = np.zeros(K + 1, dtype=np.float64)
        for k, v in zip(domain, mu[domain].astype(np.float64)):
            if k <= K:
                f_full[k] = v
        values = mu[domain].astype(np.float64)
    
    return domain, values


# EVOLVE-BLOCK-END
