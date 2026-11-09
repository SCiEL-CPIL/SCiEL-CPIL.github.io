# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


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


def _check_feasibility(domain, values, K_max, tolerance=1.0001):
    """Check if M(x) <= tolerance for all x in [1, 10*K_max]."""
    m = len(domain)
    if m == 0:
        return True
    
    x_max = 10 * K_max
    if x_max > 2000000:
        # Too large, use sampling
        x_range = np.arange(1, x_max + 1, max(1, x_max // 200000))
    else:
        x_range = np.arange(1, x_max + 1)
    
    # Compute M(x) for each x in x_range
    # M(x) = sum_k f(k) * floor(x/k)
    
    if x_max > 1000000:
        # Use direct computation with numpy for speed
        # M(x) = sum_k f(k) * floor(x/k)
        # Compute for a subset of x values
        x_sample = x_range[::max(1, len(x_range)//50000)]
        M_vals = np.zeros(len(x_sample))
        for idx, x in enumerate(x_sample):
            M_vals[idx] = np.sum(values * (x // domain))
        return np.all(M_vals <= tolerance)
    else:
        # Compute h(j) for j = 1..x_max
        h = np.zeros(x_max + 1, dtype=np.float64)
        for i, k in enumerate(domain):
            # Add f(k) to h(j) for all j that are multiples of k
            h[k::k] += values[i]
        
        # M(x) = cumsum of h[1..x]
        M_vals = np.cumsum(h[1:])
        return np.all(M_vals <= tolerance)


def _compute_objective(domain, values):
    """Compute the objective a = -sum_k f(k) * log(k) / k."""
    if len(domain) == 0:
        return 0.0
    k = domain.astype(np.float64)
    return -np.sum(values * np.log(k) / k)


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    
    Uses a Linear Programming approach to find the optimal f(k) for a given K,
    maximizing a = -sum_k f(k) log(k) / k subject to M(x) <= 1.
    Falls back to the Möbius function if LP fails or times out.
    """
    start_time = time.time()
    deadline = start_time + 580  # Leave 50s safety margin
    
    # Fallback solution (Möbius based)
    K_mob = 200000
    mu = _compute_mobius(K_mob)
    domain_mob = np.where(mu[1:] != 0)[0] + 1
    values_mob = mu[domain_mob].astype(np.float64)
    
    # Ensure fallback is feasible
    if not _check_feasibility(domain_mob, values_mob, K_mob):
        # Try smaller K for Möbius if needed (unlikely to be needed for 200k)
        # But for safety, keep the seed solution as ultimate fallback
        domain_seed = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values_seed = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        domain_mob, values_mob = domain_seed, values_seed
        K_mob = 30

    best_domain = domain_mob
    best_values = values_mob
    best_score = _compute_objective(best_domain, best_values)
    
    # Try LP for various K
    # K=1000 is a good balance. 10K=10000 constraints, K=1000 variables.
    # Matrix size 10000x1000 = 10M entries.
    # HiGHS should handle this in < 10-20s.
    
    K_candidates = [500, 1000, 2000]
    
    for K in K_candidates:
        if time.time() > deadline - 60:  # Leave 60s for finishing
            break
            
        try:
            # Variables: f_1, ..., f_K
            # Note: The problem statement says f(1) is adjusted later.
            # However, the LP constraint M(x) <= 1 depends on all f(k).
            # The objective a = -sum f(k) log(k)/k. Note log(1)=0, so f(1) doesn't affect objective.
            # But f(1) affects M(x).
            # The evaluator clips values to [-10, 10].
            # So bounds are [-10, 10].
            
            n_vars = K
            k_vals = np.arange(1, K + 1, dtype=np.float64)
            
            # Objective: Maximize -sum f(k) log(k)/k
            # Minimize sum f(k) log(k)/k
            c = np.log(k_vals) / k_vals
            c[0] = 0.0  # log(1)=0
            
            # Constraints: M(x) = sum_k f(k) floor(x/k) <= 1 for x in 1..10K
            # M(x) - 1 <= 0
            # sum_k f(k) floor(x/k) <= 1
            # This is A_ub @ f <= b_ub
            
            x_max = 10 * K
            x_vals = np.arange(1, x_max + 1)
            
            # Construct A_ub (x_max x K)
            # A_ub[i, j] = floor((i+1) / (j+1))
            # We can compute this efficiently.
            
            # To save memory/time, we can exploit structure or just compute if K is small enough.
            # For K=2000, x_max=20000. 20000*2000 = 40M entries. 40M * 8 bytes = 320MB.
            # This might be tight or slow to construct.
            # Let's stick to K=1000 first.
            
            if K > 1000:
                # Use sparse or block computation if we go higher?
                # For now, let's just try K=1000.
                continue
                
            # Dense matrix construction
            # x_vals: (10K, 1), k_vals: (1, K)
            # floor(x/k)
            A_ub = np.floor(x_vals[:, np.newaxis] / k_vals[np.newaxis, :])
            
            b_ub = np.ones(x_max)
            
            bounds = [(-10.0, 10.0)] * K
            
            # Solve LP
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if res.success:
                f_vals = res.x
                # Clip to [-10, 10] just in case of numerical drift
                f_vals = np.clip(f_vals, -10.0, 10.0)
                
                # Verify feasibility
                # M(x) = A_ub @ f_vals
                M_vals = A_ub @ f_vals
                if np.all(M_vals <= 1.0001):
                    domain = np.arange(1, K + 1, dtype=np.int64)
                    values = f_vals.astype(np.float64)
                    
                    score = _compute_objective(domain, values)
                    
                    if score > best_score:
                        best_domain = domain
                        best_values = values
                        best_score = score
                        
        except Exception as e:
            print(f"LP failed for K={K}: {e}")
            continue

    return best_domain, best_values


# EVOLVE-BLOCK-END
