import numpy as np
import time
from scipy.optimize import linprog

def _compute_mobius(K):
    """Compute the Möbius function mu(n) for n = 1..K using sieve."""
    mu = np.zeros(K + 1, dtype=np.int8)
    mu[1] = 1
    primes = []
    is_composite = np.zeros(K + 1, dtype=bool)
    for i in range(2, K + 1):
        if not is_composite[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_composite[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu

def _check_feasibility(domain, values, K_max, tolerance=1.0001):
    """Check if M(x) <= tolerance for all x in [1, 10*K_max] after clipping and f(1) adjustment."""
    # Clip values to [-10, 10]
    clipped = np.clip(values, -10.0, 10.0)
    
    # S = sum_k f(k)/k over the clipped values
    S = np.sum(clipped / domain.astype(np.float64))
    
    # f(1) is replaced by f(1) - S
    # If 1 is in domain, adjust it. If not, it's 0 - S = -S.
    # But for M(x) calculation, we use the full sum.
    # M(x) = sum_k f_adjusted(k) * floor(x/k)
    # Note: floor(x/1) = x.
    
    # Efficient check using the structure of M(x)
    # M(x) = sum_{j=1}^x sum_{k|j} f_adjusted(k)
    # Let h(j) = sum_{k|j} f_adjusted(k)
    # M(x) = cumsum(h[1..x])
    
    x_max = 10 * K_max
    h = np.zeros(x_max + 1, dtype=np.float64)
    
    for i in range(len(domain)):
        k = int(domain[i])
        f_val = clipped[i]
        if k == 1:
            f_val = f_val - S # Adjusted f(1)
        
        if k > x_max:
            continue
            
        # Add f_val to h[k], h[2k], h[3k], ...
        h[k::k] += f_val
        
    M_vals = np.cumsum(h[1:])
    return np.max(M_vals) <= tolerance

def _compute_a(domain, values):
    """Compute a = -sum_k f(k) * log(k) / k."""
    clipped = np.clip(values, -10.0, 10.0)
    k_float = domain.astype(np.float64)
    S = np.sum(clipped / k_float)
    
    # Adjusted values: f(1) is replaced by f(1) - S
    # If 1 is in domain, adjust that entry. log(1)=0 so contribution is 0 either way.
    # The metric sum is over the domain keys with adjusted values.
    adjusted = clipped.copy()
    idx_1 = np.where(domain == 1)[0]
    if len(idx_1) > 0:
        adjusted[idx_1[0]] = clipped[idx_1[0]] - S
    
    return -np.sum(adjusted * np.log(k_float) / k_float)

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the objective a = -sum_k f(k) log(k) / k subject to M(x) <= 1.
    """
    start_time = time.time()
    deadline = 580  # Leave 20s margin for the 600s limit (or 630s kill)
    
    # Fallback seed
    seed_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    seed_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    
    best_domain = seed_domain
    best_values = seed_values
    best_a = _compute_a(seed_domain, seed_values)
    
    # Try LP approach for small K
    # K=1000 worked well in current program (1.0745)
    # Can we go higher?
    # LP size is (10K) x K.
    # K=2000 -> 20000 x 2000 = 40M elements. Might be slow/memory heavy.
    # Let's try K=1500.
    
    K_candidates = [3000, 2500, 2000, 1500, 1000]
    
    for K in K_candidates:
        if time.time() - start_time > deadline - 30:
            break
            
        try:
            # Construct constraint matrix B
            # Variables: f(1), ..., f(K)
            # Constraints: M(x) <= 1 for x in 1..10K
            # M(x) = sum_{j=1}^K f(j) * floor(x/j)
            
            # Fast vectorized construction using integer remainder
            x_arr = np.arange(1, 10 * K + 1, dtype=np.int64)
            j_arr = np.arange(1, K + 1, dtype=np.int64)
            A_ub = (x_arr[:, None] - np.remainder(x_arr[:, None], j_arr[None, :])) // j_arr[None, :]
            b_ub = np.ones(10 * K, dtype=np.float64)
            
            # Objective: Maximize -sum_{k=2}^K f(k) log(k)/k
            # f(1) term is 0.
            # Minimize sum_{k=2}^K f(k) log(k)/k
            c = np.zeros(K)
            k_vals = np.arange(1, K + 1)
            c[1:] = np.log(k_vals[1:]) / k_vals[1:]
            
            # Bounds -10 <= f <= 10
            bounds = [(-10.0, 10.0)] * K
            
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if res.success:
                f_vals = res.x
                domain = np.arange(1, K + 1, dtype=np.int64)
                values = f_vals.astype(np.float64)
                
                # Verify feasibility with the exact evaluator logic (clipping + f(1) adj)
                if _check_feasibility(domain, values, K):
                    a_val = _compute_a(domain, values)
                    if a_val > best_a:
                        best_a = a_val
                        best_domain = domain
                        best_values = values
                        # print(f"K={K}, a={a_val}")
                        
        except Exception:
            continue
            
    # If LP didn't improve or failed, try Mobius fallback for larger K
    # Mobius might be feasible for larger K?
    # M(x) for Mobius is related to Mertens function.
    # Mertens function M(x) = sum_{n<=x} mu(n).
    # Our M(x) = sum_{k} mu(k) floor(x/k).
    # Note that sum_{k=1}^K mu(k) floor(x/k) is NOT the Mertens function.
    # The Mertens function is sum_{n=1}^x mu(n).
    # However, sum_{d|n} mu(d) = 1 if n=1 else 0.
    # So sum_{k=1}^x mu(k) floor(x/k) = sum_{k=1}^x mu(k) sum_{j=1}^{floor(x/k)} 1
    # = sum_{j=1}^x sum_{k=1}^{floor(x/j)} mu(k) ... no.
    # sum_{k} mu(k) floor(x/k) = sum_{k} mu(k) sum_{j: k|j, j<=x} 1
    # = sum_{j=1}^x sum_{k|j} mu(k)
    # = sum_{j=1}^x (1 if j=1 else 0)
    # = 1.
    # So if we use the full Mobius function up to K, and x <= K, M(x) = 1.
    # If x > K, it's different.
    # The constraint is M(x) <= 1 for x in [1, 10K].
    # For x > K, the sum is over k<=K.
    # sum_{k=1}^K mu(k) floor(x/k).
    # This is the partial summation.
    # It is known that sum_{k=1}^K mu(k) floor(x/k) is small?
    # Actually, if we include all k up to K, for x=10K,
    # The sum is sum_{k=1}^K mu(k) floor(10K/k).
    # This is not necessarily <= 1.
    # In fact, it grows.
    # So Mobius on [1, K] is likely infeasible for x > K.
    # The inspiration 1 tried this and had a fallback.
    # Let's stick to the LP result which is guaranteed feasible by construction (mostly).
    
    return best_domain, best_values
