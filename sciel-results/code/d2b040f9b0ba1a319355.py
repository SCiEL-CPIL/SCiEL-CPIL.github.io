# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _verify_solution(domain, values, K):
    """Verify that the solution satisfies all constraints efficiently."""
    if len(domain) != len(values):
        return False
    if len(domain) < 1:
        return False
    if np.any(domain <= 0) or np.any(domain > K):
        return False
    if not np.all(np.isfinite(values)):
        return False
    
    # Clip to [-10, 10]
    clipped = np.clip(values, -10, 10)
    
    # Compute S = sum f(k)/k
    S = np.sum(clipped / domain.astype(np.float64))
    
    # Adjust f(1)
    has_1 = 1 in domain
    if has_1:
        f1_idx = np.where(domain == 1)[0]
        f1_val = clipped[f1_idx[0]] - S
        if f1_val < -10 or f1_val > 10:
            return False
    else:
        if -S < -10 or -S > 10:
            return False
    
    # Build full f array up to K
    f = np.zeros(K + 1)
    for i, k in enumerate(domain):
        f[k] = clipped[i]
    
    # Apply adjustment
    if has_1:
        f[1] = f[1] - S
    else:
        f[1] = -S
    
    # Check M(x) for all x in [1, 10K] using efficient vectorized approach
    # M(x) = sum_{k=1}^K f[k] * floor(x/k)
    # For each k, floor(x/k) is constant for x in [k*j, k*(j+1)-1]
    # We can compute M(x) for all x by accumulating contributions
    
    max_x = 10 * K
    M = np.zeros(max_x + 1, dtype=np.float64)
    
    # For each k with non-zero f[k], add f[k] * floor(x/k) for all x
    # floor(x/k) = j for x in [k*j, k*(j+1)-1]
    for k in range(1, K + 1):
        if f[k] == 0:
            continue
        fk = f[k]
        # For each j, x ranges from k*j to min(k*(j+1)-1, max_x)
        j_max = max_x // k
        for j in range(0, j_max + 1):
            start = k * j
            end = min(k * (j + 1) - 1, max_x)
            if start <= end:
                M[start:end+1] += fk * j
    
    if np.any(M[1:] > 1.0001):
        return False
    
    return True

def _compute_metric(domain, values, K):
    """Compute the metric a = -sum f(k)*log(k)/k"""
    clipped = np.clip(values, -10, 10)
    S = np.sum(clipped / domain.astype(np.float64))
    
    f = np.zeros(K + 1)
    for i, k in enumerate(domain):
        f[k] = clipped[i]
    
    has_1 = 1 in domain
    if has_1:
        f[1] = f[1] - S
    else:
        f[1] = -S
    
    a = 0.0
    for k in range(2, K + 1):
        if f[k] != 0:
            a -= f[k] * math.log(k) / k
    return a

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    from scipy.optimize import linprog
    
    best_domain = None
    best_values = None
    best_metric = -np.inf
    
    # Try different K values, starting with larger ones
    K_candidates = [2000, 1500, 1200, 1000]
    
    for K in K_candidates:
        if time.time() > deadline:
            break
        
        n_vars = K - 1
        if n_vars <= 0:
            continue
        
        k_vals = np.arange(2, K + 1, dtype=np.float64)
        
        # Objective: maximize -sum c_k g_k => minimize sum c_k g_k
        c = np.log(k_vals) / k_vals
        
        # Constraints: M(x) = sum_{k=2}^K g(k) * (floor(x/k) - 1/k) <= 1
        # for all x in [1, 10K]
        
        # We need to be careful about which x values to include.
        # The maximum of M(x) may not be at x=10K if g(k) can be negative.
        # Let's include a strategic subset of x values.
        
        # Key x values: multiples of small k's, and x = 10K
        x_set = set()
        x_set.add(10 * K)
        x_set.add(1)
        x_set.add(2)
        
        # Add multiples of small k values
        for k in range(2, min(K, 100) + 1):
            for m in range(1, 10 * K // k + 1):
                x = m * k
                if x <= 10 * K:
                    x_set.add(x)
        
        # Add some random x values for coverage
        if time.time() < deadline - 30:
            random_x = np.random.choice(range(1, 10 * K + 1), size=min(5000, 10 * K), replace=False)
            x_set.update(random_x.tolist())
        
        x_vals = np.array(sorted(x_set), dtype=np.float64)
        n_constraints = len(x_vals)
        
        # Build A matrix: A[i, j] = floor(x_i / k_j) - 1/k_j
        A = np.floor(x_vals[:, None] / k_vals[None, :]) - 1.0 / k_vals[None, :]
        
        b = np.ones(n_constraints)
        bounds = [(-10, 10)] * n_vars
        
        try:
            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs', 
                         options={'time_limit': min(60, deadline - time.time() - 10)})
            
            if res.success and res.x is not None:
                g = res.x
                f1 = -np.sum(g / k_vals)
                
                # Check f1 is within bounds
                if abs(f1) > 10:
                    continue
                
                domain = np.arange(1, K + 1, dtype=np.int64)
                values = np.zeros(K, dtype=np.float64)
                values[0] = f1
                values[1:] = g
                
                # Quick verification
                if _verify_solution(domain, values, K):
                    metric = _compute_metric(domain, values, K)
                    if metric > best_metric:
                        best_metric = metric
                        best_domain = domain.copy()
                        best_values = values.copy()
                        
        except Exception as e:
            continue
    
    if best_domain is not None:
        return best_domain, best_values
    
    # Fallback: seed program's result
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values

# EVOLVE-BLOCK-END
