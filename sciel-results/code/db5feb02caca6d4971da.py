import numpy as np
import time
from scipy.optimize import linprog

def _check_feasibility_exact(domain, values, K_max, tolerance=1.0001):
    """
    Mimics the evaluator's logic exactly to check feasibility.
    1. Clip values to [-10, 10].
    2. Calculate S = sum(f(k)/k).
    3. Adjust f(1) = f(1) - S. (If 1 not in domain, f(1) becomes -S, but domain doesn't change for M(x) sum? 
       Wait, "M(x) = sum_k f(k) * floor(x/k)". The sum is over the domain.
       If 1 is not in domain, does the evaluator add it?
       "f(1) is replaced by f(1) - S ... (f(1) = 0 before this if 1 is not in the domain)"
       This implies f(1) is treated as a value in the function for the calculation of M(x) and the sum.
       So if 1 is not in the returned domain, the evaluator effectively adds it with value 0, then adjusts it to -S.
       Then M(x) includes the term f(1)*floor(x/1) = -S * x.
    """
    # 1. Clip
    clipped = np.clip(values, -10.0, 10.0)
    
    # 2. Calculate S
    # The sum is over the domain.
    # "sum_k f(k)/k"
    S = np.sum(clipped / domain.astype(np.float64))
    
    # 3. Adjust f(1)
    # Create a copy of the domain/values for calculation
    # If 1 is in domain, we update that specific value.
    # If 1 is NOT in domain, we must add it to the effective domain for M(x) calculation?
    # The prompt says: "M(x) = sum_k f(k) * floor(x/k)".
    # If the domain returned is D, and 1 not in D, does M(x) sum over D or D U {1}?
    # "f(1) is replaced by ... so that afterwards sum_k f(k)/k = 0".
    # This strongly implies the set of k for the sum includes 1.
    # Therefore, M(x) likely includes the term for k=1.
    
    # Let's construct the effective values for k=1..K_max
    # We assume the domain contains distinct positive integers <= K_max.
    
    effective_f = np.zeros(K_max + 1, dtype=np.float64)
    
    # Fill with clipped values
    for i, k in enumerate(domain):
        if k <= K_max:
            effective_f[k] = clipped[i]
            
    # Apply f(1) adjustment
    # "f(1) is replaced by f(1) - S"
    # effective_f[1] is currently clipped value of 1 (or 0 if not in domain).
    effective_f[1] = effective_f[1] - S
    
    # Now compute M(x) = sum_{k=1}^{K_max} effective_f[k] * floor(x/k)
    # We need to check M(x) <= tolerance for x in [1, 10*K_max]
    
    x_max = 10 * K_max
    h = np.zeros(x_max + 1, dtype=np.float64)
    
    # For each k, add effective_f[k] to h[k], h[2k], ...
    for k in range(1, K_max + 1):
        fk = effective_f[k]
        if fk != 0.0:
            h[k::k] += fk
            
    M_vals = np.cumsum(h[1:])
    
    if len(M_vals) == 0:
        return True
        
    return np.max(M_vals) <= tolerance

def _compute_score_exact(domain, values, K_max):
    """
    Compute the score a = -sum_k f(k) log(k) / k using the evaluator's logic.
    1. Clip.
    2. Adjust f(1).
    3. Sum over k in domain? Or k in 1..K?
    "a = -sum_k f(k) * log(k) / k"
    If 1 is not in domain, log(1)=0, so it contributes 0.
    So we can just sum over the provided domain, but we must use the ADJUSTED f(1) if 1 is in domain.
    If 1 is NOT in domain, f(1) is effectively 0 before adjustment, then -S.
    Does the sum include this new f(1)?
    "sum_k" usually implies the support. If 1 was not in the support, is it added?
    Given log(1)=0, it doesn't matter for the value of 'a'.
    """
    clipped = np.clip(values, -10.0, 10.0)
    S = np.sum(clipped / domain.astype(np.float64))
    
    adjusted = clipped.copy()
    for i, k in enumerate(domain):
        if k == 1:
            adjusted[i] = clipped[i] - S
            break
    
    k_float = domain.astype(np.float64)
    # If 1 is not in domain, we don't add it because log(1)=0.
    return -np.sum(adjusted * np.log(k_float) / k_float)

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the objective a = -sum_k f(k) log(k) / k subject to M(x) <= 1.
    """
    start_time = time.time()
    deadline = 580  # Leave 20s margin
    
    # Fallback seed
    seed_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    seed_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    
    best_domain = seed_domain
    best_values = seed_values
    best_a = _compute_score_exact(seed_domain, seed_values, 30)
    
    # Try LP for various K
    # K=1000 is a good balance of performance and solution quality
    K_candidates = [1000, 500, 200]
    
    for K in K_candidates:
        if time.time() - start_time > deadline - 30:
            break
            
        try:
            # Variables: g(2) ... g(K)
            n_vars = K - 1
            if n_vars <= 0:
                continue
                
            ks = np.arange(2, K + 1)
            
            # Objective: minimize sum( (log(k)/k) * g(k) )
            c = np.log(ks) / ks
            
            # Constraints:
            # M(x) = f(1)x + sum_{k=2}^K f(k) floor(x/k) <= 1
            # With f(1) = -S = -sum_{k=2}^K f(k)/k
            # M(x) = -x * sum_{k=2}^K f(k)/k + sum_{k=2}^K f(k) floor(x/k)
            # M(x) = sum_{k=2}^K f(k) (floor(x/k) - x/k)
            # M(x) = - sum_{k=2}^K f(k) {x/k}
            # Constraint: - sum_{k=2}^K f(k) {x/k} <= 1
            
            max_x = 10 * K
            
            # Build A_ub (max_x x n_vars)
            # A_ub[x-1, k-2] = - {x/k}
            
            # Vectorized construction
            x_vals = np.arange(1, max_x + 1)[:, np.newaxis]  # (max_x, 1)
            k_vals = ks[np.newaxis, :]                       # (1, n_vars)
            
            # Calculate fractional part {x/k} = (x % k) / k
            # Note: x and k are integers.
            # Using np.remainder for integer mod?
            # x_vals and k_vals are integers.
            frac_parts = (x_vals % k_vals) / k_vals.astype(np.float64)
            A_ub = -frac_parts
            
            b_ub = np.ones(max_x)
            
            # Bounds: -10 <= g(k) <= 10
            bounds = [(-10.0, 10.0)] * n_vars
            
            # Solve LP
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if res.success:
                g = res.x  # f(2)...f(K)
                
                # Construct full f vector for 1..K
                f_full = np.zeros(K + 1)
                f_full[2:K+1] = g
                
                # Calculate S = sum_{k=2}^K f(k)/k
                S = np.sum(g / ks)
                f_full[1] = -S
                
                # Prepare domain and values
                domain = np.arange(1, K + 1, dtype=np.int64)
                values = f_full[1:K+1]
                
                # Check feasibility with exact evaluator logic
                if _check_feasibility_exact(domain, values, K):
                    a_val = _compute_score_exact(domain, values, K)
                    if a_val > best_a:
                        best_a = a_val
                        best_domain = domain
                        best_values = values
                        
        except Exception:
            continue
            
    return best_domain, best_values
