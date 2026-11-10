# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    """
    Maximizes a = -sum f(k) log(k)/k subject to M(x) <= 1.0001.
    Uses LP formulation with f(k) in [-10, 10] for k >= 2.
    """
    start_time = time.time()
    deadline = start_time + 570  # 30s safety margin
    
    best_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    best_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    best_score = 1.0  # Seed score

    def evaluate(domain, values):
        """
        Compute the actual score as the evaluator would.
        Returns (score, feasible).
        """
        if len(domain) == 0:
            return 0.0, False
        
        # Clip values
        values_clipped = np.clip(values, -10, 10)
        
        # Compute S = sum f(k)/k over clipped values
        S = np.sum(values_clipped / domain.astype(float))
        
        # Adjust f(1)
        f_adj = values_clipped.copy()
        idx_1 = np.where(domain == 1)[0]
        if len(idx_1) > 0:
            f_adj[idx_1[0]] = f_adj[idx_1[0]] - S
        else:
            # If 1 not in domain, effectively f(1) = -S but it's not in domain
            # The evaluator treats f(1)=0 initially, so f(1)_new = -S
            # But since 1 is not in domain, this doesn't contribute to M(x)
            # Actually, the evaluator says "f(1) is replaced by f(1) - S"
            # and "f(1) = 0 before this if 1 is not in the domain"
            # So we need to handle this: add 1 to domain with value -S
            # But we control the domain, so let's always include 1
            pass
            
        K_max = int(domain.max())
        max_x_check = 10 * K_max
        
        # Compute M(x) = sum_k f_adj(k) * floor(x/k) for x in [1, 10K]
        x_vals = np.arange(1, max_x_check + 1)
        M_sum = np.zeros(max_x_check)
        for i in range(len(domain)):
            k = domain[i]
            if f_adj[i] != 0:
                M_sum += f_adj[i] * np.floor(x_vals / k)
        
        if np.max(M_sum) > 1.0001:
            return 0.0, False
        
        # Compute objective: a = -sum f_adj(k) * log(k) / k
        # log(1) = 0, so k=1 term is 0
        a = -np.sum(f_adj * np.log(domain.astype(float)) / domain.astype(float))
        
        return a, True

    # Try multiple K values with the LP approach
    # Using f(k) directly in [-10, 10] for k >= 2
    # Constraint: sum_{k=2}^K f(k) * (floor(x/k) - x/k) <= 1.0001 for x in [1, 10K]
    # Objective: maximize -sum_{k=2}^K f(k) * log(k)/k
    # linprog minimizes, so c = log(k)/k
    
    K_values = [100, 200, 400, 800]
    
    for K in K_values:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
            
        n_vars = K - 1
        if n_vars <= 0:
            continue
        
        # Objective coefficients: c_i = log(i+2)/(i+2) for i=0..K-2
        # We minimize sum c_i * f(i+2), which is -maximize sum f(i+2)*log(i+2)/(i+2)
        c = np.array([math.log(k) / k for k in range(2, K + 1)])
        
        # Build constraint matrix A_ub
        # For each x in [1, 10K], constraint: sum_{k=2}^K f(k)*(floor(x/k) - x/k) <= 1.0001
        # A_ub[x_idx, k_idx] = floor((x_idx+1)/(k_idx+2)) - (x_idx+1)/(k_idx+2)
        max_x = 10 * K
        
        # Vectorized construction
        x_col = np.arange(1, max_x + 1, dtype=np.float64)[:, None]  # (10K, 1)
        k_row = np.arange(2, K + 1, dtype=np.float64)[None, :]  # (1, K-1)
        
        # floor(x/k) - x/k = -frac(x/k)
        A_ub = np.floor(x_col / k_row) - x_col / k_row  # (10K, K-1)
        b_ub = np.full(max_x, 1.0001)
        
        # Bounds: f(k) in [-10, 10]
        bounds = [(-10, 10)] * n_vars
        
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        except Exception:
            continue
            
        if not result.success:
            continue
            
        f_vals = result.x  # f(2), f(3), ..., f(K)
        
        # Construct domain and values
        domain = np.arange(1, K + 1, dtype=np.int64)
        values = np.zeros(K, dtype=np.float64)
        values[0] = 0.0  # f(1) placeholder, will be adjusted by evaluator
        values[1:] = f_vals
        
        score, feasible = evaluate(domain, values)
        
        if feasible and score > best_score:
            best_score = score
            best_domain = domain.copy()
            best_values = values.copy()

    return best_domain, best_values

# EVOLVE-BLOCK-END
