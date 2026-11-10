# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = 600.0
    
    # We formulate the problem as a Linear Program.
    # Variables: f(1), ..., f(K)
    # Objective: Maximize a = -sum_{k=2}^K f(k) log(k) / k  (f(1) is adjusted by evaluator to make sum f(k)/k = 0, 
    # but the metric formula in the contract is a = -sum_k f(k) log(k) / k. 
    # Wait, the contract says: "f(1) is replaced by f(1) - S ... so that afterwards sum_k f(k)/k = 0. 
    # Otherwise the metric is a = -sum_k f(k) log(k) / k".
    # This implies the metric is calculated on the *modified* values where sum f(k)/k = 0.
    # Let f' be the modified values. f'(1) = f(1) - S, f'(k) = f(k) for k>1.
    # S = sum_{k=1}^K f(k)/k.
    # sum_{k=1}^K f'(k)/k = (f(1)-S)/1 + sum_{k=2}^K f(k)/k = f(1) - S + (S - f(1)) = 0. Correct.
    # Metric a = -sum_{k=1}^K f'(k) log(k) / k = -f'(1) log(1) - sum_{k=2}^K f(k) log(k)/k = -sum_{k=2}^K f(k) log(k)/k.
    # Since log(1)=0, the adjustment of f(1) does not affect the metric value directly, 
    # BUT it affects the feasibility constraint M(x).
    # M(x) = sum_{k=1}^K f'(k) floor(x/k).
    # f'(1) = f(1) - S.
    # So M(x) = (f(1)-S) * floor(x/1) + sum_{k=2}^K f(k) floor(x/k).
    # floor(x/1) = x.
    # So M(x) = f(1) x - S x + sum_{k=2}^K f(k) floor(x/k).
    # We need M(x) <= 1.0001 for all x in [1, 10K].
    
    # This is a linear constraint in f(1) and f(k).
    # However, solving an LP with 10K constraints and K variables is large.
    # For K=100, 10K=1000. 1000 constraints, 100 variables. This is very fast.
    # For K=500, 10K=5000. 5000 constraints, 500 variables. Still fast with scipy.
    # Let's try K=200 to be safe on time, or maybe K=1000 if fast enough.
    
    # Let's start with K=100.
    # If it works and is fast, we can try larger K in a future iteration or binary search.
    # Given the stagnation, a structural change to LP is the key.
    
    K = 100
    x_max = 10 * K
    
    # Variables: f_1, f_2, ..., f_K
    # n_vars = K
    
    # Objective: Minimize -a = sum_{k=2}^K f(k) log(k) / k
    # c = [0, log(2)/2, log(3)/3, ..., log(K)/K]
    c = np.zeros(K)
    for k in range(2, K + 1):
        c[k - 1] = np.log(k) / k
    
    # Constraints: M(x) <= 1.0001
    # M(x) = sum_{k=1}^K f'(k) floor(x/k)
    # f'(1) = f(1) - S, where S = sum_{j=1}^K f(j)/j
    # M(x) = (f(1) - S) x + sum_{k=2}^K f(k) floor(x/k)
    # M(x) = f(1) x - x sum_{j=1}^K f(j)/j + sum_{k=2}^K f(k) floor(x/k)
    # M(x) = f(1) x - x (f(1) + sum_{j=2}^K f(j)/j) + sum_{k=2}^K f(k) floor(x/k)
    # M(x) = f(1) x - x f(1) - x sum_{j=2}^K f(j)/j + sum_{k=2}^K f(k) floor(x/k)
    # M(x) = - x sum_{j=2}^K f(j)/j + sum_{k=2}^K f(k) floor(x/k)
    # Notice f(1) cancels out!
    # M(x) = sum_{k=2}^K f(k) (floor(x/k) - x/k)
    
    # This is a huge simplification. The constraint does not depend on f(1).
    # And f(1) does not appear in the objective.
    # So we can set f(1) = 0 (or any value, it gets adjusted).
    # The problem reduces to:
    # Minimize sum_{k=2}^K f(k) log(k)/k
    # Subject to:
    # 1. sum_{k=2}^K f(k) (floor(x/k) - x/k) <= 1.0001 for all x in [1, 10K]
    # 2. -10 <= f(k) <= 10 for all k (clipping)
    
    # Let's check the constraint for x=1:
    # sum_{k=2}^K f(k) (floor(1/k) - 1/k) = sum_{k=2}^K f(k) (0 - 1/k) = - sum_{k=2}^K f(k)/k <= 1.0001
    # This is a valid constraint.
    
    # Let's construct the constraint matrix A_ub.
    # Rows: x = 1 to 10K
    # Cols: k = 2 to K (index 1 to K-1 in 0-based, but let's keep 1-based for clarity in logic)
    # A_ub[x-1, k-1] = floor(x/k) - x/k
    
    # We only need to consider k from 2 to K.
    # Number of variables = K - 1.
    n_vars = K - 1
    c_lp = np.zeros(n_vars)
    for i in range(n_vars):
        k = i + 2
        c_lp[i] = np.log(k) / k
    
    # Build A_ub
    # x from 1 to 10K
    # k from 2 to K
    # A_ub has shape (10K, K-1)
    
    # We can build this efficiently.
    # For each x, for each k, A_ub[x-1, k-2] = floor(x/k) - x/k
    
    # Let's use numpy broadcasting or loops.
    # 10K = 1000. K-1 = 99. 1000 * 99 = 99,000 entries. Very small.
    
    A_ub = np.zeros((x_max, n_vars))
    for x in range(1, x_max + 1):
        for i in range(n_vars):
            k = i + 2
            A_ub[x - 1, i] = np.floor(x / k) - x / k
    
    b_ub = np.full(x_max, 1.0001)
    
    # Bounds for f(k)
    bounds = [(-10, 10)] * n_vars
    
    # Solve LP
    # Minimize c^T x
    res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        # Extract f(k) for k=2..K
        f_vals = res.x
        # Construct full values array
        values = np.zeros(K)
        values[0] = 0.0 # f(1)
        values[1:] = f_vals
        
        # Verify feasibility just in case (floating point errors)
        # Re-calculate M(x) with the solved values
        # Note: The evaluator will calculate S and adjust f(1).
        # We verified that f(1) cancels out of M(x).
        # So M(x) depends only on f(2..K).
        
        # Let's do a quick check
        max_M = 0
        for x in range(1, x_max + 1):
            m_val = 0
            for i in range(n_vars):
                k = i + 2
                m_val += f_vals[i] * (np.floor(x / k) - x / k)
            if m_val > max_M:
                max_M = m_val
        
        if max_M <= 1.0001:
            domain = np.arange(1, K + 1, dtype=np.int64)
            return domain, values.astype(np.float64)
        else:
            # Fallback if numerical error caused violation
            # Try a simpler feasible solution: all zeros?
            # If all zeros, M(x) = 0 <= 1.0001. Score = 0.
            # But we want a positive score.
            # Let's try to scale down the solution.
            # The constraint is linear. If res.x violates by epsilon, scale by (1.0001 - max_M) / max_M?
            # Actually, if max_M > 1.0001, we can scale f_vals by factor alpha < 1.
            alpha = 1.0001 / max_M
            f_vals_scaled = f_vals * alpha
            values[1:] = f_vals_scaled
            domain = np.arange(1, K + 1, dtype=np.int64)
            return domain, values.astype(np.float64)
    else:
        # LP failed. Fallback to a simple feasible solution.
        # Set f(2) = 1, others 0.
        # M(x) = 1 * (floor(x/2) - x/2).
        # Max of (floor(x/2) - x/2) is 0 (when x even) or -0.5 (when x odd).
        # So M(x) <= 0 <= 1.0001.
        # Score = -1 * log(2)/2.
        domain = np.arange(1, K + 1, dtype=np.int64)
        values = np.zeros(K)
        values[1] = 1.0
        return domain, values.astype(np.float64)

# EVOLVE-BLOCK-END
