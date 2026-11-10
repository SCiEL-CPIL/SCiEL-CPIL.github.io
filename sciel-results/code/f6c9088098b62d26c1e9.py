import numpy as np
import time
import sys
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

def prime_number_theorem_partial_function():
    start_time = time.time()
    timeout = 500  # Safety margin
    
    best_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    best_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    best_score = 1.0

    # Try a range of K values
    K_candidates = [50, 100, 200, 500, 1000]
    
    for K in K_candidates:
        if time.time() - start_time > timeout:
            break
            
        try:
            # Variables f_2 ... f_K
            n_vars = K - 1
            indices = np.arange(2, K + 1)
            
            # Objective: maximize -sum f(k) log(k)/k
            # linprog minimizes, so we minimize sum f(k) log(k)/k
            # c_k = log(k)/k
            c = np.log(indices) / indices
            
            # Constraints: sum {x/k} f_k >= -1.0001 for x in 1..10K
            # This is A @ f >= b
            # linprog handles A_ub @ x <= b_ub
            # So we use -A @ f <= 1.0001
            # A_{x,k} = {x/k}
            
            x_vals = np.arange(1, 10 * K + 1)
            
            # Construct constraint matrix A (10K x (K-1))
            # A_{x,k} = {x/k}
            # For efficiency, we can compute this in chunks if K is large
            # But for K up to 1000, 10000 x 999 is manageable in memory
            
            frac_parts = (x_vals[:, None] / indices[None, :]) % 1
            A = -frac_parts  # We want -A @ f <= 1.0001
            b = np.full(10 * K, 1.0001)
            
            bounds = [(-10, 10)] * n_vars
            
            # Solve
            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
            
            if res.status == 0:
                f_vals = res.x
                # Check feasibility with original formulation just in case
                S = np.sum(f_vals / indices)
                # f(1) eval = -S
                # M(x) = -S*x + sum f(k) floor(x/k)
                # Max M(x)
                x_check = np.arange(1, 10*K + 1)
                M_vals = -S * x_check + np.dot(f_vals, np.floor(x_check[:, None] / indices[None, :]))
                if np.max(M_vals) <= 1.0001 + 1e-6:
                    # Calculate score
                    score = -np.sum(f_vals * np.log(indices) / indices)
                    if score > best_score:
                        best_score = score
                        # Construct domain and values
                        # domain: 1, 2, ..., K
                        domain = np.arange(1, K + 1, dtype=np.int64)
                        values = np.zeros(K, dtype=np.float64)
                        values[0] = 0.0 # f(1) orig, evaluator adjusts
                        values[1:] = f_vals
                        best_domain = domain
                        best_values = values
                        
        except Exception as e:
            print(f"Error for K={K}: {e}", file=sys.stderr)
            continue
            
    return best_domain, best_values
