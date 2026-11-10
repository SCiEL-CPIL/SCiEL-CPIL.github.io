import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 600
    
    best_score = -np.inf
    best_domain = None
    best_values = None
    
    # Try K values
    # K=1000 is known to work well.
    # We can try a few values.
    for K in [1000]:
        if time.time() > deadline - 120:
            break
            
        try:
            k_vals = np.arange(1, K + 1)
            
            # Objective: Minimize sum f_k * log(k) / k
            c = np.log(k_vals) / k_vals
            
            # Equality: sum f_k / k = 0
            row_idx_eq = np.zeros(K, dtype=np.int32)
            col_idx_eq = np.arange(K)
            data_eq = 1.0 / k_vals
            A_eq = sp.coo_matrix((data_eq, (row_idx_eq, col_idx_eq)), shape=(1, K)).tocsr()
            b_eq = np.array([0.0])
            
            # Inequality: M(x) <= 1.0001
            max_x = 10 * K
            n_constraints = max_x
            
            # Estimate non-zeros
            # For each k, non-zeros are for x in [k, max_x]
            # Count is max_x - k + 1
            nz_counts = max_x - k_vals + 1
            total_nz = int(np.sum(nz_counts))
            
            row_indices = np.zeros(total_nz, dtype=np.int32)
            col_indices = np.zeros(total_nz, dtype=np.int32)
            data_values = np.zeros(total_nz, dtype=np.float64)
            
            ptr = 0
            for k in range(1, K + 1):
                j = 1
                x_start = k
                while x_start <= max_x:
                    x_end = min((j + 1) * k - 1, max_x)
                    count = x_end - x_start + 1
                    if count > 0:
                        row_indices[ptr:ptr+count] = np.arange(x_start - 1, x_end)
                        col_indices[ptr:ptr+count] = k - 1
                        data_values[ptr:ptr+count] = j
                        ptr += count
                    j += 1
                    x_start = j * k
                    
            if ptr < total_nz:
                row_indices = row_indices[:ptr]
                col_indices = col_indices[:ptr]
                data_values = data_values[:ptr]
                
            A_ub = sp.coo_matrix((data_values, (row_indices, col_indices)), shape=(n_constraints, K)).tocsr()
            b_ub = np.ones(n_constraints) * 1.0001
            
            bounds = [(-10, 10)] * K
            
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            
            if res.success:
                f_vals = res.x
                f_vals_clipped = np.clip(f_vals, -10, 10)
                
                S = np.sum(f_vals_clipped / k_vals)
                f_vals_final = f_vals_clipped.copy()
                f_vals_final[0] = f_vals_final[0] - S
                
                score = -np.sum(f_vals_final * np.log(k_vals) / k_vals)
                
                if score > best_score:
                    best_score = score
                    best_domain = k_vals
                    best_values = f_vals_final
                    
        except Exception as e:
            print(f"Error for K={K}: {e}")
            continue
            
    if best_domain is not None:
        return best_domain, best_values
    else:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
