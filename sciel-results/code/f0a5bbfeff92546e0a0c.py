import numpy as np
import time
import scipy.sparse as sp
import scipy.optimize as opt

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 580
    
    # We will try to solve the Dual LP.
    # Primal: Max c^T f s.t. A f <= 1, f free.
    # Dual: Min 1^T y s.t. A^T y >= c, y >= 0.
    # c_k = -log(k)/k
    # A_{x,k} = floor(x/k) for x in 1..10K, k in 1..K.
    
    # Since K can be up to 200,000, 10K is 2,000,000.
    # Building the full A matrix is impossible (2M * 200k entries).
    # But A is structured.
    # (A^T y)_k = sum_{x=1}^{10K} y_x floor(x/k).
    
    # We need to minimize sum(y) subject to sum(y_x floor(x/k)) >= -log(k)/k for all k.
    # And y_x >= 0.
    
    # This is a large scale problem.
    # Let's estimate the size.
    # If K is too large, the LP might be too slow or memory heavy.
    # Let's start with a moderate K, say 10000 or 5000, and see if we can scale up.
    # Or use a heuristic to guess the optimal K.
    
    # Alternative: Use the fact that the optimal solution often has support on a small set of k.
    # But let's try the LP for a smaller K first, then expand.
    
    best_score = 0.0
    best_domain = None
    best_values = None
    
    # Try a few K values
    for K in [1000, 2000, 5000, 10000, 20000]:
        if time.time() > deadline - 20:
            break
            
        print(f"Trying K={K}...")
        
        # Objective for dual: minimize sum(y) -> c_dual = ones(10K)
        # Constraints: A^T y >= c_primal
        # c_primal[k] = -log(k)/k
        
        c_primal = -np.log(np.arange(1, K+1, dtype=np.float64)) / np.arange(1, K+1, dtype=np.float64)
        
        # We need to define the constraint matrix A^T implicitly or use a callback?
        # scipy.optimize.linprog doesn't support implicit matrices well.
        # We must build A^T.
        # A^T is K x 10K.
        # Entry (k, x) is floor(x/k).
        
        # This matrix is sparse.
        # For a fixed k, floor(x/k) is non-zero for x >= k.
        # Number of non-zeros is roughly sum_{k=1}^K (10K - k) ~ 5 * 10K * K.
        # For K=1000, 10K=10000. NNZ ~ 50M. This is large.
        # For K=2000, 10K=20000. NNZ ~ 200M. Too large for memory?
        # 200M floats * 8 bytes = 1.6 GB. Might be tight.
        # Let's stick to smaller K for the LP part, or use a different method.
        
        # Let's try K=500 first to ensure it works.
        if K > 500 and time.time() > deadline - 60:
             continue
             
        try:
            # Build A^T
            # Rows: k (1..K)
            # Cols: x (1..10K)
            
            max_x = 10 * K
            rows = []
            cols = []
            data = []
            
            for k in range(1, K + 1):
                # x ranges from k to max_x
                # floor(x/k)
                # We can vectorize this row
                x_vals = np.arange(k, max_x + 1)
                vals = np.floor(x_vals / k).astype(np.float64)
                
                rows.extend([k-1] * len(x_vals))
                cols.extend(x_vals) # x is 1-indexed, so col index is x (0-indexed would be x-1, but linprog expects 0-indexed cols)
                # Wait, linprog A is (m, n). m=constraints, n=variables.
                # Dual: Constraints are k=1..K. Variables are x=1..10K.
                # So A_dual is K x 10K.
                # A_dual[k-1, x-1] = floor(x/k).
                
                data.extend(vals)
                
            # This loop is slow for K=5000.
            # Let's use a faster construction.
            
            # Actually, let's just use the Primal LP with HiGHS for small K, 
            # or a heuristic for large K.
            
            # Let's switch strategy: 
            # The inspirations got 1.0745. 
            # The seed is 1.0.
            # If I can't beat 1.0745 easily, I should at least beat 1.0 robustly.
            
            # Let's try a simpler approach: 
            # Use the Möbius function but scale it?
            # Or use a subset of Möbius?
            
            # Let's try solving the LP for K=1000 with explicit matrix.
            if K <= 1000:
                A_dual = sp.csr_matrix((data, (rows, cols)), shape=(K, max_x))
                # A_dual is K x 10K
                # Constraints: A_dual @ y >= c_primal
                # linprog requires A_ub @ x <= b_ub
                # So -A_dual @ y <= -c_primal
                
                c_dual = np.ones(max_x)
                A_ub = -A_dual
                b_ub = -c_primal
                
                # y >= 0
                bounds = [(0, None)] * max_x
                
                res = opt.linprog(c_dual, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                
                if res.success:
                    y_opt = res.x
                    # Recover f from dual? 
                    # Complementary slackness: if A^T y > c, then f_k = 0.
                    # If f_k != 0, then (A^T y)_k = c_k.
                    # This is hard to recover directly.
                    
                    # Instead, let's solve the PRIMAL for K=1000.
                    # Primal: Max c^T f s.t. A f <= 1.
                    # A is 10K x K.
                    # A[x-1, k-1] = floor(x/k).
                    
                    rows_p = []
                    cols_p = []
                    data_p = []
                    
                    for k in range(1, K + 1):
                        x_vals = np.arange(k, max_x + 1)
                        vals = np.floor(x_vals / k).astype(np.float64)
                        rows_p.extend(x_vals) # x-1
                        cols_p.extend([k-1] * len(x_vals))
                        data_p.extend(vals)
                        
                    A_primal = sp.csr_matrix((data_p, (rows_p, cols_p)), shape=(max_x, K))
                    
                    # Max c^T f s.t. A f <= 1
                    # Min -c^T f s.t. A f <= 1
                    c_primal_neg = -c_primal
                    
                    res_p = opt.linprog(c_primal_neg, A_ub=A_primal, b_ub=np.ones(max_x), bounds=[(None, None)] * K, method='highs')
                    
                    if res_p.success:
                        f_opt = res_p.x
                        # Clip to [-10, 10]
                        f_opt = np.clip(f_opt, -10, 10)
                        
                        # Check feasibility
                        M = A_primal @ f_opt
                        if np.max(M) <= 1.0001:
                            # Compute score
                            # a = -sum f(k) log(k)/k
                            a = -np.sum(f_opt * c_primal) # c_primal is -log(k)/k, so -sum f * (-log/k) = sum f log/k? 
                            # Wait. Objective is a = -sum f(k) log(k)/k.
                            # c_primal = -log(k)/k.
                            # Max c^T f = sum (-log(k)/k) f(k) = - sum f(k) log(k)/k.
                            # So res_p.fun is -a.
                            # a = -res_p.fun.
                            
                            a_val = -res_p.fun
                            if a_val > best_score:
                                best_score = a_val
                                # Construct domain and values
                                mask = np.abs(f_opt) > 1e-6
                                domain = np.nonzero(mask)[0] + 1
                                values = f_opt[mask]
                                best_domain = domain
                                best_values = values
                                print(f"K={K}, Score={a_val}")
                                
        except Exception as e:
            print(f"Error for K={K}: {e}")
            continue
            
    if best_domain is not None:
        return best_domain, best_values
    
    # Fallback to seed
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values
