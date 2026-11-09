import numpy as np
import time
from scipy.optimize import minimize
import sys

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = 600.0
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # We try to maximize a = -sum f(k) log(k)/k
    # Subject to:
    # 1. sum f(k)/k = 0 (handled by f(1) adjustment, so effectively sum_{k>=2} g(k)/k = S, f(1)=-S)
    # 2. M(x) = sum_{k} f(k) floor(x/k) <= 1 for all x in [1, 10K]
    # 3. -10 <= f(k) <= 10
    
    # The constraint M(x) <= 1 is equivalent to:
    # sum_{k=2}^K g(k) floor(x/k) + f(1) floor(x/1) <= 1
    # f(1) = -sum_{k=2}^K g(k)/k
    # So: sum_{k=2}^K g(k) (floor(x/k) - x/k) <= 1
    # Let h(x, k) = floor(x/k) - x/k. Note h(x,k) is in [-1, 0).
    # Actually, the evaluator clips values to [-10, 10] BEFORE calculating S.
    # So we must ensure |g(k)| <= 10 and |f(1)| <= 10.
    
    # Strategy:
    # Instead of LP with 10K constraints, we use a heuristic search or a smaller LP.
    # The previous LP was limited by memory. 
    # Let's try a different formulation: 
    # We want to maximize -sum g(k) log(k)/k.
    # This is equivalent to minimizing sum g(k) log(k)/k.
    # Since log(k)/k is positive for k>=2, we want g(k) to be as negative as possible (-10) for small k 
    # where log(k)/k is large, and as positive as possible (10) for large k where log(k)/k is small,
    # BUT constrained by M(x) <= 1.
    
    # The constraint M(x) <= 1 is the bottleneck.
    # M(x) = sum_{k=1}^K f(k) floor(x/k).
    
    # Let's try a greedy/iterative improvement approach or a smaller LP with better scaling.
    # The previous code tried K up to 3000. 30000 * 3000 matrix is 90M floats. 
    # Maybe we can use a sparse matrix? SciPy linprog supports sparse.
    
    from scipy.sparse import csr_matrix, diags
    
    # Let's try K values again, but using sparse matrices for the constraint matrix.
    # A_ub[i, j] corresponds to constraint x=i+1, variable g(j+2).
    # A_ub[i, j] = -(floor((i+1)/(j+2)) - (i+1)/(j+2)) ? No.
    
    # Constraint: sum_{k=2}^K g(k) (floor(x/k) - x/k) <= 1
    # Let C(x, k) = floor(x/k) - x/k.
    # We want to minimize c^T g subject to C g <= 1.
    
    # C is 10K x (K-1).
    # For K=5000, 50000 x 4999. Sparse is essential.
    
    def build_sparse_constraints(K):
        max_x = 10 * K
        n_vars = K - 1
        ks = np.arange(2, K + 1)
        
        # We need to construct the matrix A where A[i, j] = -(floor((i+1)/(j+2)) - (i+1)/(j+2))
        # Wait, the constraint is sum g(k) * (floor(x/k) - x/k) <= 1.
        # So A[i, j] = (floor((i+1)/(j+2)) - (i+1)/(j+2)).
        
        rows = []
        cols = []
        data = []
        
        # This loop might be slow in Python. We need vectorization or a faster way.
        # floor(x/k) - x/k is periodic in x with period k.
        # For a fixed k, the values repeat every k integers.
        
        # Let's try to generate the sparse matrix efficiently.
        # For each k (column j), the values in column j are determined by x mod k.
        
        # Alternative: Use the fact that we only need to check feasibility.
        # Maybe we can just run the LP for smaller K where dense is fine, and use a heuristic for larger K?
        
        # The previous score 1.0745 came from K=3000 dense LP.
        # Let's try to push K higher with sparse.
        
        # Constructing sparse matrix:
        # For each k in 2..K:
        #   for x in 1..10K:
        #      val = floor(x/k) - x/k
        #      if val != 0: add to matrix
        
        # This is still O(10K * K).
        
        # Let's stick to the dense LP but optimize the memory layout or try K=5000 with sparse if possible.
        # However, constructing the sparse matrix in Python loops is slow.
        
        # Let's try a different optimization: 
        # We know the optimal solution likely has many g(k) at the bounds (-10 or 10).
        # We can use a "boundary" heuristic.
        
        # Let's try to improve the LP by using a warm start or a better solver setting?
        # Or simply try K=4000, 5000 with sparse if we can build it fast enough.
        
        pass

    # Let's try a simpler improvement: 
    # The objective is -sum g(k) log(k)/k.
    # The constraint is M(x) <= 1.
    
    # Let's try K values up to 10000 using a sparse matrix construction that is vectorized.
    
    def solve_sparse_lp(K):
        if time.time() - start_time > 580:
            return None
            
        n_vars = K - 1
        ks = np.arange(2, K + 1)
        c = np.log(ks) / ks
        
        max_x = 10 * K
        
        # Build sparse A_ub
        # A_ub[i, j] = floor((i+1)/(j+2)) - (i+1)/(j+2)
        
        # We can generate indices for non-zeros.
        # For each k, the pattern of floor(x/k) - x/k repeats every k.
        # Let's compute the pattern for one period.
        
        rows = []
        cols = []
        vals = []
        
        # This might be too slow to build in Python for K=5000.
        # Let's estimate: 50000 * 5000 = 250M operations. Too slow in pure Python.
        
        # Alternative: Use the dense LP for K up to 5000 if memory allows?
        # 50000 * 5000 * 8 bytes = 2GB. Too much.
        
        # Let's try a different approach: 
        # Use the dual problem?
        # Maximize -sum g(k) log(k)/k
        # s.t. sum g(k) (floor(x/k) - x/k) <= 1 for all x
        #      -10 <= g(k) <= 10
        
        # Dual: Minimize sum_{x=1}^{10K} y_x * 1 + sum_{k} (10 * max(0, -c_k - sum_x y_x A_{x,k}) + 10 * max(0, c_k - sum_x y_x A_{x,k})) ?
        # This is complex.
        
        # Let's try a heuristic: 
        # Set g(k) = -10 for small k, g(k) = 10 for large k.
        # Adjust to satisfy constraints.
        
        # Actually, the LP solution is likely to push g(k) to -10 for small k.
        # Let's try a "greedy" assignment.
        # Sort k by log(k)/k descending. Assign -10 to the first few, 10 to the rest?
        # Check constraints.
        
        def heuristic_solution(K):
            ks = np.arange(2, K + 1)
            g = np.zeros(K - 1)
            # Initialize with -10 for all?
            g[:] = -10.0
            
            # Check M(x)
            # M(x) = sum_{k=2}^K (-10) floor(x/k) - S * x
            # S = sum (-10)/k = -10 H_K
            # f(1) = -S = 10 H_K
            # If 10 H_K > 10, i.e., H_K > 1, then f(1) > 10, which is invalid (clipped).
            # H_K is ~ log K. So for K > e, H_K > 1.
            # So we cannot set all to -10.
            
            # We need to balance.
            # Let's try a linear ramp or something.
            
            # Let's try the LP for K=2000, 3000, 4000 with a more efficient dense construction 
            # if we can reduce memory.
            
            pass

    # Let's try to optimize the dense LP construction.
    # Instead of building the full matrix, can we use the structure?
    # No, linprog doesn't accept callbacks for matrix construction.
    
    # Let's try K=4000, 5000 with sparse matrix using a faster construction method.
    # We can use numpy broadcasting to generate the matrix in chunks.
    
    def solve_chunked_lp(K):
        if time.time() - start_time > 590:
            return None
            
        n_vars = K - 1
        ks = np.arange(2, K + 1)
        c = np.log(ks) / ks
        max_x = 10 * K
        
        # We will build the sparse matrix in chunks of rows.
        # But linprog needs the full matrix.
        # So we must build the full sparse matrix.
        
        # Let's try to build it as fast as possible.
        rows = np.empty(max_x * n_vars, dtype=np.int32) # Too big
        # Most entries are non-zero. Sparse might not be much better than dense in terms of storage if density is high.
        # floor(x/k) - x/k is 0 only if x is divisible by k.
        # Density of zeros is ~ 1/k. For small k, density is low. For large k, density is high.
        # Average density is ~ 1/log K?
        
        # Let's try K=5000.
        # 50000 rows, 4999 cols.
        
        # We can generate the matrix using a loop over k, but vectorized over x.
        # For each k, create a vector of length max_x.
        # v = np.floor(np.arange(1, max_x+1) / k) - np.arange(1, max_x+1) / k
        # This is O(max_x) per k. Total O(max_x * K).
        
        # This is the same complexity as dense.
        
        # Let's just try K=4000, 5000, 6000 with the dense LP and see if it fits in memory/time.
        # 60000 * 6000 * 8 = 2.8 GB. Might fail.
        
        # Let's try K=3000, 4000.
        
        for K_try in [3000, 4000, 5000]:
            if time.time() - start_time > 590:
                break
                
            n_vars = K_try - 1
            ks = np.arange(2, K_try + 1)
            c = np.log(ks) / ks
            max_x = 10 * K_try
            
            try:
                # Construct A_ub
                # A_ub[i, j] = floor((i+1)/(j+2)) - (i+1)/(j+2)
                
                # Memory check: max_x * n_vars
                if max_x * n_vars * 8 > 2 * 1024**3: # 2GB limit
                    continue
                    
                x_vals = np.arange(1, max_x + 1, dtype=np.float64)
                A_ub = np.floor(x_vals[:, None] / ks[None, :]) - x_vals[:, None] / ks[None, :]
                
                # Add f(1) constraint: sum g(k)/k <= 10 and -sum g(k)/k <= 10
                row_pos = 1.0 / ks
                row_neg = -1.0 / ks
                
                A_ub_full = np.vstack([A_ub, row_pos, row_neg])
                b_ub_full = np.concatenate([np.ones(max_x), [10.0, 10.0]])
                
                bounds = [(-10, 10)] * n_vars
                
                res = linprog(c, A_ub=A_ub_full, b_ub=b_ub_full, bounds=bounds, method='highs')
                
                if res.success:
                    g = res.x
                    f = np.zeros(K_try + 1)
                    f[2:K_try+1] = g
                    S = np.sum(g / ks)
                    f[1] = -S
                    
                    if abs(f[1]) <= 10.0 + 1e-9:
                        a = -np.sum(g * c)
                        if a > best_score:
                            best_score = a
                            domain = np.arange(1, K_try + 1)
                            values = np.clip(f[1:K_try+1], -10, 10)
                            best_domain = domain
                            best_values = values
                            
            except Exception as e:
                # print(f"K={K_try} failed: {e}")
                pass

    if best_domain is not None:
        return best_domain, best_values
    else:
        # Fallback to seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

# Re-import linprog as it was used in the snippet
from scipy.optimize import linprog
