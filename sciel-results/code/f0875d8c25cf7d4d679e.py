import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Target Tensor T
    # T is a 3-tensor with dimensions (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # We want to find rank r < rank_sb
    # Using integer search for 2*U, 2*V, 2*W
    # Let U_int, V_int, W_int be integers.
    # T = sum (U_int_c / 2) (V_int_c / 2) (W_int_c / 2)
    # 8 T = sum U_int_c V_int_c W_int_c
    
    # We search for integer U_int, V_int, W_int
    # such that sum U_int[:,c] V_int[:,c] W_int[:,c] = 8 * T
    
    # Construct 8*T as a vector
    # Indices: a in [0, n*m), b in [0, m*p), c in [0, p*n)
    # Non-zero at a=i*m+j, b=j*p+k, c=k*n+i
    dim_a = n * m
    dim_b = m * p
    dim_c = p * n
    total_dim = dim_a * dim_b * dim_c
    
    target_vec = np.zeros(total_dim, dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a = i * m + j
                idx_b = j * p + k
                idx_c = k * n + i
                flat_idx = idx_a * dim_b * dim_c + idx_b * dim_c + idx_c
                target_vec[flat_idx] = 8  # Because 8*T = 8*1 = 8

    start_time = time.time()
    timeout = 110  # Safety margin
    
    # Try ranks from rank_sb - 1 down to rank_sb - 6
    for r in range(rank_sb - 1, max(rank_sb - 10, 20), -1):
        if time.time() - start_time > timeout:
            break
            
        # Number of trials
        trials = 500 if r > 30 else 200
        
        for trial in range(trials):
            if time.time() - start_time > timeout:
                break
                
            # Random integer matrices U_int, V_int
            # Entries in {-2, -1, 0, 1, 2}
            U_int = np.random.randint(-2, 3, size=(dim_a, r))
            V_int = np.random.randint(-2, 3, size=(dim_b, r))
            
            # Construct A matrix for W_int
            # A is (total_dim, dim_c * r)
            # A[flat_idx, c_idx * dim_c + w_idx] = U_int[a, c_idx] * V_int[b, c_idx]
            # This is expensive to build fully.
            # We can solve the system using least squares or check consistency.
            
            # Instead of building A, we can iterate over the 36 non-zero entries of T.
            # For each non-zero entry (a, b, c), the equation is:
            # sum_{col=0}^{r-1} U_int[a, col] * V_int[b, col] * W_int[c, col] = 8
            # This is a system of 36 equations with dim_c * r unknowns?
            # No, W_int has dim_c * r unknowns.
            # The equations only constrain specific linear combinations.
            # For a fixed col, the term is U_int[a, col] V_int[b, col] W_int[c, col].
            # This is not linear in W_int because U and V are fixed but W is variable?
            # Yes, for fixed U, V, it is linear in W.
            # Coefficient of W_int[c, col] in equation (a,b) is U_int[a, col] * V_int[b, col].
            
            # We have 36 equations (for the 36 non-zero entries of T).
            # Unknowns: W_int is dim_c x r. Total dim_c * r unknowns.
            # For r=35, dim_c=12, unknowns = 420.
            # Equations = 36.
            # Under-determined system.
            # We need a solution where W_int entries are integers.
            
            # We can set up the system:
            # M w = b
            # M is 36 x (dim_c * r)
            # b is 36 x 1 (all 8s)
            
            # Construct M
            # Row for each (i,j,k) corresponding to non-zero T entry.
            # Col for each (w_idx, col) corresponding to W_int[w_idx, col].
            
            M = np.zeros((36, dim_c * r), dtype=int)
            b = np.ones(36, dtype=int) * 8
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_a = i * m + j
                        idx_b = j * p + k
                        idx_c = k * n + i
                        
                        # For each col in 0..r-1
                        for col in range(r):
                            coeff = U_int[idx_a, col] * V_int[idx_b, col]
                            # This coeff multiplies W_int[idx_c, col]
                            # Index in w vector: col * dim_c + idx_c
                            M[eq_idx, col * dim_c + idx_c] = coeff
                        
                        eq_idx += 1
            
            # Solve M w = b
            # Since under-determined, we can use least squares or find a particular solution.
            # But we need integer solution.
            # We can use scipy.linalg.lstsq to get a real solution, then check if it's close to integer.
            # Or use a solver for integer linear programming? Too slow.
            # Given the structure, maybe we can just check if a random integer W works?
            # No, W is determined by U, V if the system is consistent.
            
            # Let's try to find a solution w.
            try:
                # Use pseudo-inverse
                w, residuals, rank, sv = np.linalg.lstsq(M, b, rcond=None)
                # Check if residuals are small
                if residuals[0] < 1e-5:
                    # Check if w is close to integers
                    w_int = np.round(w).astype(int)
                    # Verify exactness
                    if np.allclose(M @ w_int, b):
                        # Check if w_int corresponds to valid W
                        # w_int is flattened W_int
                        W_int = w_int.reshape(r, dim_c).T # Note: w is (dim_c*r), reshape to (r, dim_c) then T?
                        # My indexing: col * dim_c + idx_c.
                        # So w[col * dim_c + idx_c] is W_int[idx_c, col].
                        # Reshape to (r, dim_c) gives W_int.T?
                        # Let's check: w[0] is col=0, idx_c=0.
                        # w.reshape(r, dim_c) -> row 0 is col 0.
                        # So W_int.T = w.reshape(r, dim_c).
                        # W_int = w.reshape(r, dim_c).T
                        
                        # We found a valid decomposition!
                        U_ret = (U_int / 2).tolist()
                        V_ret = (V_int / 2).tolist()
                        W_ret = (W_int / 2).tolist()
                        
                        return {"U": U_ret, "V": V_ret, "W": W_ret}
            except Exception:
                pass

    # Return schoolbook if no better found
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
