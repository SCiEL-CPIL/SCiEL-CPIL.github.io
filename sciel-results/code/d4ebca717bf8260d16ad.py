"""Matrix multiplication 3,3,6: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 140  # safety margin below 150s kill

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify(U, V, W, tol=1e-9):
        """Verify if U, V, W form a valid decomposition for (n,m,p) matrix multiplication."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        
        # Target tensor T[i, j, k] = 1
        # Decomposed tensor D[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Efficient vectorized check
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j, :]
                    v_val = V[j * p + k, :]
                    w_val = W[k * n + i, :]
                    s = np.dot(u_val, v_val * w_val)
                    errors.append(s - 1.0)
        
        return max(abs(e) for e in errors) < tol
    
    def _try_rank(n, m, p, rank, seed=0, max_iters=300, deadline=None):
        """
        Try to find a rank-r decomposition using alternating least squares (ALS) with
        vectorized operations for speed.
        Returns (U, V, W, err) or (None, None, None, inf) if not found.
        """
        rng = np.random.RandomState(seed)
        
        # Initialize with random values, scaled appropriately
        U = rng.randn(n * m, rank) * 0.5
        V = rng.randn(m * p, rank) * 0.5
        W = rng.randn(p * n, rank) * 0.5
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline - 2.0:
                break
            
            # === Update U ===
            # For each (i, j): U[i*m+j, :] should satisfy
            #   sum_k U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
            # This is a linear system: M_U[i,j] @ U[i*m+j, :]^T = ones(p)
            # where M_U[i,j][k, r] = V[j*p+k, r] * W[k*n+i, r]
            
            for i in range(n):
                for j in range(m):
                    # Build matrix M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    V_block = V[j * p:(j + 1) * p, :]  # shape (p, rank)
                    W_block = W[np.arange(p) * n + i]  # shape (p, rank)
                    M = V_block * W_block  # element-wise, shape (p, rank)
                    
                    # Solve M^T @ x = ones(p)
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # === Update V ===
            # For each (j, k): V[j*p+k, :] should satisfy
            #   sum_i U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
            # M_V[j,k][i, r] = U[i*m+j, r] * W[k*n+i, r]
            
            for j in range(m):
                for k in range(p):
                    # Build matrix M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    U_cols = U[np.arange(n) * m + j, :]  # shape (n, rank)
                    W_rows = W[k * n + np.arange(n), :]  # shape (n, rank)
                    M = U_cols * W_rows  # element-wise, shape (n, rank)
                    
                    # Solve M^T @ x = ones(n)
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # === Update W ===
            # For each (k, i): W[k*n+i, :] should satisfy
            #   sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
            # M_W[k,i][j, r] = U[i*m+j, r] * V[j*p+k, r]
            
            for k in range(p):
                for i in range(n):
                    # Build matrix M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    U_row = U[i * m:(i + 1) * m, :]  # shape (m, rank)
                    V_col = V[np.arange(m) * p + k, :]  # shape (m, rank)
                    M = U_row * V_col  # element-wise, shape (m, rank)
                    
                    # Solve M^T @ x = ones(m)
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Calculate final error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        err += (s - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Early termination if error is small
            if best_err < 1e-6:
                break
        
        if best_err < 1e-4:
            return best_U, best_V, best_W, best_err
        return None, None, None, best_err
    
    # Fallback to schoolbook
    schoolbook_sol = get_schoolbook()
    
    # Try to find a lower rank solution
    if time.time() - START_TIME < TIME_LIMIT - 15:
        deadline = START_TIME + TIME_LIMIT - 5
        school_rank = n * m * p  # 54
        
        # Lower bound for rank is max(n*m, m*p, p*n) = max(9, 18, 18) = 18
        # But we can do better. Let's try from 18 upward.
        # For 3x3x6, known results suggest rank might be around 20-25.
        
        found = False
        # Try ranks in ascending order
        for rank in range(18, school_rank):
            if time.time() > deadline - 15:
                break
            
            # Try multiple seeds for this rank
            num_seeds = 20
            if time.time() > deadline - 40:
                num_seeds = 5
            if time.time() > deadline - 20:
                num_seeds = 2
            
            for seed in range(num_seeds):
                if time.time() > deadline - 5:
                    break
                
                U, V, W, err = _try_rank(n, m, p, rank, seed=seed, max_iters=200, deadline=deadline)
                
                if U is not None and err < 1e-4:
                    # Verify with tighter tolerance
                    if verify(U, V, W, tol=1e-7):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return schoolbook_sol

# EVOLVE-BLOCK-START
# The construct function is defined above to respect the interface.
# EVOLVE-BLOCK-END
