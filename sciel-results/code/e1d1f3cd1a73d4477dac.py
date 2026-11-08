"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _snap_to_half_int(arr):
    """Snap numpy array values to the nearest half-integer."""
    return np.round(arr * 2) / 2.0

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly satisfies the tensor equation."""
    # For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(len(U[0])):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(rank, n, m, p, max_iters=100, n_restarts=10, timeout=None, start_time=None):
    """Try to find a rank-r decomposition using ALS."""
    if start_time is None:
        start_time = time.time()
    
    for restart in range(n_restarts):
        if timeout is not None and time.time() - start_time > timeout:
            break
        
        # Random initialization
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        for it in range(max_iters):
            if timeout is not None and time.time() - start_time > timeout:
                break
            
            # Update U: for each (i,j), the equation is:
            # sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
            # Let A[i*m+j, k, r] = V[j*p+k, r] * W[k*n+i, r]
            # Then we need: sum_r U[i*m+j,r] * A[i*m+j, k, r] = 1 for all k
            # This is a linear system: U[i*m+j, :] * A[i*m+j, :, :]^T = 1 (vector)
            
            U_new = U.copy()
            for i in range(n):
                for j in range(m):
                    # Build the linear system
                    # For each k, we have: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                    # This is: U[i*m+j, :] @ (V[j*p+k, :] * W[k*n+i, :])^T = 1
                    # Stack for all k:
                    # M[i*m+j, k, r] = V[j*p+k, r] * W[k*n+i, r]
                    # We want U[i*m+j, :] @ M[i*m+j, :, :]^T = ones(p)
                    # So U[i*m+j, :] = ones(p) @ pinv(M[i*m+j, :, :]^T)
                    # But M is p x rank, so M^T is rank x p
                    # We want to solve: X @ M^T = 1 (where X is 1 x rank, 1 is 1 x p)
                    # X = 1 @ pinv(M^T)
                    
                    M = np.zeros((p, rank))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve X @ M^T = ones(p)
                    # X = ones(p) @ pinv(M^T)
                    try:
                        X = np.ones(p) @ np.linalg.pinv(M.T)
                        U_new[i*m+j, :] = X
                    except:
                        pass
            
            U = U_new
            
            # Update V: for each (j,k), sum_r V[j*p+k,r] * U[i*m+j,r] * W[k*n+i,r] = 1 for all i
            V_new = V.copy()
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, rank))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        X = np.ones(n) @ np.linalg.pinv(M.T)
                        V_new[j*p+k, :] = X
                    except:
                        pass
            V = V_new
            
            # Update W: for each (k,i), sum_r W[k*n+i,r] * U[i*m+j,r] * V[j*p+k,r] = 1 for all j
            W_new = W.copy()
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, rank))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        X = np.ones(m) @ np.linalg.pinv(M.T)
                        W_new[k*n+i, :] = X
                    except:
                        pass
            W = W_new
            
            # Check convergence
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        max_err = max(max_err, abs(val - 1.0))
            
            # Try snapping if the error is small enough that snapping might fix it
            if max_err < 0.5:
                U_snapped = _snap_to_half_int(U)
                V_snapped = _snap_to_half_int(V)
                W_snapped = _snap_to_half_int(W)
                
                U_list = U_snapped.tolist()
                V_list = V_snapped.tolist()
                W_list = W_snapped.tolist()
                
                if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110  # safety margin
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try ranks from lower bound up
    # Lower bound: max(n*m, m*p, p*n) = max(16, 24, 24) = 24
    for rank in range(24, rank_school + 1):
        if time.time() - start_time > deadline:
            break
        
        result = _try_rank(rank, n, m, p, max_iters=50, n_restarts=20, 
                          timeout=deadline, start_time=start_time)
        if result is not None:
            best_result = result
            best_rank = rank
            break  # Found a valid decomposition, stop
    
    return best_result
# EVOLVE-BLOCK-END
