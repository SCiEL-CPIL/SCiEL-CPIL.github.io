"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    best_rank = n * m * p
    best_result = None
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    sb = schoolbook()
    best_result = sb
    best_rank = n * m * p
    
    # Try to find lower rank via ALS on the matrix multiplication tensor
    # The tensor T is a 3D array with T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We want to find U (n*m x r), V (m*p x r), W (p*n x r) such that
    # for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Reshape approach: the tensor T[i,j,k] = 1 can be viewed as a 3-mode tensor
    # U mode: index i*m+j -> size n*m
    # V mode: index j*p+k -> size m*p  
    # W mode: index k*n+i -> size p*n
    # T[i,j,k] = 1
    
    # For a rank-r decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This is the standard matrix multiplication tensor.
    
    # Use ALS: fix two matrices, solve for third
    def try_als(r, max_iters=200, tol=1e-6):
        nonlocal deadline
        if time.time() > deadline:
            return None
        
        rng = np.random.RandomState(42 + r)
        # Initialize
        U = rng.randn(n * m, r) * 0.1
        V = rng.randn(m * p, r) * 0.1
        W = rng.randn(p * n, r) * 0.1
        
        # Precompute the tensor structure
        # T[i,j,k] = 1 for all valid i,j,k
        # We need: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U: for each (i,j), U[i*m+j,:] should satisfy
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
            # This is: U[i*m+j,:] * (V[j*p+0:m*p step p] outer W[:, i] ...) 
            # Actually, for fixed i,j: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
            # Let M_k_c = V[j*p+k,c] * W[k*n+i,c] for k in range(p)
            # Then U[i*m+j,:] dot M_k for each k = 1
            # So U[i*m+j,:] * M^T = ones(1,p) where M is p x r
            
            U_new = np.zeros((n * m, r))
            V_new = np.zeros((m * p, r))
            W_new = np.zeros((p * n, r))
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # Build matrix B: B[k,c] = V[j*p+k,c] * W[k*n+i,c]
                    B = np.zeros((p, r))
                    for k in range(p):
                        B[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve U[i*m+j,:] * B^T = ones(1,p)
                    # i.e., B @ U[i*m+j,:]^T = ones(p,1)
                    try:
                        sol = np.linalg.lstsq(B, np.ones(p), rcond=None)[0]
                        U_new[i * m + j, :] = sol
                    except:
                        U_new[i * m + j, :] = U[i * m + j, :]
            U = U_new
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # For fixed j,k: sum_c V[j*p+k,c] * U[i*m+j,c] * W[k*n+i,c] = 1 for all i
                    # B[i,c] = U[i*m+j,c] * W[k*n+i,c]
                    B = np.zeros((n, r))
                    for i in range(n):
                        B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        sol = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                        V_new[j * p + k, :] = sol
                    except:
                        V_new[j * p + k, :] = V[j * p + k, :]
            V = V_new
            
            # Update W
            for i in range(n):
                for k in range(p):
                    # For fixed i,k: sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1 for all j
                    # B[j,c] = U[i*m+j,c] * V[j*p+k,c]
                    B = np.zeros((m, r))
                    for j in range(m):
                        B[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        sol = np.linalg.lstsq(B, np.ones(m), rcond=None)[0]
                        W_new[k * n + i, :] = sol
                    except:
                        W_new[k * n + i, :] = W[k * n + i, :]
            W = W_new
            
            # Check error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        err += (val - 1.0) ** 2
            if err < tol:
                break
        
        # Final error check
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    err += (val - 1.0) ** 2
        
        if err < 1e-4:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist(), "rank": r}
        return None
    
    # Try decreasing rank from best_rank-1 down
    for r in range(best_rank - 1, max(40, best_rank - 20), -1):
        if time.time() > deadline:
            break
        result = try_als(r)
        if result is not None:
            best_rank = r
            best_result = {k: v for k, v in result.items() if k != "rank"}
            # Continue trying lower
        else:
            # If this rank fails, lower ranks likely also fail (but not necessarily)
            # Try a couple more then stop
            if best_rank - r > 5:
                break
    
    return best_result
# EVOLVE-BLOCK-END
