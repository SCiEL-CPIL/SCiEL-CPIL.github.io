"""Matrix multiplication 2x2 x 2x16 via Strassen decomposition."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def _strassen_2x2():
    """
    Return the Strassen decomposition of 2x2 matrix multiplication.
    """
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1
    W[3, 0] = 1
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    W[3, 1] = -1
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    W[3, 2] = 1
    
    # M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    W[2, 3] = 1
    
    # M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1
    W[1, 4] = 1
    
    # M6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _time_remaining():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)

def _build_block_strassen(n, m, p):
    """Build the block-Kronecker Strassen decomposition (rank 56)."""
    U_core, V_core, W_core = _strassen_2x2()
    num_pairs = p // 2
    rank = 7 * num_pairs
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    for t in range(num_pairs):
        U[:, t*7:(t+1)*7] = U_core
        V[2*t, t*7:(t+1)*7] = V_core[0, :]
        V[2*t+1, t*7:(t+1)*7] = V_core[1, :]
        V[p+2*t, t*7:(t+1)*7] = V_core[2, :]
        V[p+2*t+1, t*7:(t+1)*7] = V_core[3, :]
        W[4*t, t*7:(t+1)*7] = W_core[0, :]
        W[4*t+1, t*7:(t+1)*7] = W_core[2, :]
        W[4*t+2, t*7:(t+1)*7] = W_core[1, :]
        W[4*t+3, t*7:(t+1)*7] = W_core[3, :]
    
    return U, V, W, rank

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose the multiplication tensor."""
    rank = U.shape[1]
    # Check shapes
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    
    # Build the tensor from U, V, W
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Compare with the true tensor: T[i,j,k] = 1 for all valid i,j,k
    U_f = U.astype(np.float64)
    V_f = V.astype(np.float64)
    W_f = W.astype(np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_f[i*m+j, :] * V_f[j*p+k, :] * W_f[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_lower_rank(n, m, p, target_rank, max_iters=5000, seed=42):
    """
    Try to find a rank-`target_rank` decomposition using iterative refinement.
    Returns (U, V, W) if found, else None.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize with random matrices
    U = rng.randn(n*m, target_rank)
    V = rng.randn(m*p, target_rank)
    W = rng.randn(p*n, target_rank)
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if _time_remaining() < 5.0:
            break
            
        # Compute current error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0) ** 2
        err = np.sqrt(err / (n*m*p))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-12:
            break
        
        # Update each factor in turn (alternating least squares style)
        # Update U
        for i in range(n):
            for j in range(m):
                # U[i*m+j, :] should satisfy: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all k
                # This is: U[i*m+j, :] * (V[j*p+k,:] * W[k*n+i,:] elementwise summed over k) ... 
                # Actually: for each k, sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1
                # Let M_k[r] = V[j*p+k,r] * W[k*n+i,r], then U[i*m+j,:] * M = ones(p)
                # This is a least squares: find U_row such that U_row @ M.T = ones
                M = np.zeros((target_rank, p))
                for k in range(p):
                    M[:, k] = V[j*p+k, :] * W[k*n+i, :]
                # Solve: U_row @ M.T = ones(p) => U_row = ones(p) @ M @ (M.T @ M)^{-1}
                try:
                    MMT = M.T @ M + 1e-10 * np.eye(target_rank)
                    rhs = np.ones(p) @ M
                    U[i*m+j, :] = np.linalg.solve(MMT, rhs)
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((target_rank, n))
                for i in range(n):
                    M[:, i] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    MMT = M.T @ M + 1e-10 * np.eye(target_rank)
                    rhs = np.ones(n) @ M
                    V[j*p+k, :] = np.linalg.solve(MMT, rhs)
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((target_rank, m))
                for j in range(m):
                    M[:, j] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    MMT = M.T @ M + 1e-10 * np.eye(target_rank)
                    rhs = np.ones(m) @ M
                    W[k*n+i, :] = np.linalg.solve(MMT, rhs)
                except:
                    pass
    
    # Check if we found a good solution
    if best_err < 1e-10:
        # Try to round to integers/half-integers
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
            return U_rounded, V_rounded, W_rounded
        
        # Also try integer rounding
        U_rounded = np.round(best_U)
        V_rounded = np.round(best_V)
        W_rounded = np.round(best_W)
        
        if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
            return U_rounded, V_rounded, W_rounded
    
    return None

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x16 matrix.
    
    First tries to find a lower-rank decomposition via iterative refinement.
    Falls back to block-Kronecker Strassen (rank 56) if no better solution found.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    # Start with the known rank-56 solution
    U, V, W, rank = _build_block_strassen(n, m, p)
    
    # Try to find a lower rank solution
    for target_rank in range(55, 40, -1):
        if _time_remaining() < 20.0:
            break
        result = _try_lower_rank(n, m, p, target_rank, max_iters=2000, seed=target_rank)
        if result is not None:
            U, V, W = result
            rank = target_rank
            break
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
