"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _to_lists(U, V, W):
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # The decomposition must satisfy:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] == W[k*n+i, c]  ... wait no.
    # Actually the tensor T[i,j,k] = delta (it's 1 for all valid i,j,k in standard matmul tensor)
    # The factorization is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # But here U, V, W are 2D matrices. Let me re-read the interface.
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k (since (AB)_{ik} = sum_j A_{ij}B_{jk})
    # The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, that's a 3-way decomposition where each factor has one index pair.
    # Actually, looking at the schoolbook code:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for c = i*m*p + j*p + k
    # So T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # In schoolbook, for each (i,j,k), there's exactly one c where all three are 1, so T[i,j,k]=1.
    # Let me verify this interpretation.
    
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j, :]
                v_val = V[j * p + k, :]
                w_val = W[k * n + i, :]
                t_val = np.sum(u_val * v_val * w_val)
                if abs(t_val - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook()
    best_result = _to_lists(U, V, W)
    best_rank = U.shape[1]
    
    # Try to find a lower rank decomposition using numerical optimization
    # Strategy: random initialization + gradient-free optimization (Nelder-Mead style)
    # on a relaxed problem, then check for exactness.
    
    # For small problems like 2x3x7, we can try random rank-r decompositions
    # and use least squares to solve for the factors.
    
    # The tensor T[i,j,k] = 1 for all i,j,k in [0,n)x[0,m)x[0,p)
    # We want rank r such that T[i,j,k] = sum_{c=0}^{r-1} U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # This is a 3-tensor decomposition. We can use alternating least squares (ALS).
    
    def _als_decompose(n, m, p, r, n_iter=50, seed=42):
        """Alternating least squares for 3-tensor decomposition."""
        rng = np.random.default_rng(seed)
        
        # Initialize
        U = rng.standard_normal((n * m, r))
        V = rng.standard_normal((m * p, r))
        W = rng.standard_normal((p * n, r))
        
        for it in range(n_iter):
            # Update U: for each (i,j), solve for U[i*m+j, :] 
            # T[i,j,k] = sum_c U[i,j,c] * (V[j,k,:] * W[k,i,:])_c
            # Let M[i,j,k] = V[j*p+k,:] * W[k*n+i,:] (element-wise product, shape (r,))
            # Then T[i,j,:] = U[i,j,:] @ M[i,j,:]^T where M[i,j,:] is (p, r)
            # Actually: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # = U[i*m+j,:] dot (V[j*p+k,:] * W[k*n+i,:])
            # So for fixed (i,j): T[i,j,:] = U[i*m+j,:] @ (V_j * W_i)^T
            # where (V_j * W_i) is a (p, r) matrix with entry (k,c) = V[j*p+k,c]*W[k*n+i,c]
            
            for i in range(n):
                for j in range(m):
                    # Build the (p, r) matrix
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve: T[i,j,:] = U[i*m+j,:] @ M^T
                    # T[i,j,:] is shape (p,), M^T is (r, p)
                    # So U[i*m+j,:] = T[i,j,:] @ pinv(M^T)
                    # Or: U[i*m+j,:] @ M^T = T[i,j,:]
                    # Let x = U[i*m+j,:], then x @ M^T = T[i,j,:]
                    # M @ x^T = T[i,j,:]^T
                    # Solve M x^T = t where t = T[i,j,:]^T
                    t = np.ones(p)  # T[i,j,k] = 1 for all k
                    try:
                        x, _, _, _ = np.linalg.lstsq(M, t, rcond=None)
                        U[i * m + j, :] = x
                    except:
                        pass
            
            # Update V: for each (j,k), solve for V[j*p+k, :]
            # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # For fixed (j,k): T[:,j,k] = (U_i * W_k)^T @ V[j*p+k,:]^T where
            # (U_i * W_k) is (n, r) with entry (i,c) = U[i*m+j,c]*W[k*n+i,c]
            # So T[:,j,k] = (U_i * W_k) @ V[j*p+k,:]^T
            # Let A = (U_i * W_k) shape (n, r), then A @ v^T = T[:,j,k]
            # A @ v = T[:,j,k]^T where v = V[j*p+k,:]^T
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    t = np.ones(n)  # T[i,j,k] = 1 for all i
                    try:
                        v, _, _, _ = np.linalg.lstsq(A, t, rcond=None)
                        V[j * p + k, :] = v
                    except:
                        pass
            
            # Update W: for each (k,i), solve for W[k*n+i, :]
            # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # For fixed (k,i): T[i,:,k] = (U_i * V_k)^T @ W[k*n+i,:]^T where
            # (U_i * V_k) is (m, r) with entry (j,c) = U[i*m+j,c]*V[j*p+k,c]
            # So T[i,:,k] = (U_i * V_k) @ W[k*n+i,:]^T
            # Let A = (U_i * V_k) shape (m, r), then A @ w^T = T[i,:,k]
            # A @ w = T[i,:,k]^T where w = W[k*n+i,:]^T
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    t = np.ones(m)  # T[i,j,k] = 1 for all j
                    try:
                        w, _, _, _ = np.linalg.lstsq(A, t, rcond=None)
                        W[k * n + i, :] = w
                    except:
                        pass
            
            # Check convergence
            if it % 10 == 0:
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            t_val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            err += (t_val - 1.0) ** 2
                if err < 1e-6:
                    break
        
        return U, V, W, np.sqrt(err / (n * m * p))
    
    # Try ranks from lower to upper
    n, m, p = DIMENSIONS
    # Lower bound: max(n*m, m*p, p*n) / something... 
    # Actually for all-ones tensor, the rank is at least max(n, m, p) but that's for different tensors.
    # For the matrix multiplication tensor, the rank is known to be at most n*m*p and at least max(n*m, m*p, p*n).
    # Wait, that's not right either. The tensor T[i,j,k]=1 has a specific rank.
    # Let's just try various ranks.
    
    # The all-ones tensor of size n x m x p has rank 1 if we allow arbitrary factors,
    # but here the factors are constrained by the structure (U depends on (i,j), V on (j,k), W on (k,i)).
    # This is the Strassen-type problem.
    
    # Let's try ranks from some lower bound up to schoolbook rank
    lower_bound = max(n, m, p)  # very loose lower bound
    upper_bound = n * m * p
    
    # Try a few ranks
    for r in range(lower_bound, min(upper_bound + 1, 50)):
        if time.time() > deadline - 20:
            break
        
        found = False
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            
            try:
                U_r, V_r, W_r, err = _als_decompose(n, m, p, r, n_iter=30, seed=seed + 100)
                
                if err < 1e-8:
                    # Verify exactly
                    if _verify(U_r, V_r, W_r, tol=1e-9):
                        if r < best_rank:
                            best_rank = r
                            best_result = _to_lists(U_r, V_r, W_r)
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            # Found a valid decomposition at this rank, try lower
            continue
    
    return best_result
# EVOLVE-BLOCK-END
