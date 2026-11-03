"""Matrix multiplication 3,4,4 - rank minimization via tensor search."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0

def _build_tensor():
    """Build the 3x4x4 matrix multiplication tensor T[i,j,k] = 1 if i,j,k valid."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    return T

def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly reconstructs the tensor."""
    n, m, p = DIMENSIONS
    T = _build_tensor()
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    rank = U.shape[1]
    # Reconstruct: for each (i,j,k), sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    T_recon = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T_recon[i, j, k] = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
    return np.allclose(T, T_recon, atol=1e-9, rtol=1e-9)

def _rank_of_matrix(M):
    """Compute the numerical rank of a matrix."""
    if M.size == 0:
        return 0
    return np.linalg.matrix_rank(M, tol=1e-10)

def _als_step(T, U, V, W, max_iter=100, tol=1e-8):
    """Alternating least squares to refine a rank-r decomposition."""
    n, m, p = DIMENSIONS
    r = U.shape[1]
    
    for _ in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        old_U = U.copy()
        old_V = V.copy()
        old_W = W.copy()
        
        # Update U: for each (i,j), solve (V^T ⊙ W^T) x = T[i,j,:]
        # V^T ⊙ W^T is r x (m*p) ... actually let's think carefully
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # For fixed (i,j): sum_k T[i,j,k] * W[k,i,r] = sum_r U[i,j,r] * sum_k V[j,k,r] * W[k,i,r]
        # Hmm, this isn't quite right. Let me re-derive.
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Fix i,j. For all k: T[i,j,k] = sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r])
        # So we have a linear system: for each k, T[i,j,k] = sum_r U[i*m+j, r] * B_{k,r}
        # where B_{k,r} = V[j*p+k, r] * W[k*n+i, r]
        # This is: T[i,j,:] = U[i*m+j, :] @ B^T  ... no
        # T[i,j,k] = sum_r U[i*m+j, r] * B[k, r]
        # So T[i,j,:] = U[i*m+j, :] @ B.T  where B is (p, r)
        # We need: U[i*m+j, :] such that U[i*m+j, :] @ B.T = T[i,j,:]
        # i.e., (B @ U[i*m+j, :].T) = T[i,j,:].T
        # So U[i*m+j, :].T = B^+ @ T[i,j,:].T  (pseudo-inverse)
        
        for i in range(n):
            for j in range(m):
                B = np.zeros((p, r))
                for k in range(p):
                    B[k, :] = V[j * p + k, :] * W[k * n + i, :]
                rhs = T[i, j, :]
                # Solve B @ x = rhs for x (r-dim)
                try:
                    U[i * m + j, :] = np.linalg.lstsq(B, rhs, rcond=None)[0]
                except:
                    pass
        
        # Update V: for each (j,k), T[i,j,k] = sum_r V[j*p+k, r] * (U[i*m+j, r] * W[k*n+i, r])
        # Fix j,k. For all i: T[i,j,k] = sum_r V[j*p+k, r] * C_{i,r}
        # where C_{i,r} = U[i*m+j, r] * W[k*n+i, r]
        for j in range(m):
            for k in range(p):
                C = np.zeros((n, r))
                for i in range(n):
                    C[i, :] = U[i * m + j, :] * W[k * n + i, :]
                rhs = np.array([T[i, j, k] for i in range(n)])
                try:
                    V[j * p + k, :] = np.linalg.lstsq(C, rhs, rcond=None)[0]
                except:
                    pass
        
        # Update W: for each (k,i), T[i,j,k] = sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r])
        # Fix k,i. For all j: T[i,j,k] = sum_r W[k*n+i, r] * D_{j,r}
        # where D_{j,r} = U[i*m+j, r] * V[j*p+k, r]
        for k in range(p):
            for i in range(n):
                D = np.zeros((m, r))
                for j in range(m):
                    D[j, :] = U[i * m + j, :] * V[j * p + k, :]
                rhs = np.array([T[i, j, k] for j in range(m)])
                try:
                    W[k * n + i, :] = np.linalg.lstsq(D, rhs, rcond=None)[0]
                except:
                    pass
        
        # Check convergence
        diff = (np.linalg.norm(U - old_U) + np.linalg.norm(V - old_V) + np.linalg.norm(W - old_W))
        if diff < tol:
            break
    
    return U, V, W

def _try_rank(r, n_restarts=3, seed_base=0):
    """Try to find a valid rank-r decomposition."""
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return None
    
    n, m, p = DIMENSIONS
    T = _build_tensor()
    
    for restart in range(n_restarts):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return None
        rng = np.random.RandomState(seed_base + restart)
        
        U = rng.randn(n * m, r) * 0.1
        V = rng.randn(m * p, r) * 0.1
        W = rng.randn(p * n, r) * 0.1
        
        # ALS refinement
        U, V, W = _als_step(T, U, V, W, max_iter=50)
        
        # Verify
        if _verify_decomposition(U, V, W):
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_rank_fixed_pattern(r):
    """Try a structured rank-r decomposition using known patterns."""
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return None
    
    n, m, p = DIMENSIONS
    T = _build_tensor()
    
    # Try using the fact that we can share some terms
    # Schoolbook is 48. Can we do better?
    # One approach: group by i. For each i, we have a 4x4 sub-problem.
    # But the k dimension couples across i via W[k*n+i, r].
    
    # Let's try a different structured approach:
    # For each (i,k), we need to compute sum_j A[i,j]*B[j,k]
    # This is a dot product. We can use rank-1 updates.
    # But sharing across i for the same (j,k) or across k for the same (i,j) might help.
    
    # Actually, the tensor T[i,j,k]=1 for all valid indices is very symmetric.
    # The minimum rank is at least max(n*m, m*p, p*n) = max(12, 16, 12) = 16.
    # But the true minimum rank for 3x4x4 multiplication is unknown to me.
    # Let's try to find something better than 48.
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x4x4 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # First, always have the schoolbook as fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Try to find better decompositions
    # Lower bound: max(n*m, m*p, p*n) = 16
    # Try ranks from 16 upward
    for r in range(16, best_rank + 1):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        result = _try_rank(r, n_restarts=2)
        if result is not None:
            best = result
            best_rank = r
            break  # Found minimum rank
    
    return best
