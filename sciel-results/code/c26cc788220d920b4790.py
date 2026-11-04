"""Program for rank-minimal exact tensor decomposition of 2x4x4 matrix multiplication."""
import numpy as np
import random
import time

DIMENSIONS = [2, 4, 4]


def _build_tensor():
    """Build the multiplication tensor T[i, j, k] = 1 for all (i,j,k)."""
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p), dtype=np.float64)
    return T


def _tensor_to_matrix(T, axis0=0, axis1=1, axis2=2):
    """Reshape tensor into a matrix with axis0 as rows and (axis1, axis2) as columns."""
    n, m, p = DIMENSIONS
    # Move axes so that axis0 is first, then the other two
    perm = [axis0, axis1, axis2]
    T_perm = np.transpose(T, perm)
    return T_perm.reshape(T.shape[axis0], -1)


def _matrices_to_tensor(U, V, W, n, m, p):
    """Reconstruct tensor from factor matrices."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    r = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(r):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        # Tensor product contribution: T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    return T


def _verify(U, V, W, tol=1e-9):
    """Check if the factorization exactly decomposes the tensor."""
    n, m, p = DIMENSIONS
    T = _build_tensor()
    T_rec = _matrices_to_tensor(U, V, W, n, m, p)
    return np.allclose(T, T_rec, atol=tol, rtol=0)


def _round_to_half(U, V, W):
    """Round entries to nearest half-integer."""
    U = np.round(np.array(U) * 2) / 2
    V = np.round(np.array(V) * 2) / 2
    W = np.round(np.array(W) * 2) / 2
    return U, V, W


def _schoolbook(n, m, p):
    """Schoolbook decomposition."""
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
    return U, V, W


def _try_rank_reduction(U, V, W, target_rank, n, m, p, max_iters=5000, seed=42):
    """
    Try to reduce rank to target_rank by optimizing in a low-rank subspace.
    Uses iterative projection / gradient-free approach.
    """
    rng = np.random.default_rng(seed)
    current_rank = U.shape[1]
    
    # Start with a random projection from current_rank to target_rank
    P = rng.standard_normal((current_rank, target_rank))
    P = P / np.linalg.norm(P, axis=0, keepdims=True)
    
    U_curr = U @ P
    V_curr = V @ P
    W_curr = W @ P
    
    # Verify
    if _verify(U_curr, V_curr, W_curr, n, m, p):
        return U_curr, V_curr, W_curr
    
    # Iterative refinement: try to fix errors by adjusting factors
    # Use a simple iterative method: for each factor, try to reduce error
    best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
    best_err = np.inf
    
    T = _build_tensor()
    
    for it in range(max_iters):
        T_rec = _matrices_to_tensor(best_U, best_V, best_W, n, m, p)
        err = np.linalg.norm(T - T_rec)
        if err < best_err:
            best_err = err
            U_curr, V_curr, W_curr = best_U.copy(), best_V.copy(), best_W.copy()
        
        if err < 1e-10:
            break
        
        # Try updating one factor at a time using least squares
        # Update U: fix V and W, solve for U
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For fixed V, W: this is linear in U
        # Build system: for each (i,j), T[i,j,:] = U[i*m+j, :] * (V[j,:] * W[:,i])^T ... 
        # Actually: T[i,j,k] = sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c])
        # So for each (i,j): T[i,j,:] = U[i*m+j,:] @ M_{ij} where M_{ij}[k,c] = V[j*p+k,c]*W[k*n+i,c]
        
        # Update U
        for i in range(n):
            for j in range(m):
                # M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                M = np.zeros((p, target_rank))
                for k in range(p):
                    M[k, :] = V_curr[j * p + k, :] * W_curr[k * n + i, :]
                # T[i,j,:] = U[i*m+j,:] @ M
                # Solve: U[i*m+j,:] @ M = T[i,j,:]
                if target_rank <= p:
                    # Underdetermined or square
                    try:
                        U_row, _, _, _ = np.linalg.lstsq(M.T, T[i, j, :], rcond=None)
                        best_U[i * m + j, :] = U_row
                    except:
                        pass
                else:
                    try:
                        U_row, _, _, _ = np.linalg.lstsq(M.T, T[i, j, :], rcond=None)
                        best_U[i * m + j, :] = U_row
                    except:
                        pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                # M[i, c] = U[i*m+j, c] * W[k*n+i, c]
                M = np.zeros((n, target_rank))
                for i in range(n):
                    M[i, :] = best_U[i * m + j, :] * W_curr[k * n + i, :]
                # T[:,j,k] = V[j*p+k,:] @ M
                try:
                    V_row, _, _, _ = np.linalg.lstsq(M.T, T[:, j, k], rcond=None)
                    best_V[j * p + k, :] = V_row
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                # M[j, c] = U[i*m+j, c] * V[j*p+k, c]
                M = np.zeros((m, target_rank))
                for j in range(m):
                    M[j, :] = best_U[i * m + j, :] * best_V[j * p + k, :]
                # T[i,:,k] = W[k*n+i,:] @ M
                try:
                    W_row, _, _, _ = np.linalg.lstsq(M.T, T[i, :, k], rcond=None)
                    best_W[k * n + i, :] = W_row
                except:
                    pass
    
    T_rec = _matrices_to_tensor(best_U, best_V, best_W, n, m, p)
    err = np.linalg.norm(T - T_rec)
    
    if err < 1e-9:
        return best_U, best_V, best_W
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Leave margin for 120s limit
    
    # Get schoolbook solution as starting point
    U0, V0, W0 = _schoolbook(n, m, p)
    rank0 = U0.shape[1]  # 32
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank0
    
    # Try to find lower rank decompositions
    # Known lower bounds and upper bounds for 2x4x4:
    # The tensor is [2,4,4] multiplication. 
    # Let's try ranks from 16 up to rank0-1
    
    # First, try some structured approaches
    # Approach: use the fact that n=2 is small
    
    # Try rank values in decreasing order of likelihood
    for target_rank in range(rank0 - 1, 15, -1):
        if time.time() > deadline:
            break
        
        found = False
        for seed in range(20):
            if time.time() > deadline:
                break
            
            result = _try_rank_reduction(U0, V0, W0, target_rank, n, m, p, 
                                         max_iters=2000, seed=seed)
            if result is not None:
                U_r, V_r, W_r = result
                if _verify(U_r, V_r, W_r, n, m, p):
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            # Try to go even lower
            continue
    
    # Also try direct random search for low ranks
    for target_rank in range(max(16, best_rank - 5), best_rank):
        if time.time() > deadline:
            break
        
        for seed in range(10):
            if time.time() > deadline:
                break
            
            rng = np.random.default_rng(seed + 1000)
            U_rand = rng.standard_normal((n * m, target_rank))
            V_rand = rng.standard_normal((m * p, target_rank))
            W_rand = rng.standard_normal((p * n, target_rank))
            
            result = _try_rank_reduction(U0, V0, W0, target_rank, n, m, p,
                                         max_iters=3000, seed=seed + 1000)
            if result is not None:
                U_r, V_r, W_r = result
                if _verify(U_r, V_r, W_r, n, m, p):
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = target_rank
                    break
    
    # Convert to lists with half-integer rounding
    U_out, V_out, W_out = _round_to_half(best_U, best_V, best_W)
    
    # Verify after rounding
    if not _verify(U_out, V_out, W_out, n, m, p):
        # Use unrounded if rounding broke it
        U_out = best_U
        V_out = best_V
        W_out = best_W
    
    return {
        "U": U_out.tolist(),
        "V": V_out.tolist(),
        "W": W_out.tolist()
    }
