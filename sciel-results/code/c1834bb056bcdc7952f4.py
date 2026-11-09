# EVOLVE-BLOCK-START
import time
import numpy as np

DIMENSIONS = [4, 5, 6]
TIME_LIMIT = 150.0

def _strassen_2x2():
    """
    Build rank-7 Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    Returns U, V, W as 2x2 x 7 matrices (flattened to 4x7).
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11 + A22) * (B11 + B22)
    U[0, 0] = 1.0; U[3, 0] = 1.0
    V[0, 0] = 1.0; V[3, 0] = 1.0
    W[0, 0] = 1.0; W[3, 0] = 1.0
    
    # M2 = (A21 + A22) * B11
    U[2, 1] = 1.0; U[3, 1] = 1.0
    V[0, 1] = 1.0
    W[2, 1] = 1.0; W[3, 1] = -1.0
    
    # M3 = A11 * (B12 - B22)
    U[0, 2] = 1.0
    V[1, 2] = 1.0; V[3, 2] = -1.0
    W[0, 2] = 1.0; W[3, 2] = 1.0
    
    # M4 = A22 * (B21 - B11)
    U[3, 3] = 1.0
    V[2, 3] = 1.0; V[0, 3] = -1.0
    W[0, 3] = 1.0; W[2, 3] = 1.0
    
    # M5 = (A11 + A12) * B22
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = -1.0; W[1, 4] = 1.0
    
    # M6 = (A21 - A11) * (B11 + B12)
    U[2, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[1, 5] = 1.0
    W[3, 5] = 1.0
    
    # M7 = (A12 - A22) * (B21 + B22)
    U[1, 6] = 1.0; U[3, 6] = -1.0
    V[2, 6] = 1.0; V[3, 6] = 1.0
    W[0, 6] = 1.0
    
    return U, V, W

def _build_full_strassen(N):
    """
    Build full Strassen decomposition for N x N x N where N is a power of 2.
    Returns U, V, W as (N*N) x rank matrices.
    U[i*N+j, c], V[j*N+k, c], W[k*N+i, c]
    """
    if N == 2:
        return _strassen_2x2()
    
    h = N // 2
    U_h, V_h, W_h = _build_full_strassen(h)
    rank_h = U_h.shape[1]
    rank_full = 7 * rank_h
    
    U_full = np.zeros((N * N, rank_full))
    V_full = np.zeros((N * N, rank_full))
    W_full = np.zeros((N * N, rank_full))
    
    # Helper to get sub-block indices
    def block_idx(i, j, offset_i, offset_j, N, h):
        return (offset_i + i) * N + (offset_j + j)
    
    # M1 = (A11+A22)(B11+B22)
    for c in range(rank_h):
        col = c
        for i in range(h):
            for j in range(h):
                U_full[block_idx(i, j, 0, 0, N, h), col] += U_h[i * h + j, c]
                U_full[block_idx(i, j, h, h, N, h), col] += U_h[i * h + j, c]
        for j in range(h):
            for k in range(h):
                V_full[block_idx(j, k, 0, 0, N, h), col] += V_h[j * h + k, c]
                V_full[block_idx(j, k, h, h, N, h), col] += V_h[j * h + k, c]
        for k in range(h):
            for i in range(h):
                W_full[block_idx(k, i, 0, 0, N, h), col] += W_h[k * h + i, c]
                W_full[block_idx(k, i, h, h, N, h), col] += W_h[k * h + i, c]
                
    # M2 = (A21+A22)B11
    for c in range(rank_h):
        col = rank_h + c
        for i in range(h):
            for j in range(h):
                U_full[block_idx(i, j, h, 0, N, h), col] += U_h[i * h + j, c]
                U_full[block_idx(i, j, h, h, N, h), col] += U_h[i * h + j, c]
        for j in range(h):
            for k in range(h):
                V_full[block_idx(j, k, 0, 0, N, h), col] += V_h[j * h + k, c]
        for k in range(h):
            for i in range(h):
                W_full[block_idx(k, i, h, 0, N, h), col] += W_h[k * h + i, c]
                W_full[block_idx(k, i, h, h, N, h), col] -= W_h[k * h + i, c]
                
    # M3 = A11(B12-B22)
    for c in range(rank_h):
        col = 2 * rank_h + c
        for i in range(h):
            for j in range(h):
                U_full[block_idx(i, j, 0, 0, N, h), col] += U_h[i * h + j, c]
        for j in range(h):
            for k in range(h):
                V_full[block_idx(j, k, 0, h, N, h), col] += V_h[j * h + k, c]
                V_full[block_idx(j, k, h, h, N, h), col] -= V_h[j * h + k, c]
        for k in range(h):
            for i in range(h):
                W_full[block_idx(k, i, 0, h, N, h), col] += W_h[k * h + i, c]
                W_full[block_idx(k, i, h, h, N, h), col] += W_h[k * h + i, c]
                
    # M4 = A22(B21-B11)
    for c in range(rank_h):
        col = 3 * rank_h + c
        for i in range(h):
            for j in range(h):
                U_full[block_idx(i, j, h, h, N, h), col] += U_h[i * h + j, c]
        for j in range(h):
            for k in range(h):
                V_full[block_idx(j, k, h, 0, N, h), col] += V_h[j * h + k, c]
                V_full[block_idx(j, k, 0, 0, N, h), col] -= V_h[j * h + k, c]
        for k in range(h):
            for i in range(h):
                W_full[block_idx(k, i, 0, 0, N, h), col] += W_h[k * h + i, c]
                W_full[block_idx(k, i, h, 0, N, h), col] += W_h[k * h + i, c]
                
    # M5 = (A11+A12)B22
    for c in range(rank_h):
        col = 4 * rank_h + c
        for i in range(h):
            for j in range(h):
                U_full[block_idx(i, j, 0, 0, N, h), col] += U_h[i * h + j, c]
                U_full[block_idx(i, j, 0, h, N, h), col] += U_h[i * h + j, c]
        for j in range(h):
            for k in range(h):
                V_full[block_idx(j, k, h, h, N, h), col] += V_h[j * h + k, c]
        for k in range(h):
            for i in range(h):
                W_full[block_idx(k, i, 0, 0, N, h), col] -= W_h[k * h + i, c]
                W_full[block_idx(k, i, 0, h, N, h), col] += W_h[k * h + i, c]
                
    # M6 = (A21-A11)(B11+B12)
    for c in range(rank_h):
        col = 5 * rank_h + c
        for i in range(h):
            for j in range(h):
                U_full[block_idx(i, j, h, 0, N, h), col] += U_h[i * h + j, c]
                U_full[block_idx(i, j, 0, 0, N, h), col] -= U_h[i * h + j, c]
        for j in range(h):
            for k in range(h):
                V_full[block_idx(j, k, 0, 0, N, h), col] += V_h[j * h + k, c]
                V_full[block_idx(j, k, 0, h, N, h), col] += V_h[j * h + k, c]
        for k in range(h):
            for i in range(h):
                W_full[block_idx(k, i, h, h, N, h), col] += W_h[k * h + i, c]
                
    # M7 = (A12-A22)(B21+B22)
    for c in range(rank_h):
        col = 6 * rank_h + c
        for i in range(h):
            for j in range(h):
                U_full[block_idx(i, j, 0, h, N, h), col] += U_h[i * h + j, c]
                U_full[block_idx(i, j, h, h, N, h), col] -= U_h[i * h + j, c]
        for j in range(h):
            for k in range(h):
                V_full[block_idx(j, k, h, 0, N, h), col] += V_h[j * h + k, c]
                V_full[block_idx(j, k, h, h, N, h), col] += V_h[j * h + k, c]
        for k in range(h):
            for i in range(h):
                W_full[block_idx(k, i, 0, 0, N, h), col] += W_h[k * h + i, c]
                
    return U_full, V_full, W_full

def _extract_subtensor(U_full, V_full, W_full, N, n, m, p):
    """
    Extract the n x m x p sub-tensor from the N x N x N decomposition.
    """
    rank = U_full.shape[1]
    
    U_sub = np.zeros((n * m, rank))
    V_sub = np.zeros((m * p, rank))
    W_sub = np.zeros((p * n, rank))
    
    # Vectorized extraction
    # U: i in [0,n), j in [0,m) -> idx = i*N+j
    for i in range(n):
        for j in range(m):
            U_sub[i * m + j, :] = U_full[i * N + j, :]
            
    # V: j in [0,m), k in [0,p) -> idx = j*N+k
    for j in range(m):
        for k in range(p):
            V_sub[j * p + k, :] = V_full[j * N + k, :]
            
    # W: k in [0,p), i in [0,n) -> idx = k*N+i
    for k in range(p):
        for i in range(n):
            W_sub[k * n + i, :] = W_full[k * N + i, :]
            
    return U_sub, V_sub, W_sub

def _remove_zero_columns(U, V, W):
    """Remove columns where any of U, V, W is entirely zero."""
    rank = U.shape[1]
    # Vectorized check
    u_nonzero = np.any(U != 0, axis=0)
    v_nonzero = np.any(V != 0, axis=0)
    w_nonzero = np.any(W != 0, axis=0)
    
    keep_mask = u_nonzero & v_nonzero & w_nonzero
    keep = np.where(keep_mask)[0]
    
    if len(keep) == rank:
        return U, V, W
    
    return U[:, keep], V[:, keep], W[:, keep]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor decomposition using vectorized operations."""
    rank = U.shape[1]
    
    # Reshape U, V, W to (n, m, rank), (m, p, rank), (p, n, rank)
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # Compute sum over rank
    # Result[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # This is a bit tricky to vectorize directly without loops or einsum
    
    # Use einsum
    # 'ijr,jkr,kr i -> ijk' ? No, W is (p, n, rank) -> k, i, r
    # U: i, j, r
    # V: j, k, r
    # W: k, i, r
    # We want sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    
    # Let's use einsum
    try:
        result = np.einsum('ijr,jkr,kr i->ijk', U_reshaped, V_reshaped, W_reshaped)
    except Exception:
        # Fallback to loops if einsum fails
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Check if all entries are 1
    if not np.allclose(result, 1.0, atol=tol):
        return False
    return True

def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return U, V, W

def construct():
    """
    Find a low-rank exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Fallback: schoolbook
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Try embedding in 8x8x8 Strassen
    if time.time() - start_time < TIME_LIMIT - 20:
        try:
            N = 8
            U_full, V_full, W_full = _build_full_strassen(N)
            
            U_sub, V_sub, W_sub = _extract_subtensor(U_full, V_full, W_full, N, n, m, p)
            U_sub, V_sub, W_sub = _remove_zero_columns(U_sub, V_sub, W_sub)
            
            if _verify_decomposition(U_sub, V_sub, W_sub, n, m, p):
                if U_sub.shape[1] < best_rank:
                    best_U, best_V, best_W = U_sub, V_sub, W_sub
                    best_rank = U_sub.shape[1]
        except Exception:
            pass
    
    # Check deadline
    if time.time() - start_time > TIME_LIMIT - 10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
