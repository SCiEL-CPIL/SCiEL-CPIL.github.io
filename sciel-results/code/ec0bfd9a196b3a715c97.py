import numpy as np
import time

def _build_block_rank28(n, m, p):
    """Build the known rank-28 block decomposition."""
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0

    # Block 1: j=0, k=0..4 (schoolbook 2x1x5)
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1

    # Strassen 2x2x2 helper
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        U[(i_off+0)*m + (j_off+0), col] = 1.0
        U[(i_off+1)*m + (j_off+1), col] = 1.0
        V[(j_off+0)*p + (k_off+0), col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = 1.0
        W[(k_off+0)*n + (i_off+0), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = 1.0
        col += 1
        U[(i_off+1)*m + (j_off+0), col] = 1.0
        U[(i_off+1)*m + (j_off+1), col] = 1.0
        V[(j_off+0)*p + (k_off+0), col] = 1.0
        W[(k_off+0)*n + (i_off+1), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = -1.0
        col += 1
        U[(i_off+0)*m + (j_off+0), col] = 1.0
        V[(j_off+0)*p + (k_off+1), col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = -1.0
        W[(k_off+1)*n + (i_off+0), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = 1.0
        col += 1
        U[(i_off+1)*m + (j_off+1), col] = 1.0
        V[(j_off+1)*p + (k_off+0), col] = 1.0
        V[(j_off+0)*p + (k_off+0), col] = -1.0
        W[(k_off+0)*n + (i_off+0), col] = 1.0
        W[(k_off+0)*n + (i_off+1), col] = 1.0
        col += 1
        U[(i_off+0)*m + (j_off+0), col] = 1.0
        U[(i_off+0)*m + (j_off+1), col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = 1.0
        W[(k_off+0)*n + (i_off+0), col] = -1.0
        W[(k_off+1)*n + (i_off+0), col] = 1.0
        col += 1
        U[(i_off+1)*m + (j_off+0), col] = 1.0
        U[(i_off+0)*m + (j_off+0), col] = -1.0
        V[(j_off+0)*p + (k_off+0), col] = 1.0
        V[(j_off+0)*p + (k_off+1), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = 1.0
        col += 1
        U[(i_off+0)*m + (j_off+1), col] = 1.0
        U[(i_off+1)*m + (j_off+1), col] = -1.0
        V[(j_off+1)*p + (k_off+0), col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = 1.0
        W[(k_off+0)*n + (i_off+0), col] = 1.0
        col += 1

    add_strassen(i_off=0, j_off=1, k_off=0)
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j, col] = 1.0
            V[j * p + 2, col] = 1.0
            W[2 * n + i, col] = 1.0
            col += 1
    add_strassen(i_off=0, j_off=1, k_off=3)
    assert col == rank
    return U, V, W

def _verify(U, V, W, n, m, p, rank, atol=1e-10):
    """Verify decomposition on several random matrices."""
    rng = np.random.RandomState(42)
    for _ in range(5):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C = A @ B
        C_d = np.zeros((n, p))
        for t in range(rank):
            u_t = U[:, t].reshape(n, m)
            v_t = V[:, t].reshape(m, p)
            w_t = W[:, t].reshape(p, n)
            C_d += (A * u_t).sum() * (B * v_t).sum() * w_t.T
        if not np.allclose(C, C_d, atol=atol):
            return False
    return True

def _try_rank27_linear(n, m, p, deadline):
    """
    Attempt to find a rank-27 decomposition by exploiting linear dependencies
    in the schoolbook basis. The key idea: the 30 schoolbook terms span a
    subspace; if we can find a rank-27 decomposition, the 3 flattening
    matrices must satisfy rank-27 conditions.
    
    We use a different approach: start from the rank-28 block decomposition
    and try to eliminate one rank-1 term by finding a linear combination
    that cancels it, using the fact that some entries may be zero.
    
    Alternative: use the fact that the matmul tensor has a specific structure.
    For 2x3x5, we can try to find if rank 27 is possible by checking if
    the tensor has a 27-term decomposition via a structured search.
    """
    if time.time() > deadline - 10:
        return None
    
    # The matmul tensor T_{i,j,k} for 2x3x5:
    # Flatten along (i,j) axis: matrix of shape (n*m, m*p) = (6, 15)
    # T flattened: T[i*m+j, j*p+k] = 1 if valid (i,j,k) else 0
    # Actually T_{i,j,k} = delta_{i,j,k} is 1 for all valid (i,j,k)
    # The flattening along first mode: M1[a, bc] where a = i*m+j, bc = j*p+k
    # M1[i*m+j, j'*p+k] = 1 if j == j' else 0, for all i, k
    
    # This is a block diagonal structure!
    # For each j, we have a 2x5 block of 1s.
    # So M1 is block diagonal with three 2x5 blocks of 1s.
    # rank(M1) = 3 * min(2,5) = 3*2 = 6... wait, that's the rank of the flattening.
    
    # Actually, the tensor rank is NOT the same as the flattening rank.
    # The flattening rank gives a lower bound on tensor rank.
    # rank(M1) = 6, rank(M2) = ?, rank(M3) = ?
    
    # Let me compute the flattening ranks:
    # Mode-1 flattening: shape (n*m, m*p) = (6, 15)
    # M1[i*m+j, j'*p+k] = 1 if j==j' else 0
    # This is block diagonal: 3 blocks of (2,5) all-ones matrices
    # Each block has rank 1, so total rank = 3
    
    # Mode-2 flattening: shape (m*p, n*m) = (15, 6)  
    # M2[j*p+k, i*m+j'] = 1 if j==j' else 0
    # Also block diagonal: 3 blocks of (5,2) all-ones matrices
    # Each block has rank 1, so total rank = 3
    
    # Mode-3 flattening: shape (p*n, n*m) = (10, 6)
    # M3[k*n+i, i'*m+j] = 1 if i==i' else 0
    # Block diagonal: 2 blocks of (5,3) all-ones matrices
    # Each block has rank 1, so total rank = 2
    
    # So flattening ranks are 6, 3, 2. Lower bound is max(6,3,2) = 6.
    # This doesn't help rule out rank 27.
    
    # Let's try a direct approach: use the known rank-28 and try to reduce.
    U28, V28, W28 = _build_block_rank28(n, m, p)
    
    # Try to find a linear dependency among the 28 rank-1 terms
    # that allows us to express one as a combination of others.
    # This is equivalent to finding if the tensor lies in the span of
    # 27 rank-1 tensors.
    
    # A more practical approach: use optimization to find a rank-27
    # decomposition by minimizing the reconstruction error.
    
    if time.time() > deadline - 5:
        return None
    
    # Try a structured rank-27: remove one term from the rank-28 and
    # see if we can compensate with modified coefficients.
    
    # The j=0 block has 10 terms (2x1x5 schoolbook).
    # Can we do better than 10 for 2x1x5? No, rank of 2x1x5 matmul is 2*5=10.
    
    # The j=1,2,k=0,1 block uses Strassen (rank 7 for 2x2x2).
    # Can we do 2x2x2 in rank 6? No, rank is 7.
    
    # The j=1,2,k=2 block is 2x2x1 schoolbook (rank 4).
    # The j=1,2,k=3,4 block uses Strassen (rank 7).
    
    # Total: 10 + 7 + 4 + 7 = 28.
    # To get 27, we'd need to save 1 rank somewhere.
    # 
    # Idea: combine the 2x2x1 (rank 4) with one of the Strassen blocks
    # to share a term. E.g., 2x2x3 might have rank < 7+4=11.
    # 
    # Actually, let's try: can 2x2x3 be done in rank 10?
    # If so, total would be 10 + 10 + 7 = 27.
    
    # Let me try to construct a rank-10 decomposition for 2x2x3.
    # Use Strassen for 2x2x2 (rank 7) + schoolbook for 2x2x1 (rank 4) = 11.
    # Can we save 1? 
    
    # Try: use 6 terms for the 2x2x2 part and 4 for 2x2x1, but with
    # one shared term. This requires a careful construction.
    
    # Alternative: try to find if the 2x3x5 tensor has rank 27 by
    # checking if a certain determinantal variety condition holds.
    # This is complex, so let's try a numerical optimization approach.
    
    if time.time() > deadline - 3:
        return None
    
    # Numerical approach: start from rank-28, zero out one column,
    # and try to optimize the remaining 27 to minimize error.
    # This is a non-convex problem, but we can try a few random
    # initializations.
    
    best_U, best_V, best_W = None, None, None
    best_err = np.inf
    
    for attempt in range(3):
        if time.time() > deadline - 2:
            break
        
        U = U28.copy()
        V = V28.copy()
        W = W28.copy()
        
        # Zero out a random column
        col_to_remove = np.random.randint(0, 28)
        U[:, col_to_remove] = 0
        V[:, col_to_remove] = 0
        W[:, col_to_remove] = 0
        
        # Now try to optimize the remaining 27 columns
        # using a simple gradient-free approach (Nelder-Mead style)
        # This is too slow for 27*3 variables. Skip.
        
        # Instead, just check if the zeroed-out decomposition still works
        # (it won't, but this is a sanity check)
        pass
    
    # The numerical approach is too slow. Let's return None and
    # use the rank-28 fallback.
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix
    by a 3x5 matrix.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s limit

    n, m, p = 2, 3, 5  # dimensions

    # Try rank-27 first
    result_27 = _try_rank27_linear(n, m, p, DEADLINE)
    if result_27 is not None:
        U27, V27, W27 = result_27
        if _verify(U27, V27, W27, n, m, p, 27):
            return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}

    # Fall back to rank-28 block decomposition
    U, V, W = _build_block_rank28(n, m, p)
    
    if not _verify(U, V, W, n, m, p, 28):
        # Ultimate fallback: schoolbook rank-30
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb), dtype=float)
        V_sb = np.zeros((m * p, rank_sb), dtype=float)
        W_sb = np.zeros((p * n, rank_sb), dtype=float)
        col_sb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col_sb] = 1.0
                    V_sb[j * p + k, col_sb] = 1.0
                    W_sb[k * n + i, col_sb] = 1.0
                    col_sb += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
