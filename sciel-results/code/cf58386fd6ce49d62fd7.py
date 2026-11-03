import time
import numpy as np

DEADLINE_S = 110.0
N, M, P = 4, 5, 5


def _validate(U, V, W, tol=1e-9):
    """Check sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1 for all i,j,k."""
    n, m, p = N, M, P
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                if abs(s - 1.0) > tol:
                    return False
    return True


def _strassen_2x2x2():
    """Return (U,V,W) for rank-7 Strassen decomposition of 2x2x2 matrix multiplication."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[2, 0] = 1; W[1, 0] = -1; W[3, 0] = -1
    
    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3: A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    # M4: A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[1, 3] = 1
    
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[3, 4] = -1
    
    # M6: (A10-A00)(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[2, 6] = 1
    
    return U, V, W


def _schoolbook(n_b, m_b, p_b):
    """Schoolbook decomposition for n_b x m_b x p_b."""
    rank = n_b * m_b * p_b
    U = np.zeros((n_b * m_b, rank))
    V = np.zeros((m_b * p_b, rank))
    W = np.zeros((p_b * n_b, rank))
    c = 0
    for i in range(n_b):
        for j in range(m_b):
            for k in range(p_b):
                U[i * m_b + j, c] = 1.0
                V[j * p_b + k, c] = 1.0
                W[k * n_b + i, c] = 1.0
                c += 1
    return U, V, W


def _try_rank_decomposition(n_b, m_b, p_b, target_rank, max_iters=200, tol=1e-6):
    """
    Try to find a rank-target_rank decomposition for n_b x m_b x p_b.
    Uses iterative optimization on the tensor equation.
    Returns (U, V, W) if found, else None.
    """
    if target_rank < max(n_b * m_b, m_b * p_b, p_b * n_b):
        return None
    
    # Initialize with schoolbook and then try to reduce
    U_init, V_init, W_init = _schoolbook(n_b, m_b, p_b)
    
    # If target_rank equals schoolbook, just return it
    if target_rank == n_b * m_b * p_b:
        return U_init, V_init, W_init
    
    # Try random initialization with the target rank
    np.random.seed(42)
    for attempt in range(max_iters):
        U = np.zeros((n_b * m_b, target_rank))
        V = np.zeros((m_b * p_b, target_rank))
        W = np.zeros((p_b * n_b, target_rank))
        
        # Random sparse initialization
        for i in range(n_b * m_b):
            for c in range(target_rank):
                if np.random.random() < 0.4:
                    U[i, c] = np.random.choice([-1, 1])
        for j in range(m_b * p_b):
            for c in range(target_rank):
                if np.random.random() < 0.4:
                    V[j, c] = np.random.choice([-1, 1])
        for k in range(p_b * n_b):
            for c in range(target_rank):
                if np.random.random() < 0.4:
                    W[k, c] = np.random.choice([-1, 1])
        
        # Check if this works
        valid = True
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    s = sum(U[i*m_b+j, r] * V[j*p_b+k, r] * W[k*n_b+i, r] for r in range(target_rank))
                    if abs(s - 1.0) > tol:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            return U, V, W
    
    return None


def _get_decomposition(n_b, m_b, p_b):
    """Get the best known decomposition for the given block sizes."""
    # Known results:
    # 2x2x2: rank 7 (Strassen)
    # 2x2x1, 2x1x2, 1x2x2: rank 2 (trivial)
    # 2x2x3, 2x3x2, 3x2x2: rank 9 (known)
    # 2x3x3, 3x2x3, 3x3x2: rank 15 or 18?
    # 1xN x M x P: rank M*P (trivial, just dot products)
    
    if n_b == 2 and m_b == 2 and p_b == 2:
        return _strassen_2x2x2()
    
    # For 1-row or 1-column cases, it's just a dot product
    if n_b == 1:
        # C[0,k] = sum_j A[0,j] * B[j,k], so rank = m_b * p_b
        return _schoolbook(1, m_b, p_b)
    if m_b == 1:
        # C[i,k] = A[i,0] * B[0,k], so rank = n_b * p_b
        return _schoolbook(n_b, 1, p_b)
    if p_b == 1:
        # C[i,0] = sum_j A[i,j] * B[j,0], so rank = n_b * m_b
        return _schoolbook(n_b, m_b, 1)
    
    # Try to find low-rank decompositions for small cases
    # 2x2x3, 2x3x2, 3x2x2: try rank 9
    if (n_b, m_b, p_b) in [(2,2,3), (2,3,2), (3,2,2)]:
        result = _try_rank_decomposition(n_b, m_b, p_b, 9, max_iters=50)
        if result:
            return result
        return _schoolbook(n_b, m_b, p_b)
    
    # 2x3x3, 3x2x3, 3x3x2: try rank 15
    if (n_b, m_b, p_b) in [(2,3,3), (3,2,3), (3,3,2)]:
        result = _try_rank_decomposition(n_b, m_b, p_b, 15, max_iters=50)
        if result:
            return result
        # Try rank 18 (schoolbook)
        return _schoolbook(n_b, m_b, p_b)
    
    # Default: schoolbook
    return _schoolbook(n_b, m_b, p_b)


def _embed_block(n_b, m_b, p_b, i_off, j_off, k_off, U_b, V_b, W_b):
    """Embed a sub-block decomposition into the full space."""
    n, m, p = N, M, P
    rank = U_b.shape[1]
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    for r in range(rank):
        for i in range(n_b):
            for j in range(m_b):
                U[(i_off+i)*m + (j_off+j), r] = U_b[i*m_b+j, r]
        for j in range(m_b):
            for k in range(p_b):
                V[(j_off+j)*p + (k_off+k), r] = V_b[j*p_b+k, r]
        for k in range(p_b):
            for i in range(n_b):
                W[(k_off+k)*n + (i_off+i), r] = W_b[k*n_b+i, r]
    return U, V, W


def _combine_decompositions(decomps):
    """Combine a list of (U,V,W) decompositions into one."""
    if not decomps:
        return None
    
    n, m, p = N, M, P
    total_rank = sum(U.shape[1] for U, V, W in decomps)
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    c = 0
    for U_i, V_i, W_i in decomps:
        r = U_i.shape[1]
        U[:, c:c+r] = U_i
        V[:, c:c+r] = V_i
        W[:, c:c+r] = W_i
        c += r
    
    return U, V, W


def construct():
    t0 = time.time()
    n, m, p = N, M, P  # 4, 5, 5
    
    # Precompute decompositions for all block types we'll need
    # Cache to avoid recomputation
    decomp_cache = {}
    
    def get_decomp_cached(n_b, m_b, p_b):
        key = (n_b, m_b, p_b)
        if key not in decomp_cache:
            decomp_cache[key] = _get_decomposition(n_b, m_b, p_b)
        return decomp_cache[key]
    
    # Strategy 1: Split i: 2+2, j: 3+2, k: 3+2
    decomps1 = []
    for i_off in [0, 2]:
        for j_off, m_b in [(0, 3), (3, 2)]:
            for k_off, p_b in [(0, 3), (3, 2)]:
                U_b, V_b, W_b = get_decomp_cached(2, m_b, p_b)
                U_e, V_e, W_e = _embed_block(2, m_b, p_b, i_off, j_off, k_off, U_b, V_b, W_b)
                decomps1.append((U_e, V_e, W_e))
    
    U1, V1, W1 = _combine_decompositions(decomps1)
    rank1 = U1.shape[1]
    
    # Strategy 2: Split i: 2+2, j: 2+3, k: 2+3
    decomps2 = []
    for i_off in [0, 2]:
        for j_off, m_b in [(0, 2), (2, 3)]:
            for k_off, p_b in [(0, 2), (2, 3)]:
                U_b, V_b, W_b = get_decomp_cached(2, m_b, p_b)
                U_e, V_e, W_e = _embed_block(2, m_b, p_b, i_off, j_off, k_off, U_b, V_b, W_b)
                decomps2.append((U_e, V_e, W_e))
    
    U2, V2, W2 = _combine_decompositions(decomps2)
    rank2 = U2.shape[1]
    
    # Strategy 3: Split i: 1+1+1+1, j: 5, k: 5 (use 1x5x5 blocks)
    decomps3 = []
    for i_off in [0, 1, 2, 3]:
        U_b, V_b, W_b = get_decomp_cached(1, 5, 5)
        U_e, V_e, W_e = _embed_block(1, 5, 5, i_off, 0, 0, U_b, V_b, W_b)
        decomps3.append((U_e, V_e, W_e))
    
    U3, V3, W3 = _combine_decompositions(decomps3)
    rank3 = U3.shape[1]
    
    # Strategy 4: Split i: 4, j: 2+3, k: 2+3
    decomps4 = []
    for j_off, m_b in [(0, 2), (2, 3)]:
        for k_off, p_b in [(0, 2), (2, 3)]:
            U_b, V_b, W_b = get_decomp_cached(4, m_b, p_b)
            U_e, V_e, W_e = _embed_block(4, m_b, p_b, 0, j_off, k_off, U_b, V_b, W_b)
            decomps4.append((U_e, V_e, W_e))
    
    U4, V4, W4 = _combine_decompositions(decomps4)
    rank4 = U4.shape[1]
    
    # Pick the best strategy
    candidates = [
        (rank1, U1, V1, W1),
        (rank2, U2, V2, W2),
        (rank3, U3, V3, W3),
        (rank4, U4, V4, W4),
    ]
    
    # Sort by rank
    candidates.sort(key=lambda x: x[0])
    best_rank, best_U, best_V, best_W = candidates[0]
    
    # Validate
    if not _validate(best_U, best_V, best_W):
        # Fallback to schoolbook rank 100
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
                    c += 1
        best_U, best_V, best_W = U, V, W
        best_rank = rank
    
    if time.time() - t0 > DEADLINE_S:
        pass  # Still return what we have
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
