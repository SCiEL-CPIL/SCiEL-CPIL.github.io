import time
import numpy as np

_start_time = time.time()
TIME_LIMIT = 135  # safety margin before 150s kill

def _schoolbook(m, n, p):
    """Schoolbook decomposition: rank = m*n*p"""
    rank = m * n * p
    U = np.zeros((m * n, rank))
    V = np.zeros((n * p, rank))
    W = np.zeros((p * m, rank))
    col = 0
    for i in range(m):
        for j in range(n):
            for k in range(p):
                U[i * n + j, col] = 1
                V[j * p + k, col] = 1
                W[k * m + i, col] = 1
                col += 1
    return (U, V, W)

def _strassen_2x2x2():
    """Strassen decomposition for 2x2 * 2x2, rank 7"""
    U = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=float)
    V = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)
    W = np.array([
        [1, 0, 0, 1],
        [0, -1, 0, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 0, 0, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ], dtype=float)
    return (U, V, W)

def _verify(U, V, W, m, n, p, tol=1e-9):
    """Verify that U, V, W correctly decompose m×n * n×p matrix multiplication"""
    # Check shapes
    if U.shape[0] != m * n or V.shape[0] != n * p or W.shape[0] != p * m:
        return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    rank = U.shape[1]
    if rank == 0:
        return m == 0 or n == 0 or p == 0
    
    # For each output element C[i,k], check that sum_r U[i*n+j, r] * V[j*p+k, r] * W[k*m+i, r] = delta(i,j,k)
    # Actually, the tensor T_{i,j,k} = 1 if we're computing (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # The decomposition says: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where (u_r ⊗ v_r ⊗ w_r)_{i,j,k} = U[i*n+j, r] * V[j*p+k, r] * W[k*m+i, r]
    # And we need this to equal the matrix multiplication tensor.
    
    # The matrix multiplication tensor T_{i,j,k} = 1 for all i,j,k (since (AB)_{i,k} = sum_j A_{i,j} B_{j,k})
    # Wait, that's not right. The tensor for matrix multiplication is:
    # T_{i,j,k} = 1 if we're looking at the (i,k) output element's dependence on A_{i,j} and B_{j,k}
    # Actually, the standard tensor is: for each (i,j,k), the coefficient is 1
    # because (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    
    # So we need: sum_r U[i*n+j, r] * V[j*p+k, r] * W[k*m+i, r] = 1 for all i,j,k
    # Let me verify this
    
    # Create the full tensor
    T_reconstructed = np.zeros((m, n, p))
    for r in range(rank):
        u = U[:, r].reshape(m, n)
        v = V[:, r].reshape(n, p)
        w = W[:, r].reshape(p, m)
        # The contribution to T_{i,j,k} is u[i,j] * v[j,k] * w[k,i]
        for i in range(m):
            for j in range(n):
                for k in range(p):
                    T_reconstructed[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    
    # The target tensor is all ones
    if not np.allclose(T_reconstructed, 1.0, atol=tol):
        return False
    return True

def _embed_sub(U_sub, V_sub, W_sub, m_sub, n_sub, p_sub, offset_m, offset_n, offset_p, m, n, p):
    """Embed a sub-decomposition into the full m×n×p space"""
    U_full = np.zeros((m * n, U_sub.shape[1]))
    V_full = np.zeros((n * p, V_sub.shape[1]))
    W_full = np.zeros((p * m, W_sub.shape[1]))
    
    for r in range(U_sub.shape[1]):
        # Embed U_sub into U_full
        for i in range(m_sub):
            for j in range(n_sub):
                U_full[(offset_m + i) * n + (offset_n + j), r] = U_sub[i * n_sub + j, r]
        
        # Embed V_sub into V_full
        for j in range(n_sub):
            for k in range(p_sub):
                V_full[(offset_n + j) * p + (offset_p + k), r] = V_sub[j * p_sub + k, r]
        
        # Embed W_sub into W_full
        for k in range(p_sub):
            for i in range(m_sub):
                W_full[(offset_p + k) * m + (offset_m + i), r] = W_sub[k * m_sub + i, r]
    
    return (U_full, V_full, W_full)

def _combine_decomps(decomps, m, n, p):
    """Combine multiple sub-decompositions by concatenating their rank dimensions"""
    if len(decomps) == 0:
        return (np.zeros((m * n, 0)), np.zeros((n * p, 0)), np.zeros((p * m, 0)))
    
    U_parts = []
    V_parts = []
    W_parts = []
    for U, V, W in decomps:
        if U.shape[1] > 0:
            U_parts.append(U)
            V_parts.append(V)
            W_parts.append(W)
    
    if len(U_parts) == 0:
        return (np.zeros((m * n, 0)), np.zeros((n * p, 0)), np.zeros((p * m, 0)))
    
    U_combined = np.hstack(U_parts)
    V_combined = np.hstack(V_parts)
    W_combined = np.hstack(W_parts)
    return (U_combined, V_combined, W_combined)

def _decompose(m, n, p, time_budget=0.0):
    """
    Return (U, V, W) for m×n * n×p matrix multiplication using numpy.
    U: shape (m*n, rank)
    V: shape (n*p, rank)
    W: shape (p*m, rank), W[k*m+i, r] convention
    """
    if time.time() - _start_time > TIME_LIMIT:
        raise TimeoutError()
    
    # Base cases
    if m == 0 or n == 0 or p == 0:
        return (np.zeros((m * n, 0)), np.zeros((n * p, 0)), np.zeros((p * m, 0)))
    
    if m == 1 and n == 1 and p == 1:
        return (np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]]))
    
    # Degenerate cases
    if m == 1:
        rank = n * p
        U = np.zeros((n, rank))
        V = np.zeros((n * p, rank))
        W = np.zeros((p, rank))
        for j in range(n):
            for k in range(p):
                col = j * p + k
                U[j, col] = 1
                V[j * p + k, col] = 1
                W[k, col] = 1
        return (U, V, W)
    
    if p == 1:
        rank = m * n
        U = np.zeros((m * n, rank))
        V = np.zeros((n, rank))
        W = np.zeros((m, rank))
        for i in range(m):
            for j in range(n):
                col = i * n + j
                U[i * n + j, col] = 1
                V[j, col] = 1
                W[i, col] = 1
        return (U, V, W)
    
    if n == 1:
        rank = m * p
        U = np.zeros((m, rank))
        V = np.zeros((p, rank))
        W = np.zeros((m * p, rank))
        for i in range(m):
            for k in range(p):
                col = i * p + k
                U[i, col] = 1
                V[k, col] = 1
                W[k * m + i, col] = 1
        return (U, V, W)
    
    # Special case: Strassen for 2×2 * 2×2
    if m == 2 and n == 2 and p == 2:
        return _strassen_2x2x2()
    
    # Try all possible splits and pick the best
    best = None
    best_rank = m * n * p  # schoolbook
    
    # Generate all possible splits
    splits = []
    for m1 in range(1, m):
        m2 = m - m1
        for n1 in range(1, n):
            n2 = n - n1
            for p1 in range(1, p):
                p2 = p - p1
                splits.append((m1, m2, n1, n2, p1, p2))
    
    # Sort splits by estimated rank (to find good solutions early)
    def estimate_rank(m1, m2, n1, n2, p1, p2):
        # Estimate: sum of m_i * n_j * p_k for each block
        return (m1*n1*p1 + m1*n1*p2 + m1*n2*p1 + m1*n2*p2 +
                m2*n1*p1 + m2*n1*p2 + m2*n2*p1 + m2*n2*p2)
    
    splits.sort(key=lambda s: estimate_rank(s[0], s[1], s[2], s[3], s[4], s[5]))
    
    for m1, m2, n1, n2, p1, p2 in splits:
        if time.time() - _start_time > TIME_LIMIT:
            raise TimeoutError()
        
        try:
            # 8 sub-decompositions
            U1, V1, W1 = _decompose(m1, n1, p1)
            U2, V2, W2 = _decompose(m1, n1, p2)
            U3, V3, W3 = _decompose(m1, n2, p1)
            U4, V4, W4 = _decompose(m1, n2, p2)
            U5, V5, W5 = _decompose(m2, n1, p1)
            U6, V6, W6 = _decompose(m2, n1, p2)
            U7, V7, W7 = _decompose(m2, n2, p1)
            U8, V8, W8 = _decompose(m2, n2, p2)
            
            total_rank = (U1.shape[1] + U2.shape[1] + U3.shape[1] + U4.shape[1] +
                         U5.shape[1] + U6.shape[1] + U7.shape[1] + U8.shape[1])
            
            if total_rank < best_rank:
                # Embed each sub-decomposition into the full space
                U_full, V_full, W_full = _embed_sub(U1, V1, W1, m1, n1, p1, 0, 0, 0, m, n, p)
                U2f, V2f, W2f = _embed_sub(U2, V2, W2, m1, n1, p2, 0, 0, p1, m, n, p)
                U3f, V3f, W3f = _embed_sub(U3, V3, W3, m1, n2, p1, 0, n1, 0, m, n, p)
                U4f, V4f, W4f = _embed_sub(U4, V4, W4, m1, n2, p2, 0, n1, p1, m, n, p)
                U5f, V5f, W5f = _embed_sub(U5, V5, W5, m2, n1, p1, m1, 0, 0, m, n, p)
                U6f, V6f, W6f = _embed_sub(U6, V6, W6, m2, n1, p2, m1, 0, p1, m, n, p)
                U7f, V7f, W7f = _embed_sub(U7, V7, W7, m2, n2, p1, m1, n1, 0, m, n, p)
                U8f, V8f, W8f = _embed_sub(U8, V8, W8, m2, n2, p2, m1, n1, p1, m, n, p)
                
                # Combine
                U_combined, V_combined, W_combined = _combine_decomps(
                    [(U_full, V_full, W_full), (U2f, V2f, W2f), (U3f, V3f, W3f), (U4f, V4f, W4f),
                     (U5f, V5f, W5f), (U6f, V6f, W6f), (U7f, V7f, W7f), (U8f, V8f, W8f)],
                    m, n, p
                )
                
                best_rank = total_rank
                best = (U_combined, V_combined, W_combined)
                
        except TimeoutError:
            raise
        except Exception:
            continue
    
    if best is not None:
        return best
    else:
        return _schoolbook(m, n, p)

def construct():
    global _start_time
    _start_time = time.time()
    
    m, n, p = 4, 5, 5  # 4x5 * 5x5
    
    try:
        U, V, W = _decompose(m, n, p)
    except TimeoutError:
        # Fallback to schoolbook
        U, V, W = _schoolbook(m, n, p)
    
    # Convert to lists for the return interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
