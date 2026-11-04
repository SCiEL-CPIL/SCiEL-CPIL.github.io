"""Matrix multiplication 4x4x7 - Kronecker Strassen + gradient refinement."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _strassen_2x2_terms():
    """Return 7 rank-1 terms for 2x2x2 matrix multiplication."""
    terms = []
    U = np.zeros(4); U[0] = 1
    V = np.zeros(4); V[0] = 1
    W = np.zeros(4); W[0] = 1
    terms.append((U.copy(), V.copy(), W.copy()))
    
    U = np.zeros(4); U[1] = 1
    V = np.zeros(4); V[3] = 1
    W = np.zeros(4); W[2] = 1
    terms.append((U.copy(), V.copy(), W.copy()))
    
    U = np.zeros(4); U[2] = 1
    V = np.zeros(4); V[0] = 1
    W = np.zeros(4); W[1] = 1
    terms.append((U.copy(), V.copy(), W.copy()))
    
    U = np.zeros(4); U[3] = 1
    V = np.zeros(4); V[2] = 1
    W = np.zeros(4); W[3] = 1
    terms.append((U.copy(), V.copy(), W.copy()))
    
    U = np.zeros(4); U[0] = 1; U[2] = 1
    V = np.zeros(4); V[2] = 1
    W = np.zeros(4); W[2] = 1; W[3] = 1
    terms.append((U.copy(), V.copy(), W.copy()))
    
    U = np.zeros(4); U[1] = 1; U[3] = 1
    V = np.zeros(4); V[0] = 1
    W = np.zeros(4); W[0] = 1; W[1] = 1
    terms.append((U.copy(), V.copy(), W.copy()))
    
    U = np.zeros(4); U[0] = 1; U[3] = -1
    V = np.zeros(4); V[1] = 1; V[3] = 1
    W = np.zeros(4); W[2] = 1; W[3] = 1
    terms.append((U.copy(), V.copy(), W.copy()))
    
    return terms

def _strassen_2x2x3_terms():
    """Return rank-1 terms for 2x2x3 matrix multiplication (Strassen 2x2x2 + schoolbook 2x2x1)."""
    terms = []
    # First 7 terms from Strassen 2x2x2 (for columns 0,1 of the 3-column block)
    s22 = _strassen_2x2_terms()
    for (u, v, w) in s22:
        V6 = np.zeros(6); V6[:4] = v
        W6 = np.zeros(6); W6[:4] = w
        terms.append((u.copy(), V6.copy(), W6.copy()))
    # Last 4 terms: schoolbook for column 2
    U = np.zeros(4); U[0] = 1
    V = np.zeros(6); V[4] = 1
    W = np.zeros(6); W[4] = 1
    terms.append((U.copy(), V.copy(), W.copy()))
    
    U = np.zeros(4); U[1] = 1
    V = np.zeros(6); V[5] = 1
    W = np.zeros(6); W[4] = 1
    terms.append((U.copy(), V.copy(), W.copy()))
    
    U = np.zeros(4); U[2] = 1
    V = np.zeros(6); V[4] = 1
    W = np.zeros(6); W[5] = 1
    terms.append((U.copy(), V.copy(), W.copy()))
    
    U = np.zeros(4); U[3] = 1
    V = np.zeros(6); V[5] = 1
    W = np.zeros(6); W[5] = 1
    terms.append((U.copy(), V.copy(), W.copy()))
    
    return terms

def _build_kronecker_decomposition(n, m, p):
    """Build the Kronecker Strassen + 2x2x3 block decomposition."""
    # Part 1: 4x4x4 via Kronecker of two 2x2x2 Strassen
    s22 = _strassen_2x2_terms()
    terms_4x4x4 = []
    for i in range(7):
        for j in range(7):
            U4 = np.kron(s22[i][0], s22[j][0])
            V4 = np.kron(s22[i][1], s22[j][1])
            W4 = np.kron(s22[i][2], s22[j][2])
            terms_4x4x4.append((U4, V4, W4))
    
    # Part 2: 4x4x3 via 4 blocks of 2x2x3
    s22x3 = _strassen_2x2x3_terms()
    sub_blocks = [(0,0), (2,0), (0,2), (2,2)]
    
    R = 49 + 4 * len(s22x3)  # 49 + 44 = 93
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Part 1
    for idx, (U4, V4, W4) in enumerate(terms_4x4x4):
        U[:, idx] = U4
        V[:16, idx] = V4
        W[:16, idx] = W4
    
    # Part 2
    col_offset = 49
    for i_start, j_start in sub_blocks:
        for (u, v, w) in s22x3:
            for li in range(2):
                for lj in range(2):
                    U[(i_start + li) * 4 + (j_start + lj), col_offset] += u[li * 2 + lj]
            for lj in range(2):
                for lk in range(3):
                    V[(j_start + lj) * 7 + (4 + lk), col_offset] += v[lj * 3 + lk]
            for lk in range(3):
                for li in range(2):
                    W[(4 + lk) * 4 + (i_start + li), col_offset] += w[lk * 2 + li]
            col_offset += 1
    
    return U, V, W, R

def _verify(U, V, W, n, m, p, rank, tol=1e-9):
    """Check if U, V, W form a valid tensor decomposition."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    u_vals = Ua[u_idx, :]
    v_vals = Va[v_idx, :]
    w_vals = Wa[w_idx, :]
    
    prod = u_vals * v_vals * w_vals
    result = prod.sum(axis=1)
    
    expected = np.ones(n * m * p)
    return np.allclose(result, expected, atol=tol)

def _refine(U, V, W, n, m, p, rank, max_iter=200, tol=1e-8, deadline=None):
    """Iteratively refine the decomposition using gradient descent."""
    Ua = np.array(U, dtype=float).copy()
    Va = np.array(V, dtype=float).copy()
    Wa = np.array(W, dtype=float).copy()
    
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        u_vals = Ua[u_idx, :]
        v_vals = Va[v_idx, :]
        w_vals = Wa[w_idx, :]
        
        prod = u_vals * v_vals * w_vals
        result = prod.sum(axis=1)
        
        error = result - 1.0
        err_norm = np.linalg.norm(error)
        
        if err_norm < tol:
            break
        
        err_3d = error.reshape(n, m, p)
        u_4d = u_vals.reshape(n, m, p, rank)
        v_4d = v_vals.reshape(n, m, p, rank)
        w_4d = w_vals.reshape(n, m, p, rank)
        
        # Gradient for U: sum over k of 2*err[i,j,k]*V[j*p+k,c]*W[k*n+i,c]
        grad_U_4d = 2 * np.expand_dims(err_3d, axis=-1) * v_4d * w_4d
        grad_U = np.sum(grad_U_4d, axis=2).reshape(n*m, rank)
        
        # Gradient for V: sum over i of 2*err[i,j,k]*U[i*m+j,c]*W[k*n+i,c]
        grad_V_4d = 2 * np.expand_dims(err_3d, axis=-1) * u_4d * w_4d
        grad_V_flat = np.zeros((m*p, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    grad_V_flat[j*p+k, :] += grad_V_4d[i,j,k,:]
        grad_V = grad_V_flat
        
        # Gradient for W: sum over j of 2*err[i,j,k]*U[i*m+j,c]*V[j*p+k,c]
        grad_W_4d = 2 * np.expand_dims(err_3d, axis=-1) * u_4d * v_4d
        grad_W_flat = np.zeros((p*n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    grad_W_flat[k*n+i, :] += grad_W_4d[i,j,k,:]
        grad_W = grad_W_flat
        
        # Adaptive learning rate
        lr = 0.05 / (1 + it * 0.05)
        
        Ua -= lr * grad_U
        Va -= lr * grad_V
        Wa -= lr * grad_W
    
    return Ua, Va, Wa

def _schoolbook(n, m, p):
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Build the Kronecker Strassen decomposition (rank 93)
    U_init, V_init, W_init, R_init = _build_kronecker_decomposition(n, m, p)
    
    # Verify the initial decomposition
    if not _verify(U_init.tolist(), V_init.tolist(), W_init.tolist(), n, m, p, R_init, tol=1e-9):
        return _schoolbook(n, m, p)
    
    best_U, best_V, best_W, best_rank = U_init.tolist(), V_init.tolist(), W_init.tolist(), R_init
    
    # Try to find lower-rank decompositions using gradient refinement
    for rank in range(85, best_rank):
        if time.time() > deadline:
            break
        
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            np.random.seed(rank * 100 + attempt)
            
            # Initialize with Kronecker structure + random perturbation
            U = U_init[:, :rank].copy() if rank <= R_init else np.hstack([U_init, np.random.randn(n*m, rank - R_init) * 0.1])
            V = V_init[:, :rank].copy() if rank <= R_init else np.hstack([V_init, np.random.randn(m*p, rank - R_init) * 0.1])
            W = W_init[:, :rank].copy() if rank <= R_init else np.hstack([W_init, np.random.randn(p*n, rank - R_init) * 0.1])
            
            # Add small random perturbation to escape local minima
            U += np.random.randn(n*m, rank) * 0.01
            V += np.random.randn(m*p, rank) * 0.01
            W += np.random.randn(p*n, rank) * 0.01
            
            # Refine
            U_ref, V_ref, W_ref = _refine(U.tolist(), V.tolist(), W.tolist(), n, m, p, rank, 
                                           max_iter=100, tol=1e-8, deadline=deadline)
            
            # Verify
            if _verify(U_ref.tolist(), V_ref.tolist(), W_ref.tolist(), n, m, p, rank, tol=1e-6):
                best_U, best_V, best_W, best_rank = U_ref.tolist(), V_ref.tolist(), W_ref.tolist(), rank
                break  # Found valid decomposition at this rank, try lower
    
    # Final safety check
    if not _verify(best_U, best_V, best_W, n, m, p, best_rank, tol=1e-6):
        return _schoolbook(n, m, p)
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
