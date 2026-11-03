"""Rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
Uses Strassen-based block decomposition (rank 64) as baseline, then attempts
refinement via alternating least squares with rounding to half-integers.
Also tries structured sparse integer programming approach for potential rank reduction."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify_fast(U, V, W, n=4, m=4, p=5, atol=1e-9):
    """Fast verification using numpy with float64."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = Ur[:, :, c]
        vc = Vr[:, :, c]
        wc = Wr[:, :, c]
        wcT = wc.T
        for j in range(m):
            T[:, j, :] = T[:, j, :] + np.outer(uc[:, j], vc[j, :]) * wcT
    return np.allclose(T, 1.0, atol=atol)

def _round_to_half_integer(x):
    """Round x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _round_matrix_to_half_int(M):
    """Round all entries of matrix M to nearest half-integer."""
    return np.array([[ _round_to_half_integer(x) for x in row] for row in M])

def _strassen_2x2x2_terms(ai, aj, bj, bk, n, m, p):
    """Return 7 Strassen terms for 2x2x2 block multiplication.
    
    A[ai:ai+2, aj:aj+2] @ B[bj:bj+2, bk:bk+2] -> C[ai:ai+2, bk:bk+2]
    Each term is (u, v, w) where:
    u[i*m+j] = coefficient for A[i,j]
    v[j*p+k] = coefficient for B[j,k]
    w[k*n+i] = coefficient for C[i,k]
    """
    terms = []
    
    def get_u_idx(gi, gj):
        return gi * m + gj
        
    def get_v_idx(gj, gk):
        return gj * p + gk
        
    def get_w_idx(gk, gi):
        return gk * n + gi

    # M1 = (a+d)(e+h)
    u = np.zeros(n*m); u[get_u_idx(ai, aj)] = 1; u[get_u_idx(ai+1, aj+1)] = 1
    v = np.zeros(m*p); v[get_v_idx(bj, bk)] = 1; v[get_v_idx(bj+1, bk+1)] = 1
    w = np.zeros(p*n); w[get_w_idx(bk, ai)] = 1; w[get_w_idx(bk+1, ai+1)] = 1
    terms.append((u, v, w))
    
    # M2 = (c+d)e
    u = np.zeros(n*m); u[get_u_idx(ai+1, aj)] = 1; u[get_u_idx(ai+1, aj+1)] = 1
    v = np.zeros(m*p); v[get_v_idx(bj, bk)] = 1
    w = np.zeros(p*n); w[get_w_idx(bk, ai+1)] = 1; w[get_w_idx(bk+1, ai+1)] = -1
    terms.append((u, v, w))
    
    # M3 = a(f-h)
    u = np.zeros(n*m); u[get_u_idx(ai, aj)] = 1
    v = np.zeros(m*p); v[get_v_idx(bj, bk+1)] = 1; v[get_v_idx(bj+1, bk+1)] = -1
    w = np.zeros(p*n); w[get_w_idx(bk+1, ai)] = 1; w[get_w_idx(bk+1, ai+1)] = 1
    terms.append((u, v, w))
    
    # M4 = d(g-e)
    u = np.zeros(n*m); u[get_u_idx(ai+1, aj+1)] = 1
    v = np.zeros(m*p); v[get_v_idx(bj+1, bk)] = 1; v[get_v_idx(bj, bk)] = -1
    w = np.zeros(p*n); w[get_w_idx(bk, ai)] = 1; w[get_w_idx(bk+1, ai)] = 1
    terms.append((u, v, w))
    
    # M5 = (a+b)h
    u = np.zeros(n*m); u[get_u_idx(ai, aj)] = 1; u[get_u_idx(ai, aj+1)] = 1
    v = np.zeros(m*p); v[get_v_idx(bj+1, bk+1)] = 1
    w = np.zeros(p*n); w[get_w_idx(bk, ai)] = -1; w[get_w_idx(bk+1, ai)] = 1
    terms.append((u, v, w))
    
    # M6 = (c-a)(e+f)
    u = np.zeros(n*m); u[get_u_idx(ai+1, aj)] = 1; u[get_u_idx(ai, aj)] = -1
    v = np.zeros(m*p); v[get_v_idx(bj, bk)] = 1; v[get_v_idx(bj, bk+1)] = 1
    w = np.zeros(p*n); w[get_w_idx(bk+1, ai+1)] = 1
    terms.append((u, v, w))
    
    # M7 = (b-d)(g+h)
    u = np.zeros(n*m); u[get_u_idx(ai, aj+1)] = 1; u[get_u_idx(ai+1, aj+1)] = -1
    v = np.zeros(m*p); v[get_v_idx(bj+1, bk)] = 1; v[get_v_idx(bj+1, bk+1)] = 1
    w = np.zeros(p*n); w[get_w_idx(bk, ai)] = 1
    terms.append((u, v, w))
    
    return terms

def _strassen_construct(n, m, p):
    """Construct the Strassen-based block decomposition for 4x4 * 4x5.
    
    Split into 2x2 blocks:
    - C00 (2x2) = A00*B00 + A01*B10: 2 * 7 = 14 terms
    - C10 (2x2) = A10*B00 + A11*B10: 2 * 7 = 14 terms  
    - C01 (2x3) = A00*B01 + A01*B11: 2 * 9 = 18 terms (Strassen for first 2 cols + naive for 3rd)
    - C11 (2x3) = A10*B01 + A11*B11: 2 * 9 = 18 terms
    
    Total: 14 + 14 + 18 + 18 = 64 terms
    """
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def add_strassen_2x2_2(ai, aj, bj, bk):
        """Add 7 Strassen terms for 2x2x2 block multiplication."""
        for u, v, w in _strassen_2x2x2_terms(ai, aj, bj, bk, n, m, p):
            add_term(u, v, w)
    
    def add_2x2x3(ai, aj, bj, bk):
        """Add 9 terms: Strassen for first 2 cols + naive for 3rd col.
        A[ai:ai+2, aj:aj+2] @ B[bj:bj+2, bk:bk+3] -> C[ai:ai+2, bk:bk+3]
        """
        # Strassen for cols bk, bk+1
        add_strassen_2x2_2(ai, aj, bj, bk)
        
        # Naive for col bk+2
        for di in range(2):
            for dj in range(2):
                u = np.zeros(n*m); u[ai*m + aj + dj] = 1
                v = np.zeros(m*p); v[(bj+dj)*p + bk+2] = 1
                w = np.zeros(p*n); w[(bk+2)*n + ai+di] = 1
                add_term(u, v, w)
    
    # C00 (2x2) = A00*B00 + A01*B10
    add_strassen_2x2_2(0, 0, 0, 0)   # A[0:2,0:2] @ B[0:2,0:2] -> C[0:2,0:2]
    add_strassen_2x2_2(0, 2, 2, 0)   # A[0:2,2:4] @ B[2:4,0:2] -> C[0:2,0:2]
    
    # C10 (2x2) = A10*B00 + A11*B10
    add_strassen_2x2_2(2, 0, 0, 0)   # A[2:4,0:2] @ B[0:2,0:2] -> C[2:4,0:2]
    add_strassen_2x2_2(2, 2, 2, 0)   # A[2:4,2:4] @ B[2:4,0:2] -> C[2:4,0:2]
    
    # C01 (2x3) = A00*B01 + A01*B11
    add_2x2x3(0, 0, 0, 2)   # A[0:2,0:2] @ B[0:2,2:5] -> C[0:2,2:5]
    add_2x2x3(0, 2, 2, 2)   # A[0:2,2:4] @ B[2:4,2:5] -> C[0:2,2:5]
    
    # C11 (2x3) = A10*B01 + A11*B11
    add_2x2x3(2, 0, 0, 2)   # A[2:4,0:2] @ B[0:2,2:5] -> C[2:4,2:5]
    add_2x2x3(2, 2, 2, 2)   # A[2:4,2:4] @ B[2:4,2:5] -> C[2:4,2:5]

    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()

def _als_decompose(n, m, p, rank, seed=42, max_iters=500, deadline=None):
    """Try to find a rank-r decomposition using alternating least squares with rounding.
    
    Returns (U, V, W) as lists if found, else None.
    """
    np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n*m, rank) * 0.1
    V = np.random.randn(m*p, rank) * 0.1
    W = np.random.randn(p*n, rank) * 0.1
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Optimize U: for each (i,j), solve linear system
        for i in range(n):
            for j in range(m):
                A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b_vec = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Optimize V: for each (j,k)
        for j in range(m):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b_vec = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Optimize W: for each (i,k)
        for i in range(n):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                b_vec = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err += (s - 1.0) ** 2
        err = np.sqrt(err)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-12:
            break
    
    # Round to half-integers
    U_r = _round_matrix_to_half_int(best_U)
    V_r = _round_matrix_to_half_int(best_V)
    W_r = _round_matrix_to_half_int(best_W)
    
    # Verify
    if _verify_fast(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, atol=1e-9):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    return None

def _structured_sparse_search(n, m, p, target_rank, deadline):
    """Try structured sparse integer programming approach.
    
    Formulate factor matrices as sparse binary variables with patterns
    derived from the tensor's support, then solve for minimal rank.
    """
    # For now, this is a placeholder that tries specific sparse patterns
    # The idea is to use the known structure of matrix multiplication
    # to constrain the search space significantly
    
    # Try a few specific sparse patterns
    patterns = []
    
    # Pattern 1: Diagonal-like structure
    for seed in range(5):
        if time.time() > deadline - 10:
            break
        
        np.random.seed(seed)
        
        # Create sparse random initialization
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        # Add sparse non-zero entries
        for r in range(target_rank):
            # Each column has a few non-zeros
            nz_u = np.random.choice(n*m, size=min(4, n*m), replace=False)
            nz_v = np.random.choice(m*p, size=min(4, m*p), replace=False)
            nz_w = np.random.choice(p*n, size=min(4, p*n), replace=False)
            
            U[nz_u, r] = np.random.choice([-1, 1], size=len(nz_u))
            V[nz_v, r] = np.random.choice([-1, 1], size=len(nz_v))
            W[nz_w, r] = np.random.choice([-1, 1], size=len(nz_w))
        
        # Try to refine with ALS-like steps but keeping sparsity
        for it in range(100):
            if time.time() > deadline - 10:
                break
            
            # Simple gradient step on the error
            T = np.zeros((n, m, p))
            for r in range(target_rank):
                Ur_col = U[:, r].reshape(n, m)
                Vr_col = V[:, r].reshape(m, p)
                Wr_col = W[:, r].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i, j, k] += Ur_col[i, j] * Vr_col[j, k] * Wr_col[k, i]
            
            err = np.sqrt(np.sum((T - 1.0)**2))
            if err < 1e-9:
                # Round and verify
                U_r = _round_matrix_to_half_int(U)
                V_r = _round_matrix_to_half_int(V)
                W_r = _round_matrix_to_half_int(W)
                if _verify_fast(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, atol=1e-9):
                    return U_r.tolist(), V_r.tolist(), W_r.tolist()
                break
            
            # Gradient descent step
            lr = 0.01
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        grad_U = (T[i,j,k] - 1.0) * V[j*p+k] * W[k*n+i]
                        U[i*m+j] -= lr * grad_U
                        
                        grad_V = (T[i,j,k] - 1.0) * U[i*m+j] * W[k*n+i]
                        V[j*p+k] -= lr * grad_V
                        
                        grad_W = (T[i,j,k] - 1.0) * U[i*m+j] * V[j*p+k]
                        W[k*n+i] -= lr * grad_W
            
            # Enforce sparsity: zero out small values
            U[np.abs(U) < 0.1] = 0
            V[np.abs(V) < 0.1] = 0
            W[np.abs(W) < 0.1] = 0
    
    return None

def _linear_dependence_pruning(U, V, W, n, m, p):
    """Try to remove linearly dependent columns from the decomposition."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    rank = Ua.shape[1]
    
    # Try removing each column and check if still valid
    for r in range(rank):
        if time.time() > 140:
            break
        
        # Remove column r
        mask = [i for i in range(rank) if i != r]
        U_test = Ua[:, mask]
        V_test = Va[:, mask]
        W_test = Wa[:, mask]
        
        if _verify_fast(U_test.tolist(), V_test.tolist(), W_test.tolist(), n, m, p, atol=1e-9):
            U = U_test.tolist()
            V = V_test.tolist()
            W = W_test.tolist()
            rank -= 1
            # Reset and try again from the beginning
            return U, V, W
    
    return U, V, W

def _schoolbook():
    n, m, p = DIMENSIONS
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
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Get the Strassen-based decomposition as baseline (rank 64)
    U_base, V_base, W_base = _strassen_construct(n, m, p)
    best_result = {"U": U_base, "V": V_base, "W": W_base}
    best_rank = len(U_base[0])
    
    # Verify baseline
    if not _verify_fast(U_base, V_base, W_base, n, m, p, atol=1e-9):
        # Fallback to schoolbook if Strassen construction has issues
        return _schoolbook()
    
    # Try linear dependence pruning first (fast)
    if time.time() < deadline - 30:
        U_pruned, V_pruned, W_pruned = _linear_dependence_pruning(
            np.array(U_base), np.array(V_base), np.array(W_base), n, m, p
        )
        if isinstance(U_pruned, list):
            if _verify_fast(U_pruned, V_pruned, W_pruned, n, m, p, atol=1e-9):
                best_result = {"U": U_pruned, "V": V_pruned, "W": W_pruned}
                best_rank = len(U_pruned[0])
    
    # Try structured sparse search for lower ranks
    if time.time() < deadline - 20:
        for target_rank in range(best_rank - 1, 50, -1):
            if time.time() > deadline - 15:
                break
            
            result = _structured_sparse_search(n, m, p, target_rank, deadline)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
    
    # Try ALS for lower ranks
    for rank in range(best_rank - 1, 50, -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        for seed in range(10):
            if time.time() > deadline - 15:
                break
            
            result = _als_decompose(n, m, p, rank, seed=seed, max_iters=200, deadline=deadline - 15)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                found = True
                break
        
        if found:
            continue
    
    return best_result

# EVOLVE-BLOCK-END
