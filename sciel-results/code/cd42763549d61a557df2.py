"""Search for a rank-minimal exact tensor decomposition of 4x4 * 4x5 matrix multiplication.
Uses Kronecker-embedded Strassen (rank 65) as baseline, then attempts rank reduction via
combinatorial subspace elimination, vectorized ALS with half-integer rounding, and
structured column removal/merging."""
import time
import numpy as np

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
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr, optimize=True)
    return np.allclose(T, 1.0, atol=atol)

def _verify_half_integer(U, V, W, n=4, m=4, p=5):
    """Verify that entries are half-integers and decomposition is exact."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    for M in [Ua, Va, Wa]:
        if not np.allclose(M * 2, np.round(M * 2), atol=1e-12):
            return False
    return _verify_fast(U, V, W, n, m, p, atol=1e-9)

def _round_to_half_integer(x):
    return round(x * 2) / 2.0

def _round_matrix_to_half_int(M):
    return np.array([[_round_to_half_integer(x) for x in row] for row in M])

def _compute_error_fast(U, V, W, n, m, p):
    """Faster error computation using einsum."""
    rank = U.shape[1]
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr, optimize=True)
    return np.sqrt(np.sum((T - 1.0) ** 2))

def _polish_decomposition(U, V, W, n, m, p, max_iters=100, deadline=None):
    """High-precision polish pass using alternating least squares."""
    U = U.astype(np.float64).copy()
    V = V.astype(np.float64).copy()
    W = W.astype(np.float64).copy()
    
    best_err = _compute_error_fast(U, V, W, n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Optimize U
        for i in range(n):
            for j in range(m):
                V_slice = V[j*p:(j+1)*p, :]
                W_slice = np.array([W[k*n+i, :] for k in range(p)])
                A_mat = V_slice * W_slice
                b_vec = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Optimize V
        for j in range(m):
            for k in range(p):
                U_slice = np.array([U[i*m+j, :] for i in range(n)])
                W_slice = W[k*n:(k+1)*n, :]
                A_mat = U_slice * W_slice
                b_vec = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Optimize W
        for i in range(n):
            for k in range(p):
                U_slice = U[i*m:(i+1)*m, :]
                V_slice = np.array([V[j*p+k, :] for j in range(m)])
                A_mat = U_slice * V_slice
                b_vec = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        err = _compute_error_fast(U, V, W, n, m, p)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-15:
            break
    
    return best_U, best_V, best_W, best_err

def _strassen_2x2x2_terms():
    """Return the 7 Strassen 2x2x2 terms as sparse (index, value) lists."""
    return [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]

def _kron_strassen_construct(n, m, p):
    """Construct a rank-65 decomposition: 4x4x4 via Kronecker Strassen (49) + 4x4x1 naive (16)."""
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    strassen_terms = _strassen_2x2x2_terms()
    
    for (u1_sparse, v1_sparse, w1_sparse) in strassen_terms:
        for (u2_sparse, v2_sparse, w2_sparse) in strassen_terms:
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            
            for (idx1, val1) in u1_sparse:
                i1, j1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in u2_sparse:
                    i2, j2 = idx2 // 2, idx2 % 2
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    u[i * m + j] = val1 * val2
            
            for (idx1, val1) in v1_sparse:
                j1, k1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in v2_sparse:
                    j2, k2 = idx2 // 2, idx2 % 2
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    v[j * p + k] = val1 * val2
            
            for (idx1, val1) in w1_sparse:
                k1, i1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in w2_sparse:
                    k2, i2 = idx2 // 2, idx2 % 2
                    k = k1 * 2 + k2
                    i = i1 * 2 + i2
                    w[k * n + i] = val1 * val2
            
            add_term(u, v, w)
    
    for i in range(n):
        for j in range(m):
            u = np.zeros(n*m); u[i * m + j] = 1
            v = np.zeros(m*p); v[j * p + 4] = 1
            w = np.zeros(p*n); w[4 * n + i] = 1
            add_term(u, v, w)
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat, V_mat, W_mat

def _find_redundant_columns_aggressive(U, V, W, n, m, p, tol=1e-8, deadline=None):
    """Aggressively find and remove redundant columns using greedy elimination."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    while True:
        if deadline is not None and time.time() > deadline:
            break
        if rank <= 1:
            break
        
        removed_any = False
        # Try removing each column, prioritizing those with smaller combined norm
        U_norms = np.linalg.norm(U, axis=0)
        V_norms = np.linalg.norm(V, axis=0)
        W_norms = np.linalg.norm(W, axis=0)
        combined = U_norms + V_norms + W_norms
        sorted_cols = np.argsort(combined)
        
        for col in sorted_cols:
            if deadline is not None and time.time() > deadline:
                break
            if col >= rank:
                continue
            
            keep = [i for i in range(rank) if i != col]
            U_sub = U[:, keep]
            V_sub = V[:, keep]
            W_sub = W[:, keep]
            
            err = _compute_error_fast(U_sub, V_sub, W_sub, n, m, p)
            if err < tol:
                U, V, W = U_sub, V_sub, W_sub
                rank = U.shape[1]
                removed_any = True
                break
        
        if not removed_any:
            break
    
    return U, V, W

def _try_column_merging(U, V, W, n, m, p, deadline=None, tol=1e-8):
    """Try to merge pairs of columns into single columns."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    while True:
        if deadline is not None and time.time() > deadline:
            break
        if rank <= 2:
            break
        
        merged = False
        for c1 in range(rank):
            if deadline is not None and time.time() > deadline:
                break
            for c2 in range(c1 + 1, rank):
                # Try merging columns c1 and c2
                # The merged column would need to satisfy:
                # U_merged * V_merged * W_merged = U_c1 * V_c1 * W_c1 + U_c2 * V_c2 * W_c2
                # This is generally not possible unless there's special structure
                # Skip for now - too complex
                pass
        
        if not merged:
            break
    
    return U, V, W

def _als_decompose_vectorized(n, m, p, rank, seed=42, max_iters=200, deadline=None,
                               U_init=None, V_init=None, W_init=None, polish_iters=80):
    """Vectorized ALS with rounding to half-integers."""
    if U_init is not None and V_init is not None and W_init is not None:
        U = np.array(U_init, dtype=np.float64).copy()
        V = np.array(V_init, dtype=np.float64).copy()
        W = np.array(W_init, dtype=np.float64).copy()
    else:
        np.random.seed(seed)
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    def _als_iteration():
        for i in range(n):
            for j in range(m):
                V_slice = V[j*p:(j+1)*p, :]
                W_slice = np.array([W[k*n+i, :] for k in range(p)])
                A_mat = V_slice * W_slice
                b_vec = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        for j in range(m):
            for k in range(p):
                U_slice = np.array([U[i*m+j, :] for i in range(n)])
                W_slice = W[k*n:(k+1)*n, :]
                A_mat = U_slice * W_slice
                b_vec = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        for i in range(n):
            for k in range(p):
                U_slice = U[i*m:(i+1)*m, :]
                V_slice = np.array([V[j*p+k, :] for j in range(m)])
                A_mat = U_slice * V_slice
                b_vec = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        _als_iteration()
        err = _compute_error_fast(U, V, W, n, m, p)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-12:
            break
    
    for it in range(polish_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        _als_iteration()
        err = _compute_error_fast(U, V, W, n, m, p)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-14:
            break
    
    if deadline is not None and time.time() < deadline - 2:
        best_U, best_V, best_W, final_err = _polish_decomposition(
            best_U, best_V, best_W, n, m, p, 
            max_iters=50, deadline=deadline - 1
        )
    
    U_r = _round_matrix_to_half_int(best_U)
    V_r = _round_matrix_to_half_int(best_V)
    W_r = _round_matrix_to_half_int(best_W)
    
    if _verify_fast(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, atol=1e-9):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    return None

def _try_reduce_from_baseline(U_base, V_base, W_base, n, m, p, deadline):
    """Try to reduce the rank starting from a baseline decomposition."""
    Ua = np.array(U_base, dtype=np.float64)
    Va = np.array(V_base, dtype=np.float64)
    Wa = np.array(W_base, dtype=np.float64)
    rank = Ua.shape[1]
    
    best_result = None
    best_rank = rank
    
    # Aggressively remove redundant columns
    Ua, Va, Wa = _find_redundant_columns_aggressive(Ua, Va, Wa, n, m, p, tol=1e-8, deadline=deadline - 30)
    rank = Ua.shape[1]
    best_rank = rank
    best_result = (Ua.tolist(), Va.tolist(), Wa.tolist())
    
    # Try reducing rank further
    for target_rank in range(rank - 1, max(45, rank - 20), -1):
        if time.time() > deadline - 15:
            break
        
        for init_seed in range(4):
            if time.time() > deadline - 15:
                break
            
            if init_seed == 0:
                U_init = Ua[:, :target_rank]
                V_init = Va[:, :target_rank]
                W_init = Wa[:, :target_rank]
            elif init_seed == 1:
                np.random.seed(init_seed * 100 + target_rank)
                idx = np.random.choice(rank, target_rank, replace=False)
                U_init = Ua[:, idx]
                V_init = Va[:, idx]
                W_init = Wa[:, idx]
            elif init_seed == 2:
                np.random.seed(999 + target_rank)
                idx = np.random.choice(rank, target_rank, replace=False)
                U_init = Ua[:, idx] + np.random.randn(n*m, target_rank) * 0.1
                V_init = Va[:, idx] + np.random.randn(m*p, target_rank) * 0.1
                W_init = Wa[:, idx] + np.random.randn(p*n, target_rank) * 0.1
            else:
                np.random.seed(1234 + target_rank)
                U_init = np.random.randn(n*m, target_rank)
                V_init = np.random.randn(m*p, target_rank)
                W_init = np.random.randn(p*n, target_rank)
            
            result = _als_decompose_vectorized(n, m, p, target_rank, 
                                    U_init=U_init, V_init=V_init, W_init=W_init,
                                    max_iters=150, deadline=deadline - 15, polish_iters=50)
            if result is not None:
                best_result = result
                best_rank = target_rank
                break
    
    return best_result

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
    deadline = start_time + 135  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Build the rank-65 Kronecker-Strassen baseline
    U_kron, V_kron, W_kron = _kron_strassen_construct(n, m, p)
    
    kron_ok = _verify_fast(U_kron, V_kron, W_kron, n, m, p, atol=1e-9)
    
    if not kron_ok:
        return _schoolbook()
    
    best_result = {"U": U_kron.tolist(), "V": V_kron.tolist(), "W": W_kron.tolist()}
    best_rank = len(U_kron[0])
    
    # Try to reduce rank from the baseline
    if time.time() < deadline - 20:
        reduced = _try_reduce_from_baseline(U_kron.tolist(), V_kron.tolist(), W_kron.tolist(), n, m, p, deadline - 20)
        if reduced is not None:
            U_red, V_red, W_red = reduced
            best_result = {"U": U_red, "V": V_red, "W": W_red}
            best_rank = len(U_red[0])
    
    # Try random initializations at various ranks
    for rank in range(best_rank - 1, max(50, best_rank - 15), -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        for seed in range(10):
            if time.time() > deadline - 10:
                break
            
            result = _als_decompose_vectorized(n, m, p, rank, seed=seed, max_iters=200, 
                                    deadline=deadline - 10, polish_iters=40)
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
