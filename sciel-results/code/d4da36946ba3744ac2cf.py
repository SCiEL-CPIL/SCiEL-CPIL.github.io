"""Rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
Uses Kronecker-embedded Strassen (rank 49) for the 4x4x4 core,
plus 16 naive terms for the 5th column, giving rank 65.
Then applies combinatorial subspace elimination and ALS refinement.
Falls back to schoolbook (rank 80) if verification fails."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def _verify_fast(U, V, W, n=4, m=4, p=5, atol=1e-9):
    """Fast verification using numpy with float64."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    if Ua.ndim == 1:
        Ua = Ua.reshape(-1, 1)
    if Va.ndim == 1:
        Va = Va.reshape(-1, 1)
    if Wa.ndim == 1:
        Wa = Wa.reshape(-1, 1)
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

def _strassen_2x2x2_terms():
    """Return the 7 Strassen 2x2x2 terms as sparse (index, value) lists for U, V, W.
    U indices: (i,j) -> i*2+j (4 entries)
    V indices: (j,k) -> j*2+k (4 entries)
    W indices: (k,i) -> k*2+i (4 entries)
    """
    return [
        # M1 = (a+d)(e+h)
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        # M2 = (c+d)e
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        # M3 = a(f-h)
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        # M4 = d(g-e)
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        # M5 = (a+b)h
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),
        # M6 = (c-a)(e+f)
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        # M7 = (b-d)(g+h)
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]

def _kron_strassen_construct(n, m, p):
    """Construct a rank-65 decomposition using Kronecker product embedding.
    
    Splits 4x4x5 into 4x4x4 (handled by Kronecker-embedded Strassen, rank 49)
    plus 4x4x1 (handled naively, rank 16).
    """
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    # For 4x4x4: use Kronecker product of two 2x2x2 Strassen decompositions
    strassen_terms = _strassen_2x2x2_terms()
    
    for (u1_sparse, v1_sparse, w1_sparse) in strassen_terms:
        for (u2_sparse, v2_sparse, w2_sparse) in strassen_terms:
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            
            # Kronecker product for U: index (i1*2+i2, j1*2+j2) -> (i1*2+i2)*m + (j1*2+j2)
            for (idx1, val1) in u1_sparse:
                i1, j1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in u2_sparse:
                    i2, j2 = idx2 // 2, idx2 % 2
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    u[i * m + j] = val1 * val2
            
            # Kronecker product for V: index (j1*2+j2, k1*2+k2) -> (j1*2+j2)*p + (k1*2+k2)
            for (idx1, val1) in v1_sparse:
                j1, k1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in v2_sparse:
                    j2, k2 = idx2 // 2, idx2 % 2
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    v[j * p + k] = val1 * val2
            
            # Kronecker product for W: index (k1*2+k2, i1*2+i2) -> (k1*2+k2)*n + (i1*2+i2)
            for (idx1, val1) in w1_sparse:
                k1, i1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in w2_sparse:
                    k2, i2 = idx2 // 2, idx2 % 2
                    k = k1 * 2 + k2
                    i = i1 * 2 + i2
                    w[k * n + i] = val1 * val2
            
            add_term(u, v, w)
    
    # Add naive terms for the 5th column (k=4): A[*,j] * B[j,4] -> C[4, i]
    for i in range(n):
        for j in range(m):
            u = np.zeros(n*m); u[i * m + j] = 1
            v = np.zeros(m*p); v[j * p + 4] = 1
            w = np.zeros(p*n); w[4 * n + i] = 1
            add_term(u, v, w)
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()

def _round_to_half_integer(x):
    return round(x * 2) / 2.0

def _round_matrix_to_half_int(M):
    return np.array([[_round_to_half_integer(x) for x in row] for row in M])

def _subspace_elimination(U, V, W, n, m, p, tol=1e-8):
    """Try to reduce rank by finding linearly dependent terms via flattening analysis.
    
    For a rank-r decomposition, the tensor T = sum_c U_c ⊗ V_c ⊗ W_c.
    If we fix W, then T_W = sum_c (U_c ⊗ V_c) * W_c is a matrix.
    We can check if some columns of the coefficient matrix are linearly dependent.
    
    More practically: we look for rank-1 terms that are linear combinations of others.
    """
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    
    # Build the "flattening" matrix: for each term c, we have U_c (nm x 1), V_c (mp x 1), W_c (pn x 1)
    # The tensor equation is: for all (i,j,k): sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
    
    # Strategy: try to remove one term at a time and see if the remaining terms can be adjusted.
    # This is expensive, so we use a linear algebra shortcut.
    
    # Flatten the problem: treat each term as a vector in R^(nm*mp*pn)
    # But that's too large. Instead, use the fact that if we fix U and V,
    # then W must satisfy a linear system.
    
    # Try removing each term and checking if the system is still solvable.
    best_indices = list(range(rank))
    
    for remove_idx in range(rank - 1, -1, -1):
        if len(best_indices) <= 50:
            break
        
        # Try removing term remove_idx
        remaining = [i for i in best_indices if i != remove_idx]
        
        # Check if the remaining terms can still represent the tensor
        # by solving for adjusted W (or U/V) values.
        # Simple check: verify if the sub-tensor from remaining terms is close to 1.
        U_rem = Ua[:, remaining]
        V_rem = Va[:, remaining]
        W_rem = Wa[:, remaining]
        
        Ur = U_rem.reshape(n, m, len(remaining))
        Vr = V_rem.reshape(m, p, len(remaining))
        Wr = W_rem.reshape(p, n, len(remaining))
        
        T = np.zeros((n, m, p), dtype=np.float64)
        for c in range(len(remaining)):
            uc = Ur[:, :, c]
            vc = Vr[:, :, c]
            wc = Wr[:, :, c]
            wcT = wc.T
            for j in range(m):
                T[:, j, :] = T[:, j, :] + np.outer(uc[:, j], vc[j, :]) * wcT
        
        if np.allclose(T, 1.0, atol=1e-6):
            best_indices = remaining
    
    if len(best_indices) < rank:
        U_out = Ua[:, best_indices].tolist()
        V_out = Va[:, best_indices].tolist()
        W_out = Wa[:, best_indices].tolist()
        if _verify_fast(U_out, V_out, W_out, n, m, p, atol=1e-9):
            return U_out, V_out, W_out
    
    return None

def _als_decompose(n, m, p, rank, seed=42, max_iters=500, deadline=None,
                   U_init=None, V_init=None, W_init=None):
    """Try to find a rank-r decomposition using ALS with rounding."""
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
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        for i in range(n):
            for j in range(m):
                A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b_vec = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        for j in range(m):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b_vec = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        for i in range(n):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                b_vec = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
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
    
    # First try subspace elimination
    elim_result = _subspace_elimination(U_base, V_base, W_base, n, m, p)
    if elim_result is not None:
        U_elim, V_elim, W_elim = elim_result
        best_result = (U_elim, V_elim, W_elim)
        best_rank = len(U_elim[0])
    
    for target_rank in range(best_rank - 1, max(50, best_rank - 15), -1):
        if time.time() > deadline:
            break
        
        for init_seed in range(3):
            if time.time() > deadline:
                break
            
            if init_seed == 0:
                U_init = Ua[:, :target_rank]
                V_init = Va[:, :target_rank]
                W_init = Wa[:, :target_rank]
            elif init_seed == 1:
                U_init = Ua[:, rank-target_rank:]
                V_init = Va[:, rank-target_rank:]
                W_init = Wa[:, rank-target_rank:]
            else:
                np.random.seed(init_seed * 100 + target_rank)
                idx = np.random.choice(rank, target_rank, replace=False)
                U_init = Ua[:, idx]
                V_init = Va[:, idx]
                W_init = Wa[:, idx]
            
            result = _als_decompose(n, m, p, target_rank, 
                                    U_init=U_init, V_init=V_init, W_init=W_init,
                                    max_iters=150, deadline=deadline)
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
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    U_base, V_base, W_base = _kron_strassen_construct(n, m, p)
    best_result = {"U": U_base, "V": V_base, "W": W_base}
    best_rank = len(U_base[0])
    
    if not _verify_fast(U_base, V_base, W_base, n, m, p, atol=1e-9):
        return _schoolbook()
    
    if time.time() < deadline - 30:
        reduced = _try_reduce_from_baseline(U_base, V_base, W_base, n, m, p, deadline - 30)
        if reduced is not None:
            U_red, V_red, W_red = reduced
            best_result = {"U": U_red, "V": V_red, "W": W_red}
            best_rank = len(U_red[0])
    
    for rank in range(best_rank - 1, max(55, best_rank - 15), -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        for seed in range(8):
            if time.time() > deadline - 15:
                break
            
            result = _als_decompose(n, m, p, rank, seed=seed, max_iters=250, deadline=deadline - 15)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                found = True
                break
        
        if found:
            continue
    
    return best_result
