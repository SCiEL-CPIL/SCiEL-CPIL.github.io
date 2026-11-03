"""Search for a rank-minimal exact tensor decomposition of 4x4 * 4x5 matrix multiplication.
Uses Kronecker-embedded Strassen (rank 65) as baseline, then attempts refinement via
improved ALS with better convergence, syndrome-based parity-check initialization,
and high-precision polish passes. Falls back to schoolbook (rank 80) if all else fails."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify_fast(U, V, W, n=4, m=4, p=5, atol=1e-9):
    """Fast verification using numpy with float64, checking tolerance."""
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

def _compute_error(U, V, W, n, m, p, rank):
    """Compute the L2 error of the decomposition efficiently."""
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = Ur[:, :, c]
        vc = Vr[:, :, c]
        wc = Wr[:, :, c]
        wcT = wc.T
        for j in range(m):
            T[:, j, :] = T[:, j, :] + np.outer(uc[:, j], vc[j, :]) * wcT
    return np.sqrt(np.sum((T - 1.0)**2))

def _round_to_half_integer(x):
    """Round x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _round_matrix_to_half_int(M):
    """Round all entries of matrix M to nearest half-integer."""
    return np.array([[_round_to_half_integer(x) for x in row] for row in M])

def _strassen_2x2x2_terms():
    """Return the 7 Strassen 2x2x2 terms as sparse (index, value) lists for U, V, W."""
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
    """Construct a rank-65 decomposition using Kronecker product embedding.
    4x4x4 via Kronecker Strassen (rank 49) + 4x4x1 naive (rank 16)."""
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
    
    # Naive terms for 5th column
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

def _als_decompose(n, m, p, rank, seed=42, max_iters=500, deadline=None,
                   U_init=None, V_init=None, W_init=None, polish_iters=100):
    """ALS with better convergence: damped updates, multiple rounding strategies,
    and high-precision polish."""
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
    
    def _als_iteration(damping=1.0):
        """One full ALS iteration with optional damping."""
        for i in range(n):
            for j in range(m):
                A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b_vec = np.ones(p)
                try:
                    sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    U[i*m+j, :] = (1-damping)*U[i*m+j, :] + damping*sol
                except:
                    pass
        
        for j in range(m):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b_vec = np.ones(n)
                try:
                    sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    V[j*p+k, :] = (1-damping)*V[j*p+k, :] + damping*sol
                except:
                    pass
        
        for i in range(n):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                b_vec = np.ones(m)
                try:
                    sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    W[k*n+i, :] = (1-damping)*W[k*n+i, :] + damping*sol
                except:
                    pass
    
    # Phase 1: full-strength ALS
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        _als_iteration(damping=1.0)
        err = _compute_error(U, V, W, n, m, p, rank)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < 1e-12:
            break
    
    # Phase 2: damped polish for finer convergence
    for it in range(polish_iters):
        if deadline is not None and time.time() > deadline:
            break
        _als_iteration(damping=0.5)
        err = _compute_error(U, V, W, n, m, p, rank)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < 1e-14:
            break
    
    # Phase 3: high-precision polish with very small damping
    for it in range(50):
        if deadline is not None and time.time() > deadline:
            break
        _als_iteration(damping=0.1)
        err = _compute_error(U, V, W, n, m, p, rank)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < 1e-15:
            break
    
    # Try multiple rounding strategies
    candidates = []
    
    # Half-integer rounding
    U_r1 = _round_matrix_to_half_int(best_U)
    V_r1 = _round_matrix_to_half_int(best_V)
    W_r1 = _round_matrix_to_half_int(best_W)
    candidates.append((U_r1, V_r1, W_r1))
    
    # Integer rounding
    U_r2 = np.round(best_U)
    V_r2 = np.round(best_V)
    W_r2 = np.round(best_W)
    candidates.append((U_r2, V_r2, W_r2))
    
    # Offset half-integer rounding
    for offset in [0.125, -0.125, 0.25, -0.25, 0.375, -0.375]:
        U_r3 = _round_matrix_to_half_int(best_U + offset)
        V_r3 = _round_matrix_to_half_int(best_V + offset)
        W_r3 = _round_matrix_to_half_int(best_W + offset)
        candidates.append((U_r3, V_r3, W_r3))
    
    # Verify each candidate
    for Uc, Vc, Wc in candidates:
        if _verify_fast(Uc.tolist(), Vc.tolist(), Wc.tolist(), n, m, p, atol=1e-9):
            return Uc.tolist(), Vc.tolist(), Wc.tolist()
    
    # Final attempt: re-ALS from rounded values
    U_f = U_r1.copy()
    V_f = V_r1.copy()
    W_f = W_r1.copy()
    
    for it in range(30):
        if deadline is not None and time.time() > deadline:
            break
        _als_iteration(damping=0.3)
        U_close = np.allclose(U, np.round(U*2)/2.0, atol=1e-6)
        V_close = np.allclose(V, np.round(V*2)/2.0, atol=1e-6)
        W_close = np.allclose(W, np.round(W*2)/2.0, atol=1e-6)
        if U_close and V_close and W_close:
            U_r = _round_matrix_to_half_int(U)
            V_r = _round_matrix_to_half_int(V)
            W_r = _round_matrix_to_half_int(W)
            if _verify_fast(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, atol=1e-9):
                return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    return None

def _syndrome_init(n, m, p, rank, seed=42):
    """Syndrome-based parity-check initialization: use the structure of the
    multiplication tensor to create a more informed random initialization."""
    np.random.seed(seed)
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Assign each factor a random subset of the rank dimensions
    # with values drawn from {-1, 0, 1, 0.5, -0.5}
    for c in range(rank):
        # U: pick ~2/3 of entries nonzero
        mask_u = np.random.rand(n*m) < 0.65
        vals_u = np.random.choice([-1, -0.5, 0.5, 1], size=n*m)
        U[:, c] = np.where(mask_u, vals_u, 0)
        
        mask_v = np.random.rand(m*p) < 0.65
        vals_v = np.random.choice([-1, -0.5, 0.5, 1], size=m*p)
        V[:, c] = np.where(mask_v, vals_v, 0)
        
        mask_w = np.random.rand(p*n) < 0.65
        vals_w = np.random.choice([-1, -0.5, 0.5, 1], size=p*n)
        W[:, c] = np.where(mask_w, vals_w, 0)
    
    return U, V, W

def _try_reduce_from_baseline(U_base, V_base, W_base, n, m, p, deadline):
    """Try to reduce the rank starting from a baseline decomposition."""
    Ua = np.array(U_base, dtype=np.float64)
    Va = np.array(V_base, dtype=np.float64)
    Wa = np.array(W_base, dtype=np.float64)
    rank = Ua.shape[1]
    
    best_result = None
    best_rank = rank
    
    for target_rank in range(rank - 1, max(40, rank - 25), -1):
        if time.time() > deadline:
            break
        
        for init_seed in range(4):
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
            elif init_seed == 2:
                np.random.seed(init_seed * 100 + target_rank)
                idx = np.random.choice(rank, target_rank, replace=False)
                U_init = Ua[:, idx]
                V_init = Va[:, idx]
                W_init = Wa[:, idx]
            else:
                np.random.seed(999 + target_rank)
                idx = np.random.choice(rank, target_rank, replace=False)
                U_init = Ua[:, idx] + np.random.randn(n*m, target_rank) * 0.1
                V_init = Va[:, idx] + np.random.randn(m*p, target_rank) * 0.1
                W_init = Wa[:, idx] + np.random.randn(p*n, target_rank) * 0.1
            
            result = _als_decompose(n, m, p, target_rank, 
                                    U_init=U_init, V_init=V_init, W_init=W_init,
                                    max_iters=200, deadline=deadline, polish_iters=50)
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
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    
    # Build Kronecker-embedded Strassen baseline (rank 65)
    U_kron, V_kron, W_kron = _kron_strassen_construct(n, m, p)
    kron_ok = _verify_fast(U_kron, V_kron, W_kron, n, m, p, atol=1e-9)
    
    if not kron_ok:
        return _schoolbook()
    
    best_result = {"U": U_kron, "V": V_kron, "W": W_kron}
    best_rank = len(U_kron[0])
    
    # Try to reduce from baseline
    if time.time() < deadline - 60:
        reduced = _try_reduce_from_baseline(U_kron, V_kron, W_kron, n, m, p, deadline - 60)
        if reduced is not None:
            U_red, V_red, W_red = reduced
            best_result = {"U": U_red, "V": V_red, "W": W_red}
            best_rank = len(U_red[0])
    
    # Try random and syndrome initializations at various ranks
    for rank in range(best_rank - 1, max(45, best_rank - 30), -1):
        if time.time() > deadline - 25:
            break
        
        found = False
        
        # Standard random seeds
        for seed in range(10):
            if time.time() > deadline - 25:
                break
            result = _als_decompose(n, m, p, rank, seed=seed, max_iters=400, 
                                    deadline=deadline - 25, polish_iters=80)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                found = True
                break
        
        # Syndrome-based initialization
        if not found:
            for seed in range(5):
                if time.time() > deadline - 25:
                    break
                U_s, V_s, W_s = _syndrome_init(n, m, p, rank, seed=seed + 1000)
                result = _als_decompose(n, m, p, rank, 
                                        U_init=U_s, V_init=V_s, W_init=W_s,
                                        max_iters=300, 
                                        deadline=deadline - 25, polish_iters=80)
                if result is not None:
                    U, V, W = result
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    found = True
                    break
        
        if found:
            continue
        else:
            break
    
    return best_result
# EVOLVE-BLOCK-END
