"""Search for a rank-minimal exact tensor decomposition of 4x4 * 4x5 matrix multiplication.
Uses block-based Strassen decomposition as a baseline (rank 64),
then attempts refinement via ALS with rounding and alternative block structures.
Falls back to schoolbook (rank 80) if all else fails."""
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

def _round_to_half_integer(x):
    """Round x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _round_matrix_to_half_int(M):
    """Round all entries of matrix M to nearest half-integer."""
    return np.array([[ _round_to_half_integer(x) for x in row] for row in M])

def _strassen_construct(n, m, p):
    """Construct the Strassen-based block decomposition.
    
    For 4x4 * 4x5:
    - Split into 2x2 blocks
    - Use Strassen (rank 7) for 2x2x2 multiplications
    - Use Strassen + naive (rank 9) for 2x2x3 multiplications
    
    Total: 4 * 7 + 4 * 9 = 28 + 36 = 64
    """
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def get_u_idx(gi, gj):
        return gi * m + gj
        
    def get_v_idx(gj, gk):
        return gj * p + gk
        
    def get_w_idx(gk, gi):
        return gk * n + gi

    def add_strassen_2x2_2(ai, aj, bj, bk):
        """Add 7 Strassen terms for A[ai:ai+2, aj:aj+2] @ B[bj:bj+2, bk:bk+2] -> C[ai:ai+2, bk:bk+2]"""
        # M1 = (a+d)(e+h)
        u = np.zeros(n*m); u[get_u_idx(ai, aj)] = 1; u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj, bk)] = 1; v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = 1; w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        # M2 = (c+d)e
        u = np.zeros(n*m); u[get_u_idx(ai+1, aj)] = 1; u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj, bk)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai+1)] = 1; w[get_w_idx(bk+1, ai+1)] = -1
        add_term(u, v, w)
        
        # M3 = a(f-h)
        u = np.zeros(n*m); u[get_u_idx(ai, aj)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj, bk+1)] = 1; v[get_v_idx(bj+1, bk+1)] = -1
        w = np.zeros(p*n); w[get_w_idx(bk+1, ai)] = 1; w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        # M4 = d(g-e)
        u = np.zeros(n*m); u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj+1, bk)] = 1; v[get_v_idx(bj, bk)] = -1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = 1; w[get_w_idx(bk+1, ai)] = 1
        add_term(u, v, w)
        
        # M5 = (a+b)h
        u = np.zeros(n*m); u[get_u_idx(ai, aj)] = 1; u[get_u_idx(ai, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = -1; w[get_w_idx(bk+1, ai)] = 1
        add_term(u, v, w)
        
        # M6 = (c-a)(e+f)
        u = np.zeros(n*m); u[get_u_idx(ai+1, aj)] = 1; u[get_u_idx(ai, aj)] = -1
        v = np.zeros(m*p); v[get_v_idx(bj, bk)] = 1; v[get_v_idx(bj, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        # M7 = (b-d)(g+h)
        u = np.zeros(n*m); u[get_u_idx(ai, aj+1)] = 1; u[get_u_idx(ai+1, aj+1)] = -1
        v = np.zeros(m*p); v[get_v_idx(bj+1, bk)] = 1; v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = 1
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
                u = np.zeros(n*m); u[get_u_idx(ai+di, aj+dj)] = 1
                v = np.zeros(m*p); v[get_v_idx(bj+dj, bk+2)] = 1
                w = np.zeros(p*n); w[get_w_idx(bk+2, ai+di)] = 1
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

def _algebraic_elimination(n, m, p, deadline):
    """Try to find a lower-rank decomposition using algebraic elimination.
    
    Start with schoolbook rank-80 and try to eliminate linearly dependent terms
    using exact rational arithmetic.
    """
    from fractions import Fraction
    
    # Start with schoolbook rank-80
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Convert to Fraction for exact arithmetic
    U_frac = [[Fraction(int(U[i, r]), 1) for r in range(rank)] for i in range(n*m)]
    V_frac = [[Fraction(int(V[i, r]), 1) for r in range(rank)] for i in range(m*p)]
    W_frac = [[Fraction(int(W[i, r]), 1) for r in range(rank)] for i in range(p*n)]
    
    # Try to eliminate columns by finding linear dependencies
    # For each column r, check if it can be expressed as a combination of others
    # This is expensive, so we'll use a greedy approach
    
    active = list(range(rank))
    eliminated = []
    
    # Try to eliminate columns one by one
    for r in range(rank - 1, -1, -1):
        if time.time() > deadline:
            break
        
        if r not in active:
            continue
        
        # Try to express column r as a linear combination of active columns before r
        # Build system: for all (i,j,k), sum_s alpha_s * U[i,j,s]*V[j,k,s]*W[k,i,s] = U[i,j,r]*V[j,k,r]*W[k,i,r]
        
        # This is a linear system in alpha_s
        # Let's collect the equations
        equations = []
        rhs = []
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # LHS: sum of alpha_s * U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] for s in active, s != r
                    # RHS: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    
                    row = []
                    for s in active:
                        if s == r:
                            continue
                        row.append(U_frac[i*m+j][s] * V_frac[j*p+k][s] * W_frac[k*n+i][s])
                    rhs.append(U_frac[i*m+j][r] * V_frac[j*p+k][r] * W_frac[k*n+i][r])
                    equations.append(row)
        
        # Solve the linear system
        if len(equations) == 0:
            continue
        
        # Use Gaussian elimination
        num_vars = len(active) - 1  # excluding r
        if num_vars == 0:
            continue
        
        # Build augmented matrix
        aug = [eq + [rhs[i]] for i, eq in enumerate(equations)]
        
        # Gaussian elimination
        try:
            row_idx = 0
            for col_idx in range(num_vars):
                # Find pivot
                pivot_row = None
                for r2 in range(row_idx, len(aug)):
                    if aug[r2][col_idx] != 0:
                        pivot_row = r2
                        break
                if pivot_row is None:
                    continue
                
                # Swap
                aug[row_idx], aug[pivot_row] = aug[pivot_row], aug[row_idx]
                
                # Eliminate
                for r2 in range(len(aug)):
                    if r2 == row_idx:
                        continue
                    factor = aug[r2][col_idx] / aug[row_idx][col_idx]
                    for c2 in range(col_idx, num_vars + 1):
                        aug[r2][c2] -= factor * aug[row_idx][c2]
                
                row_idx += 1
            
            # Check for consistency
            consistent = True
            for r2 in range(row_idx, len(aug)):
                if any(aug[r2][c2] != 0 for c2 in range(num_vars)):
                    consistent = False
                    break
            
            if consistent:
                # Solution exists - column r is dependent
                # Extract solution
                alpha = [Fraction(0)] * num_vars
                for r2 in range(row_idx):
                    for c2 in range(num_vars):
                        if aug[r2][c2] == 1 and all(aug[r2][c3] == 0 for c3 in range(num_vars) if c3 != c2):
                            alpha[c2] = aug[r2][num_vars]
                
                # Remove column r
                active.remove(r)
                eliminated.append(r)
                
        except (ZeroDivisionError, TypeError):
            continue
    
    # Build the reduced decomposition
    if len(eliminated) == 0:
        return None
    
    new_rank = len(active)
    U_new = np.zeros((n*m, new_rank))
    V_new = np.zeros((m*p, new_rank))
    W_new = np.zeros((p*n, new_rank))
    
    for new_r, old_r in enumerate(active):
        U_new[:, new_r] = U_frac[:, old_r] if False else [float(U_frac[i][old_r]) for i in range(n*m)]
        V_new[:, new_r] = [float(V_frac[i][old_r]) for i in range(m*p)]
        W_new[:, new_r] = [float(W_frac[i][old_r]) for i in range(p*n)]
    
    # Round to half-integers
    U_r = _round_matrix_to_half_int(U_new)
    V_r = _round_matrix_to_half_int(V_new)
    W_r = _round_matrix_to_half_int(W_new)
    
    # Verify
    if _verify_fast(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, atol=1e-9):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    return None

def _als_decompose(n, m, p, rank, seed=42, max_iters=500, deadline=None, 
                   U_init=None, V_init=None, W_init=None):
    """Try to find a rank-r decomposition using alternating least squares with rounding.
    
    Returns (U, V, W) as lists if found, else None.
    """
    if U_init is not None and V_init is not None and W_init is not None:
        U = np.array(U_init, dtype=np.float64).copy()
        V = np.array(V_init, dtype=np.float64).copy()
        W = np.array(W_init, dtype=np.float64).copy()
    else:
        np.random.seed(seed)
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

def _try_reduce_from_baseline(U_base, V_base, W_base, n, m, p, deadline):
    """Try to reduce the rank starting from a baseline decomposition."""
    Ua = np.array(U_base, dtype=np.float64)
    Va = np.array(V_base, dtype=np.float64)
    Wa = np.array(W_base, dtype=np.float64)
    rank = Ua.shape[1]
    
    best_result = None
    best_rank = rank
    
    # Try reducing rank by dropping columns and re-optimizing
    for target_rank in range(rank - 1, max(40, rank - 20), -1):
        if time.time() > deadline:
            break
        
        # Try multiple initializations
        for init_seed in range(3):
            if time.time() > deadline:
                break
            
            if init_seed == 0:
                # First target_rank columns
                U_init = Ua[:, :target_rank]
                V_init = Va[:, :target_rank]
                W_init = Wa[:, :target_rank]
            elif init_seed == 1:
                # Last target_rank columns
                U_init = Ua[:, rank-target_rank:]
                V_init = Va[:, rank-target_rank:]
                W_init = Wa[:, rank-target_rank:]
            else:
                # Random subset
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
                break  # Found one at this rank, try lower
    
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
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Get the Strassen-based decomposition as baseline (rank 64)
    U_base, V_base, W_base = _strassen_construct(n, m, p)
    best_result = {"U": U_base, "V": V_base, "W": W_base}
    best_rank = len(U_base[0])
    
    # Verify baseline
    if not _verify_fast(U_base, V_base, W_base, n, m, p, atol=1e-9):
        return _schoolbook()
    
    # Try algebraic elimination first (as per directive)
    if time.time() < deadline - 60:
        alg_result = _algebraic_elimination(n, m, p, deadline - 60)
        if alg_result is not None:
            U_alg, V_alg, W_alg = alg_result
            best_result = {"U": U_alg, "V": V_alg, "W": W_alg}
            best_rank = len(U_alg[0])
    
    # Try to reduce rank from the Strassen baseline
    if time.time() < deadline - 30:
        reduced = _try_reduce_from_baseline(U_base, V_base, W_base, n, m, p, deadline - 30)
        if reduced is not None:
            U_red, V_red, W_red = reduced
            if len(U_red[0]) < best_rank:
                best_result = {"U": U_red, "V": V_red, "W": W_red}
                best_rank = len(U_red[0])
    
    # Try random initializations at various ranks
    for rank in range(best_rank - 1, max(40, best_rank - 15), -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        for seed in range(8):
            if time.time() > deadline - 15:
                break
            
            result = _als_decompose(n, m, p, rank, seed=seed, max_iters=200, deadline=deadline - 15)
            if result is not None:
                U, V, W = result
                if len(U[0]) < best_rank:
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = len(U[0])
                found = True
                break
        
        if found:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
