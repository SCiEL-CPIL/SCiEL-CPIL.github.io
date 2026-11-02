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

def _greedy_eliminate(U, V, W, n, m, p, deadline):
    """Try to eliminate terms one at a time by checking if removal preserves exactness."""
    Ua = np.array(U, dtype=np.float64).copy()
    Va = np.array(V, dtype=np.float64).copy()
    Wa = np.array(W, dtype=np.float64).copy()
    rank = Ua.shape[1]
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for c in range(rank):
            if time.time() > deadline:
                break
            U_test = np.delete(Ua, c, axis=1)
            V_test = np.delete(Va, c, axis=1)
            W_test = np.delete(Wa, c, axis=1)
            if _verify_fast(U_test.tolist(), V_test.tolist(), W_test.tolist(), n, m, p, atol=1e-9):
                Ua, Va, Wa = U_test, V_test, W_test
                rank = rank - 1
                changed = True
                break
    if rank < len(U[0]):
        return Ua.tolist(), Va.tolist(), Wa.tolist()
    return None

def _round_to_half_integer(x):
    """Round x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _round_matrix_to_half_int(M):
    """Round all entries of matrix M to nearest half-integer."""
    return np.array([[ _round_to_half_integer(x) for x in row] for row in M])

def _strassen_construct_v2(n, m, p):
    """Alternative: split B columns as 3+2 instead of 2+3 for the right block.
    C00 (2x2) from A00*B00 + A01*B10: both 2x2x2 -> 7+7=14
    C10 (2x2) from A10*B00 + A11*B10: both 2x2x2 -> 7+7=14
    C01 (2x3) from A00*B01 + A01*B11: B01 is 2x3, B11 is 2x3
      Use 2x2x3 for each: 9+9=18
    C11 (2x3) from A10*B01 + A11*B11: 9+9=18
    Total: 14+14+18+18 = 64 (same)
    
    Try different split: A rows as 1+3, B cols as 2+3
    C0 (1x2) = A0*B00 + A0*B10: 1x2x2 blocks
    C1 (1x3) = A0*B01 + A0*B11: 1x2x3 blocks
    C2 (3x2) = A1*B00 + A1*B10: 3x2x2 blocks  
    C3 (3x3) = A1*B01 + A1*B11: 3x2x3 blocks
    This gets complex. Stick with 2x2 split.
    """
    pass

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
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = 1; w[get_w_idx(bk, ai+1)] = 1
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

def _strassen_construct_v2(n, m, p):
    """Alternative block structure: split B's 5 columns as 3+2 instead of 4+1.
    
    For 4x4 * 4x5:
    - Split A into 2x2 blocks (4 blocks of size 2x2)
    - Split B into 2x3 (first 3 cols) + 2x2 (last 2 cols)
    - Use Strassen (rank 7) for 2x2x2, 2x2x3 needs custom treatment
    
    Actually, let's try: split columns of B as 2+3 for the row blocks.
    C0 (2x2 rows) gets B cols 0-1 (2x2) and B cols 2-4 (2x3)
    C1 (2x2 rows) gets B cols 0-1 (2x2) and B cols 2-4 (2x3)
    
    Same structure but let's try a different 2x2x3 algorithm.
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
        u = np.zeros(n*m); u[get_u_idx(ai, aj)] = 1; u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj, bk)] = 1; v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = 1; w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        u = np.zeros(n*m); u[get_u_idx(ai+1, aj)] = 1; u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj, bk)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai+1)] = 1; w[get_w_idx(bk+1, ai+1)] = -1
        add_term(u, v, w)
        
        u = np.zeros(n*m); u[get_u_idx(ai, aj)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj, bk+1)] = 1; v[get_v_idx(bj+1, bk+1)] = -1
        w = np.zeros(p*n); w[get_w_idx(bk+1, ai)] = 1; w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        u = np.zeros(n*m); u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj+1, bk)] = 1; v[get_v_idx(bj, bk)] = -1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = 1; w[get_w_idx(bk, ai+1)] = 1
        add_term(u, v, w)
        
        u = np.zeros(n*m); u[get_u_idx(ai, aj)] = 1; u[get_u_idx(ai, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = -1; w[get_w_idx(bk+1, ai)] = 1
        add_term(u, v, w)
        
        u = np.zeros(n*m); u[get_u_idx(ai+1, aj)] = 1; u[get_u_idx(ai, aj)] = -1
        v = np.zeros(m*p); v[get_v_idx(bj, bk)] = 1; v[get_v_idx(bj, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        u = np.zeros(n*m); u[get_u_idx(ai, aj+1)] = 1; u[get_u_idx(ai+1, aj+1)] = -1
        v = np.zeros(m*p); v[get_v_idx(bj+1, bk)] = 1; v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = 1
        add_term(u, v, w)

    def add_2x2x3_alt(ai, aj, bj, bk):
        """Alternative 2x2x3: use 8 terms instead of 9.
        A[ai:ai+2, aj:aj+2] @ B[bj:bj+2, bk:bk+3] -> C[ai:ai+2, bk:bk+3]
        
        Try: Strassen-like for the 3 columns using 8 multiplications.
        M1 = (a00+a11)*(b0k0+b1k2)  for output col bk
        M2 = (a10+a11)*b0k0          for output col bk  
        M3 = a00*(b0k1-b1k1)         for output col bk+1
        M4 = a11*(b1k1-b0k1)         for output col bk+1
        M5 = (a00+a01)*b1k2          for output col bk+2
        M6 = (a10-a00)*(b0k0+b0k1)   for output col bk+1
        M7 = (a01-a11)*(b1k0+b1k1)   for output col bk
        M8 = naive for remaining
        
        Actually this is getting complex. Let's just use the same 9-term approach
        but with a different split of the 3 columns.
        """
        # Use Strassen for cols bk, bk+1 (7 terms)
        add_strassen_2x2_2(ai, aj, bj, bk)
        
        # Naive for col bk+2 (4 terms)
        for di in range(2):
            for dj in range(2):
                u = np.zeros(n*m); u[get_u_idx(ai+di, aj+dj)] = 1
                v = np.zeros(m*p); v[get_v_idx(bj+dj, bk+2)] = 1
                w = np.zeros(p*n); w[get_w_idx(bk+2, ai+di)] = 1
                add_term(u, v, w)

    # Same structure as v1 for now
    add_strassen_2x2_2(0, 0, 0, 0)
    add_strassen_2x2_2(0, 2, 2, 0)
    add_strassen_2x2_2(2, 0, 0, 0)
    add_strassen_2x2_2(2, 2, 2, 0)
    add_2x2x3_alt(0, 0, 0, 2)
    add_2x2x3_alt(0, 2, 2, 2)
    add_2x2x3_alt(2, 0, 0, 2)
    add_2x2x3_alt(2, 2, 2, 2)

    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()

def _als_decompose(n, m, p, rank, seed=42, max_iters=500, deadline=None, 
                   U_init=None, V_init=None, W_init=None):
    """Try to find a rank-r decomposition using alternating least squares with rounding.
    Includes a high-precision polish pass for near-converged solutions.
    
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
    
    # High-precision polish: if near-converged, do additional iterations with tighter stopping
    if best_err < 1e-6:
        U_pol, V_pol, W_pol = best_U.copy(), best_V.copy(), best_W.copy()
        for polish_it in range(200):
            if deadline is not None and time.time() > deadline:
                break
            
            for i in range(n):
                for j in range(m):
                    A_mat = np.array([V_pol[j*p+k, :] * W_pol[k*n+i, :] for k in range(p)])
                    b_vec = np.ones(p)
                    try:
                        U_pol[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=1e-15)[0]
                    except:
                        pass
            
            for j in range(m):
                for k in range(p):
                    A_mat = np.array([U_pol[i*m+j, :] * W_pol[k*n+i, :] for i in range(n)])
                    b_vec = np.ones(n)
                    try:
                        V_pol[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=1e-15)[0]
                    except:
                        pass
            
            for i in range(n):
                for k in range(p):
                    A_mat = np.array([U_pol[i*m+j, :] * V_pol[j*p+k, :] for j in range(m)])
                    b_vec = np.ones(m)
                    try:
                        W_pol[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=1e-15)[0]
                    except:
                        pass
            
            err_pol = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U_pol[i*m+j, r] * V_pol[j*p+k, r] * W_pol[k*n+i, r]
                        err_pol += (s - 1.0) ** 2
            err_pol = np.sqrt(err_pol)
            
            if err_pol < best_err:
                best_err = err_pol
                best_U, best_V, best_W = U_pol.copy(), V_pol.copy(), W_pol.copy()
            
            if err_pol < 1e-14:
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
    
    # First try greedy elimination (fast)
    if time.time() < deadline:
        elim = _greedy_eliminate(Ua.tolist(), Va.tolist(), Wa.tolist(), n, m, p, deadline)
        if elim is not None:
            Ue, Ve, We = elim
            best_result = (Ue, Ve, We)
            best_rank = len(Ue[0])
            Ua = np.array(Ue, dtype=np.float64)
            Va = np.array(Ve, dtype=np.float64)
            Wa = np.array(We, dtype=np.float64)
            rank = best_rank
    
    # Target ranks to try
    targets = list(range(best_rank - 1, max(40, best_rank - 20), -1))
    
    for target_rank in targets:
        if target_rank >= rank:
            continue
        if time.time() > deadline:
            break
        
        # Try multiple initializations
        for init_seed in range(5):
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
            elif init_seed == 3:
                start = (rank - target_rank) // 2
                U_init = Ua[:, start:start+target_rank]
                V_init = Va[:, start:start+target_rank]
                W_init = Wa[:, start:start+target_rank]
            else:
                np.random.seed(init_seed * 200 + target_rank)
                idx = np.random.choice(rank, target_rank, replace=False)
                U_init = Ua[:, idx] + np.random.randn(n*m, target_rank) * 0.1
                V_init = Va[:, idx] + np.random.randn(m*p, target_rank) * 0.1
                W_init = Wa[:, idx] + np.random.randn(p*n, target_rank) * 0.1
            
            result = _als_decompose(n, m, p, target_rank, 
                                    U_init=U_init, V_init=V_init, W_init=W_init,
                                    max_iters=200, deadline=deadline)
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
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Get the Strassen-based decomposition as baseline (rank 64)
    U_base, V_base, W_base = _strassen_construct(n, m, p)
    best_result = {"U": U_base, "V": V_base, "W": W_base}
    best_rank = len(U_base[0])
    
    # Verify baseline
    if not _verify_fast(U_base, V_base, W_base, n, m, p, atol=1e-9):
        return _schoolbook()
    
    # Try to reduce rank from the Strassen baseline
    if time.time() < deadline - 30:
        reduced = _try_reduce_from_baseline(U_base, V_base, W_base, n, m, p, deadline - 30)
        if reduced is not None:
            U_red, V_red, W_red = reduced
            best_result = {"U": U_red, "V": V_red, "W": W_red}
            best_rank = len(U_red[0])
    
    # Try random initializations at various ranks
    for rank in range(best_rank - 1, max(40, best_rank - 10), -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        for seed in range(6):
            if time.time() > deadline - 15:
                break
            
            result = _als_decompose(n, m, p, rank, seed=seed, max_iters=150, deadline=deadline - 15)
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
