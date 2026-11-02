"""Search for a rank-minimal exact tensor decomposition of 4x4 * 4x5 matrix multiplication.
Uses block-based Strassen decomposition as a baseline (rank 64),
then attempts DFT-based cyclic decomposition and refinement via ALS with rounding.
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

def _dft_z5_decomposition(n, m, p):
    """Construct a DFT-based decomposition exploiting Z_5 cyclic structure.
    
    For the 5-column dimension, use DFT basis vectors to create structured rank-1 terms.
    The idea: for each (i,j) pair in A and each frequency in Z_5, create a term that
    cyclically shifts the contribution across the 5 output columns.
    
    This gives at most n*m*5 = 80 terms, but with structure that may allow
    cancellation or sharing to reduce rank.
    """
    # DFT matrix for Z_5
    omega = np.exp(2j * np.pi / 5)
    F = np.array([[omega ** (k * l) for l in range(5)] for k in range(5)], dtype=complex)
    F_inv = F.conj().T / 5.0
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def get_u_idx(i, j):
        return i * m + j
        
    def get_v_idx(j, k):
        return j * p + k
        
    def get_w_idx(k, i):
        return k * n + i

    # For each (i, j) in A and each frequency f in Z_5:
    # Create a term where V has the DFT column f (cyclic shift pattern),
    # and W has the conjugate DFT row to recover the correct output
    for i in range(n):
        for j in range(m):
            for f in range(5):
                u = np.zeros(n * m)
                u[get_u_idx(i, j)] = 1.0
                
                v = np.zeros(m * p)
                # V[j, k] = omega^(f*k) for k in 0..4
                for k in range(p):
                    v[get_v_idx(j, k)] = omega ** (f * k)
                
                w = np.zeros(p * n)
                # W[k, i] = omega^(-f*k) / 5 to normalize
                for k in range(p):
                    w[get_w_idx(k, i)] = omega ** (-f * k) / 5.0
                
                add_term(u, v, w)
    
    # Convert to real matrices (the imaginary parts should cancel in verification)
    # But we need half-integer entries, so this DFT approach gives complex values
    # Let's try a different approach: use the real and imaginary parts separately
    
    # Actually, let's try a simpler cyclic shift approach
    # For each (i, j), create 5 terms with cyclic shifts of the output column
    
    U_cols2 = []
    V_cols2 = []
    W_cols2 = []
    
    for i in range(n):
        for j in range(m):
            for shift in range(5):
                u = np.zeros(n * m)
                u[get_u_idx(i, j)] = 1.0
                
                v = np.zeros(m * p)
                # V[j, k] = 1 if k == (shift + something) mod 5
                # Actually, we want to distribute the contribution
                for k in range(p):
                    v[get_v_idx(j, k)] = 1.0
                
                w = np.zeros(p * n)
                # W[k, i] = 1 if k == shift, else 0
                w[get_w_idx(shift, i)] = 1.0
                
                add_term(u, v, w)
    
    # This gives n*m*5 = 80 terms, same as schoolbook
    # The DFT approach is more promising if we can handle the complex values
    
    # Let's try: for each (i,j), use 5 terms with DFT structure
    # The key insight: sum over f of omega^(f*k) * omega^(-f*k') = 5*delta_{k,k'}
    # So if we create terms with V having DFT columns and W having conjugate DFT rows,
    # the sum reconstructs the identity in the k dimension
    
    # But we need real half-integer entries. The DFT of Z_5 involves sqrt(5) and complex numbers.
    # Let's check if we can find a real representation
    
    # Alternative: use the fact that the 5x5 DFT matrix has a specific structure
    # F = [[1, 1, 1, 1, 1],
    #      [1, w, w^2, w^3, w^4],
    #      [1, w^2, w^4, w^6, w^8],
    #      [1, w^3, w^6, w^9, w^12],
    #      [1, w^4, w^8, w^12, w^16]]
    # where w = e^(2*pi*i/5)
    
    # The real part of F is:
    # [[1, 1, 1, 1, 1],
    #  [1, cos(2π/5), cos(4π/5), cos(6π/5), cos(8π/5)],
    #  [1, cos(4π/5), cos(8π/5), cos(12π/5), cos(16π/5)],
    #  [1, cos(6π/5), cos(12π/5), cos(18π/5), cos(24π/5)],
    #  [1, cos(8π/5), cos(16π/5), cos(24π/5), cos(32π/5)]]
    
    # cos(2π/5) = (sqrt(5)-1)/4 ≈ 0.309
    # cos(4π/5) = -(sqrt(5)+1)/4 ≈ -0.809
    # These are not half-integers, so pure DFT won't work directly
    
    # Let's try a different approach: use the Strassen baseline and try to reduce rank
    # by identifying redundant terms
    
    # For now, return the schoolbook as a fallback from this function
    return None

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

def _reduce_strassen_rank(U, V, W, n, m, p, deadline):
    """Try to reduce the rank of the Strassen decomposition by identifying
    linearly dependent or canceling terms, and by trying to merge terms.
    
    Returns (U, V, W) with reduced rank if successful, else None.
    """
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    
    # Check for columns that are linearly dependent
    # If column r is a linear combination of other columns, we can eliminate it
    # by redistributing the coefficients
    
    # Try to find a lower-rank decomposition via ALS starting from the Strassen solution
    best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
    best_rank = rank
    
    # Try reducing by 1, 2, 3, ... terms
    for target_rank in range(rank - 1, max(30, rank - 15), -1):
        if time.time() > deadline:
            break
        
        # Initialize with the first target_rank columns of the Strassen decomposition
        U_init = Ua[:, :target_rank]
        V_init = Va[:, :target_rank]
        W_init = Wa[:, :target_rank]
        
        # Try ALS from this initialization
        result = _als_decompose(n, m, p, target_rank, U_init=U_init, V_init=V_init, W_init=W_init,
                                max_iters=200, deadline=deadline)
        if result is not None:
            U_r, V_r, W_r = result
            best_U, best_V, best_W = np.array(U_r), np.array(V_r), np.array(W_r)
            best_rank = target_rank
    
    if best_rank < rank:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
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
                # Need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
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
    
    # First, try to reduce the Strassen rank by identifying dependent terms
    if time.time() < deadline - 60:
        reduced = _reduce_strassen_rank(U_base, V_base, W_base, n, m, p, deadline - 60)
        if reduced is not None:
            U_red, V_red, W_red = reduced
            best_result = {"U": U_red, "V": V_red, "W": W_red}
            best_rank = len(U_red[0])
    
    # Try to find lower-rank decompositions using ALS
    # Search from best_rank-1 down to 40
    for rank in range(best_rank - 1, 39, -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        for seed in range(15):
            if time.time() > deadline - 20:
                break
            
            result = _als_decompose(n, m, p, rank, seed=seed, max_iters=400, deadline=deadline - 20)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                found = True
                break
        
        if found:
            # If we found a lower rank, continue searching even lower
            continue
    
    return best_result
# EVOLVE-BLOCK-END
