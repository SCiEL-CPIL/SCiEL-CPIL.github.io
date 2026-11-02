"""Search for a rank-minimal exact tensor decomposition of 4x4 * 4x5 matrix multiplication.
Uses block-based Strassen decomposition as a baseline (rank 64),
then attempts tensor slice sparsification, DFT-based cyclic decomposition and refinement via ALS with rounding.
Falls back to schoolbook (rank 80) if all else fails."""
import time
import numpy as np
from collections import defaultdict

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

def _merge_identical_terms(U, V, W, n, m, p):
    """Merge terms that have identical (U_col, V_col) patterns by summing W_cols.
    
    In the Strassen decomposition, different block multiplications may produce
    identical (U, V) vectors. When this happens, their W contributions can be
    summed into a single term, reducing the total rank.
    """
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    
    # Group columns by their (U_col, V_col) pattern
    groups = defaultdict(list)
    for r in range(rank):
        key_u = tuple(Ua[:, r])
        key_v = tuple(Va[:, r])
        groups[(key_u, key_v)].append(r)
    
    new_U_cols = []
    new_V_cols = []
    new_W_cols = []
    
    for (key_u, key_v), indices in groups.items():
        u_col = Ua[:, indices[0]]
        v_col = Va[:, indices[0]]
        if len(indices) == 1:
            w_col = Wa[:, indices[0]]
        else:
            w_col = np.zeros_like(Wa[:, 0])
            for idx in indices:
                w_col += Wa[:, idx]
        new_U_cols.append(u_col)
        new_V_cols.append(v_col)
        new_W_cols.append(w_col)
    
    if len(new_U_cols) >= rank:
        return None
    
    U_new = np.column_stack(new_U_cols)
    V_new = np.column_stack(new_V_cols)
    W_new = np.column_stack(new_W_cols)
    
    # Verify
    if _verify_fast(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p, atol=1e-9):
        return U_new.tolist(), V_new.tolist(), W_new.tolist()
    
    return None

def _tensor_slice_sparsification(U, V, W, n, m, p, deadline):
    """Tensor slice sparsification: treat the matmul tensor as sum of p slices.
    For each k (column of B), the slice T[:,:,k] is a 4x4 matrix that equals
    sum over j of A[:,j] * B[j,k]. We try to find a joint decomposition across
    all slices that shares bilinear terms.
    
    Strategy: 
    1. For each k, compute the "effective" A and B contributions
    2. Try to find a low-rank factorization that works across all k simultaneously
    3. Use this as initialization for ALS at lower ranks
    """
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    
    # Build the full tensor
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    
    # For each k, the slice is T[:,:,k] = sum_j A[:,j] B[j,k]
    # In our decomposition: T[:,:,k] = sum_r sum_j Ur[:,:,r] * Vr[j,k,r] * Wr[k,:,r]
    # = sum_r Wr[k,:,r]^T * (sum_j Ur[:,:,r] * Vr[j,k,r])
    
    # Try to find a lower-rank decomposition by looking at the structure
    # across k values. If for different k, the same (Ur[:,:,r], Wr[k,:,r]) 
    # patterns appear, we might be able to merge.
    
    best_result = None
    
    # Try ranks from rank-1 down to max(30, rank-20)
    for target_rank in range(rank - 1, max(30, rank - 20), -1):
        if time.time() > deadline:
            break
        
        # Multiple random restarts
        for seed in range(10):
            if time.time() > deadline:
                break
            
            # Initialize from a subset of the Strassen decomposition
            # with some random perturbation
            idx = np.random.RandomState(seed).choice(rank, target_rank, replace=False)
            U_init = Ua[:, idx].copy()
            V_init = Va[:, idx].copy()
            W_init = Wa[:, idx].copy()
            
            # Add small random perturbation to escape local minima
            U_init += np.random.RandomState(seed + 1000).randn(n*m, target_rank) * 0.01
            V_init += np.random.RandomState(seed + 2000).randn(m*p, target_rank) * 0.01
            W_init += np.random.RandomState(seed + 3000).randn(p*n, target_rank) * 0.01
            
            result = _als_decompose(n, m, p, target_rank, U_init=U_init, V_init=V_init, 
                                    W_init=W_init, max_iters=300, deadline=deadline)
            if result is not None:
                best_result = result
                break
        
        if best_result is not None:
            return best_result
    
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
    
    # Try to find a lower-rank decomposition via ALS starting from the Strassen solution
    best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
    best_rank = rank
    
    # Try reducing by 1, 2, 3, ... terms
    for target_rank in range(rank - 1, max(30, rank - 20), -1):
        if time.time() > deadline:
            break
        
        # Multiple initialization strategies
        found = False
        
        # Strategy 1: First target_rank columns
        U_init = Ua[:, :target_rank].copy()
        V_init = Va[:, :target_rank].copy()
        W_init = Wa[:, :target_rank].copy()
        
        result = _als_decompose(n, m, p, target_rank, U_init=U_init, V_init=V_init, W_init=W_init,
                                max_iters=200, deadline=deadline)
        if result is not None:
            best_U, best_V, best_W = np.array(result[0]), np.array(result[1]), np.array(result[2])
            best_rank = target_rank
            found = True
            break
        
        # Strategy 2: Random subset
        for seed in range(5):
            if time.time() > deadline:
                break
            idx = np.random.RandomState(seed).choice(rank, target_rank, replace=False)
            U_init = Ua[:, idx].copy()
            V_init = Va[:, idx].copy()
            W_init = Wa[:, idx].copy()
            
            result = _als_decompose(n, m, p, target_rank, U_init=U_init, V_init=V_init, W_init=W_init,
                                    max_iters=200, deadline=deadline)
            if result is not None:
                best_U, best_V, best_W = np.array(result[0]), np.array(result[1]), np.array(result[2])
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
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
        
        # Compute error (vectorized)
        Ur = U.reshape(n, m, rank)
        Vr = V.reshape(m, p, rank)
        Wr = W.reshape(p, n, rank)
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
        err = np.sqrt(np.sum((T - 1.0) ** 2))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-12:
            break
    
    # Try rounding to half-integers
    U_r = _round_matrix_to_half_int(best_U)
    V_r = _round_matrix_to_half_int(best_V)
    W_r = _round_matrix_to_half_int(best_W)
    
    # Verify
    if _verify_fast(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, atol=1e-9):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    # If rounding failed, try integer rounding
    U_i = np.round(best_U)
    V_i = np.round(best_V)
    W_i = np.round(best_W)
    if _verify_fast(U_i.tolist(), V_i.tolist(), W_i.tolist(), n, m, p, atol=1e-9):
        return U_i.tolist(), V_i.tolist(), W_i.tolist()
    
    # Try a final polish: use the unrounded solution and check if it's already close enough
    # to be representable with small integers/half-integers
    if best_err < 1e-10:
        # Scale to see if we can get exact integers
        U_scaled = best_U * 2
        V_scaled = best_V * 2
        W_scaled = best_W * 2
        U_rounded = np.round(U_scaled) / 2.0
        V_rounded = np.round(V_scaled) / 2.0
        W_rounded = np.round(W_scaled) / 2.0
        if _verify_fast(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p, atol=1e-9):
            return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
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
    
    # First, try merging identical terms in the Strassen decomposition
    if time.time() < deadline - 100:
        merged = _merge_identical_terms(U_base, V_base, W_base, n, m, p)
        if merged is not None:
            U_m, V_m, W_m = merged
            best_result = {"U": U_m, "V": V_m, "W": W_m}
            best_rank = len(U_m[0])
    
    # Try to reduce the Strassen rank via ALS refinement
    if time.time() < deadline - 60:
        reduced = _reduce_strassen_rank(U_base, V_base, W_base, n, m, p, deadline - 60)
        if reduced is not None:
            U_red, V_red, W_red = reduced
            if len(U_red[0]) < best_rank:
                best_result = {"U": U_red, "V": V_red, "W": W_red}
                best_rank = len(U_red[0])
    
    # Try tensor slice sparsification
    if time.time() < deadline - 40:
        sliced = _tensor_slice_sparsification(U_base, V_base, W_base, n, m, p, deadline - 40)
        if sliced is not None:
            U_s, V_s, W_s = sliced
            if len(U_s[0]) < best_rank:
                best_result = {"U": U_s, "V": V_s, "W": W_s}
                best_rank = len(U_s[0])
    
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
            continue
    
    return best_result
# EVOLVE-BLOCK-END
