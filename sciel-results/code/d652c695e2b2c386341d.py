"""Search for a rank-minimal exact tensor decomposition of 4x4 * 4x5 matrix multiplication.
Uses block-based Strassen decomposition as baseline (rank 65 via 4x4x4+naive),
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
    return np.array([[_round_to_half_integer(x) for x in row] for row in M])

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

def _als_decompose(n, m, p, rank, seed=42, max_iters=500, deadline=None,
                   U_init=None, V_init=None, W_init=None, polish_iters=100):
    """Try to find a rank-r decomposition using alternating least squares with rounding.
    
    Returns (U, V, W) as lists if found, else None.
    """
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
        
        # Optimize U
        for i in range(n):
            for j in range(m):
                A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b_vec = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Optimize V
        for j in range(m):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b_vec = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Optimize W
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
    
    # Polish: run additional iterations with tighter convergence
    for it in range(polish_iters):
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
        
        if err < 1e-14:
            break
    
    # Round to half-integers
    U_r = _round_matrix_to_half_int(best_U)
    V_r = _round_matrix_to_half_int(best_V)
    W_r = _round_matrix_to_half_int(best_W)
    
    # Verify
    if _verify_fast(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, atol=1e-9):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    return None

def _peel_decompose(n, m, p, max_rank, deadline=None):
    """Linear algebraic peeling: iteratively extract rank-1 components via SVD of mode-1 flattening,
    round each to half-integer, subtract from residual, and continue until residual is zero or max_rank reached.
    Returns (U, V, W) as lists if successful, else None.
    """
    # Initialize residual tensor as all ones
    T = np.ones((n, m, p), dtype=np.float64)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for step in range(max_rank):
        if deadline is not None and time.time() > deadline:
            break
        
        # Check if residual is essentially zero
        if np.max(np.abs(T)) < 1e-12:
            break
        
        # Flatten mode-1: shape (n, m*p)
        T_flat = T.reshape(n, m * p)
        
        # Truncated SVD: rank-1 approximation
        try:
            U_svd, S, Vt = np.linalg.svd(T_flat, full_matrices=False)
        except:
            break
        
        if len(S) == 0 or S[0] < 1e-15:
            break
        
        # Extract rank-1 component
        u_vec = U_svd[:, 0] * np.sqrt(S[0])
        v_vec = Vt[0, :] * np.sqrt(S[0])
        
        # Reshape v_vec back to (m, p)
        v_mat = v_vec.reshape(m, p)
        
        # We need to find W such that:
        # T[i,j,k] = U[i,j] * V[j,k] * W[k,i]
        # Given U[i,j] = u_vec[i*m+j] and V[j,k] = v_mat[j,k],
        # we have W[k,i] = T[i,j,k] / (U[i,j] * V[j,k]) for some (j) where U[i,j]*V[j,k] != 0
        
        # Actually, the rank-1 tensor is: T_1[i,j,k] = u_vec[i*m+j] * v_mat[j,k] * w[k,i]
        # But we want T_1 to approximate T. The SVD gives us T ≈ u * v^T in the (n, m*p) flattening,
        # which corresponds to T[i, j*p+k] ≈ u[i] * v[j*p+k]
        # This is NOT a rank-1 tensor in 3D, it's a rank-1 in the (n, m*p) 2D flattening.
        
        # Let me rethink: for a true rank-1 3D tensor, T[i,j,k] = a[i]*b[j]*c[k]
        # The mode-1 flattening would be: T_flat[i, j*p+k] = a[i] * (b[j]*c[k])
        # So the SVD of T_flat gives us a[i] = u_vec[i] and b[j]*c[k] = v_mat[j,k]
        # But b[j]*c[k] is a rank-1 matrix, so we need to check if v_mat is rank-1.
        
        # If v_mat is not exactly rank-1, we can't directly extract a rank-1 3D component.
        # Instead, let's use a different approach: 
        # For a rank-1 3D tensor T[i,j,k] = a[i]*b[j]*c[k], we have:
        # T[i,j,k] * T[i',j',k'] / (T[i,j',k'] * T[i',j,k]) = 1 (for all i,i',j,j',k,k')
        
        # Alternative: use the fact that for the matrix multiplication tensor,
        # the "natural" rank-1 components are of the form:
        # U[i,j] = 1 for specific (i,j), V[j,k] = 1 for specific (j,k), W[k,i] = 1 for specific (k,i)
        
        # Let me try a different peeling strategy:
        # At each step, find the (i,j,k) triple that is still 1 in the residual,
        # and try to find a rank-1 component that covers it.
        
        # Actually, let's use a smarter approach: 
        # Find the row i with maximum sum, then for that i, find the best rank-1 approximation
        # of the slice T[i,:,:] in the (j,k) plane.
        
        # For a fixed i, T[i,j,k] should be sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        # If we fix W[k,i] = 1 for all k (which is a valid choice for one component),
        # then T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]
        # This is a matrix factorization of T[i,:,:]!
        
        # So for peeling: at each step, pick an i, factor T[i,:,:] as a rank-r matrix,
        # and set W[k,i] = 1 for the new component.
        
        # But this doesn't give us a clean rank-1 extraction. Let me try yet another approach.
        
        # Direct approach: try to find a single (i0, j0, k0) and construct a rank-1 tensor
        # that is 1 at (i0,j0,k0) and 0 at all other positions that are still 1 in T.
        # This is generally not possible, but we can try to find a rank-1 tensor that
        # matches T on a large subset.
        
        # Simpler: just use ALS from a random initialization but with the peeling idea
        # of starting from the residual.
        
        # For now, let's just do a simple greedy peeling:
        # Find the (i,j,k) with maximum |T[i,j,k]|, and try to fit a rank-1 tensor to it.
        
        idx = np.unravel_index(np.argmax(np.abs(T)), T.shape)
        i0, j0, k0 = idx
        
        # Try to find U_col, V_col, W_col such that:
        # U[i,j] * V[j,k] * W[k,i] ≈ T[i,j,k] for all (i,j,k)
        # This is equivalent to: U[i,j] * V[j,k] * W[k,i] = T[i,j,k]
        # Taking log (where T > 0): log U + log V + log W = log T
        # This is a linear system! But T can be negative or zero, so this doesn't work directly.
        
        # Let's just use ALS for one rank-1 component:
        # Initialize U, V, W randomly, then optimize.
        np.random.seed(step * 1000 + 42)
        U_col = np.random.randn(n*m) * 0.5
        V_col = np.random.randn(m*p) * 0.5
        W_col = np.random.randn(p*n) * 0.5
        
        # Optimize using ALS for a single component
        for it in range(20):
            # Optimize U
            for i in range(n):
                for j in range(m):
                    # U[i,j] = sum_k T[i,j,k] * V[j,k] * W[k,i] / sum_k V[j,k]^2 * W[k,i]^2
                    num = 0.0
                    den = 0.0
                    for k in range(p):
                        num += T[i,j,k] * V_col[j*p+k] * W_col[k*n+i]
                        den += V_col[j*p+k]**2 * W_col[k*n+i]**2
                    if den > 1e-15:
                        U_col[i*m+j] = num / den
            
            # Optimize V
            for j in range(m):
                for k in range(p):
                    num = 0.0
                    den = 0.0
                    for i in range(n):
                        num += T[i,j,k] * U_col[i*m+j] * W_col[k*n+i]
                        den += U_col[i*m+j]**2 * W_col[k*n+i]**2
                    if den > 1e-15:
                        V_col[j*p+k] = num / den
            
            # Optimize W
            for i in range(n):
                for k in range(p):
                    num = 0.0
                    den = 0.0
                    for j in range(m):
                        num += T[i,j,k] * U_col[i*m+j] * V_col[j*p+k]
                        den += U_col[i*m+j]**2 * V_col[j*p+k]**2
                    if den > 1e-15:
                        W_col[k*n+i] = num / den
        
        # Round to half-integers
        U_r = _round_matrix_to_half_int(U_col.reshape(1, -1))[0]
        V_r = _round_matrix_to_half_int(V_col.reshape(1, -1))[0]
        W_r = _round_matrix_to_half_int(W_col.reshape(1, -1))[0]
        
        # Check if this rank-1 component is valid (all entries are half-integers)
        # and subtract it from the residual
        T_new = T.copy()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_new[i,j,k] -= U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
        
        # Only accept if the residual doesn't get worse
        if np.sum(T_new**2) < np.sum(T**2) - 1e-10:
            T = T_new
            U_cols.append(U_r.tolist())
            V_cols.append(V_r.tolist())
            W_cols.append(W_r.tolist())
        else:
            # Try with the unrounded version to see if it's close
            T_new2 = T.copy()
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_new2[i,j,k] -= U_col[i*m+j] * V_col[j*p+k] * W_col[k*n+i]
            if np.sum(T_new2**2) < np.sum(T**2) - 1e-10:
                # The unrounded version is good, but we need half-integers
                # Try a different rounding
                pass
            break
    
    if len(U_cols) == 0:
        return None
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    
    # Verify the full decomposition
    if _verify_fast(U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), n, m, p, atol=1e-9):
        return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()
    
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
    for target_rank in range(rank - 1, max(40, rank - 25), -1):
        if time.time() > deadline:
            break
        
        # Try multiple initializations
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
                # Random perturbation of a subset
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
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Try both baseline constructions and use the better one
    U_kron, V_kron, W_kron = _kron_strassen_construct(n, m, p)
    
    # Verify
    kron_ok = _verify_fast(U_kron, V_kron, W_kron, n, m, p, atol=1e-9)
    
    if not kron_ok:
        return _schoolbook()
    
    U_base, V_base, W_base = U_kron, V_kron, W_kron
    
    best_result = {"U": U_base, "V": V_base, "W": W_base}
    best_rank = len(U_base[0])
    
    # Try to reduce rank from the baseline
    if time.time() < deadline - 40:
        reduced = _try_reduce_from_baseline(U_base, V_base, W_base, n, m, p, deadline - 40)
        if reduced is not None:
            U_red, V_red, W_red = reduced
            best_result = {"U": U_red, "V": V_red, "W": W_red}
            best_rank = len(U_red[0])
    
    # Try peeling decomposition
    if time.time() < deadline - 20:
        peeled = _peel_decompose(n, m, p, best_rank - 1, deadline - 20)
        if peeled is not None:
            U_p, V_p, W_p = peeled
            if len(U_p[0]) < best_rank:
                best_result = {"U": U_p, "V": V_p, "W": W_p}
                best_rank = len(U_p[0])
    
    # Try random initializations at various ranks
    for rank in range(best_rank - 1, max(50, best_rank - 25), -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        for seed in range(15):
            if time.time() > deadline - 20:
                break
            
            result = _als_decompose(n, m, p, rank, seed=seed, max_iters=400, 
                                    deadline=deadline - 20, polish_iters=80)
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
