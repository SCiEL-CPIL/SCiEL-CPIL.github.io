"""
Combined approach:
1. Deterministic Rank-65 Construction via Kronecker-embedded Strassen (4x4x4) + Naive (4x4x1).
2. Attempt rank reduction via structured random projection + iterative null-space elimination.
3. Final high-precision polish pass to maximize chance of valid lower-rank decomposition.
"""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify_fast(U, V, W, n=4, m=4, p=5, atol=1e-9):
    """Fast verification using numpy with float64 and einsum."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    if Ua.shape[0] != n*m or Va.shape[0] != m*p or Wa.shape[0] != p*n:
        return False
    
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    
    # Use einsum for efficient verification
    T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr, optimize=True)
    return np.allclose(T, 1.0, atol=atol)

def _round_to_half_integer(x):
    return round(x * 2) / 2.0

def _round_matrix_to_half_int(M):
    """M is a numpy array or list of lists. Returns list of lists of floats."""
    if isinstance(M, np.ndarray):
        M = M.tolist()
    return [[_round_to_half_integer(x) for x in row] for row in M]

def _strassen_2x2x2_terms():
    """
    Returns 7 terms for 2x2x2 Strassen.
    Each term is (u_sparse, v_sparse, w_sparse) where sparse is list of (index, value).
    U index: i*2+j
    V index: j*2+k
    W index: k*2+i
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

def _construct_rank_65(n=4, m=4, p=5):
    """
    Constructs a rank-65 decomposition.
    49 terms from 4x4x4 Kronecker Strassen.
    16 terms from 4x4x1 Naive.
    """
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def get_u_idx(i, j): return i * m + j
    def get_v_idx(j, k): return j * p + k
    def get_w_idx(k, i): return k * n + i

    strassen_terms = _strassen_2x2x2_terms()
    
    # 1. 4x4x4 part (k=0..3)
    # Kronecker product of two 2x2x2 Strassen decompositions
    for (u1_sparse, v1_sparse, w1_sparse) in strassen_terms:
        for (u2_sparse, v2_sparse, w2_sparse) in strassen_terms:
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            
            # U: i = i1*2 + i2, j = j1*2 + j2
            for (idx1, val1) in u1_sparse:
                i1, j1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in u2_sparse:
                    i2, j2 = idx2 // 2, idx2 % 2
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    u[get_u_idx(i, j)] = val1 * val2
            
            # V: j = j1*2 + j2, k = k1*2 + k2
            for (idx1, val1) in v1_sparse:
                j1, k1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in v2_sparse:
                    j2, k2 = idx2 // 2, idx2 % 2
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    v[get_v_idx(j, k)] = val1 * val2
            
            # W: k = k1*2 + k2, i = i1*2 + i2
            for (idx1, val1) in w1_sparse:
                k1, i1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in w2_sparse:
                    k2, i2 = idx2 // 2, idx2 % 2
                    k = k1 * 2 + k2
                    i = i1 * 2 + i2
                    w[get_w_idx(k, i)] = val1 * val2
            
            add_term(u, v, w)
            
    # 2. 4x4x1 part (k=4)
    # Naive: A[i,j] * B[j,4] -> C[4,i]
    for i in range(n):
        for j in range(m):
            u = np.zeros(n*m); u[get_u_idx(i, j)] = 1
            v = np.zeros(m*p); v[get_v_idx(j, 4)] = 1
            w = np.zeros(p*n); w[get_w_idx(4, i)] = 1
            add_term(u, v, w)
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()

def _als_decompress(n, m, p, U_init, V_init, W_init, target_rank, max_iters=200, deadline=None, seed=42):
    """
    Attempt to reduce rank by optimizing U, V, W initialized from a higher rank decomposition.
    Uses structured random projection + vectorized iterative ALS.
    """
    rng = np.random.RandomState(seed)
    
    U0 = np.array(U_init, dtype=float)
    V0 = np.array(V_init, dtype=float)
    W0 = np.array(W_init, dtype=float)
    init_rank = U0.shape[1]
    
    if target_rank > init_rank:
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        U[:, :init_rank] = U0
        V[:, :init_rank] = V0
        W[:, :init_rank] = W0
    elif target_rank == init_rank:
        U = U0.copy()
        V = V0.copy()
        W = W0.copy()
    else:
        P_U = rng.randn(init_rank, target_rank)
        P_V = rng.randn(init_rank, target_rank)
        P_W = rng.randn(init_rank, target_rank)
        
        for P in [P_U, P_V, P_W]:
            Q, _ = np.linalg.qr(P)
            P[:] = Q[:, :target_rank]
        
        U = U0 @ P_U
        V = V0 @ P_V
        W = W0 @ P_W
        
    # Precompute index arrays for vectorized updates
    # For U[i*m+j]: sum over k of V[j*p+k] * W[k*n+i]
    # For V[j*p+k]: sum over i of U[i*m+j] * W[k*n+i]
    # For W[k*n+i]: sum over j of U[i*m+j] * V[j*p+k]
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Update U vectorized
        for i in range(n):
            for j in range(m):
                num = np.zeros(target_rank)
                den = np.zeros(target_rank)
                for k in range(p):
                    num += V[j*p+k, :] * W[k*n+i, :]
                    den += V[j*p+k, :]**2 * W[k*n+i, :]**2
                mask = den > 1e-10
                if np.any(mask):
                    U[i*m+j, mask] = num[mask] / den[mask]
        
        # Update V vectorized
        for j in range(m):
            for k in range(p):
                num = np.zeros(target_rank)
                den = np.zeros(target_rank)
                for i in range(n):
                    num += U[i*m+j, :] * W[k*n+i, :]
                    den += U[i*m+j, :]**2 * W[k*n+i, :]**2
                mask = den > 1e-10
                if np.any(mask):
                    V[j*p+k, mask] = num[mask] / den[mask]
        
        # Update W vectorized
        for k in range(p):
            for i in range(n):
                num = np.zeros(target_rank)
                den = np.zeros(target_rank)
                for j in range(m):
                    num += U[i*m+j, :] * V[j*p+k, :]
                    den += U[i*m+j, :]**2 * V[j*p+k, :]**2
                mask = den > 1e-10
                if np.any(mask):
                    W[k*n+i, mask] = num[mask] / den[mask]
                    
    return U, V, W

def _exact_local_solve(U, V, W, n, m, p, target_rank, max_iters=50, deadline=None):
    """
    Final high-precision polish: alternate between exact least-solves for each factor
    given the other two, then round and verify.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Solve for U given V, W
        for i in range(n):
            for j in range(m):
                # U[i*m+j, :] = argmin sum_k (V[j*p+k,:] * W[k*n+i,:] - U[i*m+j,:])^2
                # This is a linear system: for each rank component r:
                # U[i*m+j, r] = sum_k V[j*p+k, r] * W[k*n+i, r] / sum_k V[j*p+k, r]^2 * W[k*n+i, r]^2
                num = np.zeros(target_rank)
                den = np.zeros(target_rank)
                for k in range(p):
                    num += V[j*p+k, :] * W[k*n+i, :]
                    den += V[j*p+k, :]**2 * W[k*n+i, :]**2
                mask = den > 1e-12
                if np.any(mask):
                    U[i*m+j, mask] = num[mask] / den[mask]
        
        # Solve for V given U, W
        for j in range(m):
            for k in range(p):
                num = np.zeros(target_rank)
                den = np.zeros(target_rank)
                for i in range(n):
                    num += U[i*m+j, :] * W[k*n+i, :]
                    den += U[i*m+j, :]**2 * W[k*n+i, :]**2
                mask = den > 1e-12
                if np.any(mask):
                    V[j*p+k, mask] = num[mask] / den[mask]
        
        # Solve for W given U, V
        for k in range(p):
            for i in range(n):
                num = np.zeros(target_rank)
                den = np.zeros(target_rank)
                for j in range(m):
                    num += U[i*m+j, :] * V[j*p+k, :]
                    den += U[i*m+j, :]**2 * V[j*p+k, :]**2
                mask = den > 1e-12
                if np.any(mask):
                    W[k*n+i, mask] = num[mask] / den[mask]
        
        # Check convergence: compute residual
        Ur = U.reshape(n, m, target_rank)
        Vr = V.reshape(m, p, target_rank)
        Wr = W.reshape(p, n, target_rank)
        T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr, optimize=True)
        residual = np.max(np.abs(T - 1.0))
        
        if residual < 1e-10:
            break
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin (150s limit, leave 35s buffer)
    
    # 1. Generate Rank 65 Baseline
    U65, V65, W65 = _construct_rank_65(n, m, p)
    
    # Verify baseline (should be exact integers/half-integers)
    if not _verify_fast(U65, V65, W65, n, m, p):
        # Fallback to schoolbook if construction is buggy (should not happen)
        rank_schoolbook = n * m * p
        U_school = np.zeros((n*m, rank_schoolbook))
        V_school = np.zeros((m*p, rank_schoolbook))
        W_school = np.zeros((p*n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j, col] = 1
                    V_school[j*p+k, col] = 1
                    W_school[k*n+i, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    best_U, best_V, best_W = U65, V65, W65
    best_rank = 65
    
    # 2. Attempt to reduce rank
    # Try ranks 64 down to 50 with multiple strategies
    for target_rank in range(64, 49, -1):
        if time.time() > deadline:
            break
            
        # Strategy 1: Try multiple seeds for random projection
        found = False
        for seed in range(8):
            if time.time() > deadline or found:
                break
                
            # Use rank 65 as init
            U_opt, V_opt, W_opt = _als_decompress(
                n, m, p, U65, V65, W65, target_rank, 
                max_iters=200, deadline=deadline, seed=seed
            )
            
            # Round to half-integers
            U_r = _round_matrix_to_half_int(U_opt)
            V_r = _round_matrix_to_half_int(V_opt)
            W_r = _round_matrix_to_half_int(W_opt)
            
            if _verify_fast(U_r, V_r, W_r, n, m, p):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = target_rank
                found = True
                break
        
        # Strategy 2: If not found, try with perturbations
        if not found and best_rank > target_rank and time.time() < deadline:
            found2 = False
            for seed in range(5):
                if time.time() > deadline or found2:
                    break
                
                rng = np.random.RandomState(seed)
                U_perturbed = np.array(U65, dtype=float) + rng.randn(n*m, 65) * 0.05
                V_perturbed = np.array(V65, dtype=float) + rng.randn(m*p, 65) * 0.05
                W_perturbed = np.array(W65, dtype=float) + rng.randn(p*n, 65) * 0.05
                
                U_opt, V_opt, W_opt = _als_decompress(
                    n, m, p, U_perturbed.tolist(), V_perturbed.tolist(), W_perturbed.tolist(), 
                    target_rank, 
                    max_iters=150, deadline=deadline, seed=seed
                )
                
                # Round to half-integers
                U_r = _round_matrix_to_half_int(U_opt)
                V_r = _round_matrix_to_half_int(V_opt)
                W_r = _round_matrix_to_half_int(W_opt)
                
                if _verify_fast(U_r, V_r, W_r, n, m, p):
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = target_rank
                    found2 = True
                    break
    
    # 3. Final high-precision polish with exact local solve
    if best_rank < 65 and time.time() < deadline:
        U_final = np.array(best_U, dtype=float)
        V_final = np.array(best_V, dtype=float)
        W_final = np.array(best_W, dtype=float)
        
        # Run exact local solve for polish
        U_final, V_final, W_final = _exact_local_solve(
            U_final, V_final, W_final, n, m, p, best_rank, 
            max_iters=100, deadline=deadline
        )
        
        # Round and verify
        U_r = _round_matrix_to_half_int(U_final)
        V_r = _round_matrix_to_half_int(V_final)
        W_r = _round_matrix_to_half_int(W_final)
        
        if _verify_fast(U_r, V_r, W_r, n, m, p):
            best_U, best_V, best_W = U_r, V_r, W_r
        else:
            # If polish broke it, revert to pre-polish
            pass
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
