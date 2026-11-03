"""Rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
Uses Strassen block decomposition as baseline, then optimized ALS refinement with rounding."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _verify_fast(U, V, W, n=4, m=4, p=5, atol=1e-9):
    """Verify decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        uc = Ur[:, :, r]
        vc = Vr[:, :, r]
        wc = Wr[:, :, r]
        for j in range(m):
            T[:, j, :] += np.outer(uc[:, j], vc[j, :]) * wc.T
    return np.allclose(T, 1.0, atol=atol)

def _round_to_half_integer(M):
    """Round matrix to nearest half-integer."""
    return np.round(2.0 * M) / 2.0

def _round_matrix_to_half_int(M):
    """Round matrix to nearest half-integer."""
    return _round_to_half_integer(M)

def _compute_error_fast(U, V, W, n, m, p):
    """Compute L2 error of decomposition using vectorized operations."""
    rank = U.shape[1]
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        uc = Ur[:, :, r]
        vc = Vr[:, :, r]
        wc = Wr[:, :, r]
        for j in range(m):
            T[:, j, :] += np.outer(uc[:, j], vc[j, :]) * wc.T
    return np.sqrt(np.sum((T - 1.0) ** 2))

def _strassen_construct(n, m, p):
    """Construct Strassen-based block decomposition for 4x4*4x5 using 2x2 blocks."""
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def uid(gi, gj):
        return gi * m + gj
    
    def vid(gj, gk):
        return gj * p + gk
    
    def wid(gk, gi):
        return gk * n + gi
    
    def strassen_2x2_2(ai, aj, bj, bk):
        """7 Strassen terms for 2x2 * 2x2 -> 2x2."""
        u = np.zeros(n*m); u[uid(ai,aj)]=1; u[uid(ai+1,aj+1)]=1
        v = np.zeros(m*p); v[vid(bj,bk)]=1; v[vid(bj+1,bk+1)]=1
        w = np.zeros(p*n); w[wid(bk,ai)]=1; w[wid(bk+1,ai+1)]=1
        add_term(u,v,w)
        u = np.zeros(n*m); u[uid(ai+1,aj)]=1; u[uid(ai+1,aj+1)]=1
        v = np.zeros(m*p); v[vid(bj,bk)]=1
        w = np.zeros(p*n); w[wid(bk,ai+1)]=1; w[wid(bk+1,ai+1)]=-1
        add_term(u,v,w)
        u = np.zeros(n*m); u[uid(ai,aj)]=1
        v = np.zeros(m*p); v[vid(bj,bk+1)]=1; v[vid(bj+1,bk+1)]=-1
        w = np.zeros(p*n); w[wid(bk+1,ai)]=1; w[wid(bk+1,ai+1)]=1
        add_term(u,v,w)
        u = np.zeros(n*m); u[uid(ai+1,aj+1)]=1
        v = np.zeros(m*p); v[vid(bj+1,bk)]=1; v[vid(bj,bk)]=-1
        w = np.zeros(p*n); w[wid(bk,ai)]=1; w[wid(bk,ai+1)]=1
        add_term(u,v,w)
        u = np.zeros(n*m); u[uid(ai,aj)]=1; u[uid(ai,aj+1)]=1
        v = np.zeros(m*p); v[vid(bj+1,bk+1)]=1
        w = np.zeros(p*n); w[wid(bk,ai)]=-1; w[wid(bk+1,ai)]=1
        add_term(u,v,w)
        u = np.zeros(n*m); u[uid(ai+1,aj)]=1; u[uid(ai,aj)]=-1
        v = np.zeros(m*p); v[vid(bj,bk)]=1; v[vid(bj,bk+1)]=1
        w = np.zeros(p*n); w[wid(bk+1,ai+1)]=1
        add_term(u,v,w)
        u = np.zeros(n*m); u[uid(ai,aj+1)]=1; u[uid(ai+1,aj+1)]=-1
        v = np.zeros(m*p); v[vid(bj+1,bk)]=1; v[vid(bj+1,bk+1)]=1
        w = np.zeros(p*n); w[wid(bk,ai)]=1
        add_term(u,v,w)
    
    def naive_2x2_1(ai, aj, bj, bk):
        """4 naive terms for 2x2 * 2x1 -> 2x1."""
        for di in range(2):
            for dj in range(2):
                u = np.zeros(n*m); u[uid(ai+di,aj+dj)]=1
                v = np.zeros(m*p); v[vid(bj+dj,bk)]=1
                w = np.zeros(p*n); w[wid(bk,ai+di)]=1
                add_term(u,v,w)
    
    def block_2x2_3(ai, aj, bj, bk):
        """11 terms: Strassen for 2x2x2 (cols bk,bk+1) + naive for 2x2x1 (col bk+2)."""
        strassen_2x2_2(ai, aj, bj, bk)
        naive_2x2_1(ai, aj, bj, bk+2)
    
    strassen_2x2_2(0,0,0,0)
    strassen_2x2_2(0,2,2,0)
    strassen_2x2_2(2,0,0,0)
    strassen_2x2_2(2,2,2,0)
    block_2x2_3(0,0,0,2)
    block_2x2_3(0,2,2,2)
    block_2x2_3(2,0,0,2)
    block_2x2_3(2,2,2,2)
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat, V_mat, W_mat

def _als_decompose(n, m, p, rank, U0, V0, W0, max_iters=200, deadline=None, tol=1e-10):
    """Optimized ALS with half-integer rounding. Returns (U,V,W) or None."""
    U = U0.copy().astype(np.float64)
    V = V0.copy().astype(np.float64)
    W = W0.copy().astype(np.float64)
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Optimize U - vectorized
        for i in range(n):
            for j in range(m):
                A = np.array([V[j*p+k,:] * W[k*n+i,:] for k in range(p)])
                b = np.ones(p)
                try:
                    sol = np.linalg.lstsq(A, b, rcond=None)[0]
                    U[i*m+j,:] = sol
                except:
                    pass
        
        # Optimize V - vectorized
        for j in range(m):
            for k in range(p):
                A = np.array([U[i*m+j,:] * W[k*n+i,:] for i in range(n)])
                b = np.ones(n)
                try:
                    sol = np.linalg.lstsq(A, b, rcond=None)[0]
                    V[j*p+k,:] = sol
                except:
                    pass
        
        # Optimize W - vectorized
        for i in range(n):
            for k in range(p):
                A = np.array([U[i*m+j,:] * V[j*p+k,:] for j in range(m)])
                b = np.ones(m)
                try:
                    sol = np.linalg.lstsq(A, b, rcond=None)[0]
                    W[k*n+i,:] = sol
                except:
                    pass
        
        err = _compute_error_fast(U, V, W, n, m, p)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < tol:
            break
    
    # Round and verify
    Ur = _round_to_half_integer(best_U)
    Vr = _round_to_half_integer(best_V)
    Wr = _round_to_half_integer(best_W)
    if _verify_fast(Ur.tolist(), Vr.tolist(), Wr.tolist(), n, m, p, atol=1e-9):
        return Ur, Vr, Wr
    return None

def _try_reduce_from_baseline(n, m, p, U_base, V_base, W_base, base_rank, deadline):
    """Try to reduce rank from baseline using ALS."""
    best_rank = base_rank
    best_U, best_V, best_W = U_base.copy(), V_base.copy(), W_base.copy()
    
    # First: try to optimize at same rank
    if time.time() < deadline - 30:
        result = _als_decompose(n, m, p, base_rank, U_base, V_base, W_base,
                                max_iters=150, deadline=deadline - 30)
        if result is not None:
            best_U, best_V, best_W = result
    
    # Try progressively lower ranks
    for target_rank in range(best_rank - 1, 45, -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        for attempt in range(3):
            if time.time() > deadline - 15:
                break
            
            if attempt == 0:
                U_init = U_base[:, :target_rank].copy()
                V_init = V_base[:, :target_rank].copy()
                W_init = W_base[:, :target_rank].copy()
            elif attempt == 1:
                U_init = U_base[:, :target_rank].copy() + np.random.randn(n*m, target_rank) * 0.01
                V_init = V_base[:, :target_rank].copy() + np.random.randn(m*p, target_rank) * 0.01
                W_init = W_base[:, :target_rank].copy() + np.random.randn(p*n, target_rank) * 0.01
            else:
                U_init = np.random.randn(n*m, target_rank) * 0.1
                V_init = np.random.randn(m*p, target_rank) * 0.1
                W_init = np.random.randn(p*n, target_rank) * 0.1
            
            result = _als_decompose(n, m, p, target_rank, U_init, V_init, W_init,
                                    max_iters=100, deadline=deadline - 15)
            if result is not None:
                best_U, best_V, best_W = result
                best_rank = target_rank
                found = True
                break
        
        if not found and target_rank < best_rank - 3:
            break
    
    return best_rank, best_U, best_V, best_W

def _schoolbook(n, m, p):
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
    return U, V, W

def construct():
    start = time.time()
    deadline = start + 140
    n, m, p = DIMENSIONS
    
    # Build Strassen baseline
    U_base, V_base, W_base = _strassen_construct(n, m, p)
    base_rank = U_base.shape[1]
    
    # Verify
    if not _verify_fast(U_base.tolist(), V_base.tolist(), W_base.tolist(), n, m, p):
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank
    best_rank, best_U, best_V, best_W = _try_reduce_from_baseline(
        n, m, p, U_base, V_base, W_base, base_rank, deadline
    )
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
