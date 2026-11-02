"""
Rank-minimal tensor decomposition for 2x2 @ 2x5 matrix multiplication.
Strategy: Fast ALS from diverse initializations + rational search for rank-14.
"""
import numpy as np
import time

DIMENSIONS = [2, 2, 5]

def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1
    return T

def _check_decomposition(U, V, W, T):
    U_a = np.array(U, dtype=np.float64)
    V_a = np.array(V, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    R = U_a.shape[1]
    result = np.zeros(T.shape, dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U_a[:,r], V_a[:,r], W_a[:,r])
    return np.allclose(result, T.astype(np.float64), atol=1e-9)

def _build_rank18(n, m, p):
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(col0, col1, base):
        r = base
        U[0][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 1
        U[2][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1
        W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
        
        r = base + 2
        U[0][r] = 1
        V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
        W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 3
        U[3][r] = 1
        V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
        W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
        
        r = base + 4
        U[0][r] = 1; U[1][r] = 1
        V[1*p+col1][r] = 1
        W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
        
        r = base + 5
        U[2][r] = 1; U[0][r] = -1
        V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
        W[col1*n+1][r] = 1
        
        r = base + 6
        U[1][r] = 1; U[3][r] = -1
        V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1
    
    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    r = 14
    U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15
    U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16
    U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17
    U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
    
    return U, V, W

def _fast_als(T_a, U, V, W, n, m, p, R, max_iters=200, tol=1e-12, deadline=None):
    """Fast ALS with precomputed products and early termination."""
    nm, mp, pn = n*m, m*p, p*n
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    prev_err = np.inf
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Precompute VW, UW, UV design matrices
        VW = np.einsum('vr,wr->rvw', V, W)
        VW_flat = VW.reshape(R, -1).T  # (mp*pn, R)
        
        U_new = np.zeros((nm, R), dtype=np.float64)
        for u in range(nm):
            b = T_a[u].reshape(-1)
            sol = np.linalg.lstsq(VW_flat, b, rcond=None)[0]
            U_new[u] = sol
        U = U_new
        
        UW = np.einsum('ur,wr->ruw', U, W)
        UW_flat = UW.reshape(R, -1).T  # (nm*pn, R)
        
        V_new = np.zeros((mp, R), dtype=np.float64)
        for v in range(mp):
            b = T_a[:, v, :].reshape(-1)
            sol = np.linalg.lstsq(UW_flat, b, rcond=None)[0]
            V_new[v] = sol
        V = V_new
        
        UV = np.einsum('ur,vr->ruv', U, V)
        UV_flat = UV.reshape(R, -1).T  # (nm*mp, R)
        
        W_new = np.zeros((pn, R), dtype=np.float64)
        for w in range(pn):
            b = T_a[:, :, w].reshape(-1)
            sol = np.linalg.lstsq(UV_flat, b, rcond=None)[0]
            W_new[w] = sol
        W = W_new
        
        # Compute error
        err = 0.0
        for r in range(R):
            diff = np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r]) - T_a
            err += np.sum(diff * diff)
        err = np.sqrt(err / (nm * mp * pn))
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W, err

def _try_rational(U, V, W, T, max_denom=16):
    """Try to find rational/half-integer representation."""
    for denom in [2, 4, 8, 16, 32, 64]:
        if max_denom and denom > max_denom:
            break
        U_s = np.round(U * denom) / denom
        V_s = np.round(V * denom) / denom
        W_s = np.round(W * denom) / denom
        if _check_decomposition(U_s, V_s, W_s, T):
            return U_s.tolist(), V_s.tolist(), W_s.tolist()
    if _check_decomposition(U, V, W, T):
        return U.tolist(), V.tolist(), W.tolist()
    return None

def _search_rank(T, T_a, n, m, p, R, deadline, U18_a, V18_a, W18_a):
    """Search for rank-R decomposition using multiple strategies."""
    nm, mp, pn = n*m, m*p, p*n
    
    # Strategy 1: Remove terms from rank-18 and perturb
    if R <= 17:
        for term_to_remove in range(18 - R + 1):
            if time.time() > deadline:
                break
            idx = [i for i in range(18) if i != term_to_remove]
            if len(idx) > R:
                idx = idx[:R]
            elif len(idx) < R:
                continue
            U_init = U18_a[:, idx].copy()
            V_init = V18_a[:, idx].copy()
            W_init = W18_a[:, idx].copy()
            
            for seed in range(5):
                if time.time() > deadline:
                    break
                np.random.seed(1000 + term_to_remove * 10 + seed)
                scale = 0.05
                U_pert = U_init + np.random.randn(nm, R) * scale
                V_pert = V_init + np.random.randn(mp, R) * scale
                W_pert = W_init + np.random.randn(pn, R) * scale
                
                U, V, W, err = _fast_als(T_a, U_pert, V_pert, W_pert, n, m, p, R, max_iters=100, tol=1e-12, deadline=deadline)
                if err < 1e-10:
                    result = _try_rational(U, V, W, T)
                    if result is not None:
                        return result
    
    # Strategy 2: Random initializations
    for seed in range(30):
        if time.time() > deadline:
            break
        np.random.seed(2000 + seed * 7)
        scale = 0.1 + 0.1 * (seed % 3)
        U_init = np.random.randn(nm, R) * scale
        V_init = np.random.randn(mp, R) * scale
        W_init = np.random.randn(pn, R) * scale
        
        U, V, W, err = _fast_als(T_a, U_init, V_init, W_init, n, m, p, R, max_iters=80, tol=1e-12, deadline=deadline)
        if err < 1e-10:
            result = _try_rational(U, V, W, T)
            if result is not None:
                return result
    
    # Strategy 3: SVD-informed
    try:
        T_flat = T_a.reshape(nm, mp * pn)
        U_svd, S, Vt = np.linalg.svd(T_flat, full_matrices=False)
        for seed in range(15):
            if time.time() > deadline:
                break
            np.random.seed(3000 + seed * 11)
            if U_svd.shape[1] >= R:
                U_init = 0.5 * U_svd[:, :R] + 0.5 * np.random.randn(nm, R) * 0.2
            else:
                U_init = np.random.randn(nm, R) * 0.2
            V_init = np.random.randn(mp, R) * 0.2
            W_init = np.random.randn(pn, R) * 0.2
            
            U, V, W, err = _fast_als(T_a, U_init, V_init, W_init, n, m, p, R, max_iters=80, tol=1e-12, deadline=deadline)
            if err < 1e-10:
                result = _try_rational(U, V, W, T)
                if result is not None:
                    return result
    except:
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    T = _build_target()
    T_a = T.astype(np.float64)
    nm, mp, pn = n*m, m*p, p*n
    
    # Build and verify rank-18
    U18, V18, W18 = _build_rank18(n, m, p)
    U18_a = np.array(U18, dtype=np.float64)
    V18_a = np.array(V18, dtype=np.float64)
    W18_a = np.array(W18, dtype=np.float64)
    
    if not _check_decomposition(U18, V18, W18, T):
        # Fallback to schoolbook rank-20
        rank = 20
        U = [[0]*rank for _ in range(nm)]
        V = [[0]*rank for _ in range(mp)]
        W = [[0]*rank for _ in range(pn)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
        return {"U": U, "V": V, "W": W}
    
    # Try rank-14 first (known optimal for 2x2x5)
    if time.time() < deadline - 10:
        result14 = _search_rank(T, T_a, n, m, p, 14, deadline - 10, U18_a, V18_a, W18_a)
        if result14 is not None:
            U, V, W = result14
            if _check_decomposition(U, V, W, T):
                return {"U": U, "V": V, "W": W}
    
    # Try rank-15
    if time.time() < deadline - 10:
        result15 = _search_rank(T, T_a, n, m, p, 15, deadline - 10, U18_a, V18_a, W18_a)
        if result15 is not None:
            U, V, W = result15
            if _check_decomposition(U, V, W, T):
                return {"U": U, "V": V, "W": W}
    
    # Try rank-16
    if time.time() < deadline - 10:
        result16 = _search_rank(T, T_a, n, m, p, 16, deadline - 10, U18_a, V18_a, W18_a)
        if result16 is not None:
            U, V, W = result16
            if _check_decomposition(U, V, W, T):
                return {"U": U, "V": V, "W": W}
    
    # Try rank-17
    if time.time() < deadline - 5:
        result17 = _search_rank(T, T_a, n, m, p, 17, deadline - 5, U18_a, V18_a, W18_a)
        if result17 is not None:
            U, V, W = result17
            if _check_decomposition(U, V, W, T):
                return {"U": U, "V": V, "W": W}
    
    # Fallback: rank-18
    return {"U": U18, "V": V18, "W": W18}
