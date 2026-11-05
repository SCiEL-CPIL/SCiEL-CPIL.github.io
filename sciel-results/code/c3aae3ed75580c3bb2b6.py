import numpy as np
from fractions import Fraction
import time

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array([[float(x) for x in row] for row in U], dtype=float)
    V_arr = np.array([[float(x) for x in row] for row in V], dtype=float)
    W_arr = np.array([[float(x) for x in row] for row in W], dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = np.dot(U_arr[idx_u], V_arr[idx_v] * W_arr[idx_w])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_rank28():
    """Build the known rank-28 decomposition."""
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    # Part 1: j=2, all k (10 terms)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    # Part 2: j=0,1, k=4 (4 terms)
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
    # Part 3: j=0,1, k=0,1 using Strassen (7 terms)
    U[0][r] = 1; U[4][r] = 1
    V[0][r] = 1; V[6][r] = 1
    W[0][r] = 1; W[3][r] = 1
    r += 1
    U[3][r] = 1; U[4][r] = 1
    V[0][r] = 1
    W[1][r] = 1; W[3][r] = -1
    r += 1
    U[0][r] = 1
    V[1][r] = 1; V[6][r] = -1
    W[2][r] = 1; W[3][r] = 1
    r += 1
    U[4][r] = 1
    V[5][r] = 1; V[0][r] = -1
    W[0][r] = 1; W[1][r] = 1
    r += 1
    U[0][r] = 1; U[1][r] = 1
    V[6][r] = 1
    W[0][r] = -1; W[2][r] = 1
    r += 1
    U[3][r] = 1; U[0][r] = -1
    V[0][r] = 1; V[1][r] = 1
    W[3][r] = 1
    r += 1
    U[1][r] = 1; U[4][r] = -1
    V[5][r] = 1; V[6][r] = 1
    W[0][r] = 1
    r += 1
    # Part 4: j=0,1, k=2,3 using Strassen (7 terms)
    U[0][r] = 1; U[4][r] = 1
    V[2][r] = 1; V[8][r] = 1
    W[4][r] = 1; W[7][r] = 1
    r += 1
    U[3][r] = 1; U[4][r] = 1
    V[2][r] = 1
    W[5][r] = 1; W[7][r] = -1
    r += 1
    U[0][r] = 1
    V[3][r] = 1; V[8][r] = -1
    W[6][r] = 1; W[7][r] = 1
    r += 1
    U[4][r] = 1
    V[7][r] = 1; V[2][r] = -1
    W[4][r] = 1; W[5][r] = 1
    r += 1
    U[0][r] = 1; U[1][r] = 1
    V[8][r] = 1
    W[4][r] = -1; W[6][r] = 1
    r += 1
    U[3][r] = 1; U[0][r] = -1
    V[2][r] = 1; V[3][r] = 1
    W[7][r] = 1
    r += 1
    U[1][r] = 1; U[4][r] = -1
    V[7][r] = 1; V[8][r] = 1
    W[4][r] = 1
    r += 1
    return U, V, W

def _residual_and_grad(x, rank, n, m, p):
    """Compute residual vector and its Jacobian for least_squares."""
    nm, mp, pn = n * m, m * p, p * n
    U = x[:rank*nm].reshape(rank, nm).T  # (nm, rank)
    V = x[rank*nm:rank*nm+rank*mp].reshape(rank, mp).T  # (mp, rank)
    W = x[rank*nm+rank*mp:].reshape(rank, pn).T  # (pn, rank)
    
    # Residual: for each (i,j,k), residual = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1
    res = np.zeros(n*m*p)
    idx = 0
    for i in range(n):
        for j in range(m):
            u_row = U[i*m+j]  # (rank,)
            for k in range(p):
                v_w = V[j*p+k] * W[k*n+i]  # (rank,)
                res[idx] = np.dot(u_row, v_w) - 1.0
                idx += 1
    
    # Jacobian: d(res[idx])/d(x)
    # For each (i,j,k), res = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1
    # d(res)/d(U[i*m+j,r]) = V[j*p+k,r]*W[k*n+i,r]
    # d(res)/d(V[j*p+k,r]) = U[i*m+j,r]*W[k*n+i,r]
    # d(res)/d(W[k*n+i,r]) = U[i*m+j,r]*V[j*p+k,r]
    
    J = np.zeros((n*m*p, rank*(nm+mp+pn)))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i*m*p + j*p + k
                u_row = U[i*m+j]
                v_row = V[j*p+k]
                w_row = W[k*n+i]
                for r in range(rank):
                    # d/dU[i*m+j, r]
                    J[idx, r] = v_row[r] * w_row[r]
                    # d/dV[j*p+k, r]
                    J[idx, nm*rank + (j*p+k)*rank + r] = u_row[r] * w_row[r]
                    # d/dW[k*n+i, r]
                    J[idx, (nm+mp)*rank + (k*n+i)*rank + r] = u_row[r] * v_row[r]
    
    return res, J

def _try_rank_lower(target_rank, n, m, p, base_U=None, base_V=None, base_W=None, time_budget=40):
    """Attempt to find a rank-target_rank decomposition using least_squares with multiple starts."""
    nm, mp, pn = n * m, m * p, p * n
    n_params = target_rank * (nm + mp + pn)
    start_time = time.time()
    
    best_res = np.inf
    best_x = None
    
    def _make_x0(scale=0.1, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return np.random.randn(n_params) * scale
    
    # Generate diverse initializations
    inits = []
    # Random with various scales
    for scale in [0.05, 0.1, 0.2, 0.5]:
        for s in range(3):
            inits.append(_make_x0(scale, seed=s*1000 + int(scale*100) + target_rank))
    
    # Base solution truncation
    if base_U is not None:
        base_rank = len(base_U[0])
        if base_rank > target_rank:
            U_base = np.array(base_U)[:, :target_rank]
            V_base = np.array(base_V)[:, :target_rank]
            W_base = np.array(base_W)[:, :target_rank]
            x0 = np.concatenate([U_base.flatten(), V_base.flatten(), W_base.flatten()])
            inits.append(x0)
    
    from scipy.optimize import least_squares
    
    for x0 in inits:
        if time.time() - start_time > time_budget:
            break
        try:
            result = least_squares(_residual_and_grad, x0, args=(target_rank, n, m, p),
                                   method='lm', max_nfev=10000)
            res_norm = np.linalg.norm(result.fun)
            if res_norm < best_res:
                best_res = res_norm
                best_x = result.x.copy()
                if best_res < 1e-10:
                    # Found a solution!
                    U = best_x[:target_rank*nm].reshape(target_rank, nm).T
                    V = best_x[target_rank*nm:target_rank*nm+target_rank*mp].reshape(target_rank, mp).T
                    W = best_x[target_rank*nm+target_rank*mp:].reshape(target_rank, pn).T
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    U_frac = [[Fraction(int(np.round(v * 2)), 2) for v in row] for row in U_r]
                    V_frac = [[Fraction(int(np.round(v * 2)), 2) for v in row] for row in V_r]
                    W_frac = [[Fraction(int(np.round(v * 2)), 2) for v in row] for row in W_r]
                    if _verify_decomposition(U_frac, V_frac, W_frac, n, m, p):
                        return (U_frac, V_frac, W_frac)
        except Exception:
            pass
    
    return None

def construct():
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = 130  # Safety margin before 150s kill
    
    U28, V28, W28 = _build_rank28()
    if not _verify_decomposition(U28, V28, W28, n, m, p):
        raise ValueError("Rank 28 verification failed")
    
    best_U, best_V, best_W = U28, V28, W28
    best_rank = 28
    
    # Try rank 27
    if time.time() - start_time < deadline - 50:
        result = _try_rank_lower(27, n, m, p, base_U=U28, base_V=V28, base_W=W28, time_budget=40)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 27
    
    # Try rank 26 if time permits
    if time.time() - start_time < deadline - 35 and best_rank > 26:
        result = _try_rank_lower(26, n, m, p, base_U=best_U, base_V=best_W, base_W=best_W, time_budget=30)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 26
    
    return {"U": best_U, "V": best_V, "W": best_W}
