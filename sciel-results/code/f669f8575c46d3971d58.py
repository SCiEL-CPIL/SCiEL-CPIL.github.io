import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 15]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)."""
    rank = U.shape[1]
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_strassen_2x2x2():
    """Strassen rank-7 decomposition for 2x2 x 2x2."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def _build_schoolbook_2x2x1():
    """Schoolbook rank-4 decomposition for 2x2 x 2x1."""
    n, m, p = 2, 2, 1
    rank = 4
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _build_rank53():
    """Build the guaranteed rank-53 decomposition using 7 Strassen blocks + 1 schoolbook."""
    n, m, p = DIMENSIONS
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook_2x2x1()

    total_rank = 7 * 7 + 4  # 53

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    col_offset = 0
    for b in range(7):
        start_k = 2 * b
        U[:, col_offset:col_offset+7] = U_str
        for j in range(m):
            for k_local in range(2):
                k_global = start_k + k_local
                V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
        for k_local in range(2):
            k_global = start_k + k_local
            for i in range(n):
                W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
        col_offset += 7

    # Last column: schoolbook for k=14
    U[:, col_offset:col_offset+4] = U_sb
    for j in range(m):
        V[j * p + 14, col_offset:col_offset+4] = V_sb[j, :]
    for i in range(n):
        W[14 * n + i, col_offset:col_offset+4] = W_sb[i, :]

    return U, V, W

def _residuals_223(U, V, W):
    """Residuals for 2x2x3 tensor."""
    n, m, p = 2, 2, 3
    rank = U.shape[1]
    res = np.zeros(n * m * p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                res[idx] = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :]) - 1.0
                idx += 1
    return res

def _try_rank10_223(deadline):
    """Try to find a rank-10 decomposition for 2x2x3."""
    n, m, p = 2, 2, 3
    rank = 10
    
    def objective(params):
        U = params[:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
        W = params[n*m*rank + m*p*rank :].reshape(p*n, rank)
        res = _residuals_223(U, V, W)
        return np.sum(res**2)
    
    # Warm start: use rank-11 solution (7 Strassen + 4 Schoolbook)
    # We need to map this to rank 10. 
    # A simple warm start is to take the first 10 columns of a rank-11 solution.
    # Let's build a rank-11 solution first.
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook_2x2x1()
    
    U11 = np.zeros((4, 11))
    V11 = np.zeros((6, 11))
    W11 = np.zeros((6, 11))
    
    # Fill first 7 columns with Strassen
    U11[:, :7] = U_str
    V11[:, :7] = V_str
    W11[:, :7] = W_str
    
    # Fill next 4 columns with Schoolbook for k=2
    # Schoolbook for 2x2x1 has U (4x4), V (2x4), W (2x4)
    # We need to place these in the k=2 slice of V and W
    U11[:, 7:11] = U_sb
    for j in range(2):
        V11[j*3 + 2, 7:11] = V_sb[j, :]
    for i in range(2):
        W11[2*2 + i, 7:11] = W_sb[i, :]
        
    # Now we have a valid rank-11 solution.
    # We want to optimize to rank 10.
    # We can try to drop one column or merge.
    # Let's just optimize the first 10 columns of U11, V11, W11.
    x0 = np.concatenate([
        U11[:, :10].flatten(),
        V11[:, :10].flatten(),
        W11[:, :10].flatten()
    ])
    
    best_res = None
    best_val = np.inf
    
    # Try a few random perturbations or just the warm start
    for restart in range(5):
        if time.time() > deadline - 10:
            break
            
        if restart == 0:
            x_init = x0
        else:
            x_init = x0 + np.random.normal(0, 0.1, size=x0.shape)
            
        try:
            res = minimize(objective, x_init, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_res = res.x
                
                if res.fun < 1e-10:
                    U = res.x[:4*10].reshape(4, 10)
                    V = res.x[4*10 : 4*10 + 6*10].reshape(6, 10)
                    W = res.x[4*10 + 6*10 :].reshape(6, 10)
                    
                    # Check if it's close to half-integers
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    if np.max(np.abs(_residuals_223(U_snap, V_snap, W_snap))) < 1e-6:
                        return U_snap, V_snap, W_snap
                    # If raw is good but snap is not, keep raw? No, task requires integer/half-integer.
                    # But maybe the raw solution IS half-integer?
                    if np.allclose(U*2, np.round(U*2), atol=1e-4) and \
                       np.allclose(V*2, np.round(V*2), atol=1e-4) and \
                       np.allclose(W*2, np.round(W*2), atol=1e-4):
                        return U_snap, V_snap, W_snap
                        
        except Exception:
            pass
            
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # Baseline: rank-53 solution
    U53, V53, W53 = _build_rank53()
    if not _verify_decomposition(U53, V53, W53, n, m, p):
        # Fallback to schoolbook
        U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 15) # Need to implement or use generic
        # For now, just return the rank 53 if valid, else schoolbook
        return {"U": U53.tolist(), "V": V53.tolist(), "W": W53.tolist()}

    # Try to find a rank-10 solution for 2x2x3
    if time.time() < deadline - 20:
        U10, V10, W10 = _try_rank10_223(deadline)
        if U10 is not None:
            # Build rank-50 solution from 5 blocks of 2x2x3
            total_rank = 5 * 10
            U50 = np.zeros((n * m, total_rank))
            V50 = np.zeros((m * p, total_rank))
            W50 = np.zeros((p * n, total_rank))
            
            for b in range(5):
                col_offset = b * 10
                start_k = b * 3
                U50[:, col_offset:col_offset+10] = U10
                for j in range(2):
                    for k_local in range(3):
                        k_global = start_k + k_local
                        V50[j * p + k_global, col_offset:col_offset+10] = V10[j * 3 + k_local, :]
                for k_local in range(3):
                    k_global = start_k + k_local
                    for i in range(2):
                        W50[k_global * n + i, col_offset:col_offset+10] = W10[k_local * 2 + i, :]
                        
            if _verify_decomposition(U50, V50, W50, n, m, p, tol=1e-6):
                return {"U": U50.tolist(), "V": V50.tolist(), "W": W50.tolist()}

    # Fallback: rank-53
    return {"U": U53.tolist(), "V": V53.tolist(), "W": W53.tolist()}
