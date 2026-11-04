import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                if abs(actual - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
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

def _als_decompose(T, rank, deadline, seed=0, max_sweeps=300):
    """ALS decomposition with careful convergence handling."""
    rng = np.random.RandomState(seed)
    d1, d2, d3 = T.shape
    
    # Multiple initialization strategies
    best_loss = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for init_iter in range(5):
        if time.time() > deadline - 2.0:
            break
        
        # Random initialization with varying scales
        scale = 0.5 + 0.5 * init_iter
        U = rng.randn(d1, rank) * scale
        V = rng.randn(d2, rank) * scale
        W = rng.randn(d3, rank) * scale
        
        loss_prev = float('inf')
        
        for sweep in range(max_sweeps):
            if time.time() > deadline - 1.0:
                break
            
            # Update U
            VW = np.einsum('jc,kc->jkc', V, W).reshape(-1, rank)
            T1 = T.reshape(d1, -1)
            U_new, _, _, _ = np.linalg.lstsq(VW, T1.T, rcond=None)
            U = U_new.T
            
            # Update V
            UW = np.einsum('ic,kc->ikc', U, W).reshape(-1, rank)
            T2 = np.transpose(T, (1, 0, 2)).reshape(d2, -1)
            V_new, _, _, _ = np.linalg.lstsq(UW, T2.T, rcond=None)
            V = V_new.T
            
            # Update W
            UV = np.einsum('ic,jc->ijc', U, V).reshape(-1, rank)
            T3 = np.transpose(T, (2, 0, 1)).reshape(d3, -1)
            W_new, _, _, _ = np.linalg.lstsq(UV, T3.T, rcond=None)
            W = W_new.T
            
            # Compute loss
            recon = np.einsum('ic,jc,kc->ijk', U, V, W)
            loss = np.sum((recon - T) ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Convergence check
            if abs(loss_prev - loss) < 1e-12:
                break
            loss_prev = loss
            
            if loss < 1e-10:
                break
    
    if best_U is not None and best_loss < 1e-8:
        U_list = [[float(round(x, 10)) for x in row] for row in best_U]
        V_list = [[float(round(x, 10)) for x in row] for row in best_V]
        W_list = [[float(round(x, 10)) for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_rank(T, n, m, p, rank, deadline, max_trials=20):
    for seed in range(max_trials):
        if time.time() > deadline - 5.0:
            break
        result = _als_decompose(T, rank, deadline, seed=seed, max_sweeps=200)
        if result is not None:
            if _verify_decomposition(result['U'], result['V'], result['W'], n, m, p, tol=1e-6):
                return result
    return None

def _construct_2x3x6_from_2x3x3(res_233):
    """Construct 2x3x6 decomposition from two copies of 2x3x3."""
    U_233 = np.array(res_233['U'], dtype=float)
    V_233 = np.array(res_233['V'], dtype=float)
    W_233 = np.array(res_233['W'], dtype=float)
    
    R = U_233.shape[1]
    
    # U is shared (same A matrix for both blocks of B)
    U_new = np.hstack([U_233, U_233])
    
    # V: first 9 rows for B1, last 9 rows for B2
    V_new = np.zeros((18, 2 * R))
    V_new[0:9, 0:R] = V_233
    V_new[9:18, R:2 * R] = V_233
    
    # W: first 6 rows for C1, last 6 rows for C2
    W_new = np.zeros((12, 2 * R))
    W_new[0:6, 0:R] = W_233
    W_new[6:12, R:2 * R] = W_233
    
    U_list = [[float(round(x, 10)) for x in row] for row in U_new]
    V_list = [[float(round(x, 10)) for x in row] for row in V_new]
    W_list = [[float(round(x, 10)) for x in row] for row in W_new]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Strategy 1: Find rank-7 for 2x3x3, then build 2x3x6 with rank 14
    T_233 = _build_tensor(2, 3, 3)
    res_233 = _try_rank(T_233, 2, 3, 3, 7, deadline, max_trials=30)
    
    if res_233 is not None:
        res_236 = _construct_2x3x6_from_2x3x3(res_233)
        if _verify_decomposition(res_236['U'], res_236['V'], res_236['W'], 2, 3, 6, tol=1e-6):
            return res_236
    
    # Strategy 2: Direct search for 2x3x6 with rank 14
    T_236 = _build_tensor(2, 3, 6)
    res_236 = _try_rank(T_236, 2, 3, 6, 14, deadline, max_trials=20)
    if res_236 is not None:
        return res_236
    
    # Strategy 3: Try rank 15
    res_236 = _try_rank(T_236, 2, 3, 6, 15, deadline, max_trials=15)
    if res_236 is not None:
        return res_236
    
    # Fallback: Schoolbook (rank 36)
    return _schoolbook_decomposition(2, 3, 6)
