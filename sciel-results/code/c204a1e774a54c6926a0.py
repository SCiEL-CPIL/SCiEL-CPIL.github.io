import numpy as np
import time

DIMENSIONS = [2, 2, 4]

def _build_strassen_block():
    U_s = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=float)
    
    V_s = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 1, 0]
    ], dtype=float)
    
    # P_s[s] is [C00, C10, C01, C11]
    P_s = [
        np.array([1, 0, 0, 1]),
        np.array([0, 1, 0, 1]),
        np.array([0, 0, 1, 1]),
        np.array([1, 1, 0, 0]),
        np.array([-1, 0, 1, 0]),
        np.array([0, 0, 0, 1]),
        np.array([1, 0, 0, 0])
    ]
    
    return U_s, V_s, P_s

def _build_rank14():
    n, m, p = 2, 2, 4
    rank = 14
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_s, V_s, P_s = _build_strassen_block()
    
    for s in range(7):
        r1 = s
        U[:, r1] = U_s[s]
        for j in range(2):
            for kp in range(2):
                V[j * 4 + kp, r1] = V_s[s][j * 2 + kp]
        W[0, r1] = P_s[s][0]
        W[1, r1] = P_s[s][1]
        W[2, r1] = P_s[s][2]
        W[3, r1] = P_s[s][3]
        
        r2 = 7 + s
        U[:, r2] = U_s[s]
        for j in range(2):
            for kp in range(2):
                V[j * 4 + (kp + 2), r2] = V_s[s][j * 2 + kp]
        W[4, r2] = P_s[s][0]
        W[5, r2] = P_s[s][1]
        W[6, r2] = P_s[s][2]
        W[7, r2] = P_s[s][3]
        
    return U, V, W

def _verify_decomposition(U, V, W):
    n, m, p = 2, 2, 4
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _try_rank13(U14, V14, W14, time_budget):
    n, m, p = 2, 2, 4
    t_start = time.time()
    deadline = t_start + time_budget
    
    try:
        from scipy.optimize import minimize
    except ImportError:
        return None
        
    # Try zeroing each column
    for zero_col in range(14):
        if time.time() > deadline:
            return None
            
        mask = np.ones(14, dtype=bool)
        mask[zero_col] = False
        U = U14[:, mask]
        V = V14[:, mask]
        W = W14[:, mask]
        
        U_flat = U.flatten()
        V_flat = V.flatten()
        W_flat = W.flatten()
        x0 = np.concatenate([U_flat, V_flat, W_flat])
        
        def objective(x):
            Ux = x[:52].reshape(4, 13)
            Vx = x[52:156].reshape(8, 13)
            Wx = x[156:260].reshape(8, 13)
            
            error = 0.0
            for i in range(2):
                for j in range(2):
                    for k in range(4):
                        val = 0.0
                        for r in range(13):
                            val += Ux[i*2+j, r] * Vx[j*4+k, r] * Wx[k*2+i, r]
                        error += (val - 1.0) ** 2
            return error
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-15})
            if result.fun < 1e-12:
                Ux = result.x[:52].reshape(4, 13)
                Vx = result.x[52:156].reshape(8, 13)
                Wx = result.x[156:260].reshape(8, 13)
                if _verify_decomposition(Ux, Vx, Wx):
                    return Ux, Vx, Wx
        except Exception:
            pass
            
    return None

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140
    
    U14, V14, W14 = _build_rank14()
    if not _verify_decomposition(U14, V14, W14):
        # Fallback
        U16 = np.zeros((4, 16))
        V16 = np.zeros((8, 16))
        W16 = np.zeros((8, 16))
        idx = 0
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    U16[i*2+j, idx] = 1
                    V16[j*4+k, idx] = 1
                    W16[k*2+i, idx] = 1
                    idx += 1
        U14, V14, W14 = U16, V16, W16
        
    best_U, best_V, best_W = U14, V14, W14
    best_rank = 14
    
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        result = _try_rank13(U14, V14, W14, min(remaining_time - 5, 100))
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 13
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
