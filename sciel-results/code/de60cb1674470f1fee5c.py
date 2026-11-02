import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 15]
TIME_LIMIT = 130

def _build_schoolbook(n, m, p):
    rank = n * m * p
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

def _build_strassen_2x2x2():
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Term 1: M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 2: M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # Term 3: M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # Term 4: M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # Term 5: M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # Term 6: M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 7: M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_rank53():
    n, m, p = DIMENSIONS
    block_sizes = [2, 2, 2, 2, 2, 2, 2, 1]
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
    
    total_rank = 7 * 7 + 4
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for b, block_size in enumerate(block_sizes):
        start_k = sum(block_sizes[:b])
        if block_size == 2:
            U[:, col_offset:col_offset+7] = U_str
            for j in range(m):
                for k_local in range(2):
                    V[j * p + start_k + k_local, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
            for k_local in range(2):
                for i in range(n):
                    W[(start_k + k_local) * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
            col_offset += 7
        else:
            U[:, col_offset:col_offset+4] = U_sb
            for j in range(m):
                V[j * p + start_k, col_offset:col_offset+4] = V_sb[j * 1, :]
            for i in range(n):
                W[start_k * n + i, col_offset:col_offset+4] = W_sb[0 * 2 + i, :]
            col_offset += 4
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    rank = U.shape[1]
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_find_rank16_2x2x5(time_budget=10.0):
    start_time = time.time()
    n, m, p = 2, 2, 5
    rank = 16
    
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def residuals(x):
        U = x[0:4*rank].reshape(4, rank)
        V = x[4*rank:4*rank + 10*rank].reshape(10, rank)
        W = x[4*rank + 10*rank:].reshape(10, rank)
        
        res = np.zeros(len(indices))
        for idx, (i, j, k) in enumerate(indices):
            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
            res[idx] = val - 1.0
        return res
    
    best_sol = None
    best_err = float('inf')
    
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() - start_time > time_budget:
            break
            
        U0 = np.random.randn(4, rank)
        V0 = np.random.randn(10, rank)
        W0 = np.random.randn(10, rank)
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            sol = least_squares(residuals, x0, method='lm', max_nfev=500)
            if sol.cost < best_err:
                best_err = sol.cost
                best_sol = sol.x
        except Exception:
            continue
            
    if best_sol is not None and best_err < 1e-10:
        U = best_sol[0:4*rank].reshape(4, rank)
        V = best_sol[4*rank:4*rank + 10*rank].reshape(10, rank)
        W = best_sol[4*rank + 10*rank:].reshape(10, rank)
        
        def round_to_half(x):
            return np.round(x * 2) / 2
            
        U_int = round_to_half(U)
        V_int = round_to_half(V)
        W_int = round_to_half(W)
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_int[i*m+j, :] * V_int[j*p+k, :] * W_int[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid: break
            if not valid: break
            
        if valid:
            return U_int, V_int, W_int
            
    return None

def _try_find_rank10_2x2x3(time_budget=10.0):
    start_time = time.time()
    n, m, p = 2, 2, 3
    rank = 10
    
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def residuals(x):
        U = x[0:4*rank].reshape(4, rank)
        V = x[4*rank:4*rank + 6*rank].reshape(6, rank)
        W = x[4*rank + 6*rank:].reshape(6, rank)
        
        res = np.zeros(len(indices))
        for idx, (i, j, k) in enumerate(indices):
            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
            res[idx] = val - 1.0
        return res
    
    best_sol = None
    best_err = float('inf')
    
    num_restarts = 100
    for r in range(num_restarts):
        if time.time() - start_time > time_budget:
            break
            
        U0 = np.random.randn(4, rank)
        V0 = np.random.randn(6, rank)
        W0 = np.random.randn(6, rank)
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            sol = least_squares(residuals, x0, method='lm', max_nfev=500)
            if sol.cost < best_err:
                best_err = sol.cost
                best_sol = sol.x
        except Exception:
            continue
            
    if best_sol is not None and best_err < 1e-10:
        U = best_sol[0:4*rank].reshape(4, rank)
        V = best_sol[4*rank:4*rank + 6*rank].reshape(6, rank)
        W = best_sol[4*rank + 6*rank:].reshape(6, rank)
        
        def round_to_half(x):
            return np.round(x * 2) / 2
            
        U_int = round_to_half(U)
        V_int = round_to_half(V)
        W_int = round_to_half(W)
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_int[i*m+j, :] * V_int[j*p+k, :] * W_int[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid: break
            if not valid: break
            
        if valid:
            return U_int, V_int, W_int
            
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try Rank 48 (3 blocks of rank 16 2x2x5)
    if time.time() - start_time < TIME_LIMIT - 10:
        try:
            U_16, V_16, W_16 = _try_find_rank16_2x2x5(time_budget=TIME_LIMIT - 20)
            if U_16 is not None:
                total_rank = 48
                U_full = np.zeros((4, total_rank))
                V_full = np.zeros((30, total_rank))
                W_full = np.zeros((30, total_rank))
                
                for b in range(3):
                    start_k = b * 5
                    col_off = b * 16
                    U_full[:, col_off:col_off+16] = U_16
                    for j in range(2):
                        for k_local in range(5):
                            V_full[j*15 + start_k + k_local, col_off:col_off+16] = V_16[j*5 + k_local, :]
                    for k_local in range(5):
                        for i in range(2):
                            W_full[(start_k + k_local)*2 + i, col_off:col_off+16] = W_16[k_local*2 + i, :]
                
                if _verify_decomposition(U_full, V_full, W_full, n, m, p):
                    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
        except Exception:
            pass
            
    # Try Rank 52 (6 blocks of rank 7 2x2x2 + 1 block of rank 10 2x2x3)
    if time.time() - start_time < TIME_LIMIT - 5:
        try:
            U_10, V_10, W_10 = _try_find_rank10_2x2x3(time_budget=TIME_LIMIT - 15)
            if U_10 is not None:
                total_rank = 6 * 7 + 10
                U_full = np.zeros((4, total_rank))
                V_full = np.zeros((30, total_rank))
                W_full = np.zeros((30, total_rank))
                
                U_str, V_str, W_str = _build_strassen_2x2x2()
                
                # First 6 blocks of size 2 (using Strassen)
                for b in range(6):
                    start_k = b * 2
                    col_off = b * 7
                    U_full[:, col_off:col_off+7] = U_str
                    for j in range(2):
                        for k_local in range(2):
                            V_full[j*15 + start_k + k_local, col_off:col_off+7] = V_str[j*2 + k_local, :]
                    for k_local in range(2):
                        for i in range(2):
                            W_full[(start_k + k_local)*2 + i, col_off:col_off+7] = W_str[k_local*2 + i, :]
                
                # Last block of size 3 (using rank 10)
                start_k = 12
                col_off = 42
                U_full[:, col_off:col_off+10] = U_10
                for j in range(2):
                    for k_local in range(3):
                        V_full[j*15 + start_k + k_local, col_off:col_off+10] = V_10[j*3 + k_local, :]
                for k_local in range(3):
                    for i in range(2):
                        W_full[(start_k + k_local)*2 + i, col_off:col_off+10] = W_10[k_local*2 + i, :]
                
                if _verify_decomposition(U_full, V_full, W_full, n, m, p):
                    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
        except Exception:
            pass
            
    # Fallback Rank 53
    U, V, W = _build_rank53()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
