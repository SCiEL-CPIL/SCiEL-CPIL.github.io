# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if val != 1:
                    return False
    return True

def _find_2x2x3_rank10():
    n, m, p = 2, 2, 3
    rank = 10
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return None

    def residuals(x):
        U = x[0:40].reshape(4, 10)
        V = x[40:100].reshape(6, 10)
        W = x[100:160].reshape(6, 10)
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    res.append(s - 1)
        return res

    start_time = time.time()
    deadline = time.time() + 10.0 # 10s limit for search
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        x0 = np.random.randint(-5, 5, size=160).astype(float)
        x0[0] = 1
        x0[40] = 1
        x0[100] = 1
        
        try:
            sol = least_squares(residuals, x0, method='lm', max_nfev=500)
            if sol.success:
                x_int = np.round(sol.x).astype(int)
                U_int = x_int[0:40].reshape(4, 10)
                V_int = x_int[40:100].reshape(6, 10)
                W_int = x_int[100:160].reshape(6, 10)
                
                is_exact = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for r in range(rank):
                                s += U_int[i*m+j, r] * V_int[j*p+k, r] * W_int[k*n+i, r]
                            if s != 1:
                                is_exact = False
                                break
                    if not is_exact:
                        break
                if is_exact:
                    return (U_int.tolist(), V_int.tolist(), W_int.tolist())
        except Exception:
            continue
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n, m, p = DIMENSIONS
    
    block_223 = _find_2x2x3_rank10()
    
    if block_223 is not None:
        U_block, V_block, W_block = block_223
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for block_idx in range(5):
            k_start = block_idx * 3
            col_offset = block_idx * 10
            
            for r in range(10):
                c = col_offset + r
                for i in range(2):
                    for j in range(2):
                        U[i * m + j][c] = U_block[i * 2 + j][r]
                        
                for j in range(2):
                    for local_k in range(3):
                        global_k = k_start + local_k
                        V[j * p + global_k][c] = V_block[j * 3 + local_k][r]
                        
                for local_k in range(3):
                    global_k = k_start + local_k
                    for i in range(2):
                        W[global_k * n + i][c] = W_block[local_k * 2 + i][r]
                        
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to 53 rank
        rank = 53
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        U_strassen = [
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ]
        V_strassen = [
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, 0, 1, 0, 1]
        ]
        W_strassen = [
            [1, 0, 0, 1, -1, 0, 1],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ]
        U_school = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        V_school = [
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ]
        W_school = [
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        
        col_offset = 0
        for block in range(7):
            k0 = 2 * block
            k1 = 2 * block + 1
            for r in range(7):
                c = col_offset + r
                for i in range(2):
                    for j in range(2):
                        U[i * m + j][c] = U_strassen[i * 2 + j][r]
                for j in range(2):
                    V[j * p + k0][c] = V_strassen[j * 2 + 0][r]
                    V[j * p + k1][c] = V_strassen[j * 2 + 1][r]
                for k in range(2):
                    kk = k0 + k
                    for i in range(2):
                        W[kk * n + i][c] = W_strassen[k * 2 + i][r]
            col_offset += 7
            
        k_last = 14
        for r in range(4):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U[i * m + j][c] = U_school[i * 2 + j][r]
            for j in range(2):
                V[j * p + k_last][c] = V_school[j][r]
            for i in range(2):
                W[k_last * n + i][c] = W_school[i][r]
                
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
