import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n, m, p = DIMENSIONS
    
    # Fallback: Rank 53 solution
    def get_fallback():
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

    # Try to find rank 10 decomposition for M_{2,2,3}
    # Indices: i in 0..1, j in 0..1, k in 0..2
    # U: 4x10, V: 6x10, W: 6x10
    # Tensor entries T[i,j,k] = 1 for all i,j,k
    # Equation: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    def residuals(x):
        # x is flat vector of length 160
        U_flat = x[:40]
        V_flat = x[40:100]
        W_flat = x[100:160]
        
        U = U_flat.reshape(4, 10)
        V = V_flat.reshape(6, 10)
        W = W_flat.reshape(6, 10)
        
        res = []
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    # U index: i*2 + j
                    # V index: j*3 + k
                    # W index: k*2 + i
                    val = np.sum(U[i*2 + j, :] * V[j*3 + k, :] * W[k*2 + i, :])
                    res.append(val - 1.0)
        return res

    # Initialize with random values
    np.random.seed(42)
    x0 = np.random.uniform(-1, 1, 160)
    
    try:
        # Use least_squares to minimize residuals
        # This might take some time, so we check deadline
        if time.time() < deadline - 10:
            sol = least_squares(residuals, x0, method='lm', max_nfev=1000)
            if sol.success and np.max(np.abs(sol.fun)) < 1e-6:
                # Check if solution is close to integers
                U_sol = sol.x[:40].reshape(4, 10)
                V_sol = sol.x[40:100].reshape(6, 10)
                W_sol = sol.x[100:160].reshape(6, 10)
                
                # Round to nearest integer
                U_int = np.round(U_sol)
                V_int = np.round(V_sol)
                W_int = np.round(W_sol)
                
                # Verify exactness
                valid = True
                for i in range(2):
                    for j in range(2):
                        for k in range(3):
                            val = np.sum(U_int[i*2 + j, :] * V_int[j*3 + k, :] * W_int[k*2 + i, :])
                            if not np.isclose(val, 1.0):
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                
                if valid:
                    # Construct full 2x2x15 using 5 blocks of 2x2x3
                    rank = 50
                    U_full = [[0] * rank for _ in range(n * m)]
                    V_full = [[0] * rank for _ in range(m * p)]
                    W_full = [[0] * rank for _ in range(p * n)]
                    
                    for block in range(5):
                        k0 = 3 * block
                        col_offset = 10 * block
                        for r in range(10):
                            c = col_offset + r
                            for i in range(2):
                                for j in range(2):
                                    U_full[i * m + j][c] = int(U_int[i * 2 + j][r])
                            for j in range(2):
                                for k in range(3):
                                    kk = k0 + k
                                    V_full[j * p + kk][c] = int(V_int[j * 3 + k][r])
                            for k in range(3):
                                kk = k0 + k
                                for i in range(2):
                                    W_full[kk * n + i][c] = int(W_int[k * 2 + i][r])
                    
                    return {"U": U_full, "V": V_full, "W": W_full}
    except Exception:
        pass

    # Fallback if search fails or times out
    return get_fallback()
