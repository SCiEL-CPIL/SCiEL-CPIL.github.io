"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    num_blocks = p // 2  # 8 blocks of 2 columns
    rank = num_blocks * 7  # 56
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 32 x 56
    
    for block in range(num_blocks):
        k0 = block * 2
        k1 = k0 + 1
        base = block * 7
        
        # M1 = (A00+A11)(B0k0+B1k1)
        U[0][base + 0] = 1
        U[3][base + 0] = 1
        V[k0][base + 0] = 1
        V[16 + k1][base + 0] = 1
        W[k0 * 2 + 0][base + 0] = 1
        W[k1 * 2 + 1][base + 0] = 1
        
        # M2 = (A10+A11)B0k0
        U[2][base + 1] = 1
        U[3][base + 1] = 1
        V[k0][base + 1] = 1
        W[k0 * 2 + 1][base + 1] = 1
        W[k1 * 2 + 1][base + 1] = -1
        
        # M3 = A00(B1k0-B1k1)
        U[0][base + 2] = 1
        V[16 + k0][base + 2] = 1
        V[16 + k1][base + 2] = -1
        W[k1 * 2 + 0][base + 2] = 1
        W[k1 * 2 + 1][base + 2] = 1
        
        # M4 = A11(B0k1-B0k0)
        U[3][base + 3] = 1
        V[k1][base + 3] = 1
        V[k0][base + 3] = -1
        W[k0 * 2 + 0][base + 3] = 1
        W[k0 * 2 + 1][base + 3] = 1
        
        # M5 = (A00+A01)B1k1
        U[0][base + 4] = 1
        U[1][base + 4] = 1
        V[16 + k1][base + 4] = 1
        W[k0 * 2 + 0][base + 4] = -1
        W[k1 * 2 + 0][base + 4] = 1
        
        # M6 = (A10-A00)(B0k0+B0k1)
        U[2][base + 5] = 1
        U[0][base + 5] = -1
        V[k0][base + 5] = 1
        V[k1][base + 5] = 1
        W[k1 * 2 + 1][base + 5] = 1
        
        # M7 = (A01-A11)(B1k0+B1k1)
        U[1][base + 6] = 1
        U[3][base + 6] = -1
        V[16 + k0][base + 6] = 1
        V[16 + k1][base + 6] = 1
        W[k0 * 2 + 0][base + 6] = 1
    
    # Vectorized verification
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check all (i,j,k) entries efficiently
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            u_row = U_arr[i * m + j]  # shape (rank,)
            for k in range(p):
                v_row = V_arr[j * p + k]  # shape (rank,)
                w_row = W_arr[k * n + i]  # shape (rank,)
                val = np.dot(u_row, v_row * w_row)
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-9:
        # Fall back to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_fb[ii * m + jj][col] = 1
                    V_fb[jj * p + kk][col] = 1
                    W_fb[kk * n + ii][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
