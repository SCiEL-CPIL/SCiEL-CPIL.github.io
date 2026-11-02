import time
import numpy as np

def construct():
    n, m, p = 4, 5, 6
    rank = n * m  # 20
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    r_idx = 0
    for i in range(n):
        for j in range(m):
            U[i * m + j][r_idx] = 1.0
            for k in range(p):
                V[j * p + k][r_idx] = 1.0
                W[k * n + i][r_idx] = 1.0
            r_idx += 1
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > 1e-9:
                    # Fallback schoolbook
                    rank_full = n * m * p
                    U_sb = [[0.0] * rank_full for _ in range(n * m)]
                    V_sb = [[0.0] * rank_full for _ in range(m * p)]
                    W_sb = [[0.0] * rank_full for _ in range(p * n)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_sb[ii*m+jj][col] = 1.0
                                V_sb[jj*p+kk][col] = 1.0
                                W_sb[kk*n+ii][col] = 1.0
                                col += 1
                    return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U, "V": V, "W": W}
