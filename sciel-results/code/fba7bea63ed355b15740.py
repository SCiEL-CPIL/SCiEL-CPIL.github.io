import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # Strassen 7 products for A[0:2, 0:2] * B[0:2, k0:k1+1]
        
        # M1: (a00 + a11) * (B[0,k0] + B[1,k1])
        U[0][col] += 1  # a00
        U[4][col] += 1  # a11
        V[k0][col] += 1  # B[0,k0]
        V[10 + k1][col] += 1  # B[1,k1]
        W[k0 * 2 + 0][col] += 1  # C[0,k0]
        W[k1 * 2 + 1][col] += 1  # C[1,k1]
        col += 1
        
        # M2: (a10 + a11) * B[0,k0]
        U[3][col] += 1  # a10
        U[4][col] += 1  # a11
        V[k0][col] += 1  # B[0,k0]
        W[k0 * 2 + 1][col] += 1  # C[1,k0]
        W[k1 * 2 + 1][col] -= 1  # -C[1,k1]
        col += 1
        
        # M3: a00 * (B[0,k1] - B[1,k1])
        U[0][col] += 1  # a00
        V[k1][col] += 1  # B[0,k1]
        V[10 + k1][col] -= 1  # -B[1,k1]
        W[k1 * 2 + 0][col] += 1  # C[0,k1]
        W[k1 * 2 + 1][col] += 1  # C[1,k1]
        col += 1
        
        # M4: a11 * (B[1,k0] - B[0,k0])
        U[4][col] += 1  # a11
        V[10 + k0][col] += 1  # B[1,k0]
        V[k0][col] -= 1  # -B[0,k0]
        W[k0 * 2 + 0][col] += 1  # C[0,k0]
        W[k0 * 2 + 1][col] += 1  # C[1,k0]
        col += 1
        
        # M5: (a00 + a01) * B[1,k1]
        U[0][col] += 1  # a00
        U[1][col] += 1  # a01
        V[10 + k1][col] += 1  # B[1,k1]
        W[k0 * 2 + 0][col] -= 1  # -C[0,k0]
        W[k1 * 2 + 0][col] += 1  # C[0,k1]
        col += 1
        
        # M6: (a10 - a00) * (B[0,k0] + B[0,k1])
        U[3][col] += 1  # a10
        U[0][col] -= 1  # -a00
        V[k0][col] += 1  # B[0,k0]
        V[k1][col] += 1  # B[0,k1]
        W[k1 * 2 + 1][col] += 1  # C[1,k1]
        col += 1
        
        # M7: (a01 - a11) * (B[1,k0] + B[1,k1])
        U[1][col] += 1  # a01
        U[4][col] -= 1  # -a11
        V[10 + k0][col] += 1  # B[1,k0]
        V[10 + k1][col] += 1  # B[1,k1]
        W[k0 * 2 + 0][col] += 1  # C[0,k0]
        col += 1
        
        # Schoolbook for A[:,2] * B[2, k0:k1+1]
        U[2][col] += 1  # a02
        V[20 + k0][col] += 1  # B[2,k0]
        W[k0 * 2 + 0][col] += 1  # C[0,k0]
        col += 1
        
        U[2][col] += 1  # a02
        V[20 + k1][col] += 1  # B[2,k1]
        W[k1 * 2 + 0][col] += 1  # C[0,k1]
        col += 1
        
        U[5][col] += 1  # a12
        V[20 + k0][col] += 1  # B[2,k0]
        W[k0 * 2 + 1][col] += 1  # C[1,k0]
        col += 1
        
        U[5][col] += 1  # a12
        V[20 + k1][col] += 1  # B[2,k1]
        W[k1 * 2 + 1][col] += 1  # C[1,k1]
        col += 1
    
    # Verify: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check all (i,j,k) tuples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0.0
                for r in range(rank):
                    val += U_np[u_idx, r] * V_np[v_idx, r] * W_np[w_idx, r]
                if abs(val - 1.0) > 1e-9:
                    # Decomposition is wrong; fall back to schoolbook
                    rank_sb = n * m * p
                    U_sb = [[0] * rank_sb for _ in range(n * m)]
                    V_sb = [[0] * rank_sb for _ in range(m * p)]
                    W_sb = [[0] * rank_sb for _ in range(p * n)]
                    col_sb = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_sb[ii * m + jj][col_sb] = 1
                                V_sb[jj * p + kk][col_sb] = 1
                                W_sb[kk * n + ii][col_sb] = 1
                                col_sb += 1
                    return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U, "V": V, "W": W}
