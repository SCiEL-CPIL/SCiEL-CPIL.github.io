import numpy as np
import random
import time

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for jb in range(3):
        for s in range(7):
            col = jb * 7 + s
            # U
            if s == 0:
                U[0][col] = 1; U[3][col] = 1
            elif s == 1:
                U[2][col] = 1; U[3][col] = 1
            elif s == 2:
                U[0][col] = 1
            elif s == 3:
                U[3][col] = 1
            elif s == 4:
                U[0][col] = 1; U[1][col] = 1
            elif s == 5:
                U[2][col] = 1; U[0][col] = -1
            elif s == 6:
                U[1][col] = 1; U[3][col] = -1
            
            # V
            e_idx = 2 * jb
            f_idx = 2 * jb + 1
            g_idx = 6 + 2 * jb
            h_idx = 6 + 2 * jb + 1
            
            if s == 0:
                V[e_idx][col] = 1; V[h_idx][col] = 1
            elif s == 1:
                V[e_idx][col] = 1
            elif s == 2:
                V[g_idx][col] = 1; V[h_idx][col] = -1
            elif s == 3:
                V[f_idx][col] = 1; V[e_idx][col] = -1
            elif s == 4:
                V[h_idx][col] = 1
            elif s == 5:
                V[e_idx][col] = 1; V[f_idx][col] = 1
            elif s == 6:
                V[g_idx][col] = 1; V[h_idx][col] = 1
            
            # W
            c00 = 4 * jb
            c01 = 4 * jb + 2
            c10 = 4 * jb + 1
            c11 = 4 * jb + 3
            
            if s == 0:
                W[c00][col] = 1; W[c11][col] = 1
            elif s == 1:
                W[c10][col] = -1; W[c11][col] = -1
            elif s == 2:
                W[c10][col] = 1; W[c11][col] = 1
            elif s == 3:
                W[c00][col] = -1; W[c01][col] = 1
            elif s == 4:
                W[c00][col] = -1; W[c01][col] = 1
            elif s == 5:
                W[c11][col] = -1
            elif s == 6:
                W[c00][col] = 1
    
    # Verify
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for _ in range(10):
        A = np.random.randint(-5, 5, (n, m)).astype(float)
        B = np.random.randint(-5, 5, (m, p)).astype(float)
        C_expected = A @ B
        
        A_flat = A.flatten()  # [a00, a01, a10, a11]
        B_flat = B.flatten()  # [b00..b05, b10..b15]
        
        C_computed = np.zeros((n, p))
        for c in range(rank):
            prod = np.dot(U_np[:, c], A_flat) * np.dot(V_np[:, c], B_flat)
            for k in range(p):
                for i in range(n):
                    C_computed[i, k] += W_np[k * n + i, c] * prod
        
        if not np.allclose(C_computed, C_expected):
            # Fallback to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
            return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
