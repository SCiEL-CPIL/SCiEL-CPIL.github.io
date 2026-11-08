import numpy as np
import time

DIMENSIONS = [2, 3, 9]
_DEADLINE = 140.0
_START_TIME = None

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE - 5.0:
        # Return a fallback valid decomposition (rank 54 direct) if we are out of time
        # Though construction is instant, this is a safety net.
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][idx] = 1
                    V[j * p + k][idx] = 1
                    W[k * n + i][idx] = 1
                    idx += 1
        return {"U": U, "V": V, "W": W}

    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_L (2x2) * B_T (2x9)
    # A_L indices: 0, 1, 3, 4 (A00, A01, A10, A11)
    # B_T indices: 0..8, 9..17 (B0k, B1k)
    
    # 4 blocks of 2x2 Strassen
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        v_e = k0       # B_{0, k0}
        v_f = k1       # B_{0, k1}
        v_g = p + k0   # B_{1, k0}
        v_h = p + k1   # B_{1, k1}
        
        u_a = 0        # A_{00}
        u_b = 1        # A_{01}
        u_c = 3        # A_{10}
        u_d = 4        # A_{11}
        
        # Strassen products
        # M1 = (a+d)(e+h)
        U[u_a][col] += 1; U[u_d][col] += 1
        V[v_e][col] += 1; V[v_h][col] += 1
        
        # M2 = (c+d)e
        U[u_c][col+1] += 1; U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        
        # M3 = a(f-h)
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1; V[v_h][col+2] -= 1
        
        # M4 = d(g-e)
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1; V[v_e][col+3] -= 1
        
        # M5 = (a+b)h
        U[u_a][col+4] += 1; U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        
        # M6 = (c-a)(e+f)
        U[u_c][col+5] += 1; U[u_a][col+5] -= 1
        V[v_e][col+5] += 1; V[v_f][col+5] += 1
        
        # M7 = (b-d)(g+h)
        U[u_b][col+6] += 1; U[u_d][col+6] -= 1
        V[v_g][col+6] += 1; V[v_h][col+6] += 1
        
        # W indices: k*n + i
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        # C00 = M1 + M4 - M5 + M7
        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1
        
        # C01 = M3 + M5
        W[w_01][col+2] += 1
        W[w_01][col+4] += 1
        
        # C10 = M2 + M4
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1
        
        # C11 = M1 - M2 + M3 + M6
        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1
        
        col += 7
        
    # 1 block of 2x1 (last column k=8)
    k8 = 8
    v_e = k8       # B_{0, 8}
    v_g = p + k8   # B_{1, 8}
    
    u_a = 0
    u_b = 1
    u_c = 3
    u_d = 4
    
    # P1 = a e
    U[u_a][col] += 1; V[v_e][col] += 1
    # P2 = b g
    U[u_b][col+1] += 1; V[v_g][col+1] += 1
    # P3 = c e
    U[u_c][col+2] += 1; V[v_e][col+2] += 1
    # P4 = d g
    U[u_d][col+3] += 1; V[v_g][col+3] += 1
    
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    
    # C08 = P1 + P2
    W[w_08][col] += 1
    W[w_08][col+1] += 1
    
    # C18 = P3 + P4
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1
    
    col += 4
    
    # Part 2: A_R (2x1) * B_B (1x9)
    # A_R indices: 2, 5 (A02, A12)
    # B_B indices: 18..26 (B2k)
    
    u_ar0 = 2
    u_ar1 = 5
    
    for k in range(9):
        v_b = 2 * p + k
        
        # Q_k = A_{02} B_{2k}
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        
        # R_k = A_{12} B_{2k}
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        
        col += 2
    
    return {"U": U, "V": V, "W": W}
