import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition attempt
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 4 Strassen blocks for B columns (0,1), (2,3), (4,5), (6,7)
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        U[u_a][col] += 1; U[u_d][col] += 1; V[v_e][col] += 1; V[v_h][col] += 1
        U[u_c][col+1] += 1; U[u_d][col+1] += 1; V[v_e][col+1] += 1
        U[u_a][col+2] += 1; V[v_f][col+2] += 1; V[v_h][col+2] -= 1
        U[u_d][col+3] += 1; V[v_g][col+3] += 1; V[v_e][col+3] -= 1
        U[u_a][col+4] += 1; U[u_b][col+4] += 1; V[v_h][col+4] += 1
        U[u_c][col+5] += 1; U[u_a][col+5] -= 1; V[v_e][col+5] += 1; V[v_f][col+5] += 1
        U[u_b][col+6] += 1; U[u_d][col+6] -= 1; V[v_g][col+6] += 1; V[v_h][col+6] += 1
        
        w_00 = k0 * n + 0; w_01 = k1 * n + 0
        w_10 = k0 * n + 1; w_11 = k1 * n + 1
        W[w_00][col] += 1; W[w_00][col+3] += 1; W[w_00][col+4] -= 1; W[w_00][col+6] += 1
        W[w_01][col+2] += 1; W[w_01][col+4] += 1
        W[w_10][col+1] += 1; W[w_10][col+3] += 1
        W[w_11][col] += 1; W[w_11][col+1] -= 1; W[w_11][col+2] += 1; W[w_11][col+5] += 1
        col += 7
    
    # Column 8: 4 direct terms for j=0,1
    k8 = 8
    v_e8 = 0 * p + k8
    v_g8 = 1 * p + k8
    u_a = 0; u_b = 1; u_c = 3; u_d = 4
    U[u_a][col] += 1; V[v_e8][col] += 1; W[k8*n+0][col] += 1
    U[u_b][col+1] += 1; V[v_g8][col+1] += 1; W[k8*n+0][col+1] += 1
    U[u_c][col+2] += 1; V[v_e8][col+2] += 1; W[k8*n+1][col+2] += 1
    U[u_d][col+3] += 1; V[v_g8][col+3] += 1; W[k8*n+1][col+3] += 1
    col += 4
    
    # j=2: 18 terms, but try to save 1 by merging
    u_ar0 = 2; u_ar1 = 5
    for k in range(9):
        v_b = 2 * p + k
        U[u_ar0][col] += 1; V[v_b][col] += 1; W[k*n+0][col] += 1
        U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[k*n+1][col+1] += 1
        col += 2
    
    return {"U": U, "V": V, "W": W}
