import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Current best known rank for 2x5x6 using Strassen blocks
    # Split m=5 into 1 + 4
    # 2x1x6: rank 12
    # 2x4x6: two 2x2x6 blocks, each rank 21 (three 2x2x2 Strassen)
    # Total rank: 12 + 42 = 54
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 2x1x6 block (j=0)
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    # Part 2: 2x4x6 block (j=1..4)
    # Split j into {1,2} and {3,4}
    # Split k into {0,1}, {2,3}, {4,5}
    # 6 Strassen blocks of 2x2x2
    
    def add_strassen_block(j_base, k_base, col_offset):
        # i is always 0,1
        u00 = 0 * m + j_base
        u01 = 0 * m + (j_base + 1)
        u10 = 1 * m + j_base
        u11 = 1 * m + (j_base + 1)
        
        v00 = j_base * p + k_base
        v01 = j_base * p + (k_base + 1)
        v10 = (j_base + 1) * p + k_base
        v11 = (j_base + 1) * p + (k_base + 1)
        
        w00 = k_base * n + 0
        w01 = (k_base + 1) * n + 0
        w10 = k_base * n + 1
        w11 = (k_base + 1) * n + 1
        
        c = col_offset
        
        # P1
        U[u00][c] += 1; U[u11][c] += 1
        V[v00][c] += 1; V[v11][c] += 1
        W[w00][c] += 1; W[w11][c] += 1
        c += 1
        # P2
        U[u10][c] += 1; U[u11][c] += 1
        V[v00][c] += 1
        W[w10][c] += 1; W[w11][c] += -1
        c += 1
        # P3
        U[u00][c] += 1
        V[v01][c] += 1; V[v11][c] += -1
        W[w01][c] += 1; W[w11][c] += 1
        c += 1
        # P4
        U[u11][c] += 1
        V[v10][c] += 1; V[v00][c] += -1
        W[w00][c] += 1; W[w10][c] += 1
        c += 1
        # P5
        U[u00][c] += 1; U[u01][c] += 1
        V[v11][c] += 1
        W[w00][c] += -1; W[w01][c] += 1
        c += 1
        # P6
        U[u10][c] += 1; U[u00][c] += -1
        V[v00][c] += 1; V[v01][c] += 1
        W[w11][c] += 1
        c += 1
        # P7
        U[u01][c] += 1; U[u11][c] += -1
        V[v10][c] += 1; V[v11][c] += 1
        W[w00][c] += 1
        c += 1
        return c

    j_pairs = [(1, 2), (3, 4)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j1, j2 in j_pairs:
        for k1, k2 in k_pairs:
            col = add_strassen_block(j1, k1, col)
            
    # Verify
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_np[i * m + j, :]
                v_row = V_np[j * p + k, :]
                w_row = W_np[k * n + i, :]
                result = np.sum(u_row * v_row * w_row)
                if abs(result - 1) > 1e-9:
                    # Fallback to trivial
                    rank_fallback = n * m * p
                    U_f = [[0] * rank_fallback for _ in range(n * m)]
                    V_f = [[0] * rank_fallback for _ in range(m * p)]
                    W_f = [[0] * rank_fallback for _ in range(p * n)]
                    col_f = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                U_f[i * m + j][col_f] = 1
                                V_f[j * p + k][col_f] = 1
                                W_f[k * n + i][col_f] = 1
                                col_f += 1
                    return {"U": U_f, "V": V_f, "W": W_f}
                    
    return {"U": U, "V": V, "W": W}
