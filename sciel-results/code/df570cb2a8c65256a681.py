"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0
    
    n, m, p = DIMENSIONS
    # Rank 42 decomposition
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # u[i][j], v[j][k], w[k][i]
    # i, j, k in {0, 1}
    # s=1 to 7
    
    strassen_u = [
        [[1, 0], [0, 1]], # s=1
        [[0, 0], [1, 1]], # s=2
        [[1, 0], [0, 0]], # s=3
        [[0, 0], [0, 1]], # s=4
        [[1, 1], [0, 0]], # s=5
        [[0, -1], [1, 0]],# s=6
        [[0, 1], [-1, 0]] # s=7
    ]
    
    strassen_v = [
        [[1, 0], [0, 1]], # s=1
        [[1, 0], [0, 0]], # s=2
        [[0, 1], [0, -1]],# s=3
        [[0, -1], [1, 0]],# s=4
        [[0, 0], [0, 1]], # s=5
        [[1, 1], [0, 0]], # s=6
        [[0, 0], [1, 1]]  # s=7
    ]
    
    strassen_w = [
        [[1, 0], [0, 1]], # s=1: w[0][0]=1, w[1][1]=1
        [[0, 0], [1, -1]],# s=2: w[0][1]=0? No.
        # Let's re-verify w indices. w[k][i]
        # s=1: C00+=1, C11+=1. w[0][0]=1, w[1][1]=1.
        # s=2: C10+=1, C11+=-1. w[0][1]=1, w[1][1]=-1.
        # s=3: C01+=1, C11+=1. w[1][0]=1, w[1][1]=1.
        # s=4: C00+=1, C10+=1. w[0][0]=1, w[0][1]=1.
        # s=5: C00+=-1, C01+=1. w[0][0]=-1, w[1][0]=1.
        # s=6: C11+=1. w[1][1]=1.
        # s=7: C00+=1. w[0][0]=1.
    ]
    
    # Correcting strassen_w based on derivation:
    # w[k][i]
    # s=1: w[0][0]=1, w[1][1]=1
    # s=2: w[0][1]=1, w[1][1]=-1
    # s=3: w[1][0]=1, w[1][1]=1
    # s=4: w[0][0]=1, w[0][1]=1
    # s=5: w[0][0]=-1, w[1][0]=1
    # s=6: w[1][1]=1
    # s=7: w[0][0]=1
    
    strassen_w = [
        [[1, 0], [0, 1]], # s=1
        [[0, 1], [0, -1]],# s=2
        [[0, 0], [1, 1]], # s=3
        [[1, 1], [0, 0]], # s=4
        [[-1, 0], [1, 0]],# s=5
        [[0, 0], [0, 1]], # s=6
        [[1, 0], [0, 0]]  # s=7
    ]
    
    # Populate U, V, W
    for c in range(3):
        for s_idx in range(7):
            s = s_idx + 1
            r1 = c * 7 + s_idx
            r2 = 21 + c * 7 + s_idx
            
            u_coeff = strassen_u[s_idx]
            v_coeff = strassen_v[s_idx]
            w_coeff = strassen_w[s_idx]
            
            for i in range(2):
                for j_local in range(2):
                    val_u = u_coeff[i][j_local]
                    if val_u != 0:
                        # Part 1: j_global = j_local
                        U[i * m + j_local][r1] += val_u
                        # Part 2: j_global = j_local + 2
                        U[i * m + (j_local + 2)][r2] += val_u
                        
                        # W depends on i, k.
                        # We need to update W for all k.
                        # But W is updated in the k loop.
                        
            for j_local in range(2):
                for k_local in range(2):
                    val_v = v_coeff[j_local][k_local]
                    if val_v != 0:
                        k_global = 2 * c + k_local
                        # Part 1: j_global = j_local
                        V[j_local * p + k_global][r1] += val_v
                        # Part 2: j_global = j_local + 2
                        V[(j_local + 2) * p + k_global][r2] += val_v
                        
            for k_local in range(2):
                for i in range(2):
                    val_w = w_coeff[k_local][i]
                    if val_w != 0:
                        k_global = 2 * c + k_local
                        # W index: k * n + i
                        W[k_global * n + i][r1] += val_w
                        W[k_global * n + i][r2] += val_w
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
