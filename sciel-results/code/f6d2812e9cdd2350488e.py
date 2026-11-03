import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition into 2x2x2 sub-problems.
    # m=4 -> 2 blocks of 2
    # p=8 -> 4 blocks of 2
    # Total 8 sub-problems, each rank 7 (Strassen).
    # Total rank = 8 * 7 = 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen linear forms for X (2x2) and Y (2x2)
    # X indices: (0,0), (0,1), (1,0), (1,1)
    # Y indices: (0,0), (0,1), (1,0), (1,1)
    
    # L_X[s] is dict { (i,j): coeff }
    # L_Y[s] is dict { (i,j): coeff }
    
    L_X = [
        {}, # 0
        {(0,0): 1, (1,1): 1}, # 1
        {(1,0): 1, (1,1): 1}, # 2
        {(0,0): 1},           # 3
        {(1,1): 1},           # 4
        {(0,0): 1, (1,0): 1}, # 5
        {(1,0): 1, (0,0): -1},# 6
        {(0,1): 1, (1,1): -1} # 7
    ]
    
    L_Y = [
        {}, # 0
        {(0,0): 1, (1,1): 1}, # 1
        {(0,0): 1},           # 2
        {(1,0): 1, (1,1): -1},# 3
        {(0,1): 1, (0,0): -1},# 4
        {(1,1): 1},           # 5
        {(0,0): 1, (1,0): 1}, # 6
        {(1,1): 1, (0,1): 1}  # 7
    ]
    
    # Output coefficients for Z[i,j] from M_s
    # Z = X Y
    # Z_00 = M1 + M4 - M5 + M7
    # Z_01 = M3 + M5
    # Z_10 = M2 + M4
    # Z_11 = M1 - M2 + M3 + M6
    
    # C_out[s][i][j] = coeff
    C_out = [
        [[0,0],[0,0]], # 0
        [[1,0],[0,1]], # 1: M1 adds to Z00, Z11
        [[0,0],[1,0]], # 2: M2 adds to Z10, Z11(-1) -> wait
        # Let's re-verify Z11 coeff for M2.
        # Z11 = M1 - M2 + M3 + M6
        # So M2 coeff is -1.
        # Let's list them properly.
    ]
    
    # s=1: Z00+=1, Z11+=1
    # s=2: Z10+=1, Z11+=-1
    # s=3: Z01+=1, Z11+=1
    # s=4: Z00+=1, Z10+=1
    # s=5: Z00+=-1, Z01+=1
    # s=6: Z11+=1
    # s=7: Z00+=1
    
    C_out = [
        [[0,0],[0,0]],
        [[1,0],[0,1]],
        [[0,0],[1,-1]],
        [[0,1],[0,1]],
        [[1,0],[1,0]],
        [[-1,1],[0,0]],
        [[0,0],[0,1]],
        [[1,0],[0,0]]
    ]
    
    col = 0
    # a: block index for m (0, 1)
    # c: block index for p (0, 1, 2, 3)
    for c in range(4):
        for a in range(2):
            for s in range(1, 8):
                # Update U
                for (xi, xj), coeff in L_X[s].items():
                    # X is A^{(a)}
                    # A index: i, j_global = 2*a + xj
                    row_u = xi * m + (2 * a + xj)
                    U[row_u][col] += coeff
                
                # Update V
                for (yi, yj), coeff in L_Y[s].items():
                    # Y is B^{(a,c)}
                    # B index: j_global = 2*a + yi, k_global = 2*c + yj
                    row_v = (2 * a + yi) * p + (2 * c + yj)
                    V[row_v][col] += coeff
                
                # Update W
                for zi in range(2):
                    for zj in range(2):
                        coeff = C_out[s][zi][zj]
                        if coeff != 0:
                            # Z is C^{(c)}
                            # C index: i=zi, k_global = 2*c + zj
                            # W row index: k*n + i
                            row_w = (2 * c + zj) * n + zi
                            W[row_w][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
