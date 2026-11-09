"""Matrix multiplication 2,4,8 tensor decomposition"""
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # A is n x m, B is m x p, C is n x p
    
    # Rank 56 using Strassen for 2x2x2 blocks
    # 8 blocks: s in {0,1} for j-split, t in {0,1,2,3} for k-split
    # Each block uses 7 Strassen multiplications
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # u: (i, j) -> coeff
    # v: (j, k) -> coeff
    # w: (k, i) -> coeff (since W index is k*n+i)
    
    # M1: (a00+a11)(b00+b11) -> C00 +=, C11 +=
    # u: (0,0):1, (1,1):1
    # v: (0,0):1, (1,1):1
    # w: (0,0):1, (1,1):1  (k=0,i=0 and k=1,i=1)
    
    # M2: (a10+a11)b00 -> C10 +=, C11 -=
    # u: (1,0):1, (1,1):1
    # v: (0,0):1
    # w: (0,1):1, (1,1):-1 (k=0,i=1 and k=1,i=1)
    
    # M3: a00(b01-b11) -> C01 +=, C11 +=
    # u: (0,0):1
    # v: (0,1):1, (1,1):-1
    # w: (1,0):1, (1,1):1 (k=1,i=0 and k=1,i=1)
    
    # M4: a11(b10-b00) -> C00 +=, C10 +=
    # u: (1,1):1
    # v: (1,0):1, (0,0):-1
    # w: (0,0):1, (0,1):1 (k=0,i=0 and k=0,i=1)
    
    # M5: (a00+a01)b11 -> C00 -=, C01 +=
    # u: (0,0):1, (0,1):1
    # v: (1,1):1
    # w: (0,0):-1, (1,0):1 (k=0,i=0 and k=1,i=0)
    
    # M6: (a10-a00)(b00+b01) -> C11 +=
    # u: (1,0):1, (0,0):-1
    # v: (0,0):1, (0,1):1
    # w: (1,1):1 (k=1,i=1)
    
    # M7: (a01-a11)(b10+b11) -> C00 +=
    # u: (0,1):1, (1,1):-1
    # v: (1,0):1, (1,1):1
    # w: (0,0):1 (k=0,i=0)

    strassen_u = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1}
    ]
    
    strassen_v = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1}
    ]
    
    strassen_w = [
        {(0,0): 1, (1,1): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(0,0): -1, (1,0): 1},
        {(1,1): 1},
        {(0,0): 1}
    ]
    
    for s in range(2):  # j split: 0 for j in {0,1}, 1 for j in {2,3}
        for t in range(4):  # k split: 0..3 for k in {0,1}, {2,3}, ...
            for r in range(7):
                # Update U
                for (i_loc, j_loc), val in strassen_u[r].items():
                    i_glob = i_loc
                    j_glob = 2 * s + j_loc
                    U[i_glob * m + j_glob][col] += val
                
                # Update V
                for (j_loc, k_loc), val in strassen_v[r].items():
                    j_glob = 2 * s + j_loc
                    k_glob = 2 * t + k_loc
                    V[j_glob * p + k_glob][col] += val
                
                # Update W
                for (k_loc, i_loc), val in strassen_w[r].items():
                    k_glob = 2 * t + k_loc
                    i_glob = i_loc
                    W[k_glob * n + i_glob][col] += val
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
