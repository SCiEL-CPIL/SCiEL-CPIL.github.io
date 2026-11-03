"""Matrix multiplication 3x4 x 4x6 with block-structured Strassen decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Returns (U, V, W) for 2x2 * 2x2 -> 2x2 using Strassen's algorithm.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (a0+a3)*(b0+b3)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # M2 = (a2+a3)*b0
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # M3 = a0*(b1-b2)
    U[0][2] = 1
    V[1][2] = 1; V[2][2] = -1
    # M4 = a3*(b2-b3)
    U[3][3] = 1
    V[2][3] = 1; V[3][3] = -1
    # M5 = (a0+a1)*b3
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # M6 = (a2-a0)*(b0+b1)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # M7 = (a1-a3)*(b2+b3)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    
    # W: output reconstruction
    # C[0,0] = M1 + M4 - M5 + M7
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    # C[0,1] = M3 + M5
    W[1][2] = 1; W[1][4] = 1
    # C[1,0] = M2 + M4
    W[2][1] = 1; W[2][3] = 1
    # C[1,1] = M1 - M2 + M3 + M6
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    return U, V, W

def _block_1x2x2():
    """
    Returns (U, V, W) for 1x2 * 2x2 -> 1x2. Rank 4.
    """
    rank = 4
    U = [[0]*rank for _ in range(2)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(2)]
    
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    U[1][1] = 1; V[2][1] = 1; W[0][1] = 1
    U[0][2] = 1; V[1][2] = 1; W[1][2] = 1
    U[1][3] = 1; V[3][3] = 1; W[1][3] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    input_blocks = [[0, 1], [2]]
    mid_blocks = [[0, 1], [2, 3]]
    out_blocks = [[0, 1], [2, 3], [4, 5]]
    
    strassen_U, strassen_V, strassen_W = _strassen_2x2x2()
    small_U, small_V, small_W = _block_1x2x2()
    
    total_rank = 0
    for ib in input_blocks:
        for mb in mid_blocks:
            for ob in out_blocks:
                if len(ib) == 2 and len(mb) == 2 and len(ob) == 2:
                    total_rank += 7
                else:
                    total_rank += 4
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for ib in input_blocks:
        for mb in mid_blocks:
            for ob in out_blocks:
                if len(ib) == 2 and len(mb) == 2 and len(ob) == 2:
                    bU, bV, bW = strassen_U, strassen_V, strassen_W
                    b_rank = 7
                else:
                    bU, bV, bW = small_U, small_V, small_W
                    b_rank = 4
                
                for local_i, gi in enumerate(ib):
                    for local_j, gj in enumerate(mb):
                        global_u_row = gi * m + gj
                        for c in range(b_rank):
                            U[global_u_row][col_offset + c] += bU[local_i * len(mb) + local_j][c]
                
                for local_j, gj in enumerate(mb):
                    for local_k, gk in enumerate(ob):
                        global_v_row = gj * p + gk
                        for c in range(b_rank):
                            V[global_v_row][col_offset + c] += bV[local_j * len(ob) + local_k][c]
                
                for local_i, gi in enumerate(ib):
                    for local_k, gk in enumerate(ob):
                        global_w_row = gk * n + gi
                        for c in range(b_rank):
                            W[global_w_row][col_offset + c] += bW[local_i * len(ob) + local_k][c]
                
                col_offset += b_rank
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
