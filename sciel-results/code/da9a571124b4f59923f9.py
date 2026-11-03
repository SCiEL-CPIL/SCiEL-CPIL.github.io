import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def _build_strassen7(n, m, p):
    # Rank 7 decomposition for 2x2x2
    # n=2, m=2, p=2
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's algorithm
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    # M1
    U[0*2+0][0] = 1
    U[1*2+1][0] = 1
    V[0*2+0][0] = 1
    V[1*2+1][0] = 1
    W[0*2+0][0] = 1
    W[1*2+1][0] = 1
    
    # M2
    U[1*2+0][1] = 1
    U[1*2+1][1] = 1
    V[0*2+0][1] = 1
    W[0*2+1][1] = 1
    
    # M3
    U[0*2+0][2] = 1
    V[0*2+1][2] = 1
    V[1*2+1][2] = -1
    W[1*2+0][2] = 1
    
    # M4
    U[1*2+1][3] = 1
    V[1*2+0][3] = 1
    V[0*2+0][3] = -1
    W[0*2+0][3] = 1
    
    # M5
    U[0*2+0][4] = 1
    U[0*2+1][4] = 1
    V[1*2+1][4] = 1
    W[1*2+1][4] = 1
    
    # M6
    U[1*2+0][5] = 1
    U[0*2+0][5] = -1
    V[0*2+0][5] = 1
    V[0*2+1][5] = 1
    W[1*2+1][5] = 1
    
    # M7
    U[0*2+1][6] = 1
    U[1*2+1][6] = -1
    V[1*2+0][6] = 1
    V[1*2+1][6] = 1
    W[1*2+0][6] = 1
    
    return U, V, W

def _build_outer2(n, m, p):
    # Rank 2 decomposition for 2x1x2
    # n=2, m=1, p=2
    # j is fixed to 0 (which will be mapped to j=2 in the 3x2 case)
    rank = 2
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # M1 = A00 * B00
    U[0*1+0][0] = 1
    V[0*2+0][0] = 1
    W[0*2+0][0] = 1
    W[1*2+0][0] = 0 # Wait, W index is k*n+i.
    # C_00 = A_00 B_00. So W[0*2+0] = 1.
    # C_10 = A_10 B_00. So W[0*2+1] = 1.
    # My previous derivation:
    # W[0 i, 1] = 1 for all i.
    # So W[0*2+0][0] = 1, W[0*2+1][0] = 1.
    # W[1 i, 1] = 0.
    
    W[0*2+0][0] = 1
    W[0*2+1][0] = 1
    
    # M2 = A00 * B01
    U[0*1+0][1] = 1
    U[1*1+0][1] = 1
    V[0*2+1][1] = 1
    W[1*2+0][1] = 1
    W[1*2+1][1] = 1
    
    # Wait, U index is i*m+j. Here m=1, j=0.
    # U[0][0]=1, U[1][0]=1.
    # U[0][1]=1, U[1][1]=1.
    
    return U, V, W

def _build_block_2x3x2():
    # Combine Strassen (j=0,1) and Outer (j=2)
    U_s, V_s, W_s = _build_strassen7(2, 2, 2)
    U_o, V_o, W_o = _build_outer2(2, 1, 2)
    
    rank = 7 + 2
    U = [[0] * rank for _ in range(2 * 3)]
    V = [[0] * rank for _ in range(3 * 2)]
    W = [[0] * rank for _ in range(2 * 2)]
    
    # Copy Strassen part (j=0,1)
    for i in range(2):
        for j in range(2):
            for r in range(7):
                U[i*3 + j][r] = U_s[i*2 + j][r]
    for j in range(2):
        for k in range(2):
            for r in range(7):
                V[j*2 + k][r] = V_s[j*2 + k][r]
    for k in range(2):
        for i in range(2):
            for r in range(7):
                W[k*2 + i][r] = W_s[k*2 + i][r]
                
    # Copy Outer part (j=2)
    # Map j=0 in outer to j=2 in block
    for i in range(2):
        for r in range(2):
            U[i*3 + 2][7 + r] = U_o[i*1 + 0][r]
    for k in range(2):
        for r in range(2):
            V[2*2 + k][7 + r] = V_o[0*2 + k][r]
    for k in range(2):
        for i in range(2):
            for r in range(2):
                W[k*2 + i][7 + r] = W_o[k*2 + i][r]
                
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # p=10, so 5 blocks of size 2
    num_blocks = p // 2
    block_rank = 9
    total_rank = num_blocks * block_rank
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    U_b, V_b, W_b = _build_block_2x3x2()
    
    for b in range(num_blocks):
        offset = b * block_rank
        # U is independent of k, so it's the same for all blocks?
        # No, the decomposition is per block.
        # The terms r are distinct for each block.
        # So U for block b uses columns offset to offset+8.
        for i in range(n):
            for j in range(m):
                U[i*m + j][offset : offset + block_rank] = U_b[i*3 + j]
                
        for j in range(m):
            for k in range(2):
                V[j*p + (2*b + k)][offset : offset + block_rank] = V_b[j*2 + k]
                
        for k in range(2):
            for i in range(n):
                W[(2*b + k)*n + i][offset : offset + block_rank] = W_b[k*2 + i]
                
    return {"U": U, "V": V, "W": W}
