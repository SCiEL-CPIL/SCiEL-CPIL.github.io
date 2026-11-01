"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    
    def schoolbook():
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
    
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-10:
                        return False
        return True
    
    # Strassen 2x2x2 decomposition (rank 7)
    # For multiplying 2x2 matrix A by 2x2 matrix B:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # 
    # 7 products:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    #
    # Output:
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M6
    # C11 = M1 - M3 + M6 + M7
    #
    # Tensor decomposition form:
    # (AB)_{ik} = sum_j A_{ij} B_{jk} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # For each Strassen term c, we need:
    # U[i*m+j, c] = coefficient of A_{ij} in the first factor of M_c
    # V[j*p+k, c] = coefficient of B_{jk} in the second factor of M_c  
    # W[k*n+i, c] = coefficient of M_c in C_{ik}
    
    # Strassen U factors for A (shape 4x7, indexed as U[i*2+j, c]):
    # M1: a00+a11 -> [1, 0, 0, 1]
    # M2: a10+a11 -> [0, 0, 1, 1]
    # M3: a00 -> [1, 0, 0, 0]
    # M4: a11 -> [0, 0, 0, 1]
    # M5: a00+a01 -> [1, 1, 0, 0]
    # M6: a10-a00 -> [-1, 0, 1, 0]
    # M7: a01-a11 -> [0, 1, 0, -1]
    strassen_U = [
        [1, 0, 1, 0, 1, -1, 0],   # a00
        [0, 0, 0, 0, 1, 0, 1],    # a01
        [0, 1, 0, 0, 0, 1, 0],    # a10
        [1, 1, 0, 1, 0, 0, -1]    # a11
    ]
    
    # Strassen V factors for B (shape 4x7, indexed as V[j*2+k, c]):
    # M1: b00+b11 -> [1, 0, 0, 1]
    # M2: b00 -> [1, 0, 0, 0]
    # M3: b10-b11 -> [0, 0, 1, -1]
    # M4: b01-b00 -> [-1, 1, 0, 0]
    # M5: b11 -> [0, 0, 0, 1]
    # M6: b00+b10 -> [1, 0, 1, 0]
    # M7: b01+b11 -> [0, 1, 0, 1]
    strassen_V = [
        [1, 1, 0, -1, 0, 1, 0],   # b00
        [0, 0, 0, 1, 0, 0, 1],    # b01
        [0, 0, 1, 0, 0, 1, 0],    # b10
        [1, 0, -1, 0, 1, 0, 1]    # b11
    ]
    
    # Strassen W factors for output C (shape 4x7, indexed as W[k*2+i, c]):
    # C00 = M1 - M4 + M5 + M7 -> W[0*2+0, c] = W[0, c]
    # C01 = M3 + M4 -> W[1*2+0, c] = W[2, c]
    # C10 = M2 + M6 -> W[0*2+1, c] = W[1, c]
    # C11 = M1 - M3 + M6 + M7 -> W[1*2+1, c] = W[3, c]
    #
    # W[0, c] (C00): [1, 0, 0, -1, 1, 0, 1]
    # W[1, c] (C10): [0, 1, 0, 0, 0, 1, 0]
    # W[2, c] (C01): [0, 0, 1, 1, 0, 0, 0]
    # W[3, c] (C11): [1, 0, -1, 0, 0, 1, 1]
    strassen_W = [
        [1, 0, 0, -1, 1, 0, 1],   # C00 (k=0, i=0)
        [0, 1, 0, 0, 0, 1, 0],    # C10 (k=0, i=1)
        [0, 0, 1, 1, 0, 0, 0],    # C01 (k=1, i=0)
        [1, 0, -1, 0, 0, 1, 1]    # C11 (k=1, i=1)
    ]
    
    # Verify Strassen decomposition for 2x2x2
    def verify_strassen_block():
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    total = 0
                    for c in range(7):
                        total += strassen_U[i*2+j][c] * strassen_V[j*2+k][c] * strassen_W[k*2+i][c]
                    if abs(total - 1) > 1e-10:
                        return False
        return True
    
    # Build rank-32 decomposition:
    # Columns 0,1: Strassen block 1 (rank 7)
    # Columns 2,3: Strassen block 2 (rank 7)
    # Columns 4,5: Strassen block 3 (rank 7)
    # Columns 6,7: Strassen block 4 (rank 7)
    # Column 8: schoolbook (rank 4)
    # Total: 32
    
    total_rank = 32
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    # Four Strassen blocks for column pairs (0,1), (2,3), (4,5), (6,7)
    for block in range(4):
        k0 = block * 2
        k1 = block * 2 + 1
        for c in range(7):
            # U factors
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = strassen_U[i * m + j][c]
            # V factors - only for columns k0 and k1
            for j in range(m):
                V[j * p + k0][col] = strassen_V[j * 2 + 0][c]  # b_{j,0} maps to column k0
                V[j * p + k1][col] = strassen_V[j * 2 + 1][c]  # b_{j,1} maps to column k1
            # W factors - only for columns k0 and k1
            for i in range(n):
                W[k0 * n + i][col] = strassen_W[0 * 2 + i][c]  # output column k0
                W[k1 * n + i][col] = strassen_W[1 * 2 + i][c]  # output column k1
            col += 1
    
    # Schoolbook for column 8 (rank 4)
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 8][col] = 1
            W[8 * n + i][col] = 1
            col += 1
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify the full decomposition
    if verify(result["U"], result["V"], result["W"]):
        return result
    
    # If verification fails, fall back to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
