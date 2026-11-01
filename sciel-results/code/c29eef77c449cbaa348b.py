import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    """
    Return a tensor decomposition for 2x3 * 3x8 matrix multiplication.
    
    Strategy:
    - Split the middle index j (0..2) into two parts: j in {0,1} and j=2.
    - For j=2: rank-1 term (outer product of column 2 of A and row 2 of B).
    - For j in {0,1}: A is 2x2, B is 2x8. Split B into 4 blocks of 2 columns.
      Each 2x2 * 2x2 block uses Strassen (rank 7).
    - Total rank = 1 + 4*7 = 29.
    
    The output W follows the trace convention: W[k*n + i, r] where C[k, i] gets
    contribution from rank-r term.
    """
    n, m, p = 2, 3, 8
    rank = 29
    
    U = [[0] * rank for _ in range(n * m)]  # (2*3, 29)
    V = [[0] * rank for _ in range(m * p)]  # (3*8, 29)
    W = [[0] * rank for _ in range(p * n)]  # (8*2, 29)
    
    col = 0
    
    # --- Part 1: j=2 ---
    # A indices: i in {0,1}, j=2 -> row = i*m + 2 = i*3 + 2
    # B indices: j=2, k in {0..7} -> row = 2*p + k = 16 + k
    # C indices: k in {0..7}, i in {0,1} -> row = k*n + i = k*2 + i
    
    for i in range(n):
        U[i * m + 2][col] = 1
    for k in range(p):
        V[2 * p + k][col] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][col] = 1
    col += 1
    
    # --- Part 2: j in {0,1} ---
    # A indices: i in {0,1}, j in {0,1} -> row = i*m + j
    # B indices: j in {0,1}, k in {0..7} -> row = j*p + k
    # C indices: k in {0..7}, i in {0,1} -> row = k*n + i
    
    # Split k into 4 groups of 2: (0,1), (2,3), (4,5), (6,7)
    for block in range(4):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # Strassen for 2x2 * 2x2:
        # A = [[a00, a01], [a10, a11]] where a_ij = A[i,j] for i in {0,1}, j in {0,1}
        # B = [[b00, b01], [b10, b11]] where b_jk = B[j,k] for j in {0,1}, k in {k0,k1}
        # C = [[c00, c01], [c10, c11]] where c_ik = C[k,i] for k in {k0,k1}, i in {0,1}
        #
        # Standard Strassen:
        # M1 = (a00 + a11) * (b00 + b11)
        # M2 = (a10 + a11) * b00
        # M3 = a00 * (b01 - b11)
        # M4 = a11 * (b10 - b00)
        # M5 = (a00 + a01) * b11
        # M6 = (a10 - a00) * (b00 + b01)
        # M7 = (a01 - a11) * (b10 + b11)
        #
        # c00 = M1 - M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 - M6
        
        # Term 1: M1 = (a00 + a11) * (b00 + b11)
        U[0 * m + 0][col] = 1  # a00
        U[1 * m + 1][col] = 1  # a11
        V[0 * p + k0][col] = 1  # b00 = B[0, k0]
        V[1 * p + k0][col] = 1  # b10 = B[1, k0]... wait
        # Actually b00 = B[0,k0], b11 = B[1,k1]
        # M1 = (a00+a11)*(b00+b11)
        V[0 * p + k0][col] = 1  # b00
        V[1 * p + k1][col] = 1  # b11
        # c00 += M1, c11 += M1
        W[k0 * n + 0][col] = 1  # c00 = C[k0, 0]
        W[k1 * n + 1][col] = 1  # c11 = C[k1, 1]
        col += 1
        
        # Term 2: M2 = (a10 + a11) * b00
        U[1 * m + 0][col] = 1  # a10
        U[1 * m + 1][col] = 1  # a11
        V[0 * p + k0][col] = 1  # b00
        # c10 += M2, c11 -= M2
        W[k0 * n + 1][col] = 1  # c10 = C[k0, 1]
        W[k1 * n + 1][col] = -1  # c11 -= M2
        col += 1
        
        # Term 3: M3 = a00 * (b01 - b11)
        U[0 * m + 0][col] = 1  # a00
        V[0 * p + k1][col] = 1  # b01 = B[0, k1]
        V[1 * p + k1][col] = -1  # b11 = B[1, k1]
        # c01 += M3, c11 += M3
        W[k1 * n + 0][col] = 1  # c01 = C[k1, 0]
        W[k1 * n + 1][col] = 1  # c11 += M3
        col += 1
        
        # Term 4: M4 = a11 * (b10 - b00)
        U[1 * m + 1][col] = 1  # a11
        V[1 * p + k0][col] = 1  # b10 = B[1, k0]
        V[0 * p + k0][col] = -1  # b00 = B[0, k0]
        # c00 -= M4, c10 += M4
        W[k0 * n + 0][col] = -1  # c00 -= M4
        W[k0 * n + 1][col] = 1  # c10 += M4
        col += 1
        
        # Term 5: M5 = (a00 + a01) * b11
        U[0 * m + 0][col] = 1  # a00
        U[0 * m + 1][col] = 1  # a01
        V[1 * p + k1][col] = 1  # b11 = B[1, k1]
        # c00 -= M5, c01 += M5
        W[k0 * n + 0][col] = -1  # c00 -= M5
        W[k1 * n + 0][col] = 1  # c01 += M5
        col += 1
        
        # Term 6: M6 = (a10 - a00) * (b00 + b01)
        U[1 * m + 0][col] = 1  # a10
        U[0 * m + 0][col] = -1  # a00
        V[0 * p + k0][col] = 1  # b00
        V[0 * p + k1][col] = 1  # b01
        # c11 -= M6
        W[k1 * n + 1][col] = -1  # c11 -= M6
        col += 1
        
        # Term 7: M7 = (a01 - a11) * (b10 + b11)
        U[0 * m + 1][col] = 1  # a01
        U[1 * m + 1][col] = -1  # a11
        V[1 * p + k0][col] = 1  # b10 = B[1, k0]
        V[1 * p + k1][col] = 1  # b11 = B[1, k1]
        # c00 += M7
        W[k0 * n + 0][col] = 1  # c00 += M7
        col += 1
    
    # Verify: col should be 29
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
