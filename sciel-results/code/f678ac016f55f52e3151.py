import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Strategy:
    # Split j index (0..2) into {0,1} and {2}.
    # Part 1: j=2. A is 2x1, B is 1x8. Rank 1.
    # Part 2: j in {0,1}. A is 2x2, B is 2x8.
    # Split B (2x8) into 4 blocks of 2x2.
    # Each 2x2 by 2x2 uses Strassen (Rank 7).
    # Total Rank = 1 + 4 * 7 = 29.
    
    rank = 29
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: j=2 ---
    # A indices: i in {0,1}, j=2. Index in U: i*m + 2.
    # B indices: j=2, k in {0..7}. Index in V: 2*p + k.
    # C indices: k in {0..7}, i in {0,1}. Index in W: k*n + i.
    
    # Rank 1 term
    for i in range(n):
        U[i * m + 2][col] = 1
    for k in range(p):
        V[2 * p + k][col] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][col] = 1
    col += 1
    
    # --- Part 2: j in {0,1} ---
    # A indices: i in {0,1}, j in {0,1}.
    # B indices: j in {0,1}, k in {0..7}.
    # C indices: k in {0..7}, i in {0,1}.
    
    # Split k into 4 groups of 2: (0,1), (2,3), (4,5), (6,7).
    for block in range(4):
        k_start = block * 2
        k0 = k_start
        k1 = k_start + 1
        
        # Strassen for 2x2 by 2x2
        # a11 = A[0,0], a12 = A[0,1]
        # a21 = A[1,0], a22 = A[1,1]
        # b11 = B[0,k0], b12 = B[0,k1]
        # b21 = B[1,k0], b22 = B[1,k1]
        
        # M1 = (a11+a22)(b11+b22)
        # U: A[0,0], A[1,1]
        # V: B[0,k0], B[1,k0]
        # W: C[0,0] (k0,i=0), C[0,1] (k0,i=1)? No, W maps to C.
        # The term contributes to C.
        # M1 contributes to c11, c22.
        # c11 = C[k0, 0], c22 = C[k1, 1].
        
        # We need to fill U, V, W for this rank-1 term.
        # U[row, col] = coeff for A[row]
        # V[row, col] = coeff for B[row]
        # W[row, col] = coeff for C[row]
        
        # Term 1: (A[0,0] + A[1,1]) * (B[0,k0] + B[1,k0])
        U[0 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        V[1 * p + k0][col] = 1
        # Result: C[k0, 0] += 1, C[k1, 1] += 1
        # Wait, M1 = (a11+a22)(b11+b22).
        # c11 = M1 + ...
        # c22 = M1 - ...
        # So W[k0*n + 0][col] = 1
        # W[k1*n + 1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # Term 2: (A[1,0] + A[1,1]) * B[0,k0]
        U[1 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        # c21 = M2 + ... => C[k0, 1] += 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # Term 3: A[0,0] * (B[0,k1] - B[1,k1])
        U[0 * m + 0][col] = 1
        V[0 * p + k1][col] = 1
        V[1 * p + k1][col] = -1
        # c12 = M3 + ... => C[k1, 0] += 1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # Term 4: A[1,1] * (B[1,k0] - B[0,k0])
        U[1 * m + 1][col] = 1
        V[1 * p + k0][col] = 1
        V[0 * p + k0][col] = -1
        # c11 += M4, c21 += M4
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # Term 5: (A[0,0] + A[0,1]) * B[1,k1]
        U[0 * m + 0][col] = 1
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        # c11 -= M5, c12 += M5
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # Term 6: (A[1,0] - A[0,0]) * (B[0,k0] + B[0,k1])
        U[1 * m + 0][col] = 1
        U[0 * m + 0][col] = -1
        V[0 * p + k0][col] = 1
        V[0 * p + k1][col] = 1
        # c22 += M6
        W[k1 * n + 1][col] = 1
        col += 1
        
        # Term 7: (A[0,1] - A[1,1]) * (B[1,k0] + B[1,k1])
        U[0 * m + 1][col] = 1
        U[1 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        # c11 += M7
        W[k0 * n + 0][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
