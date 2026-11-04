"""
Rewritten program for 4x4x4 matrix multiplication tensor decomposition.
Uses Kronecker product of Strassen rank-7 decomposition to achieve rank 49.
This is the currently known minimal rank for 4x4x4 matrix multiplication.
"""

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as lists of lists with shape (4, 7), (4, 7), (4, 7).
    
    Indices:
    U[i*2+j, r] = coeff of A[i,j]
    V[j*2+k, r] = coeff of B[j,k]
    W[k*2+i, r] = coeff of C[i,k] (Trace convention: W[k*n+i, r])
    """
    rank = 7
    # Initialize with zeros
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    # A: A00(0,0), A11(1,1) -> U[0,0], U[3,0]
    # B: B00(0,0), B11(1,1) -> V[0,0], V[3,0]
    # C: C00(0,0), C11(1,1) -> W[0,0], W[3,0]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 2: M2 = (A10+A11)B00
    # A: A10(1,0), A11(1,1) -> U[2,1], U[3,1]
    # B: B00(0,0) -> V[0,1]
    # C: C10(1,0), C11(1,1) -> W[1,1], W[3,1]
    # C10 = M2 + ... -> W[1,1] = 1
    # C11 = ... - M2 ... -> W[3,1] = -1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # Term 3: M3 = A00(B01-B11)
    # A: A00(0,0) -> U[0,2]
    # B: B01(0,1), B11(1,1) -> V[1,2], V[3,2]
    # C: C01(0,1), C11(1,1) -> W[2,2], W[3,2]
    # C01 = M3 + ... -> W[2,2] = 1
    # C11 = ... + M3 ... -> W[3,2] = 1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # Term 4: M4 = A11(B10-B00)
    # A: A11(1,1) -> U[3,3]
    # B: B10(1,0), B00(0,0) -> V[2,3], V[0,3]
    # C: C00(0,0), C10(1,0) -> W[0,3], W[1,3]
    # C00 = ... + M4 ... -> W[0,3] = 1
    # C10 = ... + M4 ... -> W[1,3] = 1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # Term 5: M5 = (A00+A01)B11
    # A: A00(0,0), A01(0,1) -> U[0,4], U[1,4]
    # B: B11(1,1) -> V[3,4]
    # C: C00(0,0), C01(0,1) -> W[0,4], W[2,4]
    # C00 = ... - M5 ... -> W[0,4] = -1
    # C01 = ... + M5 ... -> W[2,4] = 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    # A: A10(1,0), A00(0,0) -> U[2,5], U[0,5]
    # B: B00(0,0), B01(0,1) -> V[0,5], V[1,5]
    # C: C11(1,1) -> W[3,5]
    # C11 = ... + M6 ... -> W[3,5] = 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    # A: A01(0,1), A11(1,1) -> U[1,6], U[3,6]
    # B: B10(1,0), B11(1,1) -> V[2,6], V[3,6]
    # C: C00(0,0) -> W[0,6]
    # C00 = ... + M7 ... -> W[0,6] = 1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _kronecker_decomposition(n, m, p):
    """
    Construct the rank-49 decomposition for 4x4x4 using Kronecker product
    of the rank-7 Strassen decomposition for 2x2x2.
    """
    U2, V2, W2 = _strassen_2x2()
    
    rank = 49
    # Shapes: U (n*m, rank) = (16, 49)
    #         V (m*p, rank) = (16, 49)
    #         W (p*n, rank) = (16, 49)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for 4x4:
    # i = 2*i1 + i2
    # j = 2*j1 + j2
    # k = 2*k1 + k2
    # where i1,i2,j1,j2,k1,k2 in {0,1}
    
    # Kronecker structure:
    # r = r1 * 7 + r2
    # U[i*m+k, r] = U2[i1*2+k1, r1] * U2[i2*2+k2, r2]
    # V[k*p+j, r] = V2[k1*2+j1, r1] * V2[k2*2+j2, r2]
    # W[j*n+i, r] = W2[j1*2+i1, r1] * W2[j2*2+i2, r2]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            for k1 in range(2):
                                for k2 in range(2):
                                    i = 2 * i1 + i2
                                    j = 2 * j1 + j2
                                    k = 2 * k1 + k2
                                    
                                    u_val = U2[i1 * 2 + k1][r1] * U2[i2 * 2 + k2][r2]
                                    v_val = V2[k1 * 2 + j1][r1] * V2[k2 * 2 + j2][r2]
                                    w_val = W2[j1 * 2 + i1][r1] * W2[j2 * 2 + i2][r2]
                                    
                                    U[i * m + k][r] = u_val
                                    V[k * p + j][r] = v_val
                                    W[j * n + i][r] = w_val
                                    
    return {"U": U, "V": V, "W": W, "rank": rank}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Rank 49 is the optimal known rank for 4x4x4 matrix multiplication.
    # We construct it directly.
    try:
        result = _kronecker_decomposition(n, m, p)
        return {"U": result["U"], "V": result["V"], "W": result["W"]}
    except Exception as e:
        # Fallback to schoolbook (rank 64) if something goes wrong, though unlikely
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
