import numpy as np
from fractions import Fraction

def construct():
    n, m, p = 4, 4, 5
    
    # Use Strassen-like block decomposition for 4x4x4
    # Split into 2x2 blocks, each 2x2x2 uses Strassen (rank 7)
    # Total for 4x4x4: 4 blocks * 7 = 28
    # For 4x4x5: the 5th column is handled by schoolbook (rank 16)
    # Total: 28 + 16 = 44
    
    # Strassen for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M2 + M3 - M4
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M7
    
    rank = 28 + 16
    
    U = [[0] * rank for _ in range(n * m)]  # (16, rank)
    V = [[0] * rank for _ in range(m * p)]  # (20, rank)
    W = [[0] * rank for _ in range(p * n)]  # (20, rank)
    
    col = 0
    
    # 4x4x4 part: 4 blocks of 2x2x2, each with Strassen
    # Block (bi, bj) where bi, bj in {0,1}
    # A is 4x4, B is 4x4 (first 4 cols), C is 4x4 (first 4 cols)
    # A[i,j], B[j,k], C[i,k]
    # Block A[2*bi:2*bi+2, 2*bj:2*bj+2], B[2*bj:2*bj+2, 2*bk:2*bk+2], C[2*bi:2*bi+2, 2*bk:2*bk+2]
    
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                # Strassen terms for this 2x2x2 block
                # Local indices: A[bi,bj], B[bj,bk], C[bi,bk]
                # Global: A[2*bi+li, 2*bj+lj], B[2*bj+lj, 2*bk+lk], C[2*bi+li, 2*bk+lk]
                
                # M1: (A00+A11)(B00+B11)
                # A00 = A[2*bi, 2*bj], A11 = A[2*bi+1, 2*bj+1]
                # B00 = B[2*bj, 2*bk], B11 = B[2*bj+1, 2*bk+1]
                # C00 = C[2*bi, 2*bk], C11 = C[2*bi+1, 2*bk+1]
                
                # M1 contributes to C00 and C11
                U[2*bi*m + 2*bj][col] = 1
                U[2*bi*m + 2*bj + 1][col] = 1
                V[2*bj*p + 2*bk][col] = 1
                V[2*bj*p + 2*bk + 1][col] = 1
                W[2*bk*n + 2*bi][col] = 1
                W[2*bk*n + 2*bi + 1][col] = 1
                col += 1
                
                # M2: (A10+A11)B00
                # A10 = A[2*bi+1, 2*bj], A11 = A[2*bi+1, 2*bj+1]
                # B00 = B[2*bj, 2*bk]
                # C10 = C[2*bi+1, 2*bk]
                
                U[2*bi*m + 2*bj + 1][col] = 1
                V[2*bj*p + 2*bk][col] = 1
                W[2*bk*n + 2*bi + 1][col] = 1
                col += 1
                
                # M3: A00(B01-B11)
                # A00 = A[2*bi, 2*bj]
                # B01 = B[2*bj, 2*bk+1], B11 = B[2*bj+1, 2*bk+1]
                # C01 = C[2*bi, 2*bk+1]
                
                U[2*bi*m + 2*bj][col] = 1
                V[2*bj*p + 2*bk + 1][col] = 1
                V[2*bj*p + 2*bk + 2][col] = -1
                W[2*bk*n + 2*bi][col] = 1
                col += 1
                
                # M4: A11(B10-B00)
                # A11 = A[2*bi+1, 2*bj+1]
                # B10 = B[2*bj+1, 2*bk], B00 = B[2*bj, 2*bk]
                # C10 = C[2*bi+1, 2*bk]
                
                U[2*bi*m + 2*bj + 2][col] = 1
                V[2*bj*p + 2*bk + 2][col] = 1
                V[2*bj*p + 2*bk][col] = -1
                W[2*bk*n + 2*bi + 1][col] = 1
                col += 1
                
                # M5: (A00+A01)B11
                # A00 = A[2*bi, 2*bj], A01 = A[2*bi, 2*bj+1]
                # B11 = B[2*bj+1, 2*bk+1]
                # C01 = C[2*bi, 2*bk+1]
                
                U[2*bi*m + 2*bj][col] = 1
                U[2*bi*m + 2*bj + 1][col] = 1
                V[2*bj*p + 2*bk + 2][col] = 1
                W[2*bk*n + 2*bi][col] = 1
                col += 1
                
                # M6: (A10-A00)(B00+B01)
                # A10 = A[2*bi+1, 2*bj], A00 = A[2*bi, 2*bj]
                # B00 = B[2*bj, 2*bk], B01 = B[2*bj, 2*bk+1]
                # C10 = C[2*bi+1, 2*bk]
                
                U[2*bi*m + 2*bj + 1][col] = 1
                U[2*bi*m + 2*bj][col] = -1
                V[2*bj*p + 2*bk][col] = 1
                V[2*bj*p + 2*bk + 1][col] = 1
                W[2*bk*n + 2*bi + 1][col] = 1
                col += 1
                
                # M7: (A01-A11)(B10+B11)
                # A01 = A[2*bi, 2*bj+1], A11 = A[2*bi+1, 2*bj+1]
                # B10 = B[2*bj+1, 2*bk], B11 = B[2*bj+1, 2*bk+1]
                # C00 = C[2*bi, 2*bk], C11 = C[2*bi+1, 2*bk+1]
                
                U[2*bi*m + 2*bj + 1][col] = 1
                U[2*bi*m + 2*bj + 2][col] = -1
                V[2*bj*p + 2*bk + 2][col] = 1
                V[2*bj*p + 2*bk + 3][col] = 1
                W[2*bk*n + 2*bi][col] = 1
                W[2*bk*n + 2*bi + 1][col] = 1
                col += 1
    
    # 4x4x1 part: schoolbook for the 5th column of B (index 4)
    # A is 4x4, B[:,4] is 4x1, C[:,4] is 4x1
    # C[i,4] = sum_j A[i,j] * B[j,4]
    # W index: k*n + i where k=4, i in 0..3
    # So W[4*n + i] = W[16 + i]
    
    for i in range(n):
        for j in range(m):
            U[i*m + j][col] = 1
            V[j*p + 4][col] = 1
            W[4*n + i][col] = 1
            col += 1
    
    # Verify col == rank
    assert col == rank, f"Column count {col} != rank {rank}"
    
    return {"U": U, "V": V, "W": W}
