import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Use Strassen-based decomposition for rank 56
    # For each of the 8 columns, use 7 Strassen terms
    rank = 7 * p  # 56
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    # Strassen coefficients for 3x3 matrix multiplication
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Using 7 products per column
    
    # For column k, the 7 Strassen products are:
    # M1 = (A[0,0]+A[1,1]+A[2,2]) * (B[0,k]+B[1,k]+B[2,k])
    # M2 = (A[0,0]+A[1,1]) * (B[0,k]+B[1,k])
    # M3 = (A[0,0]+A[2,2]) * (B[0,k]+B[2,k])
    # M4 = (A[1,1]+A[2,2]) * (B[1,k]+B[2,k])
    # M5 = (A[0,0]-A[1,1]) * (B[1,k]+B[1,k])  # Wait, this should be (B[1,k-1]+B[1,k]) but for single column...
    
    # Actually, for 3x3 x 3x1 (single column), we can use a simpler 7-term decomposition
    # Let me use the known rank-7 decomposition for 3x3 matrix multiplication
    
    # The 7 Strassen terms for computing C = A * B (both 3x3):
    # M1 = (A00+A11+A22) * (B00+B11+B22)
    # M2 = (A00+A11) * (B00+B11)
    # M3 = (A00+A22) * (B00+B22)  
    # M4 = (A11+A22) * (B11+B22)
    # M5 = (A00-A11) * (B10+B11)
    # M6 = (A11-A22) * (B20+B21)
    # M7 = (A22-A00) * (B01+B02)
    
    # And C entries:
    # C00 = M1 - M2 + M3
    # C01 = M4 - M5
    # C02 = M6 - M7
    # C10 = M5 + M7
    # C11 = M1 - M3 + M4
    # C12 = M2 - M4
    # C20 = M6 + M2
    # C21 = M3 - M1
    # C22 = M1 - M2 + M4
    
    # For 3x3 x 3x8, we apply this per column
    # For column k, B[:,k] = [B0k, B1k, B2k]
    
    # The 7 products for column k:
    # P1_k = (A00+A11+A22) * (B0k+B1k+B2k)
    # P2_k = (A00+A11) * (B0k+B1k)
    # P3_k = (A00+A22) * (B0k+B2k)
    # P4_k = (A11+A22) * (B1k+B2k)
    # P5_k = (A00-A11) * (B1k+B1k)  # This doesn't work for single column
    
    # I realize the standard Strassen for 3x3 uses overlapping submatrices, not just diagonal elements
    # Let me use a different approach: the known rank-7 tensor decomposition for 3x3
    
    # Actually, the minimal rank for 3x3 matrix multiplication is 19, not 7!
    # Strassen's algorithm works for 2x2 (rank 7 vs 8 schoolbook)
    # For 3x3, the schoolbook is 9, and the best known is 19? No, that's not right either.
    
    # Let me reconsider. For 3x3 matrix multiplication:
    # Schoolbook rank = 9
    # The tensor rank of 3x3 matrix multiplication is actually 19 (lower bound) 
    # But we can use Strassen-style block decomposition
    
    # For 3x3 x 3x8, let's just use the schoolbook approach but verify it's correct
    # The schoolbook rank is 3*3*8 = 72
    
    # Since finding a rank < 72 exact decomposition with integer/half-integer entries
    # is non-trivial, let's return the schoolbook decomposition which is guaranteed valid
    
    rank = n * m * p  # 72
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
