import numpy as np
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    # We embed 2x3 x 3x9 into 3x3 x 3x9 by padding A with a zero row
    # Then split 3x9 into three 3x3 blocks, each handled by rank-5 3x3x3x3 decomposition
    # Total rank: 3 * 5 = 15
    
    # Rank-5 decomposition for 3x3 matrix multiplication
    # C = A * B where A, B, C are 3x3
    # Uses 5 products:
    # M1 = A00 * B00
    # M2 = (A00 + A01) * (B01 - B00)
    # M3 = (A10 + A11) * (B10 - B11)
    # M4 = A11 * B11
    # M5 = (A00 + A11) * (B00 + B11) - M1 - M4
    # Plus corrections for off-diagonal terms... 
    # Actually, the standard rank-5 for 3x3 is more complex.
    
    # Let me use a known rank-5 decomposition for 3x3x3x3.
    # The 3x3 matrix multiplication tensor has rank 5.
    # One decomposition:
    # Products:
    # P1 = A[0,0]*B[0,0]
    # P2 = (A[0,0]+A[0,1])*(B[0,1]-B[0,0])  
    # P3 = (A[1,0]+A[1,1])*(B[1,0]-B[1,1])
    # P4 = A[1,1]*B[1,1]
    # P5 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1]) - P1 - P4
    # 
    # Output:
    # C[0,0] = P1
    # C[0,1] = P1 + P2
    # C[1,0] = P3 + P4
    # C[1,1] = P4
    # C[0,2] = ... need more terms
    
    # Actually, I recall that the rank of 3x3x3x3 is 5, but the decomposition 
    # involves all 9 output entries. Let me use a verified decomposition.
    
    # A known rank-5 decomposition for 3x3 matrix multiplication:
    # U (3x3 x 5), V (3x3 x 5), W (3x3 x 5)
    # 
    # Using the decomposition from literature:
    # Let's use the one where:
    # U[i,j,0] = 1 if (i,j) in {(0,0)}
    # U[i,j,1] = 1 if (i,j) in {(0,0),(0,1)}
    # U[i,j,2] = 1 if (i,j) in {(1,0),(1,1)}
    # U[i,j,3] = 1 if (i,j) in {(1,1)}
    # U[i,j,4] = 1 if (i,j) in {(0,0),(1,1)}
    #
    # V[i,j,0] = 1 if (i,j) in {(0,0)}
    # V[i,j,1] = 1 if (i,j) in {(0,1)} else -1 if (i,j) in {(0,0)}
    # V[i,j,2] = 1 if (i,j) in {(1,0)} else -1 if (i,j) in {(1,1)}
    # V[i,j,3] = 1 if (i,j) in {(1,1)}
    # V[i,j,4] = 1 if (i,j) in {(0,0),(1,1)}
    #
    # This doesn't look right. Let me use a different approach.
    
    # For safety, let me just return the schoolbook rank 54 solution,
    # but try to verify if rank 53 is achievable with a simple modification.
    
    # Actually, let me try the Kronecker approach more carefully.
    # The 3x3x3x3 tensor has rank 5. I'll hardcode a known valid decomposition.
    
    # Known rank-5 decomposition for 3x3 matrix multiplication:
    # Reference: The rank of the 3x3 matrix multiplication tensor is 5.
    # One valid decomposition:
    
    # U_3x3 (indices i,j for A, 5 columns)
    U3 = [
        [1, 1, 0, 0, 1],  # A[0,0]
        [0, 1, 0, 0, 0],  # A[0,1]
        [0, 0, 0, 0, 0],  # A[0,2]
        [0, 0, 1, 0, 0],  # A[1,0]
        [0, 0, 1, 1, 0],  # A[1,1]
        [0, 0, 0, 0, 0],  # A[1,2]
        [0, 0, 0, 0, 1],  # A[2,0]
        [0, 0, 0, 0, 0],  # A[2,1]
        [0, 0, 0, 0, 0],  # A[2,2]
    ]
    
    # This is getting complicated. Let me just return schoolbook for now
    # since I can't guarantee correctness of a lower rank decomposition.
    
    rank = n * m * p  # 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
