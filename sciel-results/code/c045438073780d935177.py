import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Try rank 16 decomposition for 2x3x3 matrix multiplication
    # Using a more efficient combination of Strassen and schoolbook
    
    rank = 16
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # A: i in 0..1, j in 0..2 -> idx = i*3 + j
    # B: j in 0..2, k in 0..2 -> idx = j*3 + k  
    # C: k in 0..2, i in 0..1 -> idx = k*2 + i (Trace convention W[k*n+i, r])
    
    # We'll use Strassen for the 2x2x2 core (rows 0,1 of A and B; cols 0,1 of B)
    # and handle the third column/row with shared terms
    
    # Part 1: Strassen for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # This gives us C[0,0], C[0,1], C[1,0], C[1,1]
    
    # r=0: (A00+A11)(B00+B11) -> contributes to C00, C11
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1: (A10+A11)B00 -> C10, -C11
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # r=2: A00(B01-B11) -> C01, C11
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # r=3: A11(B10-B00) -> C00, C10
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # r=4: (A00+A01)B11 -> -C00, C01
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # r=5: (A10-A00)(B00+B01) -> C11
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # r=6: (A01-A11)(B10+B11) -> C00
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Part 2: Handle the third column of B (k=2) and third row interactions
    # We need C[0,2] = A00*B02 + A01*B12 + A02*B22
    # and C[1,2] = A10*B02 + A11*B12 + A12*B22
    
    # Also need to account for A[0:2,2] * B[2,0:2] which contributes to C[0:2,0:2]
    # C[i,k] += A[i,2]*B[2,k] for k=0,1
    
    # Let's use 9 more terms (r=7..15) for the remaining 9 products:
    # A02*B20, A02*B21, A12*B20, A12*B21 (4 terms for row 2 of B, cols 0,1)
    # A00*B02, A01*B12, A02*B22, A10*B02, A11*B12, A12*B22 (6 terms for col 2 of B)
    # Total: 10 terms, but we only have 9 slots. Let's see if we can merge.
    
    # Actually, let's use rank 17 to be safe, same as before but let me verify correctness
    rank = 17
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Redo with rank 17
    # Part 1: Strassen (7 terms)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Part 2: A[0:2,2] * B[2,0:2] -> C[0:2,0:2] (4 terms)
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1   # A02*B20 -> C00
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1   # A02*B21 -> C01
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1   # A12*B20 -> C10
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1 # A12*B21 -> C11
    
    # Part 3: A[0:2,0:3] * B[0:3,2] -> C[0:2,2] (6 terms)
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1  # A00*B02 -> C02
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1  # A01*B12 -> C02
    U[2, 13] = 1; V[8, 13] = 1; W[4, 13] = 1  # A02*B22 -> C02
    U[3, 14] = 1; V[2, 14] = 1; W[5, 14] = 1  # A10*B02 -> C12
    U[4, 15] = 1; V[5, 15] = 1; W[5, 15] = 1  # A11*B12 -> C12
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1  # A12*B22 -> C12
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
