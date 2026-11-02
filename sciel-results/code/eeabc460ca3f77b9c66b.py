import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strategy: Recursive block decomposition
    # Split 4x4 * 4x6 into four 2x2 * 2x3 subproblems
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]] where each is 2x3
    # C = [[C00, C01], [C10, C11]] where each is 2x3
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # For each 2x2 * 2x3 multiplication, we need a decomposition.
    # Schoolbook for 2x2 * 2x3 has rank 12 (each of 6 outputs is sum of 2 products)
    # But we can potentially do better.
    #
    # Actually, let's think about this more carefully.
    # The tensor for 2x2 * 2x3: T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,1,2}
    # This is a 2x2x3 tensor. The rank of this tensor...
    #
    # For 2x2*2x2, the rank is 7 (Winograd).
    # For 2x2*2x3, we can view it as two independent 2x2*2x1 problems plus... no.
    # Actually, C[i,k] = sum_j A[i,j]*B[j,k]. For fixed k, this is a 2x2*2x1 problem.
    # But the B matrix is shared across k.
    #
    # Let's just use the schoolbook for the subproblems and see the total rank.
    # 4 subproblems, each 2x2*2x3 with schoolbook rank 12? No wait.
    # Each subproblem is computing C_block = A_block * B_block where A is 2x2, B is 2x3.
    # The number of terms in schoolbook is 2*2*3 = 12? No, it's sum over j of A[i,j]*B[j,k].
    # There are 2*2*3 = 12 terms total (for all i,j,k combinations).
    # So each subproblem contributes rank 12 if we use schoolbook independently.
    # Total = 4 * 12 = 48? That's already better than 96.
    #
    # But wait, the subproblems are NOT independent in the tensor decomposition sense.
    # Let me reconsider.
    #
    # The full tensor T[i,j,k] for 4x4*4x6 has T[i,j,k]=1 for all i,j,k.
    # If I decompose it as a sum of 4 tensors, one for each block pair, then each sub-tensor
    # is supported on a 2x2x3 region. The rank of each sub-tensor (2x2x3 all-ones) is what matters.
    #
    # The rank of the 2x2x3 all-ones tensor:
    # This is the matrix multiplication tensor for 2x2 * 2x3.
    # I believe the rank is 8 (same as 2x2*2x2 since we can pad to 2x2*2x2 and use rank 7, 
    # but 2x3 is different). Let me think...
    #
    # Actually, for 2x2 * 2x3:
    # C00 = a00*b00 + a01*b10
    # C01 = a00*b01 + a01*b11
    # C02 = a00*b02 + a01*b12
    # C10 = a10*b00 + a11*b10
    # C11 = a10*b01 + a11*b11
    # C12 = a10*b02 + a11*b12
    #
    # Schoolbook: 12 terms. Can we do better?
    # Using Winograd-style for the 2x2 part:
    # M1 = (a00+a11)(b00+b10) -> contributes to C00, C10
    # M2 = (a10+a11)(b00) -> contributes to C10
    # M3 = a00(b10-b00) -> contributes to C00
    # M4 = a11(b00-b10) -> contributes to C10
    # M5 = (a00+a01)(b01+b11) -> hmm, this mixes k=1 and k=2
    #
    # This is getting complicated. Let me just use the straightforward approach:
    # For each of the 4 block pairs, use a rank-12 schoolbook decomposition.
    # Total rank = 48.
    #
    # But actually, I can be smarter. Let me group by the B blocks.
    # B has blocks B00, B01, B10, B11 each 2x3.
    # 
    # Let me just build the rank-48 solution and verify it.
    
    rank = 48
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    # Block (0,0): A00 * B00 -> C00
    # A00 is rows 0-1, cols 0-1 of A. Index in U: i*4+j for i in {0,1}, j in {0,1}
    # B00 is rows 0-1, cols 0-2 of B. Index in V: j*6+k for j in {0,1}, k in {0,1,2}
    # C00 is rows 0-1, cols 0-2 of C. Index in W: k*4+i for k in {0,1,2}, i in {0,1}
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i*4 + j][col] = 1.0
                V[j*6 + k][col] = 1.0
                W[k*4 + i][col] = 1.0
                col += 1
    
    # Block (0,1): A01 * B10 -> C01
    # A01 is rows 0-1, cols 2-3 of A. Index in U: i*4+j for i in {0,1}, j in {2,3}
    # B10 is rows 2-3, cols 0-2 of B. Index in V: j*6+k for j in {2,3}, k in {0,1,2}
    # C01 is rows 0-1, cols 3-5 of C. Index in W: k*4+i for k in {3,4,5}, i in {0,1}
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i*4 + (j+2)][col] = 1.0
                V[(j+2)*6 + k][col] = 1.0
                W[(k+3)*4 + i][col] = 1.0
                col += 1
    
    # Block (1,0): A10 * B00 -> C10
    # A10 is rows 2-3, cols 0-1 of A. Index in U: i*4+j for i in {2,3}, j in {0,1}
    # B00 is rows 0-1, cols 0-2 of B. Index in V: j*6+k for j in {0,1}, k in {0,1,2}
    # C10 is rows 2-3, cols 0-2 of C. Index in W: k*4+i for k in {0,1,2}, i in {2,3}
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[(i+2)*4 + j][col] = 1.0
                V[j*6 + k][col] = 1.0
                W[k*4 + (i+2)][col] = 1.0
                col += 1
    
    # Block (1,1): A11 * B10 -> C11
    # A11 is rows 2-3, cols 2-3 of A. Index in U: i*4+j for i in {2,3}, j in {2,3}
    # B11 is rows 2-3, cols 3-5 of B. Index in V: j*6+k for j in {2,3}, k in {3,4,5}
    # C11 is rows 2-3, cols 3-5 of C. Index in W: k*4+i for k in {3,4,5}, i in {2,3}
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[(i+2)*4 + (j+2)][col] = 1.0
                V[(j+2)*6 + (k+3)][col] = 1.0
                W[(k+3)*4 + (i+2)][col] = 1.0
                col += 1
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        rank = Ua.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fall back to schoolbook if verification fails
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
