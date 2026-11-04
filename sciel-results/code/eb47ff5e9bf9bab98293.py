"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x9 matrix multiplication.
    
    The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,...,8}.
    We seek U (6 x r), V (27 x r), W (18 x r) such that:
    T[i,j,k] = sum_c U[i*3+j, c] * V[j*9+k, c] * W[k*2+i, c]
    
    Strategy: Exploit the structure where A has only 2 rows.
    For each column k of C, we have:
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    We can use a decomposition that shares computations across the two rows of A.
    """
    n, m, p = 2, 3, 9
    start_time = time.time()
    
    # The schoolbook rank is n*m*p = 54.
    # Let's try to find a lower rank decomposition.
    
    # One approach: For each pair of rows in A, we can use a rank-3 decomposition
    # for the 1x3 by 3x9 multiplication, but we need to handle both rows simultaneously.
    
    # Let's try a structured decomposition.
    # For each k, we need to compute two linear combinations of B[0,k], B[1,k], B[2,k].
    
    # Define intermediate products:
    # For each k, let's define:
    # X_k = A[0,0] + A[1,0]
    # Y_k = A[0,0] - A[1,0]
    # Z_k = A[0,1]
    # W_k = A[0,2]
    # V_k = A[1,1]
    # U_k = A[1,2]
    
    # This doesn't directly help. Let's think differently.
    
    # Actually, for a 2x3 by 3x9 multiplication, we can view it as:
    # C = A * B where A is 2x3 and B is 3x9.
    
    # The rank of the multiplication tensor for (2,3,9) is known to be at most 54 (schoolbook).
    # Can we do better?
    
    # Let's try a decomposition based on the fact that we can factor through a smaller space.
    
    # One known technique: use the fact that 2x3 multiplication can be done with rank 5
    # (instead of 6), and then extend to 3x9.
    
    # For 2x3 by 3x1, the rank is 5 (using Strassen-like methods).
    # For 2x3 by 3x9, we might be able to use rank 5*9 = 45? Or less?
    
    # Let's try a direct construction.
    
    # For each column k of B, we have a 3x1 vector. We multiply 2x3 by 3x1.
    # This can be done with rank 5.
    
    # Let's define the rank-5 decomposition for 2x3 by 3x1:
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1] + A[0,2]*B[2]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1] + A[1,2]*B[2]
    
    # Using rank 5:
    # P1 = A[0,0]
    # P2 = A[1,0]
    # P3 = A[0,1] + A[1,1]
    # P4 = A[0,2] + A[1,2]
    # P5 = A[0,1] - A[1,1]
    
    # Q1 = B[0]
    # Q2 = B[0]
    # Q3 = B[1]
    # Q4 = B[2]
    # Q5 = B[1]
    
    # R1 = 1, 0
    # R2 = 0, 1
    # R3 = 1, 1
    # R4 = 1, 1
    # R5 = 1, -1
    
    # Then:
    # C[0] = P1*Q1*R1[0] + P2*Q2*R2[0] + P3*Q3*R3[0] + P4*Q4*R4[0] + P5*Q5*R5[0]
    #      = A[0,0]*B[0] + 0 + (A[0,1]+A[1,1])*B[1] + (A[0,2]+A[1,2])*B[2] + (A[0,1]-A[1,1])*B[1]
    #      = A[0,0]*B[0] + A[0,1]*B[1] + A[1,1]*B[1] + A[0,2]*B[2] + A[1,2]*B[2] + A[0,1]*B[1] - A[1,1]*B[1]
    #      = A[0,0]*B[0] + 2*A[0,1]*B[1] + A[0,2]*B[2]
    
    # This is not correct. Let me redo this.
    
    # For 2x3 by 3x1, the standard rank is 6. Can we do rank 5?
    # Actually, for 2xn by nx1, the rank is 2n - 1 for n >= 2? No, that's not right either.
    
    # Let's just use the schoolbook decomposition for now, which has rank 54.
    # But let's try to find a better one.
    
    # Actually, let's try a different approach.
    # For each k, we can use a rank-3 decomposition (one term per j).
    # Total rank = 3 * 9 = 27? No, that's not right because we have 2 rows in A.
    
    # Let's try: for each (j,k), we have one term.
    # U[i*3+j, c] = 1 for all i? No, that doesn't work.
    
    # Let's just return the schoolbook decomposition.
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Convert to lists of lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
