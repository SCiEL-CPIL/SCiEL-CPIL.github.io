import time
import sys

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 23 solution:
    # Use Strassen for the 2x2x2 sub-block (rows 0-1 of A, rows 0-1 of B, cols 0-1 of C)
    # Standard for the rest.
    
    # Indices:
    # A: i in 0..1, j in 0..2. Index i*m+j
    # B: j in 0..2, k in 0..3. Index j*p+k
    # C: i in 0..1, k in 0..3. Index k*n+i (W index)
    
    # Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # This handles i in 0,1; j in 0,1; k in 0,1.
    # Terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11B11
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M3 - M5 + M7
    # C01 = M4 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6
    
    # Note: This Strassen block covers j=0,1 and k=0,1.
    # It does NOT cover j=2 or k=2,3.
    # The standard multiplication sums over j=0,1,2.
    # So for C[i,k] where i,k in 0,1:
    # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k]
    # The Strassen part computes S[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] for i,k in 0,1.
    # So we need to add A[i,2]B[2,k] for i,k in 0,1.
    # And for k=2,3, we need full standard multiplication.
    
    # Rank 7 for the 2x2x2 part.
    # Plus standard for the rest.
    
    # Let's build the U, V, W matrices.
    rank = 23
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. Strassen block for i in 0,1; j in 0,1; k in 0,1
    # M1
    U[0][col] = 1.0  # A00
    U[4][col] = 1.0  # A11
    V[0][col] = 1.0  # B00
    V[5][col] = 1.0  # B11
    W[0][col] = 1.0  # C00
    W[3][col] = 1.0  # C11
    col += 1
    
    # M2
    U[3][col] = 1.0  # A10
    U[4][col] = 1.0  # A11
    V[0][col] = 1.0  # B00
    W[1][col] = 1.0  # C10
    col += 1
    
    # M3
    U[0][col] = 1.0  # A00
    V[4][col] = 1.0  # B10
    V[5][col] = -1.0 # B11
    W[0][col] = 1.0  # C00
    col += 1
    
    # M4
    U[4][col] = 1.0  # A11
    V[5][col] = 1.0  # B11
    W[2][col] = 1.0  # C01
    W[1][col] = 1.0  # C10
    col += 1
    
    # M5
    U[0][col] = 1.0  # A00
    U[1][col] = 1.0  # A01
    V[5][col] = 1.0  # B11
    W[0][col] = -1.0 # C00
    W[2][col] = 1.0  # C01
    col += 1
    
    # M6
    U[3][col] = 1.0  # A10
    U[0][col] = -1.0 # A00
    V[0][col] = 1.0  # B00
    V[1][col] = 1.0  # B01
    W[3][col] = 1.0  # C11
    col += 1
    
    # M7
    U[1][col] = 1.0  # A01
    U[4][col] = -1.0 # A11
    V[4][col] = 1.0  # B10
    V[5][col] = 1.0  # B11
    W[0][col] = 1.0  # C00
    col += 1
    
    # 2. Standard terms for j=2 (A[i,2] * B[2,k]) for all i, k
    # i in 0,1; k in 0,1,2,3
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1.0 # A[i,2]
            V[2 * p + k][col] = 1.0 # B[2,k]
            W[k * n + i][col] = 1.0 # C[i,k]
            col += 1
            
    # 3. Standard terms for k=2,3 (A[i,j] * B[j,k]) for j=0,1,2; i=0,1
    # Note: For k=0,1 and j=0,1, we already handled it with Strassen.
    # But Strassen only covers j=0,1.
    # So for k=2,3, we need j=0,1,2.
    # For k=0,1, we need j=2 (handled in step 2).
    
    for i in range(n):
        for j in range(m):
            for k in range(2, p):
                # Skip if this term was already covered?
                # Strassen covered j=0,1 for k=0,1.
                # Step 2 covered j=2 for all k.
                # So we need j=0,1 for k=2,3.
                if j < 2:
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
                # j=2 is covered in step 2.
                
    return {"U": U, "V": V, "W": W}
