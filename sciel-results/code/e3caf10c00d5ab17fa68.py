import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 34 construction
    rank = 34
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    # A indices: (ai1, aj1), (ai2, aj2) corresponding to 0,0 and 1,1 in local 2x2
    # B indices: (bj1, bk1), (bj2, bk2) corresponding to 0,0 and 1,1 in local 2x2
    # C indices: (ci1, ck1), (ci2, ck2) corresponding to 0,0 and 1,1 in local 2x2
    # Note: Global B index is j*p + k. Global C index is k*n + i.
    
    def add_strassen(ai1, aj1, ai2, aj2, bj1, bk1, bj2, bk2, ci1, ck1, ci2, ck2):
        nonlocal col
        # M1 = (A00 + A11)(B00 + B11)
        U[ai1*m + aj1][col] = 1
        U[ai2*m + aj2][col] = 1
        V[bj1*p + bk1][col] = 1
        V[bj2*p + bk2][col] = 1
        W[ck1*n + ci1][col] = 1
        W[ck2*n + ci2][col] = 1
        col += 1
        
        # M2 = (A10 + A11)B00
        U[ai2*m + aj1][col] = 1  # A10
        U[ai2*m + aj2][col] = 1  # A11
        V[bj1*p + bk1][col] = 1  # B00
        W[ck2*n + ci2][col] = 1  # C10
        col += 1
        
        # M3 = A00(B01 - B11)
        U[ai1*m + aj1][col] = 1  # A00
        V[bj1*p + bk2][col] = 1  # B01
        V[bj2*p + bk2][col] = -1 # B11
        W[ck1*n + ci2][col] = 1  # C01
        col += 1
        
        # M4 = A11(B10 - B00)
        U[ai2*m + aj2][col] = 1  # A11
        V[bj2*p + bk1][col] = 1  # B10
        V[bj1*p + bk1][col] = -1 # B00
        W[ck2*n + ci1][col] = 1  # C10
        col += 1
        
        # M5 = (A00 + A01)B11
        U[ai1*m + aj1][col] = 1  # A00
        U[ai1*m + aj2][col] = 1  # A01
        V[bj2*p + bk2][col] = 1  # B11
        W[ck1*n + ci2][col] = 1  # C01
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[ai2*m + aj1][col] = 1  # A10
        U[ai1*m + aj1][col] = -1 # A00
        V[bj1*p + bk1][col] = 1  # B00
        V[bj1*p + bk2][col] = 1  # B01
        W[ck2*n + ci2][col] = 1  # C11
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[ai1*m + aj2][col] = 1  # A01
        U[ai2*m + aj2][col] = -1 # A11
        V[bj2*p + bk1][col] = 1  # B10
        V[bj2*p + bk2][col] = 1  # B11
        W[ck1*n + ci1][col] = 1  # C00
        col += 1

    # Block 1: A(0:2, 0:2) * B(0:2, 0:2) -> C(0:2, 0:2)
    # A indices: (0,0), (1,1) are corners? No, local 0,0 is global (0,0), local 1,1 is global (1,1).
    # Local 0,0: A[0,0], B[0,0], C[0,0]
    # Local 1,1: A[1,1], B[1,1], C[1,1]
    # Local 0,1: A[0,1], B[0,1], C[0,1]
    # Local 1,0: A[1,0], B[1,0], C[1,0]
    
    add_strassen(0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1)
    
    # Block 2: A(0:2, 0:2) * B(0:2, 2:4) -> C(0:2, 2:4)
    # B columns 2,3 correspond to k=2,3.
    # Local 0,0: B[0,2], Local 1,1: B[1,3]
    # Local 0,1: B[0,3], Local 1,0: B[1,2]
    add_strassen(0, 0, 1, 1, 0, 2, 1, 3, 0, 2, 1, 3)
    
    # Linear terms
    # Y = a e^T, a = A[:,2] (i=0,1), e = B[2,:] (k=0,1,2,3)
    # Y[i,k] = A[i,2] B[2,k]
    for i in range(2):
        for k in range(4):
            U[i*m + 2][col] = 1
            V[2*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    # Z = b^T B(0:2, :), b = A[2,:] (j=0,1)
    # Z[k] = A[2,0]B[0,k] + A[2,1]B[1,k]
    # We need to add these to C[2,k]
    for k in range(4):
        # Term A[2,0] B[0,k]
        U[2*m + 0][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 2][col] = 1
        col += 1
        # Term A[2,1] B[1,k]
        U[2*m + 1][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 2][col] = 1
        col += 1
        
    # W = c e^T, c = A[2,2], e = B[2,:]
    # W[k] = A[2,2] B[2,k]
    for k in range(4):
        U[2*m + 2][col] = 1
        V[2*p + k][col] = 1
        W[k*n + 2][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
