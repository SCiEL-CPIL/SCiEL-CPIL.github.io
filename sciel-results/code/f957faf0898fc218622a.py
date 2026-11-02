import numpy as np
import time

DIMENSIONS = [3, 3, 4]
n, m, p = DIMENSIONS

def _verify_decomposition(U, V, W, n, m, p, tol=1e-12):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    R = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    U_reshaped = U.T.reshape(R, n, m)
    V_reshaped = V.T.reshape(R, m, p)
    W_reshaped = W.T.reshape(R, p, n)
    
    T_computed = np.einsum('rij,rjk,rki->ikj', U_reshaped, V_reshaped, W_reshaped)
    T_target = np.ones((n, p, m))
    
    return np.allclose(T_computed, T_target, atol=tol)

def _schoolbook():
    """Return schoolbook decomposition with rank n*m*p."""
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def _strassen_2x2x2():
    """
    Return Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    Rank 7.
    
    U: shape (4, 7) - represents 2x2 matrices
    V: shape (4, 7) - represents 2x2 matrices  
    W: shape (4, 7) - represents 2x2 output matrices
    
    Strassen's 7 products:
    M1 = (A11+A22)*(B11+B22)
    M2 = (A21+A22)*B11
    M3 = A11*(B12-B22)
    M4 = A22*(B21-B11)
    M5 = (A11+A12)*B22
    M6 = (A21-A11)*(B11+B12)
    M7 = (A12-A22)*(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # U encodes A matrix entries: U[i*2+j, r] = coefficient for A[i,j] in product r
    # V encodes B matrix entries: V[i*2+j, r] = coefficient for B[i,j] in product r
    # W encodes C matrix entries: W[i*2+j, r] = coefficient for C[i,j] from product r
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)*(B11+B22)
    U[0, 0] = 1  # A11
    U[3, 0] = 1  # A22
    V[0, 0] = 1  # B11
    V[3, 0] = 1  # B22
    W[0, 0] = 1  # C11
    W[3, 0] = 1  # C22
    
    # M2 = (A21+A22)*B11
    U[2, 1] = 1  # A21
    U[3, 1] = 1  # A22
    V[0, 1] = 1  # B11
    W[2, 1] = 1  # C21
    W[3, 1] = -1 # C22
    
    # M3 = A11*(B12-B22)
    U[0, 2] = 1  # A11
    V[1, 2] = 1  # B12
    V[3, 2] = -1 # B22
    W[1, 2] = 1  # C12
    W[3, 2] = 1  # C22
    
    # M4 = A22*(B21-B11)
    U[3, 3] = 1  # A22
    V[2, 3] = 1  # B21
    V[0, 3] = -1 # B11
    W[0, 3] = 1  # C11
    W[2, 3] = 1  # C21
    
    # M5 = (A11+A12)*B22
    U[0, 4] = 1  # A11
    U[1, 4] = 1  # A12
    V[3, 4] = 1  # B22
    W[0, 4] = -1 # C11
    W[1, 4] = 1  # C12
    
    # M6 = (A21-A11)*(B11+B12)
    U[2, 5] = 1  # A21
    U[0, 5] = -1 # A11
    V[0, 5] = 1  # B11
    V[1, 5] = 1  # B12
    W[3, 5] = 1  # C22
    
    # M7 = (A12-A22)*(B21+B22)
    U[1, 6] = 1  # A12
    U[3, 6] = -1 # A22
    V[2, 6] = 1  # B21
    V[3, 6] = 1  # B22
    W[0, 6] = 1  # C11
    
    return U, V, W

def _construct_rank19():
    """
    Construct rank-19 decomposition using subspace decomposition.
    
    Partition 3x3 x 3x4 into:
    - 2x2 x 2x3 block (rows 0-1, cols 0-2 of A; rows 0-1, cols 0-2 of B; output rows 0-1, cols 0-2)
      Use Strassen for the 2x2x2 part + 2 extra terms for the third column
    - 1x3 x 1x4 block (row 2 of A; all of B; output row 2)
      This is just 3*4 = 12 terms
    
    Actually, let me think more carefully.
    
    We have A (3x3) * B (3x4) = C (3x4)
    
    Partition:
    A = [[A11, A12, A13],
         [A21, A22, A23],
         [A31, A32, A33]]
    
    B = [[B11, B12, B13, B14],
         [B21, B22, B23, B24],
         [B31, B32, B33, B34]]
    
    C = [[C11, C12, C13, C14],
         [C21, C22, C23, C24],
         [C31, C32, C33, C34]]
    
    Strategy:
    - Use Strassen on the top-left 2x2 x 2x2 subproblem: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
      This gives rank 7.
    
    - Handle the remaining terms:
      * A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4] : 2*2*2 = 8 terms (schoolbook)
      * A[0:2,2:3] * B[0:3,0:4] -> C[0:2,0:4] : 2*1*4 = 8 terms (schoolbook)  
      * A[2:3,0:3] * B[0:3,0:4] -> C[2:3,0:4] : 1*3*4 = 12 terms (schoolbook)
    
    Total: 7 + 8 + 8 + 12 = 35. That's not better than schoolbook 36.
    
    Let me try a different approach. Use Strassen on 2x2x3 blocks.
    
    Actually, for 2x2 x 2x3:
    A (2x2) * B (2x3) = C (2x3)
    
    We can view this as two 2x2x2 problems plus some cross terms, but that's not clean.
    
    Alternative: Use the known result that the minimal rank for 3x3 x 3x4 is at most 25-27 based on literature.
    
    Let me try a simpler approach: use Strassen recursively.
    
    For 3x3 x 3x4:
    Split A into 2x2 + 1x1 blocks (top-left 2x2, and the rest)
    Split B into 2x2 + 1x2 blocks
    
    This gets complicated. Let me just implement a careful schoolbook + one Strassen application.
    
    Actually, let me just do a clean schoolbook and verify it works, then try to improve.
    """
    pass

def _construct_rank25():
    """
    Construct a rank-25 decomposition.
    
    Use Strassen on the 2x2 x 2x2 top-left corner (rank 7),
    then handle the remaining terms with schoolbook.
    
    A (3x3) * B (3x4) = C (3x4)
    
    Terms:
    1. A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]: Strassen, rank 7
    2. A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]: schoolbook, 2*2*2=8 terms
    3. A[0:2,2:3] * B[0:3,0:4] -> C[0:2,0:4]: schoolbook, 2*1*4=8 terms
    4. A[2:3,0:3] * B[0:3,0:4] -> C[2:3,0:4]: schoolbook, 1*3*4=12 terms
    
    Total: 7 + 8 + 8 + 12 = 35. Still not great.
    
    Let me try: what if I use Strassen on a 2x2 x 2x4 block?
    That would be A[0:2,0:2] * B[0:2,0:4] -> C[0:2,0:4]
    But Strassen is for square matrices.
    
    I can split B[0:2,0:4] into B[0:2,0:2] and B[0:2,2:4], apply Strassen to each 2x2x2,
    giving rank 7+7=14 for this part.
    
    Then:
    1. A[0:2,0:2] * B[0:2,0:4] -> C[0:2,0:4]: 2 Strassen blocks, rank 14
    2. A[0:2,2:3] * B[0:3,0:4] -> C[0:2,0:4]: schoolbook, 2*1*4=8 terms
    3. A[2:3,0:3] * B[0:3,0:4] -> C[2:3,0:4]: schoolbook, 1*3*4=12 terms
    
    Total: 14 + 8 + 12 = 34.
    
    The knowledge says rank 34 was achieved. Let me implement that.
    """
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Part 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] using Strassen
    S_U, S_V, S_W = _strassen_2x2x2()
    
    # Map indices:
    # S_U[i*2+j, r] corresponds to A[i,j] where i,j in {0,1}
    # In full A (3x3), A[i,j] is at row i, col j. U index = i*m + j = i*3 + j
    # S_V[i*2+j, r] corresponds to B[i,j] where i,j in {0,1}
    # In full B (3x4), B[i,j] is at row i, col j. V index = i*p + j = i*4 + j
    # S_W[i*2+j, r] corresponds to C[i,j] where i,j in {0,1}
    # In full C (3x4), C[i,j] is at row i, col j. W index = k*n + i where k=j (col of B), i=row of A
    # Wait, W index is k*n + i where k is the column index of B (0..p-1) and i is row of A (0..n-1)
    # So for C[i,j], the W index is j*n + i
    
    U1 = np.zeros((n * m, 7))
    V1 = np.zeros((m * p, 7))
    W1 = np.zeros((p * n, 7))
    
    for r in range(7):
        for i in range(2):
            for j in range(2):
                # A[i,j] -> U index i*3 + j
                U1[i * m + j, r] = S_U[i * 2 + j, r]
                # B[i,j] -> V index i*4 + j
                V1[i * p + j, r] = S_V[i * 2 + j, r]
                # C[i,j] -> W index j*3 + i
                W1[j * n + i, r] = S_W[i * 2 + j, r]
    
    U_parts.append(U1)
    V_parts.append(V1)
    W_parts.append(W1)
    
    # Part 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4] using Strassen
    # This is A[0:2,0:2] * B[0:2,2:4] where B[0:2,2:4] is 2x2
    S_U2, S_V2, S_W2 = _strassen_2x2x2()
    
    U2 = np.zeros((n * m, 7))
    V2 = np.zeros((m * p, 7))
    W2 = np.zeros((p * n, 7))
    
    for r in range(7):
        for i in range(2):
            for j in range(2):
                # A[i,j] -> U index i*3 + j
                U2[i * m + j, r] = S_U2[i * 2 + j, r]
                # B[i, j+2] -> V index i*4 + (j+2)
                V2[i * p + (j + 2), r] = S_V2[i * 2 + j, r]
                # C[i, j+2] -> W index (j+2)*3 + i
                W2[(j + 2) * n + i, r] = S_W2[i * 2 + j, r]
    
    U_parts.append(U2)
    V_parts.append(V2)
    W_parts.append(W2)
    
    # Part 3: A[0:2,2:3] * B[0:3,0:4] -> C[0:2,0:4] schoolbook
    # A[0:2,2] is 2x1, B[0:3,0:4] is 3x4, output C[0:2,0:4] is 2x4
    # Terms: for i in 0..1, j in 0..2, k in 0..3: A[i,2] * B[j,k] contributes to C[i,k]
    # But wait, this isn't a standard matrix multiply. A[0:2,2:3] is 2x1, B is 3x4.
    # The product is (2x1)*(3x4) which doesn't make sense dimensionally.
    
    # I made an error. Let me reconsider.
    
    # A (3x3) * B (3x4) = C (3x4)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The terms are:
    # C[0:2,0:4] = A[0:2,0:3] * B[0:3,0:4]
    # C[2:3,0:4] = A[2:3,0:3] * B[0:3,0:4]
    
    # For C[0:2,0:4]:
    #   = A[0:2,0:2] * B[0:2,0:4] + A[0:2,2:3] * B[2:3,0:4]
    #   First part: 2x2 * 2x4 -> can use 2 Strassen blocks (rank 14)
    #   Second part: 2x1 * 1x4 -> schoolbook, 2*1*4 = 8 terms
    
    # For C[2:3,0:4]:
    #   = A[2:3,0:3] * B[0:3,0:4]
    #   1x3 * 3x4 -> schoolbook, 1*3*4 = 12 terms
    
    # Total: 14 + 8 + 12 = 34
    
    # Part 3: A[0:2,2:3] * B[2:3,0:4] -> C[0:2,0:4]
    # A[0:2,2] is 2x1 (column 2 of A, rows 0-1)
    # B[2:3,0:4] is 1x4 (row 2 of B)
    # C[i,k] += A[i,2] * B[2,k] for i in 0..1, k in 0..3
    # This is 2*4 = 8 terms (one for each (i,k) pair)
    
    U3 = np.zeros((n * m, 8))
    V3 = np.zeros((m * p, 8))
    W3 = np.zeros((p * n, 8))
    
    col = 0
    for i in range(2):
        for k in range(4):
            # A[i,2] * B[2,k] -> C[i,k]
            U3[i * m + 2, col] = 1
            V3[2 * p + k, col] = 1
            W3[k * n + i, col] = 1
            col += 1
    
    U_parts.append(U3)
    V_parts.append(V3)
    W_parts.append(W3)
    
    # Part 4: A[2:3,0:3] * B[0:3,0:4] -> C[2:3,0:4]
    # A[2,0:3] is 1x3, B[0:3,0:4] is 3x4
    # C[2,k] = sum_j A[2,j] * B[j,k] for j in 0..2, k in 0..3
    # 3*4 = 12 terms
    
    U4 = np.zeros((n * m, 12))
    V4 = np.zeros((m * p, 12))
    W4 = np.zeros((p * n, 12))
    
    col = 0
    for j in range(3):
        for k in range(4):
            # A[2,j] * B[j,k] -> C[2,k]
            U4[2 * m + j, col] = 1
            V4[j * p + k, col] = 1
            W4[k * n + 2, col] = 1
            col += 1
    
    U_parts.append(U4)
    V_parts.append(V4)
    W_parts.append(W4)
    
    U = np.hstack(U_parts)
    V = np.hstack(V_parts)
    W = np.hstack(W_parts)
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try rank 34 construction
    try:
        U, V, W = _construct_rank25()  # Actually rank 34
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U, V, W = _schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
