"""
Rewritten program for 4x4x4 matrix multiplication tensor decomposition.
Uses Kronecker product of Strassen rank-7 decomposition to achieve rank 49.
"""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as lists of lists with shape (4, 7), (4, 7), (4, 7).
    
    The 7 Strassen terms:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    U[i*2+j, r] = coefficient of A[i,j] in term r
    V[j*2+k, r] = coefficient of B[j,k] in term r
    W[k*2+i, r] = coefficient of C[i,k] in term r (trace convention)
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    # A factors: A00 (i=0,j=0), A11 (i=1,j=1)
    # B factors: B00 (j=0,k=0), B11 (j=1,k=1)
    # C output: C00 (i=0,k=0), C11 (i=1,k=1)
    U[0*2+0][0] = 1  # A00
    U[1*2+1][0] = 1  # A11
    V[0*2+0][0] = 1  # B00
    V[1*2+1][0] = 1  # B11
    W[0*2+0][0] = 1  # C00
    W[1*2+1][0] = 1  # C11
    
    # Term 2: M2 = (A10+A11)B00
    # A factors: A10 (i=1,j=0), A11 (i=1,j=1)
    # B factors: B00 (j=0,k=0)
    # C output: C10 (i=1,k=0), C11 (i=1,k=1)
    U[1*2+0][1] = 1  # A10
    U[1*2+1][1] = 1  # A11
    V[0*2+0][1] = 1  # B00
    W[0*2+1][1] = 1  # C10
    W[1*2+1][1] = -1 # C11
    
    # Term 3: M3 = A00(B01-B11)
    # A factors: A00 (i=0,j=0)
    # B factors: B01 (j=0,k=1), B11 (j=1,k=1)
    # C output: C01 (i=0,k=1), C11 (i=1,k=1)
    U[0*2+0][2] = 1  # A00
    V[0*2+1][2] = 1  # B01
    V[1*2+1][2] = -1 # B11
    W[1*2+0][2] = 1  # C01
    W[1*2+1][2] = 1  # C11
    
    # Term 4: M4 = A11(B10-B00)
    # A factors: A11 (i=1,j=1)
    # B factors: B10 (j=1,k=0), B00 (j=0,k=0)
    # C output: C00 (i=0,k=0), C10 (i=1,k=0)
    U[1*2+1][3] = 1  # A11
    V[1*2+0][3] = 1  # B10
    V[0*2+0][3] = -1 # B00
    W[0*2+0][3] = 1  # C00
    W[0*2+1][3] = 1  # C10
    
    # Term 5: M5 = (A00+A01)B11
    # A factors: A00 (i=0,j=0), A01 (i=0,j=1)
    # B factors: B11 (j=1,k=1)
    # C output: C00 (i=0,k=0), C01 (i=0,k=1)
    U[0*2+0][4] = 1  # A00
    U[0*2+1][4] = 1  # A01
    V[1*2+1][4] = 1  # B11
    W[0*2+0][4] = -1 # C00
    W[1*2+0][4] = 1  # C01
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    # A factors: A10 (i=1,j=0), A00 (i=0,j=0)
    # B factors: B00 (j=0,k=0), B01 (j=0,k=1)
    # C output: C11 (i=1,k=1)
    U[1*2+0][5] = 1  # A10
    U[0*2+0][5] = -1 # A00
    V[0*2+0][5] = 1  # B00
    V[0*2+1][5] = 1  # B01
    W[1*2+1][5] = 1  # C11
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    # A factors: A01 (i=0,j=1), A11 (i=1,j=1)
    # B factors: B10 (j=1,k=0), B11 (j=1,k=1)
    # C output: C00 (i=0,k=0)
    U[0*2+1][6] = 1  # A01
    U[1*2+1][6] = -1 # A11
    V[1*2+0][6] = 1  # B10
    V[1*2+1][6] = 1  # B11
    W[0*2+0][6] = 1  # C00
    
    return U, V, W

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1, else 0.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomp(n, m, p, U, V, W, tol=1e-9):
    """Verify that sum_r U[a,r]*V[b,r]*W[c,r] matches the tensor."""
    T = _build_tensor(n, m, p)
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    T_decomp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    max_diff = np.max(np.abs(T_decomp - T))
    return max_diff < tol, max_diff

def _kronecker_decomposition(n, m, p):
    """
    Construct the rank-49 decomposition for 4x4x4 using Kronecker product
    of the rank-7 Strassen decomposition for 2x2x2.
    
    The idea: view 4x4 matrices as 2x2 block matrices of 2x2 submatrices.
    Apply Strassen at the block level (7 block multiplications), and each
    block multiplication is itself a 2x2*2x2 product decomposed via Strassen
    (rank 7). Total rank = 7 * 7 = 49.
    """
    # Get the base Strassen decomposition for 2x2x2
    U2, V2, W2 = _strassen_2x2()
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # (16, 49)
    V = [[0] * rank for _ in range(m * p)]  # (16, 49)
    W = [[0] * rank for _ in range(p * n)]  # (16, 49)
    
    # For 4x4 matrices:
    # i = 2*i1 + i2 (row of A, row of C)
    # j = 2*j1 + j2 (col of A, row of B)
    # k = 2*k1 + k2 (col of B, col of C)
    # where i1, i2, j1, j2, k1, k2 in {0, 1}
    
    # The Kronecker product structure:
    # U[i, j, r1*7+r2] = U2[i1, j1, r1] * U2[i2, j2, r2]
    # V[j, k, r1*7+r2] = V2[j1, k1, r1] * V2[j2, k2, r2]
    # W[k, i, r1*7+r2] = W2[k1, i1, r1] * W2[k2, i2, r2]
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            i = 2 * i1 + i2
                            j = 2 * j1 + j2
                            k = 2 * k1 + k2
                            for r1 in range(7):
                                for r2 in range(7):
                                    r = r1 * 7 + r2
                                    U[i * m + j][r] = U2[i1 * 2 + j1][r1] * U2[i2 * 2 + j2][r2]
                                    V[j * p + k][r] = V2[j1 * 2 + k1][r1] * V2[j2 * 2 + k2][r2]
                                    W[k * n + i][r] = W2[k1 * 2 + i1][r1] * W2[k2 * 2 + i2][r2]
    
    return {"U": U, "V": V, "W": W, "rank": rank}

def _rank28_decomposition(n, m, p):
    """
    Construct a rank-28 decomposition for 4x4x4 using Strassen at block level
    with schoolbook (rank 4) for each 2x2 block product.
    
    Total rank = 7 * 4 = 28.
    
    View 4x4 matrices as 2x2 block matrices of 2x2 submatrices.
    Block-level: C[i1,k1] = sum_j1 A[i1,j1] * B[j1,k1]
    Strassen computes 7 linear combinations at block level.
    Each block product (2x2 * 2x2) is computed via schoolbook (4 scalar products).
    
    For each Strassen term r1 and each scalar position (i2, j2, k2) in the 2x2 blocks:
      U[ (2*i1+i2)*m + (2*j1+j2) ][ r1*4 + i2*2 + k2 ] = U2[i1*2+j1][r1]
      V[ (2*j1+j2)*p + (2*k1+k2) ][ r1*4 + i2*2 + k2 ] = V2[j1*2+k1][r1]  
      W[ (2*k1+k2)*n + (2*i1+i2) ][ r1*4 + i2*2 + k2 ] = W2[k1*2+i1][r1]
    
    Wait, that's not right either. Let me think more carefully.
    
    Actually, the correct construction:
    - The 4x4 multiplication tensor decomposes as:
      sum_{r1=0}^{6} sum_{i2,j2,k2} U2[i1*2+j1][r1] * V2[j1*2+k1][r1] * W2[k1*2+i1][r1]
      * A[2*i1+i2, 2*j1+j2] * B[2*j1+j2, 2*k1+k2] * C[2*i1+i2, 2*k1+k2]
    
    But this isn't a valid rank-1 decomposition because the scalar positions (i2,j2,k2) 
    are coupled across the three matrices.
    
    The correct rank-28 approach:
    For each Strassen term r1, the block product A[i1,j1]*B[j1,k1] is a 2x2 matrix product.
    This 2x2 product has rank 4 (schoolbook), giving 4 scalar products per block position.
    So for each r1, we have 4 rank-1 terms, one for each (i2, k2) pair where
    i2 is the row within the 2x2 block and k2 is the col within the 2x2 block.
    
    For scalar product: A[i1*2+i2][j1*2+j2] * B[j1*2+j2][k1*2+k2] contributes to C[i1*2+i2][k1*2+k2]
    with coefficient 1 (schoolbook).
    
    But in the Strassen framework, the coefficient for A[i1,j1] in term r1 is U2[i1*2+j1][r1],
    and for B[j1,k1] is V2[j1*2+k1][r1], and for C[i1,k1] is W2[k1*2+i1][r1].
    
    So the full decomposition is:
    For each r1 in 0..6, for each (i2, j2, k2) in {0,1}^3:
      This is NOT valid because j2 appears in both U and V but they must be independent.
    
    The key insight: the rank-28 decomposition works because we can separate the 
    block-level Strassen from the sub-block schoolbook. The correct form is:
    
    For each r1 in 0..6 (Strassen term) and each (i2, k2) in {0,1}^2 (sub-block position):
      r = r1 * 4 + i2 * 2 + k2
      For each (i1, j1, k1) in {0,1}^3:
        U[(2*i1+i2)*m + (2*j1+j2)][r] ... but j2 is not determined!
    
    I think the correct approach is different. Let me use the tensor product structure:
    The 4x4x4 multiplication tensor is the tensor product of two 2x2x2 multiplication tensors.
    rank(MM_4) <= rank(MM_2) * rank(MM_2) = 7 * 7 = 49 (Kronecker)
    
    For rank 28, we need: rank(MM_4) <= rank(MM_2) * rank(MM_2_schoolbook) = 7 * 4 = 28
    This is valid because MM_2 has rank <= 4 (schoolbook), so:
    MM_4 = MM_2 ⊗ MM_2, and rank(A ⊗ B) <= rank(A) * rank(B)
    Using rank(MM_2) = 7 for the first factor and rank(MM_2) = 4 for the second:
    rank(MM_4) <= 7 * 4 = 28.
    
    The construction:
    Let MM_2 = sum_{r=0}^{6} u_r ⊗ v_r ⊗ w_r (Strassen, rank 7)
    Let MM_2' = sum_{s=0}^{3} u'_s ⊗ v'_s ⊗ w'_s (schoolbook, rank 4)
    
    Then MM_4 = MM_2 ⊗ MM_2' = sum_{r,s} (u_r ⊗ u'_s) ⊗ (v_r ⊗ v'_s) ⊗ (w_r ⊗ w'_s)
    
    This gives rank 7*4 = 28.
    """
    U2, V2, W2 = _strassen_2x2()
    
    # Schoolbook 2x2x2 decomposition (rank 4)
    # MM_2: sum_{i,j,k} A[i,j] * B[j,k] * C[i,k]
    # Rank-1 terms: for each (i,j,k), U[i*2+j] = 1, V[j*2+k] = 1, W[k*2+i] = 1
    U2s = [[0]*4 for _ in range(4)]
    V2s = [[0]*4 for _ in range(4)]
    W2s = [[0]*4 for _ in range(4)]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = i * 4 + j * 2 + k
                U2s[i*2+j][s] = 1
                V2s[j*2+k][s] = 1
                W2s[k*2+i][s] = 1
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Tensor product: index r = r1 * 4 + s
    # U[(2*i1+i2)*m + (2*j1+j2)][r] = U2[i1*2+j1][r1] * U2s[i2*2+j2][s]
    # V[(2*j1+j2)*p + (2*k1+k2)][r] = V2[j1*2+k1][r1] * V2s[j2*2+k2][s]
    # W[(2*k1+k2)*n + (2*i1+i2)][r] = W2[k1*2+i1][r1] * W2s[k2*2+i2][s]
    
    for r1 in range(7):
        for s in range(4):
            r = r1 * 4 + s
            for i1 in range(2):
                for j1 in range(2):
                    for k1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                for k2 in range(2):
                                    i = 2 * i1 + i2
                                    j = 2 * j1 + j2
                                    k = 2 * k1 + k2
                                    u_val = U2[i1*2+j1][r1] * U2s[i2*2+j2][s]
                                    v_val = V2[j1*2+k1][r1] * V2s[j2*2+k2][s]
                                    w_val = W2[k1*2+i1][r1] * W2s[k2*2+i2][s]
                                    if u_val != 0 and v_val != 0 and w_val != 0:
                                        U[i * m + j][r] = u_val
                                        V[j * p + k][r] = v_val
                                        W[k * n + i][r] = w_val
    
    return {"U": U, "V": V, "W": W, "rank": rank}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Deadline guard: must return before 120s (evaluator kills at 150s)
    deadline = start_time + 110  # 10s safety margin
    
    # Try rank-28 (Strassen block × schoolbook subblock) - best candidate
    try:
        result28 = _rank28_decomposition(n, m, p)
        valid28, diff28 = _verify_decomp(n, m, p, result28["U"], result28["V"], result28["W"])
        
        if valid28:
            return {"U": result28["U"], "V": result28["V"], "W": result28["W"]}
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline:
        # Fall back to schoolbook
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
    
    # Try rank-49 (Kronecker product of Strassen) - fallback
    try:
        result49 = _kronecker_decomposition(n, m, p)
        valid49, diff49 = _verify_decomp(n, m, p, result49["U"], result49["V"], result49["W"])
        
        if valid49:
            return {"U": result49["U"], "V": result49["V"], "W": result49["W"]}
    except Exception:
        pass
    
    # Final fallback: schoolbook
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
