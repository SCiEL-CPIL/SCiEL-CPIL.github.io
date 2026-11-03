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
    # i = 2*i1 + i2, k = 2*k1 + k2, j = 2*j1 + j2
    # where i1, i2, k1, k2, j1, j2 in {0, 1}
    
    # The Kronecker product structure:
    # U[i, k, r1*7+r2] = U2[i1, k1, r1] * U2[i2, k2, r2]
    # V[k, j, r1*7+r2] = V2[k1, j1, r1] * V2[k2, j2, r2]
    # W[j, i, r1*7+r2] = W2[j1, i1, r1] * W2[j2, i2, r2]
    
    for i1 in range(2):
        for i2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    i = 2 * i1 + i2
                    k = 2 * k1 + k2
                    for j1 in range(2):
                        for j2 in range(2):
                            j = 2 * j1 + j2
                            for r1 in range(7):
                                for r2 in range(7):
                                    r = r1 * 7 + r2
                                    U[i * m + k][r] = U2[i1 * 2 + k1][r1] * U2[i2 * 2 + k2][r2]
                                    V[k * p + j][r] = V2[k1 * 2 + j1][r1] * V2[k2 * 2 + j2][r2]
                                    W[j * n + i][r] = W2[j1 * 2 + i1][r1] * W2[j2 * 2 + i2][r2]
    
    return {"U": U, "V": V, "W": W, "rank": rank}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Construct the rank-49 decomposition using Kronecker product of Strassen
    result = _kronecker_decomposition(n, m, p)
    
    # Verify the decomposition
    valid, max_diff = _verify_decomp(n, m, p, result["U"], result["V"], result["W"])
    
    if not valid:
        # Fall back to schoolbook if verification fails (should not happen)
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
    
    return {"U": result["U"], "V": result["V"], "W": result["W"]}
# EVOLVE-BLOCK-END
