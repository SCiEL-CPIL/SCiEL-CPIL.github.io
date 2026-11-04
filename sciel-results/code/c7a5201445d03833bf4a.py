import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START

def _schoolbook_2x3x6():
    """
    Schoolbook matrix multiplication tensor decomposition for 2x3x6.
    Rank = n * m * p = 36.
    
    U shape: (n*m, rank) = (6, 36)
    V shape: (m*p, rank) = (18, 36)
    W shape: (p*n, rank) = (12, 36)
    
    Index conventions:
      U[i*m + j, r]
      V[j*p + k, r]
      W[k*n + i, r]
    
    The tensor decomposition must satisfy:
      sum_r U[i*m+j, r] * V[l*p+k, r] * W[k*n+i, r] = delta_{j,l}
    for all i, k, j, l.
    """
    n, m, p = DIMENSIONS
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


def _strassen_2x2x2_block(U_base, V_base, W_base, i0, k0, rank_offset, n, m, p):
    """
    Add a Strassen 2x2x2 block to the decomposition.
    i0: row offset in A (2 rows: i0, i0+1)
    k0: col offset in B (2 cols: k0, k0+1)
    
    Strassen products:
      M1 = (A11+A22)(B11+B22)
      M2 = (A21+A22)B11
      M3 = A11(B12-B22)
      M4 = A22(B21-B11)
      M5 = (A11+A12)B22
      M6 = (A21-A11)(B11+B12)
      M7 = (A12-A22)(B21+B22)
      
    Output:
      C11 = M1 + M4 - M5 - M7
      C12 = M3 + M5
      C21 = M2 + M7
      C22 = M1 - M2 + M3 + M4
    """
    # U factors for A submatrix (2x2): rows i0, i0+1; cols 0,1
    # V factors for B submatrix (2x2): rows 0,1; cols k0, k0+1
    # W factors for C submatrix (2x2): rows k0, k0+1; cols i0, i0+1
    
    # M1 = (A11+A22)(B11+B22)
    # U: A11 + A22 -> row i0: 1, row i0+1: 1
    # V: B11 + B22 -> row 0: 1, row 1: 1
    # W: C11 + C22 -> (k0, i0): 1, (k0+1, i0+1): 1
    U_base[i0 * m + 0][rank_offset] = 1
    U_base[(i0 + 1) * m + 1][rank_offset] = 1
    V_base[0 * p + k0][rank_offset] = 1
    V_base[1 * p + (k0 + 1)][rank_offset] = 1
    W_base[k0 * n + i0][rank_offset] = 1
    W_base[(k0 + 1) * n + (i0 + 1)][rank_offset] = 1
    
    # M2 = (A21+A22)B11
    # U: A21 + A22 -> row i0+1: col 1: 1, col 2: 1
    # V: B11 -> row 0, col k0
    # W: C21 -> (k0, i0+1)
    U_base[(i0 + 1) * m + 1][rank_offset + 1] = 1
    U_base[(i0 + 1) * m + 2][rank_offset + 1] = 1
    V_base[0 * p + k0][rank_offset + 1] = 1
    W_base[k0 * n + (i0 + 1)][rank_offset + 1] = 1
    
    # M3 = A11(B12-B22)
    # U: A11 -> row i0, col 0
    # V: B12 - B22 -> row 0, col k0+1: 1; row 1, col k0+1: -1
    # W: C12 -> (k0+1, i0)
    U_base[i0 * m + 0][rank_offset + 2] = 1
    V_base[0 * p + (k0 + 1)][rank_offset + 2] = 1
    V_base[1 * p + (k0 + 1)][rank_offset + 2] = -1
    W_base[(k0 + 1) * n + i0][rank_offset + 2] = 1
    
    # M4 = A22(B21-B11)
    # U: A22 -> row i0+1, col 1
    # V: B21 - B11 -> row 1, col k0: 1; row 0, col k0: -1
    # W: C11 -> (k0, i0)
    U_base[(i0 + 1) * m + 1][rank_offset + 3] = 1
    V_base[1 * p + k0][rank_offset + 3] = 1
    V_base[0 * p + k0][rank_offset + 3] = -1
    W_base[k0 * n + i0][rank_offset + 3] = 1
    
    # M5 = (A11+A12)B22
    # U: A11 + A12 -> row i0: col 0: 1, col 1: 1
    # V: B22 -> row 1, col k0+1
    # W: C11 -> (k0, i0) with -1; C12 -> (k0+1, i0) with 1
    U_base[i0 * m + 0][rank_offset + 4] = 1
    U_base[i0 * m + 1][rank_offset + 4] = 1
    V_base[1 * p + (k0 + 1)][rank_offset + 4] = 1
    W_base[k0 * n + i0][rank_offset + 4] = -1
    W_base[(k0 + 1) * n + i0][rank_offset + 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    # U: A21 - A11 -> row i0+1, col 1: 1; row i0, col 0: -1
    # V: B11 + B12 -> row 0, col k0: 1; row 0, col k0+1: 1
    # W: C21 -> (k0, i0+1)
    U_base[(i0 + 1) * m + 1][rank_offset + 5] = 1
    U_base[i0 * m + 0][rank_offset + 5] = -1
    V_base[0 * p + k0][rank_offset + 5] = 1
    V_base[0 * p + (k0 + 1)][rank_offset + 5] = 1
    W_base[k0 * n + (i0 + 1)][rank_offset + 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    # U: A12 - A22 -> row i0, col 1: 1; row i0+1, col 1: -1
    # V: B21 + B22 -> row 1, col k0: 1; row 1, col k0+1: 1
    # W: C21 -> (k0, i0+1) with 1; C11 -> (k0, i0) with -1
    U_base[i0 * m + 1][rank_offset + 6] = 1
    U_base[(i0 + 1) * m + 1][rank_offset + 6] = -1
    V_base[1 * p + k0][rank_offset + 6] = 1
    V_base[1 * p + (k0 + 1)][rank_offset + 6] = 1
    W_base[k0 * n + (i0 + 1)][rank_offset + 6] = 1
    W_base[k0 * n + i0][rank_offset + 6] = -1


def _assemble_strassen():
    """
    Assemble rank-33 decomposition for 2x3x6:
    Split the middle dimension (3) into 2+1:
    - 2x2x6 part: A[:,0:2] * B[0:2,:] -> split 6 cols into three 2-col blocks, each 2x2x2 Strassen (rank 7) -> 21
    - 2x1x6 part: A[:,2] * B[2,:] -> schoolbook -> 12
    Total rank: 21 + 12 = 33
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    rank = 33
    U = [[0] * rank for _ in range(n * m)]  # (6, 33)
    V = [[0] * rank for _ in range(m * p)]  # (18, 33)
    W = [[0] * rank for _ in range(p * n)]  # (12, 33)
    
    # Part 1: 2x2x6 using three Strassen 2x2x2 blocks
    # A rows 0-1, cols 0-1; B rows 0-1, cols k0..k0+1; C rows k0..k0+1, cols 0-1
    for block in range(3):
        k0 = block * 2
        ro = block * 7  # rank offset
        
        # M1 = (A11+A22)(B11+B22) -> C11 + C22
        U[0 * m + 0][ro] = 1      # A[0,0]
        U[1 * m + 1][ro] = 1      # A[1,1]
        V[0 * p + k0][ro] = 1     # B[0,k0]
        V[1 * p + (k0+1)][ro] = 1 # B[1,k0+1]
        W[k0 * n + 0][ro] = 1     # C[k0,0]
        W[(k0+1) * n + 1][ro] = 1 # C[k0+1,1]
        
        # M2 = (A21+A22)B11 -> C21(+1), C22(-1)
        U[1 * m + 1][ro+1] = 1    # A[1,1]
        U[1 * m + 0][ro+1] = 1    # A[1,0]
        V[0 * p + k0][ro+1] = 1   # B[0,k0]
        W[k0 * n + 1][ro+1] = 1   # C[k0,1]
        W[(k0+1) * n + 1][ro+1] = -1 # C[k0+1,1]
        
        # M3 = A11(B12-B22) -> C12(+1), C22(+1)
        U[0 * m + 0][ro+2] = 1    # A[0,0]
        V[0 * p + (k0+1)][ro+2] = 1  # B[0,k0+1]
        V[1 * p + (k0+1)][ro+2] = -1 # B[1,k0+1]
        W[(k0+1) * n + 0][ro+2] = 1  # C[k0+1,0]
        W[(k0+1) * n + 1][ro+2] = 1  # C[k0+1,1]
        
        # M4 = A22(B21-B11) -> C11
        U[1 * m + 1][ro+3] = 1    # A[1,1]
        V[1 * p + k0][ro+3] = 1   # B[1,k0]
        V[0 * p + k0][ro+3] = -1  # B[0,k0]
        W[k0 * n + 0][ro+3] = 1   # C[k0,0]
        
        # M5 = (A11+A12)B22 -> -C11 + C12
        U[0 * m + 0][ro+4] = 1    # A[0,0]
        U[0 * m + 1][ro+4] = 1    # A[0,1]
        V[1 * p + (k0+1)][ro+4] = 1 # B[1,k0+1]
        W[k0 * n + 0][ro+4] = -1  # C[k0,0]
        W[(k0+1) * n + 0][ro+4] = 1 # C[k0+1,0]
        
        # M6 = (A21-A11)(B11+B12) -> C21
        U[1 * m + 0][ro+5] = 1    # A[1,0]
        U[0 * m + 0][ro+5] = -1   # A[0,0]
        V[0 * p + k0][ro+5] = 1   # B[0,k0]
        V[0 * p + (k0+1)][ro+5] = 1 # B[0,k0+1]
        W[k0 * n + 1][ro+5] = 1   # C[k0,1]
        
        # M7 = (A12-A22)(B21+B22) -> C21 - C11
        U[0 * m + 1][ro+6] = 1    # A[0,1]
        U[1 * m + 1][ro+6] = -1   # A[1,1]
        V[1 * p + k0][ro+6] = 1   # B[1,k0]
        V[1 * p + (k0+1)][ro+6] = 1 # B[1,k0+1]
        W[k0 * n + 1][ro+6] = 1   # C[k0,1]
        W[k0 * n + 0][ro+6] = -1  # C[k0,0]
    
    # Part 2: 2x1x6 schoolbook: A[:,2] * B[2,:]
    # C[i,k] += A[i,2] * B[2,k]
    # U[i*m+2, r] = A[i,2], V[2*p+k, r] = B[2,k], W[k*n+i, r] = C[k,i]
    for i in range(n):
        for k in range(p):
            r = 21 + i * p + k
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}


def _verify_full(U, V, W):
    """
    Verify that the tensor decomposition is correct.
    sum_r U[i*m+j, r] * V[l*p+k, r] * W[k*n+i, r] = delta_{j,l}
    """
    from fractions import Fraction
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(m):
                    total = Fraction(0)
                    for r in range(rank):
                        total += Fraction(U[i * m + j][r]) * Fraction(V[l * p + k][r]) * Fraction(W[k * n + i][r])
                    expected = Fraction(1) if j == l else Fraction(0)
                    if total != expected:
                        return False
    return True


def construct():
    deadline = time.time() + 110.0
    try:
        result = _assemble_strassen()
        if time.time() < deadline:
            if _verify_full(result["U"], result["V"], result["W"]):
                return result
    except Exception:
        pass
    # Fallback to schoolbook
    try:
        fb = _schoolbook_2x3x6()
        if _verify_full(fb["U"], fb["V"], fb["W"]):
            return fb
    except Exception:
        pass
    return _schoolbook_2x3x6()

# EVOLVE-BLOCK-END
