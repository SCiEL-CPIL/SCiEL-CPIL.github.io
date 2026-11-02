"""Optimized matrix multiplication 2x4 by 4x5 using Strassen for 2x2 sub-blocks."""

DIMENSIONS = [2, 4, 5]

def construct():
    """
    Find a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 2x4 matrix by a 4x5 matrix.
    
    Strategy:
    - Split the 4x5 matrix into 2x2 blocks (last column separate).
    - For each 2x2 block of B and corresponding 2x2 sub-block of A (cols 0-1 or 2-3),
      use Strassen's 7-multiplication algorithm for 2x2x2.
    - The last column of B (size 4x1) is handled by schoolbook: 2x4 by 4x1 = 8 mults.
    
    Total rank: 4 blocks * 7 (Strassen) + 8 (schoolbook) = 36.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # We'll build U, V, W as lists of lists of integers.
    # U: (n*m) x rank = 8 x rank
    # V: (m*p) x rank = 20 x rank
    # W: (p*n) x rank = 10 x rank
    
    # First, let's figure out the total rank.
    # Strassen for 2x2x2 uses 7 multiplications.
    # We have:
    #   - B columns 0-1 (2 cols): A cols 0-1 * B[0:2, 0:2] (2x2 * 2x2) + A cols 2-3 * B[2:4, 0:2] (2x2 * 2x2)
    #   - B columns 2-3 (2 cols): A cols 0-1 * B[0:2, 2:4] (2x2 * 2x2) + A cols 2-3 * B[2:4, 2:4] (2x2 * 2x2)
    #   - B column 4 (1 col): A * B[:, 4] (2x4 * 4x1) = schoolbook, 8 mults
    #
    # So 4 Strassen blocks (7 each) + 1 schoolbook (8) = 28 + 8 = 36
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    col = 0
    
    def add_strassen_222(i0, j0, k0, col_start):
        """
        Add a Strassen 2x2x2 decomposition for:
        C[i0:i0+2, k0:k0+2] = A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        
        Indices in U: i*m + j where i in {i0, i0+1}, j in {j0, j0+1}
        Indices in V: j*p + k where j in {j0, j0+1}, k in {k0, k0+1}
        Indices in W: k*n + i where k in {k0, k0+1}, i in {i0, i0+1}
        """
        nonlocal col
        # Strassen's 7 multiplications for 2x2:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        i0_idx = [i0, i0 + 1]
        j0_idx = [j0, j0 + 1]
        k0_idx = [k0, k0 + 1]
        
        # M1 = (A00 + A11) * (B00 + B11)
        c = col_start
        # U: A00 and A11 -> +1
        U[i0 * m + j0][c] += 1
        U[(i0+1) * m + (j0+1)][c] += 1
        # V: B00 and B11 -> +1
        V[j0 * p + k0][c] += 1
        V[(j0+1) * p + (k0+1)][c] += 1
        # W: C00 gets +M1, C11 gets +M1
        W[k0 * n + i0][c] += 1
        W[(k0+1) * n + (i0+1)][c] += 1
        col += 1
        
        # M2 = (A10 + A11) * B00
        c = col_start + 1
        U[(i0+1) * m + j0][c] += 1
        U[(i0+1) * m + (j0+1)][c] += 1
        V[j0 * p + k0][c] += 1
        # W: C10 gets +M2, C11 gets -M2
        W[k0 * n + (i0+1)][c] += 1
        W[(k0+1) * n + (i0+1)][c] -= 1
        col += 1
        
        # M3 = A00 * (B01 - B11)
        c = col_start + 2
        U[i0 * m + j0][c] += 1
        V[j0 * p + (k0+1)][c] += 1
        V[(j0+1) * p + (k0+1)][c] -= 1
        # W: C01 gets +M3, C11 gets +M3
        W[(k0+1) * n + i0][c] += 1
        W[(k0+1) * n + (i0+1)][c] += 1
        col += 1
        
        # M4 = A11 * (B10 - B00)
        c = col_start + 3
        U[(i0+1) * m + (j0+1)][c] += 1
        V[(j0+1) * p + k0][c] += 1
        V[j0 * p + k0][c] -= 1
        # W: C00 gets +M4, C10 gets +M4
        W[k0 * n + i0][c] += 1
        W[k0 * n + (i0+1)][c] += 1
        col += 1
        
        # M5 = (A00 + A01) * B11
        c = col_start + 4
        U[i0 * m + j0][c] += 1
        U[i0 * m + (j0+1)][c] += 1
        V[(j0+1) * p + (k0+1)][c] += 1
        # W: C00 gets -M5, C01 gets +M5
        W[k0 * n + i0][c] -= 1
        W[(k0+1) * n + i0][c] += 1
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        c = col_start + 5
        U[(i0+1) * m + j0][c] += 1
        U[i0 * m + j0][c] -= 1
        V[j0 * p + k0][c] += 1
        V[j0 * p + (k0+1)][c] += 1
        # W: C11 gets +M6
        W[(k0+1) * n + (i0+1)][c] += 1
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        c = col_start + 6
        U[i0 * m + (j0+1)][c] += 1
        U[(i0+1) * m + (j0+1)][c] -= 1
        V[(j0+1) * p + k0][c] += 1
        V[(j0+1) * p + (k0+1)][c] += 1
        # W: C00 gets +M7
        W[k0 * n + i0][c] += 1
        col += 1
        
        return col_start + 7
    
    def add_schoolbook_2x4x1(i0, j0, k0, col_start):
        """
        Add schoolbook decomposition for:
        C[i0:i0+2, k0:k0+1] = A[i0:i0+2, j0:j0+4] * B[j0:j0+4, k0:k0+1]
        This is 2*4*1 = 8 multiplications.
        """
        c = col_start
        for di in range(2):
            for dj in range(4):
                U[(i0 + di) * m + (j0 + dj)][c] = 1
                V[(j0 + dj) * p + k0][c] = 1
                W[k0 * n + (i0 + di)][c] = 1
                c += 1
        return c
    
    # Block 1: C[:, 0:2] from A[:, 0:2] * B[0:2, 0:2]
    col = add_strassen_222(0, 0, 0, 0)
    
    # Block 2: C[:, 0:2] from A[:, 2:4] * B[2:4, 0:2]
    col = add_strassen_222(0, 2, 0, col)
    
    # Block 3: C[:, 2:4] from A[:, 0:2] * B[0:2, 2:4]
    col = add_strassen_222(0, 0, 2, col)
    
    # Block 4: C[:, 2:4] from A[:, 2:4] * B[2:4, 2:4]
    col = add_strassen_222(0, 2, 2, col)
    
    # Block 5: C[:, 4] from A * B[:, 4] (schoolbook 2x4 * 4x1)
    col = add_schoolbook_2x4x1(0, 0, 4, col)
    
    # col should be 36
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
