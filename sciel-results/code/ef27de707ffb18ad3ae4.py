"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-46 exact tensor decomposition for 2x2 * 2x13 matrix multiplication.
    
    Uses Strassen's algorithm on the 2x2 block structure where each block is 1x13.
    For each pair of columns (2k, 2k+1) of B, we apply a 7-multiplication Strassen
    scheme to the 2x2 block, giving 7*6 = 42 multiplications for the 12 paired
    columns plus 1 schoolbook multiplication for the unpaired column 12, total rank 43.
    
    Wait - let me reconsider. The blocks are 1x13, so we can't directly apply Strassen
    to 1x13 blocks. Instead, we treat the 2x13 matrix B as having 13 scalar columns,
    and the 2x2 matrix A multiplies them. The result C is 2x13.
    
    For a 2x2 * 2x13 multiplication:
    C[0][k] = A[0][0]*B[0][k] + A[0][1]*B[1][k]
    C[1][k] = A[1][0]*B[0][k] + A[1][1]*B[1][k]
    
    For each k, this is a 2x2 * 1x1 (scalar column) multiplication, which has rank 3
    using Strassen:
    M1 = (A00+A11)*(B0+B1)
    M2 = (A10+A11)*B0
    M3 = A00*(B1-B0)
    M4 = A11*(B0-B1)
    M5 = (A00+A01)*B1
    M6 = (A10-A00)*(B0+B1)
    M7 = (A01-A11)*(B0+B1)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    But here B is 2x13, so for each column k, B0 = B[0][k], B1 = B[1][k].
    This gives 7 multiplications per column, for 13 columns = 91 total.
    
    However, we can do better by grouping columns. Let's use pairs of columns.
    
    For columns 2k and 2k+1 (k=0..5, so 6 pairs = 12 columns), plus column 12.
    
    For a pair of columns (B[0][2k], B[1][2k], B[0][2k+1], B[1][2k+1]),
    we need to compute:
    C[0][2k]   = A00*B0_2k + A01*B1_2k
    C[1][2k]   = A10*B0_2k + A11*B1_2k
    C[0][2k+1] = A00*B0_2k1 + A01*B1_2k1
    C[1][2k+1] = A10*B0_2k1 + A11*B1_2k1
    
    This is a 2x2 * 2x2 multiplication where the "2x2" for B is actually
    two separate column vectors. We can apply Strassen to each column independently
    (7 multiplications per column, 14 per pair), but that's the same as doing it
    per-column.
    
    Alternatively, we can view this as multiplying a 2x2 matrix by a 2x2 matrix
    where the second matrix has columns [B_col_2k, B_col_2k1]. This is just
    standard 2x2 matrix multiplication with rank 7 (Strassen).
    
    So for 6 pairs of columns: 6 * 7 = 42 multiplications.
    For column 12: 3 multiplications (Strassen on a single column).
    Total: 45.
    
    Actually, for a single column (2x1), Strassen gives rank 3:
    M1 = (A00+A11)*(B0+B1)
    M2 = (A10+A11)*B0  
    M3 = A00*(B1-B0)
    C00 = M1 - M2 + M3
    C10 = M2 - M3
    
    Wait, let me re-derive. For 2x2 * 1x2 (treating the column as a 1x2 row vector
    transposed):
    
    Actually, let's just use the standard Strassen for 2x2 * 2x2 and apply it to
    each pair of columns as a 2x2 block, and for the single column 12, use a
    2x2 * 2x1 decomposition.
    
    For 2x2 * 2x2 (Strassen, rank 7):
    Let A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    M1 = (a00+a11)(b00+b11)
    M2 = (a10+a11)(b00+b01)
    M3 = (a00-a10)(b10+b11)
    M4 = a11(b11-b01)
    M5 = a00(b01-b10)
    M6 = (a01+a10)(b00+b10)
    M7 = (a01-a11)(b01+b11)
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    For our case, the "2x2 B block" for columns 2k, 2k+1 is:
    B_block = [[B[0][2k], B[0][2k+1]],
               [B[1][2k], B[1][2k+1]]]
    
    So b00 = B[0][2k], b01 = B[0][2k+1], b10 = B[1][2k], b11 = B[1][2k+1].
    
    For the single column 12, we need 2x2 * 1x1 (column vector):
    C[0][12] = A00*B[0][12] + A01*B[1][12]
    C[1][12] = A10*B[0][12] + A11*B[1][12]
    
    This is a bilinear map from (2x2) * (2x1) to (2x1), which has rank 3.
    Using:
    M1 = (A00+A11)*(B0+B1)
    M2 = (A10+A11)*B0
    M3 = A00*(B1-B0)
    C0 = M1 - M2 + M3
    C1 = M2 - M3
    
    So total rank = 6*7 + 3 = 45.
    
    Let me verify the index conventions:
    U has shape (n*m, rank) = (4, rank)
    V has shape (m*p, rank) = (26, rank)
    W has shape (p*n, rank) = (26, rank)
    
    U indices: i*m+j where i in [0,1], j in [0,1]
      U[0] = a00, U[1] = a01, U[2] = a10, U[3] = a11
    
    V indices: j*p+k where j in [0,1], k in [0,12]
      V[0..12] = B[0][0..12], V[13..25] = B[1][0..12]
    
    W indices: k*n+i where k in [0,12], i in [0,1]
      W[0] = C[0][0], W[1] = C[1][0], W[2] = C[0][1], W[3] = C[1][1], ...
      W[2k] = C[0][k], W[2k+1] = C[1][k]
    
    Let's build the rank-45 decomposition.
    """
    n, m, p = 2, 2, 13
    rank = 45
    U = [[0] * rank for _ in range(n * m)]  # 4 x 45
    V = [[0] * rank for _ in range(m * p)]  # 26 x 45
    W = [[0] * rank for _ in range(p * n)]  # 26 x 45
    
    col = 0
    # For each pair of columns (2k, 2k+1) for k=0..5
    for k in range(6):
        c0 = 2 * k
        c1 = 2 * k + 1
        # B_block = [[B[0][c0], B[0][c1]], [B[1][c0], B[1][c1]]]
        # b00 = B[0][c0] -> V[0*13 + c0] = V[c0]
        # b01 = B[0][c1] -> V[c1]
        # b10 = B[1][c0] -> V[13 + c0]
        # b11 = B[1][c1] -> V[13 + c1]
        
        # M1 = (a00+a11)(b00+b11)
        U[0][col] += 1  # a00
        U[3][col] += 1  # a11
        V[c0][col] += 1   # b00
        V[13 + c1][col] += 1  # b11
        # C00 = M1 + M4 - M5 + M7 => C[0][c0] gets +1 from M1
        W[c0 * 2 + 0][col] += 1  # C[0][c0]
        # C11 = M1 - M2 + M3 + M6 => C[1][c1] gets +1 from M1
        W[c1 * 2 + 1][col] += 1  # C[1][c1]
        col += 1
        
        # M2 = (a10+a11)(b00+b01)
        U[2][col] += 1  # a10
        U[3][col] += 1  # a11
        V[c0][col] += 1   # b00
        V[c1][col] += 1   # b01
        # C10 = M2 + M4 => C[1][c0] gets +1 from M2
        W[c0 * 2 + 1][col] += 1  # C[1][c0]
        # C11 = M1 - M2 + M3 + M6 => C[1][c1] gets -1 from M2
        W[c1 * 2 + 1][col] -= 1
        col += 1
        
        # M3 = (a00-a10)(b10+b11)
        U[0][col] += 1  # a00
        U[2][col] -= 1  # a10
        V[13 + c0][col] += 1  # b10
        V[13 + c1][col] += 1  # b11
        # C01 = M3 + M5 => C[0][c1] gets +1 from M3
        W[c1 * 2 + 0][col] += 1  # C[0][c1]
        # C11 = M1 - M2 + M3 + M6 => C[1][c1] gets +1 from M3
        W[c1 * 2 + 1][col] += 1
        col += 1
        
        # M4 = a11(b11-b01)
        U[3][col] += 1  # a11
        V[13 + c1][col] += 1  # b11
        V[c1][col] -= 1   # b01
        # C00 = M1 + M4 - M5 + M7 => C[0][c0] gets +1 from M4
        W[c0 * 2 + 0][col] += 1  # C[0][c0]
        # C10 = M2 + M4 => C[1][c0] gets +1 from M4
        W[c0 * 2 + 1][col] += 1  # C[1][c0]
        col += 1
        
        # M5 = a00(b01-b10)
        U[0][col] += 1  # a00
        V[c1][col] += 1   # b01
        V[13 + c0][col] -= 1  # b10
        # C00 = M1 + M4 - M5 + M7 => C[0][c0] gets -1 from M5
        W[c0 * 2 + 0][col] -= 1  # C[0][c0]
        # C01 = M3 + M5 => C[0][c1] gets +1 from M5
        W[c1 * 2 + 0][col] += 1  # C[0][c1]
        col += 1
        
        # M6 = (a01+a10)(b00+b10)
        U[1][col] += 1  # a01
        U[2][col] += 1  # a10
        V[c0][col] += 1   # b00
        V[13 + c0][col] += 1  # b10
        # C11 = M1 - M2 + M3 + M6 => C[1][c1] gets +1 from M6
        W[c1 * 2 + 1][col] += 1  # C[1][c1]
        col += 1
        
        # M7 = (a01-a11)(b01+b11)
        U[1][col] += 1  # a01
        U[3][col] -= 1  # a11
        V[c1][col] += 1   # b01
        V[13 + c1][col] += 1  # b11
        # C00 = M1 + M4 - M5 + M7 => C[0][c0] gets +1 from M7
        W[c0 * 2 + 0][col] += 1  # C[0][c0]
        col += 1
    
    # For column 12 (single column, rank 3)
    # C[0][12] = A00*B[0][12] + A01*B[1][12]
    # C[1][12] = A10*B[0][12] + A11*B[1][12]
    # Using:
    # M1 = (A00+A11)*(B0+B1)
    # M2 = (A10+A11)*B0
    # M3 = A00*(B1-B0)
    # C0 = M1 - M2 + M3
    # C1 = M2 - M3
    
    c = 12
    # M1 = (a00+a11)*(B[0][c]+B[1][c])
    U[0][col] += 1  # a00
    U[3][col] += 1  # a11
    V[c][col] += 1   # B[0][c]
    V[13 + c][col] += 1  # B[1][c]
    W[c * 2 + 0][col] += 1  # C[0][c] gets +1 from M1
    W[c * 2 + 1][col] -= 1  # C[1][c] gets -1 from M1 (since C1 = M2 - M3, and M1 doesn't appear... wait)
    
    # Let me re-derive. For 2x2 * 2x1:
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # 
    # Strassen-like:
    # M1 = (A00+A11)*(B0+B1)
    # M2 = (A10+A11)*B0
    # M3 = A00*(B1-B0)
    # 
    # C0 = M1 - M2 + M3 = (A00+A11)(B0+B1) - (A10+A11)B0 + A00(B1-B0)
    #    = A00*B0 + A00*B1 + A11*B0 + A11*B1 - A10*B0 - A11*B0 + A00*B1 - A00*B0
    #    = A00*B1 + A11*B1 - A10*B0 + A00*B1
    #    = A00*B1 + A11*B1 - A10*B0 + A00*B1
    #    That doesn't seem right. Let me recompute.
    # 
    # M1 = A00*B0 + A00*B1 + A11*B0 + A11*B1
    # M2 = A10*B0 + A11*B0
    # M3 = A00*B1 - A00*B0
    # 
    # M1 - M2 + M3 = A00*B0 + A00*B1 + A11*B0 + A11*B1 - A10*B0 - A11*B0 + A00*B1 - A00*B0
    #               = A00*B1 + A11*B1 - A10*B0 + A00*B1
    #               = 2*A00*B1 + A11*B1 - A10*B0
    # That's not C0. I made an error.
    # 
    # Correct Strassen for 2x2 * 2x1:
    # Actually, the standard 2x2 * 2x2 Strassen doesn't directly apply to 2x2 * 2x1.
    # The tensor [2x2 * 2x1] has rank 3. Let's use:
    # M1 = (A00+A11)*(B0+B1)
    # M2 = (A10+A11)*B0
    # M3 = A00*(B1-B0)
    # C0 = M1 + M4 - M5 + M7 style... 
    # 
    # Let me just use the known rank-3 decomposition:
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # 
    # M1 = (A00+A11)*(B0+B1)
    # M2 = (A10+A11)*B0
    # M3 = A00*(B1-B0)
    # 
    # C0 = M1 - M2 + M3? Let's check:
    # = (A00+A11)(B0+B1) - (A10+A11)B0 + A00(B1-B0)
    # = A00*B0 + A00*B1 + A11*B0 + A11*B1 - A10*B0 - A11*B0 + A00*B1 - A00*B0
    # = A00*B1 + A11*B1 - A10*B0 + A00*B1
    # = 2*A00*B1 + A11*B1 - A10*B0
    # Not correct.
    # 
    # Let me try:
    # C0 = M1 + A00*B1 - A00*B0 - M2 + A11*B0... this is getting messy.
    # 
    # The correct rank-3 decomposition for 2x2 * 2x1 -> 2x1:
    # M1 = (A00+A11)*(B0+B1)
    # M2 = (A10+A11)*B0
    # M3 = (A01+A00)*B1  -- no, let me think differently.
    # 
    # Actually, for the matrix multiplication tensor [2,2,1] * [2,1] = [2,1],
    # the rank is 3. A valid decomposition:
    # M1 = (A00+A11)*(B0+B1)
    # M2 = (A10+A11)*B0
    # M3 = A00*(B1-B0)
    # C0 = M1 + M4 - M5 + M7 -- no, there's no M4,M5,M7.
    # 
    # Let me just use schoolbook for the last column: rank 4.
    # Total = 6*7 + 4 = 46.
    
    # Reset: use schoolbook for column 12 (4 multiplications)
    # C[0][12] = A00*B[0][12] + A01*B[1][12]
    # C[1][12] = A10*B[0][12] + A11*B[1][12]
    
    # A00*B[0][12]
    U[0][col] += 1
    V[c][col] += 1
    W[c * 2 + 0][col] += 1  # C[0][12]
    col += 1
    
    # A01*B[1][12]
    U[1][col] += 1
    V[13 + c][col] += 1
    W[c * 2 + 0][col] += 1  # C[0][12]
    col += 1
    
    # A10*B[0][12]
    U[2][col] += 1
    V[c][col] += 1
    W[c * 2 + 1][col] += 1  # C[1][12]
    col += 1
    
    # A11*B[1][12]
    U[3][col] += 1
    V[13 + c][col] += 1
    W[c * 2 + 1][col] += 1  # C[1][12]
    col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
