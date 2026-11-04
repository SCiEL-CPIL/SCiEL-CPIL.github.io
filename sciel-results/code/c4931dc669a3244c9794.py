"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-98 decomposition of 4x4 by 4x8 matrix multiplication.
    
    Strategy: Embed the 4x4 and 4x8 matrices into 8x8 matrices,
    use the rank-70 Strassen-based decomposition for 8x8x8,
    then extract the top-left 4x4 block of the 8x8 result.
    
    Actually, let me use a simpler and more reliable approach:
    Use Kronecker products of the rank-7 Strassen decomposition.
    
    For 4x4x4: rank = 7*7 = 49
    For 4x4x8: we can view as two 4x4x4 blocks, so rank = 49*2 = 98
    
    This matches the measured improvement from the knowledge.
    """
    import time
    start_time = time.time()
    
    def deadline_guard():
        return (time.time() - start_time) > 140.0
    
    # Base case: Rank-7 Strassen decomposition for 2x2x2
    # U, V, W are 4x7 matrices
    # Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # where i,j,k in {0,1}
    
    # Strassen formulas:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B11)
    # M4 = A22(B12-B22)
    # M5 = (A11+A12)B22
    # M6 = (A21-A22)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # Let's define the rank-7 tensors.
    # For each of the 7 terms, we need U, V, W vectors.
    
    # Term 1: (A11+A22)(B11+B22)
    # U: A11 and A22 both contribute +1
    # V: B11 and B22 both contribute +1
    # W: C11 gets +1, C22 gets +1
    
    # Let me be more systematic. The tensor for 2x2x2 matrix multiplication
    # has entries T[i,j,k] = 1 if i,j,k correspond to a valid product term.
    
    # Actually, let me just hardcode the known rank-7 Strassen decomposition.
    
    # U matrix (4x7): rows indexed by (i,j) in {0,1}x{0,1} -> 0,1,2,3
    #   row 0: (0,0) = A11
    #   row 1: (0,1) = A12
    #   row 2: (1,0) = A21
    #   row 3: (1,1) = A22
    
    # V matrix (4x7): rows indexed by (j,k) in {0,1}x{0,1} -> 0,1,2,3
    #   row 0: (0,0) = B11
    #   row 1: (0,1) = B12
    #   row 2: (1,0) = B21
    #   row 3: (1,1) = B22
    
    # W matrix (4x7): rows indexed by (k,i) in {0,1}x{0,1} -> 0,1,2,3
    #   row 0: (0,0) = C11
    #   row 1: (1,0) = C21
    #   row 2: (0,1) = C12
    #   row 3: (1,1) = C22
    
    # Wait, the convention is W[k*n+i, r] where n=2, p=2
    # So W has p*n = 2*2 = 4 rows
    # Row k*n+i: k is column index of B (and output), i is row index of A (and output)
    # So:
    #   row 0: k=0,i=0 -> C11 (output row 0, col 0)
    #   row 1: k=0,i=1 -> C21 (output row 1, col 0)
    #   row 2: k=1,i=0 -> C12 (output row 0, col 1)
    #   row 3: k=1,i=1 -> C22 (output row 1, col 1)
    
    U_strassen = [
        [1, 0, 1, 0, 1, 0, 0],  # A11
        [0, 0, 0, 0, 1, 0, 1],  # A12
        [0, 1, 0, 0, 0, 1, 0],  # A21
        [1, 1, 0, 1, 0, -1, -1],  # A22
    ]
    
    V_strassen = [
        [1, 1, 1, 0, 0, 1, 0],  # B11
        [0, 0, 0, 1, 0, 1, 0],  # B12
        [0, 0, -1, 0, 0, 0, 1],  # B21
        [1, 0, 0, -1, 1, 0, 1],  # B22
    ]
    
    W_strassen = [
        [1, 0, 0, 1, -1, 0, -1],  # C11
        [0, 1, 0, 1, 0, 0, 0],    # C21
        [0, 0, 1, 0, 1, 0, 0],    # C12
        [1, -1, 1, 0, 0, 0, 1],   # C22
    ]
    
    # Verify: U^T * W should give the 2x2x2 tensor
    # Actually, the decomposition satisfies:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    # Let me verify this is correct by checking a few entries.
    # T[0,0,0] = 1 (A11 * B11 -> C11)
    # Sum_r U[0,r]*V[0,r]*W[0,r] = 1*1*1 + 0*1*0 + 1*1*0 + 0*0*1 + 1*0*(-1) + 0*1*0 + 0*0*(-1) = 1 + 0 + 0 + 0 + 0 + 0 + 0 = 1 ✓
    
    # T[0,0,1] = 0 (A11 * B12 should not contribute to C11)
    # Sum_r U[0,r]*V[1,r]*W[0,r] = 1*0*1 + 0*0*0 + 1*0*0 + 0*1*1 + 1*0*(-1) + 0*1*0 + 0*0*(-1) = 0 ✓
    
    # T[0,1,0] = 0 (A12 * B11 should not contribute to C11)
    # Sum_r U[1,r]*V[0,r]*W[0,r] = 0*1*1 + 0*1*0 + 0*1*0 + 0*0*1 + 1*0*(-1) + 0*1*0 + 1*0*(-1) = 0 ✓
    
    # T[0,1,1] = 1 (A12 * B22 -> C12, but wait, this should be 0 for C11)
    # Actually T[i,j,k] is 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # So T[0,1,0] = 0, T[0,1,1] = 1 (A12*B22 -> C12, which is C[0,1])
    
    # Let me re-check the W indexing.
    # W[k*n+i, r]: k is the column of B/output, i is the row of A/output
    # So W[0, r] corresponds to output position (i=0, k=0) = C[0,0]
    # W[1, r] corresponds to output position (i=1, k=0) = C[1,0]
    # W[2, r] corresponds to output position (i=0, k=1) = C[0,1]
    # W[3, r] corresponds to output position (i=1, k=1) = C[1,1]
    
    # T[0,1,1]: A[0,1] * B[1,1] -> C[0,1]
    # Sum_r U[0*2+1, r] * V[1*2+1, r] * W[1*2+0, r]
    # = Sum_r U[1, r] * V[3, r] * W[2, r]
    # = 0*1*0 + 0*1*0 + 0*0*0 + 0*0*0 + 1*1*1 + 0*0*0 + 1*1*0
    # = 0 + 0 + 0 + 0 + 1 + 0 + 0 = 1 ✓
    
    # Great, the Strassen decomposition looks correct.
    
    # Now use Kronecker product to build 4x4x4 with rank 49.
    # For Kronecker product of two rank-r decompositions:
    # U_kron = U1 ⊗ U2 (in some appropriate ordering)
    # The row ordering for U is (i1*m1+j1)*m2+j2 or similar
    
    # For 4x4x4: n=4, m=4, p=4
    # U has 4*4=16 rows, V has 4*4=16 rows, W has 4*4=16 rows
    # Rank = 7*7 = 49
    
    # The Kronecker product approach:
    # If we have a decomposition of size (n1,m1,p1) with rank r1,
    # and a decomposition of size (n2,m2,p2) with rank r2,
    # then we get a decomposition of size (n1*n2, m1*m2, p1*p2) with rank r1*r2.
    
    # Row indexing for U: (i1*m1+j1) * m2 + j2, where i1 in [0,n1), j1 in [0,m1), j2 in [0,m2)
    # Wait, that's not quite right. Let me think more carefully.
    
    # Actually, for the Kronecker product:
    # A is n1*n2 by m1*m2, B is m1*m2 by p1*p2, C is n1*n2 by p1*p2
    # A[i1*m1*m2 + i2*m2 + j1*m2 + j2... ]
    
    # Let me use a simpler approach. For 4x4x4:
    # View A as a 2x2 block matrix where each block is 2x2
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    # Similarly for B and C.
    
    # Then C = A*B is computed via Strassen at the block level:
    # C11 = (A11+A22)*(B11+B22) + (A21+A22)*B11 - A11*(B21-B11) - (A11+A12)*B22 - (A12-A22)*(B21+B22)
    # etc.
    
    # Each block product is a 2x2 by 2x2 multiplication, which has rank 7.
    # So each of the 7 Strassen terms at the block level requires 7 scalar multiplications.
    # Total rank = 7*7 = 49.
    
    # Let me construct this properly.
    
    # For the 4x4x4 decomposition:
    # U has 16 rows (indexed by i*4+j, i,j in 0..3), 49 columns
    # V has 16 rows (indexed by j*4+k, j,k in 0..3), 49 columns  
    # W has 16 rows (indexed by k*4+i, k,i in 0..3), 49 columns
    
    # Column index: (c1, c2) where c1 in 0..6 (Strassen term at block level), c2 in 0..6 (Strassen term within block)
    # So column = c1*7 + c2
    
    # Row index for U: (i1, j1, j2) where i1 in 0..1, j1 in 0..1, j2 in 0..1
    # i = i1*2 + i2, j = j1*2 + j2
    # Row = i*4 + j = (i1*2+i2)*4 + (j1*2+j2) = i1*8 + i2*4 + j1*2 + j2
    
    # U[i1*8 + i2*4 + j1*2 + j2, c1*7 + c2] = U_strassen[i1*2+j1, c1] * U_strassen[i2*2+j2, c2]
    
    # Similarly for V and W.
    
    n1, m1, p1 = 2, 2, 2
    n2, m2, p2 = 2, 2, 2
    r1, r2 = 7, 7
    
    n = n1 * n2  # 4
    m = m1 * m2  # 4
    p = p1 * p2  # 4
    rank_444 = r1 * r2  # 49
    
    U_444 = [[0] * rank_444 for _ in range(n * m)]
    V_444 = [[0] * rank_444 for _ in range(m * p)]
    W_444 = [[0] * rank_444 for _ in range(p * n)]
    
    for i1 in range(n1):
        for j1 in range(m1):
            for c1 in range(r1):
                for i2 in range(n2):
                    for j2 in range(m2):
                        for c2 in range(r2):
                            u1 = U_strassen[i1 * m1 + j1][c1]
                            u2 = U_strassen[i2 * m2 + j2][c2]
                            if u1 != 0 and u2 != 0:
                                row_u = (i1 * n2 + i2) * m + (j1 * m2 + j2)
                                col = c1 * r2 + c2
                                U_444[row_u][col] += u1 * u2
    
    for j1 in range(m1):
        for k1 in range(p1):
            for c1 in range(r1):
                for j2 in range(m2):
                    for k2 in range(p2):
                        for c2 in range(r2):
                            v1 = V_strassen[j1 * p1 + k1][c1]
                            v2 = V_strassen[j2 * p2 + k2][c2]
                            if v1 != 0 and v2 != 0:
                                row_v = (j1 * p2 + j2) * p + (k1 * p2 + k2)
                                col = c1 * r2 + c2
                                V_444[row_v][col] += v1 * v2
    
    for k1 in range(p1):
        for i1 in range(n1):
            for c1 in range(r1):
                for k2 in range(p2):
                    for i2 in range(n2):
                        for c2 in range(r2):
                            w1 = W_strassen[k1 * n1 + i1][c1]
                            w2 = W_strassen[k2 * n2 + i2][c2]
                            if w1 != 0 and w2 != 0:
                                row_w = (k1 * n2 + k2) * n + (i1 * n2 + i2)
                                col = c1 * r2 + c2
                                W_444[row_w][col] += w1 * w2
    
    # Now for 4x4x8:
    # View this as multiplying a 4x4 matrix A by a 4x8 matrix B.
    # We can split B into two 4x4 blocks: B = [B_left | B_right]
    # Then C = A * B = [A*B_left | A*B_right]
    # Each A*B_left and A*B_right is a 4x4x4 multiplication with rank 49.
    # Total rank = 49 + 49 = 98.
    
    # For the 4x4x8 decomposition:
    # n=4, m=4, p=8
    # U has 4*4=16 rows, 98 columns
    # V has 4*8=32 rows, 98 columns
    # W has 8*4=32 rows, 98 columns
    
    # Columns 0-47: left block (B_left), columns 48-97: right block (B_right)
    
    n = 4
    m = 4
    p = 8
    rank = 98
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Left block: B columns 0-3
    for r in range(16):
        for c in range(49):
            if U_444[r][c] != 0:
                U[r][c] = U_444[r][c]
    
    for r in range(16):
        for c in range(49):
            if V_444[r][c] != 0:
                V[r][c] = V_444[r][c]
    
    for r in range(16):
        for c in range(49):
            if W_444[r][c] != 0:
                W[r][c] = W_444[r][c]
    
    # Right block: B columns 4-7
    # B_right has rows j=0..3, cols k=4..7
    # In V, row j*8+k where k=4..7
    # In W, row k*4+i where k=4..7
    
    for r in range(16):
        for c in range(49):
            if U_444[r][c] != 0:
                U[r][49 + c] = U_444[r][c]
    
    for j in range(4):
        for k in range(4):
            for c in range(49):
                if V_444[j * 4 + k][c] != 0:
                    V[j * 8 + (k + 4)][49 + c] = V_444[j * 4 + k][c]
    
    for k in range(4):
        for i in range(4):
            for c in range(49):
                if W_444[k * 4 + i][c] != 0:
                    W[(k + 4) * 4 + i][49 + c] = W_444[k * 4 + i][c]
    
    if deadline_guard():
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
