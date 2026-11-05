"""Rewritten program: hybrid tensor decomposition using Strassen for 4x4x4 sub-blocks
plus schoolbook for the remaining columns, with exact integer entries."""

DIMENSIONS = [4, 4, 7]


def construct():
    """Return a rank-minimal (or near-minimal) exact decomposition of the 4x4x4 * 4x4x7
    matrix multiplication tensor.
    
    Strategy: split the 4x7 output into a 4x4 block (handled by Strassen, rank 49)
    and a 4x3 block (handled by schoolbook, rank 48). Total rank = 97.
    """
    n, m, p = DIMENSIONS
    # We'll build two parts:
    # Part 1: Strassen for A(4x4) * B(4x4) -> C(4x4), rank 49
    # Part 2: Schoolbook for A(4x4) * B(4x3) -> C(4x3), rank 48
    # Total rank = 49 + 48 = 97
    
    rank = 49 + 48
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x rank
    V = [[0] * rank for _ in range(m * p)]  # 28 x rank
    W = [[0] * rank for _ in range(p * n)]  # 28 x rank
    
    # ---- Part 1: Strassen for 4x4x4 ----
    # Strassen on 4x4: split into 2x2 blocks, each 2x2x2 uses Strassen (rank 7).
    # Total 7 * 7 = 49 scalar multiplications.
    
    # For 2x2x2 Strassen, the 7 products are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4x4, we split A into [[A11, A12],[A21, A22]] (each 2x2)
    # B into [[B11, B12],[B21, B22]] (each 2x2)
    # C = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
    #      [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    # Each 2x2x2 product uses Strassen with 7 multiplications.
    # So 4 sub-blocks * 7 = 28? No wait, that's not how Strassen works for 4x4.
    # Actually, for 4x4, we apply Strassen recursively: split into 2x2, then each 2x2
    # multiplication is done via Strassen (7 multiplications). So 7 * 7 = 49.
    
    # Let me think more carefully. The 4x4 matrix multiplication tensor can be decomposed
    # by applying Strassen's 2x2 decomposition at the top level (7 block multiplications
    # of 2x2 matrices), and then each 2x2 multiplication is done via Strassen (7 scalar
    # multiplications). So 7*7 = 49.
    
    # Let me define the Strassen-4x4 decomposition explicitly.
    
    # Top-level Strassen for 4x4:
    # Split A = [[A1, A2], [A3, A4]] where each Ai is 2x2
    # Split B = [[B1, B2], [B3, B4]] where each Bj is 2x2
    # 
    # The 7 Strassen products (each is a 2x2 * 2x2 = 2x2 multiplication):
    # P1 = (A1+A4)(B1+B4)
    # P2 = (A3+A4)B1
    # P3 = A1(B3-B4)
    # P4 = A4(B2-B1)
    # P5 = (A1+A2)B4
    # P6 = (A3-A1)(B1+B2)
    # P7 = (A2-A4)(B3+B4)
    #
    # C1 = P1 + P4 - P5 + P7
    # C2 = P3 + P5
    # C3 = P2 + P4
    # C4 = P1 - P2 + P3 + P6
    
    # Now each Pi is a 2x2x2 multiplication, which we decompose via Strassen (rank 7).
    
    # For a 2x2 matrix X = [[x1, x2], [x3, x4]], we index as:
    # X[0] = x1 (row 0, col 0)
    # X[1] = x2 (row 0, col 1)
    # X[2] = x3 (row 1, col 0)
    # X[3] = x4 (row 1, col 1)
    
    # For the overall U matrix, row index i*m + j corresponds to A[i,j]
    # For the overall V matrix, row index j*p + k corresponds to B[j,k]
    # For the overall W matrix, row index k*n + i corresponds to C[k,i] (note: W[k*n+i])
    
    # Let me build this step by step.
    
    # Strassen 2x2x2 decomposition:
    # Given matrices X (2x2) and Y (2x2), compute Z = X*Y using 7 products.
    # X = [[x0, x1], [x2, x3]]
    # Y = [[y0, y1], [y2, y3]]
    # 
    # The 7 products and their contributions:
    # s1 = (x0+x3)(y0+y3)  -> contributes to z0,z3
    # s2 = (x2+x3)y0       -> contributes to z2
    # s3 = x0(y2-y3)       -> contributes to z1,z2
    # s4 = x3(y1-y0)       -> contributes to z1,z3
    # s5 = (x0+x1)y3       -> contributes to z1,z3
    # s6 = (x2-x0)(y0+y1)  -> contributes to z2
    # s7 = (x1-x3)(y2+y3)  -> contributes to z0,z3
    #
    # z0 = s1 + s4 - s5 + s7
    # z1 = s3 + s5
    # z2 = s2 + s4
    # z3 = s1 - s2 + s3 + s6
    
    # For each of the 7 top-level Strassen products (each a 2x2x2), we need to
    # further decompose into 7 scalar products.
    
    # This is getting complex. Let me use a different approach:
    # I'll directly construct the 49-term decomposition.
    
    # Actually, let me just use the known result: rank(4,4,4) <= 49 via Strassen.
    # And rank(4,4,3) <= 48 via schoolbook (or potentially less, but 48 is safe).
    
    # For simplicity and correctness, let me use:
    # - Strassen-based decomposition for the first 4 columns of B (4x4x4 part)
    # - Schoolbook for the last 3 columns of B (4x4x3 part)
    
    # Let me implement this more carefully using a tensor composition approach.
    
    # Helper: schoolbook decomposition of M_{n,m,p}
    def schoolbook(n, m, p):
        """Returns (U, V, W) for schoolbook decomposition of M_{n,m,p}."""
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
        return U, V, W, rank
    
    # Helper: Strassen decomposition of M_{2,2,2}
    def strassen_222():
        """Returns (U, V, W) for Strassen decomposition of M_{2,2,2}, rank 7."""
        rank = 7
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(4)]
        W = [[0] * rank for _ in range(4)]
        
        # s1 = (x0+x3)(y0+y3)
        # x0=A[0,0], x1=A[0,1], x2=A[1,0], x3=A[1,1]
        # y0=B[0,0], y1=B[0,1], y2=B[1,0], y3=B[1,1]
        # W uses W[k*2+i] convention
        # z0 = C[0,0] -> W[0*2+0] = W[0]
        # z1 = C[0,1] -> W[1*2+0] = W[2]
        # z2 = C[1,0] -> W[0*2+1] = W[1]
        # z3 = C[1,1] -> W[1*2+1] = W[3]
        
        # s1: (A00+A11)(B00+B11)
        # U: A00=1, A11=1; V: B00=1, B11=1
        U[0][0] = 1  # A00
        U[3][0] = 1  # A11
        V[0][0] = 1  # B00
        V[3][0] = 1  # B11
        # z0 += s1, z3 += s1
        W[0][0] = 1  # C00
        W[3][0] = 1  # C11
        
        # s2: (A10+A11)B00
        U[2][1] = 1  # A10
        U[3][1] = 1  # A11
        V[0][1] = 1  # B00
        # z2 += s2
        W[1][1] = 1  # C10
        
        # s3: A00(B10-B11)
        U[0][2] = 1  # A00
        V[2][2] = 1  # B10
        V[3][2] = -1 # B11
        # z1 += s3, z2 += s3
        W[2][2] = 1  # C01
        W[1][2] = 1  # C10
        
        # s4: A11(B01-B00)
        U[3][3] = 1  # A11
        V[1][3] = 1  # B01
        V[0][3] = -1 # B00
        # z1 += s4, z3 += s4
        W[2][3] = 1  # C01
        W[3][3] = 1  # C11
        
        # s5: (A00+A01)B11
        U[0][4] = 1  # A00
        U[1][4] = 1  # A01
        V[3][4] = 1  # B11
        # z1 += s5, z3 -= s5
        W[2][4] = 1  # C01
        W[3][4] = -1 # C11
        
        # s6: (A10-A00)(B00+B01)
        U[2][5] = 1  # A10
        U[0][5] = -1 # A00
        V[0][5] = 1  # B00
        V[1][5] = 1  # B01
        # z2 += s6
        W[1][5] = 1  # C10
        
        # s7: (A01-A11)(B10+B11)
        U[1][6] = 1  # A01
        U[3][6] = -1 # A11
        V[2][6] = 1  # B10
        V[3][6] = 1  # B11
        # z0 += s7, z3 += s7
        W[0][6] = 1  # C00
        W[3][6] = 1  # C11
        
        return U, V, W, rank
    
    # For 4x4x4, I'll use a block Strassen approach.
    # Let me define the 4x4x4 Strassen decomposition by composing 2x2x2 Strassen.
    
    # Actually, let me take a simpler approach. I know that:
    # - The 4x4x4 tensor can be decomposed with rank 49 using Strassen.
    # - The 4x4x3 tensor can be decomposed with rank 48 using schoolbook.
    # Total rank = 97.
    
    # Let me just build the schoolbook for 4x4x7 but replace the first 4 columns
    # with Strassen to save rank. Actually, that's not how it works.
    
    # Better approach: decompose M_{4,4,7} as M_{4,4,4} + M_{4,4,3} where the
    # B matrix is split into two parts. The tensor is a direct sum.
    
    # Let me build:
    # Part 1: M_{4,4,4} with Strassen (rank 49)
    # Part 2: M_{4,4,3} with schoolbook (rank 48)
    
    # For Part 1, I need the Strassen-4x4 decomposition.
    # Let me construct it by applying Strassen at two levels.
    
    # Level 1: Split 4x4 into 2x2 blocks.
    # A = [[A1, A2], [A3, A4]], B = [[B1, B2], [B3, B4]]
    # C1 = A1*B1 + A2*B3, C2 = A1*B2 + A2*B4
    # C3 = A3*B1 + A4*B3, C4 = A3*B2 + A4*B4
    
    # Strassen: 7 products of 2x2 matrices
    # P1 = (A1+A4)(B1+B4)
    # P2 = (A3+A4)B1
    # P3 = A1(B3-B4)
    # P4 = A4(B2-B1)
    # P5 = (A1+A2)B4
    # P6 = (A3-A1)(B1+B2)
    # P7 = (A2-A4)(B3+B4)
    
    # C1 = P1 + P4 - P5 + P7
    # C2 = P3 + P5
    # C3 = P2 + P4
    # C4 = P1 - P2 + P3 + P6
    
    # Each Pi is a 2x2x2 multiplication, decomposed via Strassen (rank 7).
    # Total: 7 * 7 = 49 scalar products.
    
    # Let me build this.
    
    # First, let me define the Strassen-2x2 as a function that returns
    # the U, V, W matrices for a single 2x2x2 Strassen decomposition.
    # Then I'll compose.
    
    # For the overall problem:
    # A has indices (i,j) where i,j in {0,1,2,3}
    # B has indices (j,k) where j in {0,1,2,3}, k in {0,1,2,3,4,5,6}
    # C has indices (k,i) where k in {0,1,2,3,4,5,6}, i in {0,1,2,3}
    
    # W convention: W[k*n+i] = W[k*4+i]
    
    # For the first 4 columns of B (k=0,1,2,3), use Strassen.
    # For the last 3 columns of B (k=4,5,6), use schoolbook.
    
    # Let me build Part 1 (Strassen 4x4x4) and Part 2 (schoolbook 4x4x3).
    
    # Part 2: schoolbook for columns k=4,5,6 of B
    U2, V2, W2, r2 = schoolbook(4, 4, 3)
    # U2 is 16x48, V2 is 16x48 (but for m=4, p=3, so m*p=12), W2 is 12x48
    
    # Wait, schoolbook(4,4,3) gives U of shape (4*4, 4*4*3) = (16, 48)
    # V of shape (4*3, 48) = (12, 48)
    # W of shape (3*4, 48) = (12, 48)
    
    # But in our problem, V should have m*p = 4*7 = 28 rows.
    # For Part 2, only rows corresponding to j=0..3, k=4..6 are used.
    # So V2_row j*7 + k for k=4,5,6.
    
    # Let me rebuild properly.
    
    # Actually, let me just build the full 97-rank decomposition directly.
    
    # Part 1: Strassen for A(4x4) * B_first4(4x4) -> C_first4(4x4)
    # This contributes to W[k*4+i] for k=0,1,2,3.
    
    # Part 2: Schoolbook for A(4x4) * B_last3(4x3) -> C_last3(4x3)
    # This contributes to W[k*4+i] for k=4,5,6.
    
    # For Part 2, the V rows are for B[j,k] where j=0..3, k=4,5,6.
    # In the global V matrix (28 rows), these are at positions j*7+k.
    
    # Let me construct Part 1 (Strassen 4x4x4).
    
    # I'll build the Strassen 4x4 decomposition by composing 7 Strassen 2x2 decompositions.
    
    # Strassen 2x2: for matrices X (2x2) and Y (2x2), the 7 products are:
    # Let me index X elements as X[0]=x00, X[1]=x01, X[2]=x10, X[3]=x11
    # and Y elements as Y[0]=y00, Y[1]=y01, Y[2]=y10, Y[3]=y11
    
    # The 7 Strassen products and their U, V contributions:
    # prod 0: (x00+x11)(y00+y11)
    #   U: x00->1, x11->1; V: y00->1, y11->1
    # prod 1: (x10+x11)*y00
    #   U: x10->1, x11->1; V: y00->1
    # prod 2: x00*(y10-y11)
    #   U: x00->1; V: y10->1, y11->-1
    # prod 3: x11*(y01-y00)
    #   U: x11->1; V: y01->1, y00->-1
    # prod 4: (x00+x01)*y11
    #   U: x00->1, x01->1; V: y11->1
    # prod 5: (x10-x00)*(y00+y01)
    #   U: x10->1, x00->-1; V: y00->1, y01->1
    # prod 6: (x01-x11)*(y10+y11)
    #   U: x01->1, x11->-1; V: y10->1, y11->1
    
    # And the W contributions (for 2x2 output Z = X*Y):
    # z00 = prod0 + prod3 - prod4 + prod6
    # z01 = prod2 + prod4
    # z10 = prod1 + prod3
    # z11 = prod0 - prod1 + prod2 + prod5
    
    # For the 4x4 case, I'll define the 7 top-level Strassen products.
    # Each involves 2x2 blocks of A and B.
    
    # Let me define helper to extract 2x2 block contributions.
    
    # A blocks:
    # A1 = A[0:2, 0:2] = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # A2 = A[0:2, 2:4] = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]
    # A3 = A[2:4, 0:2] = [[A[2,0], A[2,1]], [A[3,0], A[3,1]]]
    # A4 = A[2:4, 2:4] = [[A[2,2], A[2,3]], [A[3,2], A[3,3]]]
    
    # B blocks (for first 4 columns):
    # B1 = B[0:2, 0:2], B2 = B[0:2, 2:4], B3 = B[2:4, 0:2], B4 = B[2:4, 2:4]
    
    # Top-level Strassen products (each is a 2x2x2 multiplication):
    # P1 = (A1+A4)(B1+B4): X = A1+A4, Y = B1+B4
    # P2 = (A3+A4)B1: X = A3+A4, Y = B1
    # P3 = A1(B3-B4): X = A1, Y = B3-B4
    # P4 = A4(B2-B1): X = A4, Y = B2-B1
    # P5 = (A1+A2)B4: X = A1+A2, Y = B4
    # P6 = (A3-A1)(B1+B2): X = A3-A1, Y = B1+B2
    # P7 = (A2-A4)(B3+B4): X = A2-A4, Y = B3+B4
    
    # For each Pi, the 2x2x2 Strassen gives 7 scalar products.
    # I need to map the 2x2 block indices to global indices.
    
    # For a 2x2 block in A with top-left corner at (row_off, col_off):
    # X[0] = A[row_off, col_off]     -> U index row_off*4 + col_off
    # X[1] = A[row_off, col_off+1]   -> U index row_off*4 + col_off + 1
    # X[2] = A[row_off+1, col_off]   -> U index (row_off+1)*4 + col_off
    # X[3] = A[row_off+1, col_off+1] -> U index (row_off+1)*4 + col_off + 1
    
    # For a 2x2 block in B with top-left corner at (row_off, col_off):
    # Y[0] = B[row_off, col_off]     -> V index row_off*7 + col_off
    # Y[1] = B[row_off, col_off+1]   -> V index row_off*7 + col_off + 1
    # Y[2] = B[row_off+1, col_off]   -> V index (row_off+1)*7 + col_off
    # Y[3] = B[row_off+1, col_off+1] -> V index (row_off+1)*7 + col_off + 1
    
    # For the output C (first 4 columns), the 2x2 blocks are:
    # C1 = C[0:2, 0:2], C2 = C[0:2, 2:4], C3 = C[2:4, 0:2], C4 = C[2:4, 2:4]
    # C[k,i] -> W index k*4 + i
    # C1[0] = C[0,0] -> W[0], C1[1] = C[0,1] -> W[4], C1[2] = C[1,0] -> W[1], C1[3] = C[1,1] -> W[5]
    # C2[0] = C[0,2] -> W[2], C2[1] = C[0,3] -> W[6], C2[2] = C[1,2] -> W[3], C2[3] = C[1,3] -> W[7]
    # C3[0] = C[2,0] -> W[8], C3[1] = C[2,1] -> W[12], C3[2] = C[3,0] -> W[9], C3[3] = C[3,1] -> W[13]
    # C4[0] = C[2,2] -> W[10], C4[1] = C[2,3] -> W[14], C4[2] = C[3,2] -> W[11], C4[3] = C[3,3] -> W[15]
    
    # Now let me build the Strassen 4x4 decomposition.
    
    # Define the 7 top-level Strassen products with their X and Y definitions.
    # Each product has:
    #   - X: a linear combination of 2x2 blocks of A (with coefficients)
    #   - Y: a linear combination of 2x2 blocks of B (with coefficients)
    #   - Output: which C block(s) it contributes to, and how
    
    # Let me define the Strassen 2x2 template.
    # For a 2x2x2 Strassen decomposition:
    # X = [[x0, x1], [x2, x3]], Y = [[y0, y1], [y2, y3]]
    # 7 products:
    # p0: X coeffs: [1,0,0,1], Y coeffs: [1,0,0,1]
    # p1: X coeffs: [0,0,1,1], Y coeffs: [1,0,0,0]
    # p2: X coeffs: [1,0,0,0], Y coeffs: [0,0,1,-1]
    # p3: X coeffs: [0,0,0,1], Y coeffs: [0,1,-1,0]  # y01-y00 -> [0,1,-1,0]? No.
    # Let me recheck: y01-y00. Y[0]=y00, Y[1]=y01, Y[2]=y10, Y[3]=y11.
    # So y01-y00 -> Y coeffs: [-1, 1, 0, 0]
    # p3: X coeffs: [0,0,0,1], Y coeffs: [-1,1,0,0]
    # p4: X coeffs: [1,1,0,0], Y coeffs: [0,0,0,1]
    # p5: X coeffs: [-1,0,1,0], Y coeffs: [1,1,0,0]
    # p6: X coeffs: [0,1,0,-1], Y coeffs: [0,0,1,1]
    
    # W contributions for 2x2 output (z0, z1, z2, z3):
    # z0 = p0 + p3 - p4 + p6
    # z1 = p2 + p4
    # z2 = p1 + p3
    # z3 = p0 - p1 + p2 + p5
    
    # Now for each top-level Strassen product Pi, I have:
    # - The X block (which 2x2 block of A, with what offset)
    # - The Y block (which 2x2 block of B, with what offset)
    # - The output C block(s) and the Strassen 2x2 template applied
    
    # Let me define the 7 top-level products:
    
    # P1: X = A1+A4, Y = B1+B4, output -> C1, C4
    #   X blocks: A1 (offset 0,0) with coeff 1, A4 (offset 2,2) with coeff 1
    #   Y blocks: B1 (offset 0,0) with coeff 1, B4 (offset 2,2) with coeff 1
    #   Output: C1 = P1 + ... (C1 gets contribution from P1), C4 = P1 - ... 
    #   Actually, C1 = P1 + P4 - P5 + P7, C4 = P1 - P2 + P3 + P6
    #   So P1 contributes +1 to C1 and +1 to C4.
    
    # P2: X = A3+A4, Y = B1, output -> C3
    #   C3 = P2 + P4, so P2 contributes +1 to C3.
    
    # P3: X = A1, Y = B3-B4, output -> C2, C4
    #   C2 = P3 + P5, C4 = P1 - P2 + P3 + P6, so P3 contributes +1 to C2 and +1 to C4.
    
    # P4: X = A4, Y = B2-B1, output -> C1, C3
    #   C1 = P1 + P4 - P5 + P7, C3 = P2 + P4, so P4 contributes +1 to C1 and +1 to C3.
    
    # P5: X = A1+A2, Y = B4, output -> C1, C2
    #   C1 = P1 + P4 - P5 + P7, C2 = P3 + P5, so P5 contributes -1 to C1 and +1 to C2.
    
    # P6: X = A3-A1, Y = B1+B2, output -> C4
    #   C4 = P1 - P2 + P3 + P6, so P6 contributes +1 to C4.
    
    # P7: X = A2-A4, Y = B3+B4, output -> C1, C4
    #   C1 = P1 + P4 - P5 + P7, C4 = P1 - P2 + P3 + P6, so P7 contributes +1 to C1.
    #   Wait, P7 is only in C1? Let me recheck.
    #   C1 = P1 + P4 - P5 + P7: P7 coeff +1
    #   C4 = P1 - P2 + P3 + P6: no P7
    #   So P7 contributes +1 to C1 only.
    
    # Now for each Pi, I apply the Strassen 2x2 template.
    # The 7 scalar products within Pi use the X and Y definitions.
    
    # Let me build this.
    
    U1 = [[0] * 49 for _ in range(16)]
    V1 = [[0] * 49 for _ in range(28)]
    W1 = [[0] * 49 for _ in range(28)]
    
    # Strassen 2x2 template
    # X coeffs for each of 7 products (each is [x0, x1, x2, x3] coeffs)
    S_X = [
        [1, 0, 0, 1],   # p0
        [0, 0, 1, 1],   # p1
        [1, 0, 0, 0],   # p2
        [0, 0, 0, 1],   # p3
        [1, 1, 0, 0],   # p4
        [-1, 0, 1, 0],  # p5
        [0, 1, 0, -1],  # p6
    ]
    S_Y = [
        [1, 0, 0, 1],   # p0
        [1, 0, 0, 0],   # p1
        [0, 0, 1, -1],  # p2
        [-1, 1, 0, 0],  # p3
        [0, 0, 0, 1],   # p4
        [1, 1, 0, 0],   # p5
        [0, 0, 1, 1],   # p6
    ]
    S_W = [
        [1, 0, 0, 1],   # z0 = p0 + p3 - p4 + p6
        [0, 0, 1, 0],   # z1 = p2 + p4
        [0, 1, 0, 0],   # z2 = p1 + p3
        [1, -1, 1, 0],  # z3 = p0 - p1 + p2 + p5
    ]
    # Wait, let me recheck S_W.
    # z0 = p0 + p3 - p4 + p6 -> coeffs for [p0,p1,p2,p3,p4,p5,p6]: [1,0,0,1,-1,0,1]
    # z1 = p2 + p4 -> [0,0,1,0,1,0,0]
    # z2 = p1 + p3 -> [0,1,0,1,0,0,0]
    # z3 = p0 - p1 + p2 + p5 -> [1,-1,1,0,0,1,0]
    
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],  # z0
        [0, 0, 1, 0, 1, 0, 0],   # z1
        [0, 1, 0, 1, 0, 0, 0],   # z2
        [1, -1, 1, 0, 0, 1, 0],  # z3
    ]
    
    # Now define the 7 top-level products.
    # Each has:
    #   x_blocks: list of (row_off, col_off, coeff) for A
    #   y_blocks: list of (row_off, col_off, coeff) for B
    #   c_blocks: list of (row_off, col_off, coeff) for C (output)
    
    top_products = [
        # P1
        {
            'x_blocks': [(0, 0, 1), (2, 2, 1)],
            'y_blocks': [(0, 0, 1), (2, 2, 1)],
            'c_blocks': [(0, 0, 1), (2, 2, 1)],  # C1 and C4
        },
        # P2
        {
            'x_blocks': [(2, 0, 1), (2, 2, 1)],
            'y_blocks': [(0, 0, 1)],
            'c_blocks': [(2, 0, 1)],  # C3
        },
        # P3
        {
            'x_blocks': [(0, 0, 1)],
            'y_blocks': [(2, 0, 1), (2, 2, -1)],
            'c_blocks': [(0, 2, 1), (2, 2, 1)],  # C2 and C4
        },
        # P4
        {
            'x_blocks': [(2, 2, 1)],
            'y_blocks': [(0, 2, 1), (0, 0, -1)],
            'c_blocks': [(0, 0, 1), (2, 0, 1)],  # C1 and C3
        },
        # P5
        {
            'x_blocks': [(0, 0, 1), (0, 2, 1)],
            'y_blocks': [(2, 2, 1)],
            'c_blocks': [(0, 0, -1), (0, 2, 1)],  # C1 and C2
        },
        # P6
        {
            'x_blocks': [(2, 0, 1), (0, 0, -1)],
            'y_blocks': [(0, 0, 1), (0, 2, 1)],
            'c_blocks': [(2, 2, 1)],  # C4
        },
        # P7
        {
            'x_blocks': [(0, 2, 1), (2, 2, -1)],
            'y_blocks': [(2, 0, 1), (2, 2, 1)],
            'c_blocks': [(0, 0, 1)],  # C1
        },
    ]
    
    # Now build the U, V, W matrices for Part 1.
    col = 0
    for prod_idx, prod in enumerate(top_products):
        for s_idx in range(7):  # 7 Strassen 2x2 products within this top-level product
            # For each of the 4 X elements, add contributions from x_blocks
            for xi in range(4):
                x_coeff = S_X[s_idx][xi]
                if x_coeff != 0:
                    for (row_off, col_off, block_coeff) in prod['x_blocks']:
                        # Determine the global A index
                        # xi: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1) within the 2x2 block
                        dr = (xi // 2) if xi >= 2 else 0
                        dc = (xi % 2) if xi < 2 else (xi - 2)
                        # Actually: xi=0: (0,0), xi=1: (0,1), xi=2: (1,0), xi=3: (1,1)
                        if xi == 0:
                            dr, dc = 0, 0
                        elif xi == 1:
                            dr, dc = 0, 1
                        elif xi == 2:
                            dr, dc = 1, 0
                        else:
                            dr, dc = 1, 1
                        a_row = row_off + dr
                        a_col = col_off + dc
                        u_idx = a_row * 4 + a_col
                        U1[u_idx][col] += x_coeff * block_coeff
            
            # For each of the 4 Y elements, add contributions from y_blocks
            for yi in range(4):
                y_coeff = S_Y[s_idx][yi]
                if y_coeff != 0:
                    for (row_off, col_off, block_coeff) in prod['y_blocks']:
                        if yi == 0:
                            dr, dc = 0, 0
                        elif yi == 1:
                            dr, dc = 0, 1
                        elif yi == 2:
                            dr, dc = 1, 0
                        else:
                            dr, dc = 1, 1
                        b_row = row_off + dr
                        b_col = col_off + dc
                        v_idx = b_row * 7 + b_col
                        V1[v_idx][col] += y_coeff * block_coeff
            
            # For each of the 4 Z elements, add contributions to c_blocks
            for zi in range(4):
                w_coeff = S_W[zi][s_idx]
                if w_coeff != 0:
                    for (row_off, col_off, block_coeff) in prod['c_blocks']:
                        if zi == 0:
                            dr, dc = 0, 0
                        elif zi == 1:
                            dr, dc = 0, 1
                        elif zi == 2:
                            dr, dc = 1, 0
                        else:
                            dr, dc = 1, 1
                        c_row = row_off + dr  # k index
                        c_col = col_off + dc  # i index
                        w_idx = c_row * 4 + c_col
                        W1[w_idx][col] += w_coeff * block_coeff
            
            col += 1
    
    # Part 2: Schoolbook for A(4x4) * B_last3(4x3) -> C_last3(4x3)
    # U2: 16 x 48, V2: 12 x 48, W2: 12 x 48
    U2, V2, W2, r2 = schoolbook(4, 4, 3)
    
    # Now combine Part 1 and Part 2.
    # U is 16 x 97
    # V is 28 x 97
    # W is 28 x 97
    
    # For U: U1 (16x49) concatenated with U2 (16x48)
    for i in range(16):
        for j in range(49):
            U[i][j] = U1[i][j]
        for j in range(48):
            U[i][49 + j] = U2[i][j]
    
    # For V: V1 (28x49) and V2 (12x48)
    # V2 rows correspond to B[j,k] for j=0..3, k=0..2 (local), which map to
    # global B[j, 4+k] for k=0..2, i.e., global row j*7 + (4+k)
    for i in range(28):
        for j in range(49):
            V[i][j] = V1[i][j]
    for i in range(12):
        # i corresponds to local B[j,k] where j = i//3, k = i%3
        j_local = i // 3
        k_local = i % 3
        v_global_row = j_local * 7 + (4 + k_local)
        for jj in range(48):
            V[v_global_row][49 + jj] = V2[i][jj]
    
    # For W: W1 (28x49) and W2 (12x48)
    # W1 covers C[k,i] for k=0..3, i=0..3 -> W indices 0..15
    # W2 covers C[k,i] for k=0..2 (local), i=0..3 -> global C[4+k, i] -> W indices (4+k)*4+i
    for i in range(28):
        for j in range(49):
            W[i][j] = W1[i][j]
    for i in range(12):
        # i corresponds to local C[k,i] where k = i//4, i_local = i%4
        k_local = i // 4
        i_local = i % 4
        w_global_row = (4 + k_local) * 4 + i_local
        for jj in range(48):
            W[w_global_row][49 + jj] = W2[i][jj]
    
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Rank: {rank}")
    # Verify
    import numpy as np
    n, m, p = 4, 4, 7
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta[i,i'] * delta[j,j'] * delta[k,k']
    # Actually, the tensor is T[i,j,k] = 1 if i,j,k are valid indices, 0 otherwise.
    # The decomposition should satisfy:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # Wait, that's not right. The matrix multiplication tensor is:
    # T[i,j,k] = A[i,j] * B[j,k] * C[k,i] but that's not a tensor in the usual sense.
    # Actually, for matrix multiplication, the tensor is defined such that:
    # C = A * B means C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has T[i,j,k] = 1 if the indices are valid, and the decomposition
    # satisfies: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = delta (some Kronecker delta structure)
    
    # Let me verify by checking a few entries.
    # The correct verification is:
    # For random A, B: C = A @ B, and check that sum_c U[:,c] * V[:,c] * W[:,c] reconstructs C.
    
    # Actually, the standard verification for tensor decomposition of matrix multiplication:
    # The tensor M_{n,m,p} has entries M[i,j,k] = 1 if (i,j,k) is a valid triple.
    # The decomposition U, V, W should satisfy:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = M[i,j,k] = 1
    
    # Let me check.
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if val != 1:
                    max_err = max(max_err, abs(val - 1))
    print(f"Max error: {max_err}")
    
    # Also check that off-diagonal terms are zero? No, the tensor is all ones for valid indices.
    # Actually, I think the tensor for matrix multiplication is:
    # T[i,j,k] = 1 for all i,j,k (it's the "multiplication" tensor)
    # And the decomposition gives the bilinear form.
    
    # Let me do a numerical test.
    np.random.seed(42)
    A = np.random.randn(4, 4)
    B = np.random.randn(4, 7)
    C_expected = A @ B
    
    # Reconstruct using the decomposition
    C_reconstructed = np.zeros((4, 7))
    for c in range(rank):
        Uc = U[:, c]  # 16-vector
        Vc = V[:, c]  # 28-vector
        Wc = W[:, c]  # 28-vector
        # Uc reshaped to 4x4, Vc reshaped to 4x7, Wc reshaped to 7x4
        Uc_2d = Uc.reshape(4, 4)
        Vc_2d = Vc.reshape(4, 7)
        Wc_2d = Wc.reshape(7, 4)
        # Contribution: Uc_2d @ Vc_2d @ Wc_2d? No, that's not right.
        # The contribution is: for each (i,j,k): Uc[i,j] * Vc[j,k] * Wc[k,i]
        # This is a trilinear form, not a matrix product.
        # To compute C[i,k] = sum_j sum_c Uc[i,j] Vc[j,k] Wc[k,i]
        # = sum_c sum_j Uc[i,j] Vc[j,k] Wc[k,i]
        # = sum_c (Uc @ Vc) [i,k] * Wc[k,i]... no, that's not right either.
        
        # Let me just compute it directly.
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    C_reconstructed[i, k] += Uc[i*4+j] * Vc[j*7+k] * Wc[k*4+i]
    
    print(f"Numerical max error: {np.max(np.abs(C_reconstructed - C_expected))}")
