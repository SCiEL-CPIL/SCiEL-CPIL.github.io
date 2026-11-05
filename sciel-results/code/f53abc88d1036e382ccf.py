"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0

    def add_block(i0, i1, j0, j1, k0, k1, block_type='schoolbook'):
        nonlocal column
        # block_type: 'schoolbook' or 'strassen' (only for 2x2x2)
        if block_type == 'schoolbook':
            for i in range(i0, i1):
                for j in range(j0, j1):
                    for k in range(k0, k1):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
        elif block_type == 'strassen':
            # 2x2x2 Strassen decomposition
            # i in [i0, i0+2), j in [j0, j0+2), k in [k0, k0+2)
            # M1 = (A11+A22)(B12+B21)
            # M2 = (A21+A22)B11
            # M3 = A11(B11-B21)
            # M4 = A22(B22-B12)
            # M5 = (A11+A12)B22
            # M6 = (A21-A11)(B11+B12)
            # M7 = (A12-A22)(B21+B22)
            #
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            #
            # Each M is a rank-1 tensor for a 2x2x2 block.
            # We need to map (i, j, k) local indices to global.
            # Local i: 0,1 -> global i0, i0+1
            # Local j: 0,1 -> global j0, j0+1
            # Local k: 0,1 -> global k0, k0+1

            def add_m12(i_local, j_local, k_local, coeff=1):
                nonlocal column
                if coeff == 0:
                    return
                gi = i0 + i_local
                gj = j0 + j_local
                gk = k0 + k_local
                U[gi * m + gj][column] = coeff
                V[gj * p + gk][column] = coeff
                W[gk * n + gi][column] = coeff
                column += 1

            # M1 = (A11+A22)(B12+B21)
            # A indices: (0,0),(1,1) for A11,A22; B indices: (0,1),(1,0) for B12,B21
            # M1 contributes to C11 with +1, C22 with +1
            # For C11: i_local=0, k_local=0; B: j_local=0,1; A: i_local=0,1
            # This is a 2x2x2 block where A has nonzero at (0,0),(1,1) and B at (0,1),(1,0)
            # Actually, M1 is a product of two 2x2 matrices, so it's rank 1 for the 2x2x2 subproblem?
            # No, M1 is a 2x2 matrix, and we're computing its contribution to C.
            # The tensor for M1 is: for each (i,j,k), the coefficient is A[i,j]*B[j,k].
            # M1_A = [[1,0],[0,1]], M1_B = [[0,1],[1,0]]
            # So M1[i,j,k] = M1_A[i,j] * M1_B[j,k]
            # This is a rank-1 tensor in the 2x2x2 space.
            
            def add_strassen_term(A_mat, B_mat):
                """Add a rank-1 tensor term defined by A_mat (2x2) and B_mat (2x2)."""
                nonlocal column
                # Check if all entries are zero
                all_zero = True
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            if A_mat[i][j] * B_mat[j][k] != 0:
                                all_zero = False
                                break
                        if not all_zero:
                            break
                    if not all_zero:
                        break
                if all_zero:
                    return
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            coeff = A_mat[i][j] * B_mat[j][k]
                            if coeff != 0:
                                gi = i0 + i
                                gj = j0 + j
                                gk = k0 + k
                                U[gi * m + gj][column] = coeff
                                V[gj * p + gk][column] = coeff
                                W[gk * n + gi][column] = coeff
                column += 1

            # Define the 7 Strassen matrices
            M1_A = [[1, 0], [0, 1]]  # A11+A22
            M1_B = [[0, 1], [1, 0]]  # B12+B21
            
            M2_A = [[0, 0], [1, 1]]  # A21+A22
            M2_B = [[1, 0], [0, 0]]  # B11
            
            M3_A = [[1, 0], [0, 0]]  # A11
            M3_B = [[1, 0], [0, -1]]  # B11-B21
            
            M4_A = [[0, 0], [0, 1]]  # A22
            M4_B = [[0, -1], [0, 1]]  # B22-B12
            
            M5_A = [[1, 1], [0, 0]]  # A11+A12
            M5_B = [[0, 0], [0, 1]]  # B22
            
            M6_A = [[-1, 0], [1, 0]]  # A21-A11
            M6_B = [[1, 1], [0, 0]]  # B11+B12
            
            M7_A = [[0, 1], [-1, 0]]  # A12-A22
            M7_B = [[0, 1], [1, 1]]  # B21+B22

            add_strassen_term(M1_A, M1_B)
            add_strassen_term(M2_A, M2_B)
            add_strassen_term(M3_A, M3_B)
            add_strassen_term(M4_A, M4_B)
            add_strassen_term(M5_A, M5_B)
            add_strassen_term(M6_A, M6_B)
            add_strassen_term(M7_A, M7_B)
        else:
            raise ValueError(f"Unknown block type: {block_type}")

    # Partition: I1={0,1,2}, I2={3,4}
    # C = A*B where A is 5x5, B is 5x5, C is 5x5
    # Block structure:
    # A = [[A33, A32], [A23, A22]] where A33 is 3x3, A32 is 3x2, A23 is 2x3, A22 is 2x2
    # B = [[B33, B32], [B23, B22]]
    # C = [[C33, C32], [C23, C22]]
    # C33 = A33*B33 + A32*B23 (both 3x3 and 3x2*2x3)
    # C32 = A33*B32 + A32*B22 (both 3x2)
    # C23 = A23*B33 + A22*B23 (both 2x3)
    # C22 = A23*B32 + A22*B22 (both 2x2)

    # We'll decompose each sub-block using either schoolbook or strassen.
    # For 2x2x2 blocks, use Strassen (rank 7 instead of 8).
    # For other blocks, use schoolbook.

    # C33 = A33*B33 (3x3x3) + A32*B23 (3x2x3)
    # For 3x3x3: partition 3 into 2+1
    #   A33 = [[A22, A21], [A12, A11]] where A22 is 2x2, A21 is 2x1, A12 is 1x2, A11 is 1x1
    #   B33 = [[B22, B21], [B12, B11]]
    #   C33 = [[A22*B22+A21*B12, A22*B21+A21*B11], [A12*B22+A11*B12, A12*B21+A11*B11]]
    #   C33[0:2,0:2] = A22*B22 (2x2x2, Strassen) + A21*B12 (2x1x2, schoolbook rank 4)
    #   C33[0:2,2:3] = A22*B21 (2x2x1, schoolbook rank 4) + A21*B11 (2x1x1, schoolbook rank 2)
    #   C33[2:3,0:2] = A12*B22 (1x2x2, schoolbook rank 4) + A11*B12 (1x1x2, schoolbook rank 2)
    #   C33[2:3,2:3] = A12*B21 (1x2x1, schoolbook rank 2) + A11*B11 (1x1x1, schoolbook rank 1)
    
    # Let's implement this recursively.
    
    # Actually, let's just use schoolbook for all 3x3 sub-blocks and Strassen for 2x2.
    # The total rank will be:
    # C33 (3x3x3): schoolbook = 27
    # C32 (3x3x2 + 3x2x2): A33*B32 (3x3x2=18) + A32*B22 (3x2x2=12) = 30
    # C23 (2x3x3 + 2x2x3): A23*B33 (2x3x3=18) + A22*B23 (2x2x3=12) = 30
    # C22 (2x3x2 + 2x2x2): A23*B32 (2x3x2=12) + A22*B22 (2x2x2=7 Strassen) = 19
    # Total = 27+30+30+19 = 106. That's worse than 83.
    
    # Let me reconsider. The BCDFR rank for 3x3 is 16, not 27.
    # I need to implement a rank-16 decomposition for 3x3x3.
    
    # Actually, let me just use schoolbook everywhere first to get a valid baseline,
    # then optimize the 2x2 blocks with Strassen.
    
    # Use schoolbook for all blocks:
    # C33: 3x3x3 = 27
    # C32: A33*B32 (3x3x2=18) + A32*B22 (3x2x2=12) = 30
    # C23: A23*B33 (2x3x3=18) + A22*B23 (2x2x3=12) = 30
    # C22: A23*B32 (2x3x2=12) + A22*B22 (2x2x2=8) = 20
    # Total = 27+30+30+20 = 107
    
    # With Strassen for 2x2x2 blocks:
    # C33: 27 (all schoolbook)
    # C32: 18 + 12 = 30
    # C23: 18 + 12 = 30
    # C22: 12 + 7 = 19
    # Total = 27+30+30+19 = 106
    
    # Let me implement this:
    
    # C33 = A33*B33: i in [0,3], j in [0,3], k in [0,3]
    add_block(0, 3, 0, 3, 0, 3, 'schoolbook')
    
    # C32 = A33*B32: i in [0,3], j in [0,3], k in [3,5]
    add_block(0, 3, 0, 3, 3, 5, 'schoolbook')
    # C32 += A32*B22: i in [0,3], j in [3,5], k in [3,5]
    add_block(0, 3, 3, 5, 3, 5, 'schoolbook')
    
    # C23 = A23*B33: i in [3,5], j in [0,3], k in [0,3]
    add_block(3, 5, 0, 3, 0, 3, 'schoolbook')
    # C23 += A22*B23: i in [3,5], j in [3,5], k in [0,3]
    add_block(3, 5, 3, 5, 0, 3, 'schoolbook')
    
    # C22 = A23*B32: i in [3,5], j in [0,3], k in [3,5]
    add_block(3, 5, 0, 3, 3, 5, 'schoolbook')
    # C22 += A22*B22: i in [3,5], j in [3,5], k in [3,5]
    add_block(3, 5, 3, 5, 3, 5, 'strassen')
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

