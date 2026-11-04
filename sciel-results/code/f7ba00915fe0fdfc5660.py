"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    # Decomposition of M_{2,3,7} as direct sum of M_{2,2,2} (Strassen),
    # M_{2,3,2} (Strassen+outer), M_{2,3,3} (outer).
    # Total rank = 7 + 7 + 4 + 6 + 9 = 33.
    #
    # U shape: (n*m, rank) = (6, 33)
    # V shape: (m*p, rank) = (21, 33)
    # W shape: (p*n, rank) = (14, 33)
    #
    # Index conventions:
    #   A is 2x3: A[i,j] with i in 0..1, j in 0..2
    #   B is 3x7: B[j,k] with j in 0..2, k in 0..6
    #   C = A*B is 2x7: C[i,k] with i in 0..1, k in 0..6
    #
    # U[i*m + j, c] = coefficient of A[i,j] in rank-c component
    # V[j*p + k, c] = coefficient of B[j,k] in rank-c component
    # W[k*n + i, c] = coefficient of C[i,k] in rank-c component (trace convention)

    import numpy as np
    from fractions import Fraction

    n, m, p = DIMENSIONS
    total_rank = 33
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    # --- Block 1: M_{2,2,2} using Strassen, columns 0-6 ---
    # Uses A[i,0..1], B[0..1, 0..1], produces C[0..1, 0..1]
    # Strassen for 2x2 matrices:
    # A sub-matrix: A[0:2, 0:2] -> indices in U: i*3 + j for i in 0..1, j in 0..1
    #   A00 = U[0], A01 = U[1], A10 = U[3], A11 = U[4]
    # B sub-matrix: B[0:2, 0:2] -> indices in V: j*7 + k for j in 0..1, k in 0..1
    #   B00 = V[0], B01 = V[1], B10 = V[7], B11 = V[8]
    # C sub-matrix: C[0:2, 0:2] -> indices in W: k*2 + i for k in 0..1, i in 0..1
    #   C00 = W[0], C01 = W[1], C10 = W[2], C11 = W[3]

    # Strassen formulas (standard):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B01-B11)
    # M4 = (A11)(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # For each Strassen term, we have a rank-1 tensor with:
    # U-coefficients for A entries, V-coefficients for B entries, W-coefficients for C entries.

    # Term 1: M1 = (A00+A11)(B00+B11) -> contributes to C00(+1), C11(+1)
    # Actually, let me re-derive. The standard Strassen gives:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B01-B11)
    # M4 = (A11)(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Each M_t is a product of a linear form in A and a linear form in B.
    # So each M_t is a rank-1 tensor in the (A_sub, B_sub, C_sub) space.

    # Let's define the 7 Strassen components:
    # For component t (0-6), we have:
    #   u_t[i,j] for the A-linear form (i in 0..1, j in 0..1)
    #   v_t[j,k] for the B-linear form (j in 0..1, k in 0..1)
    #   w_t[k,i] for the C-output (k in 0..1, i in 0..1), which is a sum over t

    # Actually, the standard way: each M_t is one scalar multiplication.
    # The output C is a linear combination of the M_t's.
    # So for the tensor decomposition, we have 7 rank-1 terms, each with:
    #   A-coefficients (from the A-factor of M_t)
    #   B-coefficients (from the B-factor of M_t)
    #   C-coefficients (which are the contributions of M_t to each C entry)

    # Let me list them:
    # t=0 (M1): A: (1,0,0,1) [A00=1,A01=0,A10=0,A11=1], B: (1,0,0,1) [B00=1,B01=0,B10=0,B11=1]
    #   C-contribution: C00 += 1, C11 += 1
    # t=1 (M2): A: (0,0,1,1) [A10=1,A11=1], B: (1,0,0,0) [B00=1]
    #   C-contribution: C10 += 1
    # t=2 (M3): A: (1,0,0,0) [A00=1], B: (0,1,0,-1) [B01=1,B11=-1]
    #   C-contribution: C01 += 1
    # t=3 (M4): A: (0,0,0,1) [A11=1], B: (0,0,1,-1) [B10=1,B00=-1]
    #   C-contribution: C00 += 1, C10 += 1
    # t=4 (M5): A: (1,1,0,0) [A00=1,A01=1], B: (0,0,0,1) [B11=1]
    #   C-contribution: C00 -= 1, C01 += 1
    # t=5 (M6): A: (0,0,1,-1) [A10=1,A00=-1], B: (1,1,0,0) [B00=1,B01=1]
    #   C-contribution: C11 += 1
    # t=6 (M7): A: (0,1,-1,0) [A01=1,A11=-1], B: (0,0,1,1) [B10=1,B11=1]
    #   C-contribution: C00 += 1, C11 -= 1

    # Verify:
    # C00 = M1 + M4 - M5 + M7 = 1 + 1 - 1 + 1 = 2? No...
    # Let me recheck. Standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # So the C-coefficients for each M_t:
    # M1: C00=+1, C11=+1
    # M2: C10=+1
    # M3: C01=+1
    # M4: C00=+1, C10=+1
    # M5: C00=-1, C01=+1
    # M6: C11=+1
    # M7: C00=+1, C11=-1

    strassen = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # A_coeffs: (A00, A01, A10, A11)
        # B_coeffs: (B00, B01, B10, B11)
        # C_coeffs: (C00, C01, C10, C11)
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),    # M1
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),    # M2
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),   # M3
        ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),   # M4
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),   # M5
        ([0,0,1,-1], [1,1,0,0], [0,0,0,1]),   # M6
        ([0,1,-1,0], [0,0,1,1], [1,0,0,-1]),  # M7
    ]

    col = 0
    for a_coeffs, b_coeffs, c_coeffs in strassen:
        # A indices: A00=U[0*3+0]=U[0], A01=U[1*3+0]=U[3]? No.
        # U[i*m + j] where i is row, j is col of A.
        # A is 2x3. For the 2x2 submatrix (cols 0,1):
        # A00: i=0,j=0 -> U[0]
        # A01: i=0,j=1 -> U[1]
        # A10: i=1,j=0 -> U[3]
        # A11: i=1,j=1 -> U[4]
        a_idx = [0, 1, 3, 4]  # A00, A01, A10, A11
        for ai, ac in zip(a_idx, a_coeffs):
            U[ai][col] += ac

        # B indices: B[j,k], V[j*p + k]
        # B is 3x7. For the 2x2 submatrix (rows 0,1; cols 0,1):
        # B00: j=0,k=0 -> V[0]
        # B01: j=0,k=1 -> V[1]
        # B10: j=1,k=0 -> V[7]
        # B11: j=1,k=1 -> V[8]
        b_idx = [0, 1, 7, 8]  # B00, B01, B10, B11
        for bi, bc in zip(b_idx, b_coeffs):
            V[bi][col] += bc

        # C indices: C[i,k], W[k*n + i]
        # C is 2x7. For the 2x2 submatrix (rows 0,1; cols 0,1):
        # C00: i=0,k=0 -> W[0]
        # C01: i=0,k=1 -> W[1]
        # C10: i=1,k=0 -> W[2]
        # C11: i=1,k=1 -> W[3]
        c_idx = [0, 1, 2, 3]  # C00, C01, C10, C11
        for ci, cc in zip(c_idx, c_coeffs):
            W[ci][col] += cc

        col += 1

    # --- Block 2: M_{2,3,2} using Strassen(2x2) + outer(1x2), columns 7-13 ---
    # Uses A[i,0..2] (all 3 cols), B[0..1, 0..1] (2x2 submatrix)
    # This is decomposed as: Strassen on A[0:2, 0:2] * B[0:2, 0:2] 
    # plus outer product A[:,2] * B[0:2, :]
    #
    # Wait, that's not quite right. Let me think again.
    # M_{2,3,2}: A is 2x3, B is 3x2, C is 2x2.
    # We can split A into A_left (2x2) and A_right (2x1), and B into B_top (2x2) and B_bottom (1x2).
    # C = A_left * B_top + A_right * B_bottom
    # A_left * B_top is M_{2,2,2} -> Strassen (7 multiplications)
    # A_right * B_bottom is M_{2,1,2} -> 2*1*2 = 4 multiplications (outer product)
    # Total: 7 + 4 = 11... but I said 7+4=11, and earlier I calculated 33 = 7+7+4+6+9.
    # Let me recount: Block1=7, Block2=7+4=11, Block3=6+9=15? No, Block3 is M_{2,3,3} = 18.
    # 7 + 11 + 18 = 36, not 33.
    
    # Let me reconsider. Maybe I should use a different split.
    # Actually, for M_{2,3,2}, we can do:
    # C = A * B where A is 2x3, B is 3x2.
    # Split B into B1 (first 2 rows) and B2 (last 1 row).
    # C = A[:, :2] * B1 + A[:, 2:3] * B2
    # A[:, :2] * B1 is M_{2,2,2} -> 7 mults (Strassen)
    # A[:, 2:3] * B2 is M_{2,1,2} -> 4 mults
    # Total: 11
    #
    # For M_{2,3,3}: A is 2x3, B is 3x3.
    # C = A * B, 2x3 result.
    # This is 2*3*3 = 18 mults schoolbook.
    #
    # Total: 7 + 11 + 18 = 36.
    #
    # Hmm, can we do better? What if we split differently?
    # M_{2,3,7}: 
    # Option: split B into 3 parts: B1 (3x2), B2 (3x2), B3 (3x3).
    # C = A*B1 + A*B2 + A*B3
    # A*B1: M_{2,3,2} = 11
    # A*B2: M_{2,3,2} = 11  
    # A*B3: M_{2,3,3} = 18
    # Total: 40. Worse.
    #
    # Better: split into B1 (3x1), B2 (3x1), ..., B7 (3x1).
    # Each A*Bj is M_{2,3,1} = 6 mults.
    # 7 * 6 = 42. Same as schoolbook.
    #
    # What about: B1 (3x2), B2 (3x2), B3 (3x2), B4 (3x1)?
    # 3 * 11 + 6 = 39.
    #
    # Or: B1 (3x2), B2 (3x2), B3 (3x3)?
    # 2 * 11 + 18 = 40.
    #
    # So the best split seems to be: one 2x2 Strassen block + remaining.
    # Actually, let me reconsider M_{2,3,2}.
    # A is 2x3, B is 3x2.
    # We can also view this as:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    #
    # If we use Strassen on the 2x2 part (A[0:2, 0:2] and B[0:2, 0:2]), we get 7 mults for the 2x2 block.
    # Then we need to add the contributions from A[:,2] and B[2,:].
    # C += A[:,2:3] * B[2:3, :]
    # This is a 2x1 times 1x2 = 2x2 matrix, which is 4 mults.
    # Total: 7 + 4 = 11.
    #
    # Can we do better? The tensor rank of M_{2,3,2} is known to be at most 11.
    # Actually, I recall that the rank of M_{2,3,2} is exactly 11.
    
    # So total = 7 + 11 + 18 = 36.
    # Let me just go with this.

    # Actually wait, let me reconsider the split. What if I split B into:
    # B1 (3x2), B2 (3x2), B3 (3x2), B4 (3x1)?
    # That gives 3*M_{2,3,2} + M_{2,3,1} = 3*11 + 6 = 39.
    #
    # Or B1 (3x3), B2 (3x3), B3 (3x1)?
    # 2*M_{2,3,3} + M_{2,3,1} = 2*18 + 6 = 42.
    #
    # So 7 + 11 + 18 = 36 is the best I can find with simple splits.
    
    # Let me just implement the 7 + 11 + 18 = 36 decomposition.
    
    # Actually, I realize I should double-check: is there a way to get below 36?
    # The trivial bound is 42. Strassen-style splits give 36.
    # Let me just go with 36.

    # Let me redo this properly.
    
    # Reset and rebuild.
    total_rank = 7 + 11 + 18  # 36
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0

    # Block 1: M_{2,2,2} Strassen on A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # Columns 0-6
    for a_coeffs, b_coeffs, c_coeffs in strassen:
        a_idx = [0, 1, 3, 4]
        b_idx = [0, 1, 7, 8]
        c_idx = [0, 1, 2, 3]
        for ai, ac in zip(a_idx, a_coeffs):
            U[ai][col] += ac
        for bi, bc in zip(b_idx, b_coeffs):
            V[bi][col] += bc
        for ci, cc in zip(c_idx, c_coeffs):
            W[ci][col] += cc
        col += 1

    # Block 2: M_{2,3,2} = Strassen(A[0:2,0:2]*B[0:2,0:2]) + outer(A[:,2]*B[2,:])
    # Columns 7-17
    # First, Strassen part: same as block 1 but for B cols 2,3 (k=2,3)
    # A[0:2, 0:2]: U indices same: [0,1,3,4]
    # B[0:2, 2:4]: V indices: j*7+k for j in 0,1; k in 2,3
    #   B00 (j=0,k=2): V[2]
    #   B01 (j=0,k=3): V[3]
    #   B10 (j=1,k=2): V[9]
    #   B11 (j=1,k=3): V[10]
    # C[0:2, 2:4]: W indices: k*2+i for k in 2,3; i in 0,1
    #   C00 (k=2,i=0): W[4]
    #   C01 (k=3,i=0): W[5]
    #   C10 (k=2,i=1): W[6]
    #   C11 (k=3,i=1): W[7]

    b_idx2 = [2, 3, 9, 10]  # B00,B01,B10,B11 for cols 2,3
    c_idx2 = [4, 5, 6, 7]   # C00,C01,C10,C11 for cols 2,3

    for a_coeffs, b_coeffs, c_coeffs in strassen:
        a_idx = [0, 1, 3, 4]
        for ai, ac in zip(a_idx, a_coeffs):
            U[ai][col] += ac
        for bi, bc in zip(b_idx2, b_coeffs):
            V[bi][col] += bc
        for ci, cc in zip(c_idx2, c_coeffs):
            W[ci][col] += cc
        col += 1

    # Outer product part: A[:,2] * B[2,:]
    # A[:,2]: A[0,2]=U[2], A[1,2]=U[5]
    # B[2,:]: B[2,0]=V[14], B[2,1]=V[15]
    # C[0:2, 0:2]: C00=W[0], C01=W[1], C10=W[2], C11=W[3]
    #
    # 4 terms:
    # A[0,2]*B[2,0] -> C[0,0]
    # A[0,2]*B[2,1] -> C[0,1]
    # A[1,2]*B[2,0] -> C[1,0]
    # A[1,2]*B[2,1] -> C[1,1]

    outer_terms = [
        (2, 14, 0),   # A[0,2], B[2,0], C[0,0]
        (2, 15, 1),   # A[0,2], B[2,1], C[0,1]
        (5, 14, 2),   # A[1,2], B[2,0], C[1,0]
        (5, 15, 3),   # A[1,2], B[2,1], C[1,1]
    ]
    for u_i, v_i, w_i in outer_terms:
        U[u_i][col] = 1
        V[v_i][col] = 1
        W[w_i][col] = 1
        col += 1

    # Block 3: M_{2,3,3} schoolbook for B[0:3, 4:7]
    # Columns 18-35
    # A[i,j] for i in 0,1; j in 0,1,2
    # B[j,k] for j in 0,1,2; k in 4,5,6
    # C[i,k] for i in 0,1; k in 4,5,6
    #
    # W[k*n + i] = W[k*2 + i]
    # k=4: W[8], W[9]
    # k=5: W[10], W[11]
    # k=6: W[12], W[13]

    for i in range(n):
        for j in range(m):
            for k in range(4, 7):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Convert to list of lists (already are)
    # Ensure all values are integers
    for row in U:
        for idx in range(len(row)):
            if isinstance(row[idx], Fraction):
                row[idx] = int(row[idx])
    for row in V:
        for idx in range(len(row)):
            if isinstance(row[idx], Fraction):
                row[idx] = int(row[idx])
    for row in W:
        for idx in range(len(row)):
            if isinstance(row[idx], Fraction):
                row[idx] = int(row[idx])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
