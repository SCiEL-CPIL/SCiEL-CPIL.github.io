"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 6

    # Strassen rank-7 decomposition for 2x2x2 matrix multiplication.
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B where C[i,j] = sum_k A[i,k]*B[k,j]
    #
    # The 7 multiplications:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b10-b00)
    # M4 = (a11)(b01-b11)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # Reconstruction:
    # C00 = M1 - M5 + M4 + M7
    # C10 = M2 + M6
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6

    # U vectors (for A), indexed by (row, col) of 2x2: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    strassen_U = [
        [1, 0, 0, 1],   # M1: a00+a11
        [0, 0, 1, 1],   # M2: a10+a11
        [1, 0, 0, 0],   # M3: a00
        [0, 0, 0, 1],   # M4: a11
        [1, 1, 0, 0],   # M5: a00+a01
        [1, 0, -1, 0],  # M6: a10-a00
        [0, 1, 0, -1],  # M7: a01-a11
    ]

    # V vectors (for B), indexed by (row, col) of 2x2: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    strassen_V = [
        [1, 0, 0, 1],   # M1: b00+b11
        [1, 0, 0, 0],   # M2: b00
        [0, 0, 1, -1],  # M3: b10-b00
        [0, 1, 0, -1],  # M4: b01-b11
        [0, 0, 0, 1],   # M5: b11
        [1, 1, 0, 0],   # M6: b00+b01
        [0, 0, 1, 1],   # M7: b10+b11
    ]

    # W vectors (for C), indexed by (row, col) of 2x2: (0,0)=0, (1,0)=1, (0,1)=2, (1,1)=3
    # Note: trace convention means W[k*n+i, r] where output index = k*n + i
    # For 2x2 block: local indices i (row), k (col) -> local output index = k*2 + i
    # So (0,0)=0, (1,0)=1, (0,1)=2, (1,1)=3
    strassen_W = [
        [1, 0, 0, 1],   # M1: C00 += M1, C11 += M1
        [0, 1, 0, -1],  # M2: C10 += M2, C11 -= M2
        [0, 0, 1, 1],   # M3: C01 += M3, C11 += M3
        [1, 0, 0, 0],   # M4: C00 += M4
        [-1, 0, 1, 0],  # M5: C00 -= M5, C01 += M5
        [0, 1, 0, 1],   # M6: C10 += M6, C11 += M6
        [1, 0, 0, 0],   # M7: C00 += M7
    ]

    # Verify Strassen: C00 = M1 - M5 + M4 + M7
    # M1[0]=1, -M5[0]=-(-1)=1, M4[0]=1, M7[0]=1 -> sum = 4? That's wrong.
    # Let me re-check. The W vectors should sum to give the correct C entries.
    # For C00 (local index 0): M1 contributes 1, M4 contributes 1, M5 contributes -1, M7 contributes 1
    # Sum = 1 + 1 + (-1) + 1 = 2. But C00 should be 1.
    # I think the issue is that each W vector represents the coefficient for that term in the output.
    # Let me re-derive.
    #
    # Actually, the way this works: for each term t (0..6), we have a scalar multiplication
    # U_t^T * A_block * V_t, and then W_t tells us how to add it to the output.
    # The full output is: C = sum_t W_t * (U_t^T * A * V_t)
    # where W_t is a 4-vector (flattened 2x2) and the multiplication is element-wise.
    #
    # So for C00: sum_t W_t[0] * (U_t^T * A * V_t)
    # We need this to equal A00*B00 + A01*B10.
    #
    # Let's just verify numerically with a simple example.
    # A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    # C should be [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19, 22],[43, 50]]
    #
    # M1 = (1+4)(5+8) = 5*13 = 65
    # M2 = (3+4)*5 = 35
    # M3 = 1*(7-5) = 2
    # M4 = 4*(6-8) = -8
    # M5 = (1+2)*8 = 24
    # M6 = (3-1)*(5+6) = 2*11 = 22
    # M7 = (2-4)*(7+8) = -2*15 = -30
    #
    # C00 = M1 - M5 + M4 + M7 = 65 - 24 + (-8) + (-30) = 3. Should be 19. Wrong!
    #
    # I have the Strassen formulas wrong. Let me look up the correct ones.
    #
    # Correct Strassen:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)(B00)
    # M3 = (A00)(B10 - B00)
    # M4 = (A11)(B01 - B11)
    # M5 = (A00 + A01)(B11)
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 - M5 + M4 + M7
    # C10 = M2 + M6
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    #
    # Let me recompute:
    # C00 = 65 - 24 + (-8) + (-30) = 3. Still wrong.
    #
    # Wait, I think I'm confusing the indices. Let me be very careful.
    # A = [[a00, a01], [a10, a11]] = [[1, 2], [3, 4]]
    # B = [[b00, b01], [b10, b11]] = [[5, 6], [7, 8]]
    #
    # C = A*B = [[a00*b00 + a01*b10, a00*b01 + a01*b11],
    #            [a10*b00 + a11*b10, a10*b01 + a11*b11]]
    # = [[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]]
    # = [[19, 22], [43, 50]]
    #
    # M1 = (a00+a11)(b00+b11) = (1+4)(5+8) = 65
    # M2 = (a10+a11)(b00) = (3+4)*5 = 35
    # M3 = a00(b10-b00) = 1*(7-5) = 2
    # M4 = a11(b01-b11) = 4*(6-8) = -8
    # M5 = (a00+a01)(b11) = (1+2)*8 = 24
    # M6 = (a10-a00)(b00+b01) = (3-1)*(5+6) = 22
    # M7 = (a01-a11)(b10+b11) = (2-4)*(7+8) = -30
    #
    # C00 = M1 - M5 + M4 + M7 = 65 - 24 - 8 - 30 = 3. NOT 19.
    #
    # I must have the wrong formulas. Let me try a different set.
    #
    # Actually, I think the standard Strassen is:
    # C00 = M1 - M5 + M4 + M7  -- this doesn't work with my numbers.
    #
    # Let me try: maybe it's C00 = M1 + M4 - M5 + M7? Same thing.
    #
    # Let me just use the schoolbook for 2x2x2 (rank 4) to be safe, and then we can optimize later.
    # For now, let's implement the block structure correctly.

    # Schoolbook 2x2x2: rank 4
    schoolbook_U = [
        [1, 0, 0, 0],  # a00
        [0, 1, 0, 0],  # a01
        [0, 0, 1, 0],  # a10
        [0, 0, 0, 1],  # a11
    ]
    schoolbook_V = [
        [1, 0, 0, 0],  # b00
        [0, 0, 1, 0],  # b10
        [0, 1, 0, 0],  # b01
        [0, 0, 0, 1],  # b11
    ]
    schoolbook_W = [
        [1, 0, 0, 0],  # C00 += a00*b00
        [0, 1, 0, 0],  # C10 += a10*b00
        [0, 0, 1, 0],  # C01 += a01*b10
        [0, 0, 0, 1],  # C11 += a11*b11
    ]

    # Now let's implement the block structure for 4x4x6.
    # A is 4x4, B is 4x6, C is 4x6.
    # Block A into 2x2 blocks of size 2x2: A[i,j] for i,j in {0,1}
    # Block B into 2x3 blocks of size 2x2: B[j,k] for j in {0,1}, k in {0,1,2}
    # C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k] for i in {0,1}, k in {0,1,2}
    #
    # Each C[i,k] is a 2x2 matrix, computed as sum of 2 products of 2x2 matrices.
    # We can use Strassen for each product, but that's complex.
    # For now, let's use schoolbook for each 2x2x2 sub-multiplication.
    #
    # Total rank: For each (i,k) pair, we have 2 j-values, and for each j, we have 4 schoolbook terms.
    # So 2*2*2*4 = 32? No, that's not right.
    #
    # Actually, let's just do the full schoolbook for 4x4x6: rank = 4*4*6 = 96.
    # Then we can try to improve.

    rank = 4 * 4 * 6
    U = [[0] * rank for _ in range(4 * 4)]
    V = [[0] * rank for _ in range(4 * 6)]
    W = [[0] * rank for _ in range(6 * 4)]
    column = 0
    for i in range(4):
        for j in range(4):
            for k in range(6):
                U[i * 4 + j][column] = 1
                V[j * 6 + k][column] = 1
                W[k * 4 + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
