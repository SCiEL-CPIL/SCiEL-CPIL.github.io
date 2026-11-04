"""
Rank-minimal tensor decomposition for 5x5x5 matrix multiplication.
Uses 4x4 Strassen core (rank 49) + boundary terms (rank 61) = rank 110.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = 5, 5, 5
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    # Total rank: 49 (Strassen 4x4) + 61 (boundary) = 110
    R = 110

    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)

    col = 0

    # ---- Strassen 2x2 helper ----
    # For a 2x2x2 sub-problem with A at rows (i1,i2) cols (j1,j2),
    # B at rows (j1,j2) cols (k1,k2), C at rows (i1,i2) cols (k1,k2)
    def strassen_2x2(i1, j1, k1):
        nonlocal col
        i2 = i1 + 1
        j2 = j1 + 1
        k2 = k1 + 1

        # M1 = (A00+A11)(B00+B11) -> C00(+1), C11(+1)
        U[i1*m+j1, col] = 1.0; U[i2*m+j2, col] = 1.0
        V[j1*p+k1, col] = 1.0; V[j2*p+k2, col] = 1.0
        W[k1*n+i1, col] = 1.0; W[k2*n+i2, col] = 1.0
        col += 1

        # M2 = (A10+A11)B00 -> C10(+1)
        U[i2*m+j1, col] = 1.0; U[i2*m+j2, col] = 1.0
        V[j1*p+k1, col] = 1.0
        W[k1*n+i2, col] = 1.0
        col += 1

        # M3 = A00(B01-B11) -> C01(+1)
        U[i1*m+j1, col] = 1.0
        V[j1*p+k2, col] = 1.0; V[j2*p+k2, col] = -1.0
        W[k2*n+i1, col] = 1.0
        col += 1

        # M4 = A11(B10-B00) -> C10(+1), C00(-1)
        U[i2*m+j2, col] = 1.0
        V[j2*p+k1, col] = 1.0; V[j1*p+k1, col] = -1.0
        W[k1*n+i2, col] = 1.0; W[k1*n+i1, col] = -1.0
        col += 1

        # M5 = (A00+A01)B11 -> C01(+1), C00(-1)
        U[i1*m+j1, col] = 1.0; U[i1*m+j2, col] = 1.0
        V[j2*p+k2, col] = 1.0
        W[k2*n+i1, col] = 1.0; W[k1*n+i1, col] = -1.0
        col += 1

        # M6 = (A10-A00)(B00+B01) -> C11(+1)
        U[i2*m+j1, col] = 1.0; U[i1*m+j1, col] = -1.0
        V[j1*p+k1, col] = 1.0; V[j1*p+k2, col] = 1.0
        W[k2*n+i2, col] = 1.0
        col += 1

        # M7 = (A01-A11)(B10+B11) -> C00(+1), C11(-1)
        U[i1*m+j2, col] = 1.0; U[i2*m+j2, col] = -1.0
        V[j2*p+k1, col] = 1.0; V[j2*p+k2, col] = 1.0
        W[k1*n+i1, col] = 1.0; W[k2*n+i2, col] = -1.0
        col += 1

    # ---- Strassen 4x4 helper (7 top-level mults, each 2x2x2 = 7 terms) ----
    # 4x4 split into 2x2 blocks:
    # A = [[A00, A01],[A10, A11]] where A00 rows 0-1 cols 0-1, etc.
    # B = [[B00, B01],[B10, B11]]
    # C = [[C00, C01],[C10, C11]]
    #
    # Top-level Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # Each M is a 2x2*2x2 product. We need a generalized Strassen where
    # the input matrices are linear combinations of blocks.
    #
    # For a 2x2x2 Strassen where A = alpha1*P + alpha2*Q, B = beta1*R + beta2*S,
    # the 7 rank-1 terms have u vectors with coefficients from {alpha1, alpha2},
    # v vectors with coefficients from {beta1, beta2}.
    #
    # Let me define a helper that does Strassen for A = sum of two 2x2 matrices
    # (specified by their (i,j) positions) and B = sum of two 2x2 matrices.

    def strassen_general(u_coeff_a1, u_coeff_a2, v_coeff_b1, v_coeff_b2,
                         a1_pos, a2_pos, b1_pos, b2_pos, c_pos):
        """
        a1_pos, a2_pos: (i1,j1) top-left of each 2x2 A block
        b1_pos, b2_pos: (j1,k1) top-left of each 2x2 B block
        c_pos: (i1,k1) top-left of 2x2 C output
        u_coeff_a1, u_coeff_a2: coefficients for A blocks
        v_coeff_b1, v_coeff_b2: coefficients for B blocks
        """
        nonlocal col
        ci1, ck1 = c_pos
        ci2 = ci1 + 1
        ck2 = ck1 + 1

        # A1 block: rows a1_pos[0], a1_pos[0]+1; cols a1_pos[1], a1_pos[1]+1
        a1i1, a1j1 = a1_pos
        a1i2, a1j2 = a1i1+1, a1j1+1
        # A2 block
        a2i1, a2j1 = a2_pos
        a2i2, a2j2 = a2i1+1, a2j1+1
        # B1 block
        b1j1, b1k1 = b1_pos
        b1j2, b1k2 = b1j1+1, b1k1+1
        # B2 block
        b2j1, b2k1 = b2_pos
        b2j2, b2k2 = b2j1+1, b2k1+1

        ca1, ca2 = u_coeff_a1, u_coeff_a2
        cb1, cb2 = v_coeff_b1, v_coeff_b2

        # M1 = (A1+A2)(B1+B2) -> C00(+1), C11(+1)
        # u: ca1 at A1 positions, ca2 at A2 positions
        # v: cb1 at B1 positions, cb2 at B2 positions
        U[a1i1*m+a1j1, col] = ca1; U[a1i2*m+a1j2, col] = ca1
        U[a2i1*m+a2j1, col] = ca2; U[a2i2*m+a2j2, col] = ca2
        V[b1j1*p+b1k1, col] = cb1; V[b1j2*p+b1k2, col] = cb1
        V[b2j1*p+b2k1, col] = cb2; V[b2j2*p+b2k2, col] = cb2
        W[ck1*n+ci1, col] = 1.0; W[ck2*n+ci2, col] = 1.0
        col += 1

        # M2 = (A2_row1 + A2_row2) * B1_col1
        # A10+A11 in local = row1 of combined A
        # Actually for general case: (A1[1,:]+A2[1,:]) * B1[0,:]
        # u: ca1 at A1 row1, ca2 at A2 row1
        # v: cb1 at B1 col0 (rows b1j1, cols b1k1)
        # w: C10 (+1)
        U[a1i2*m+a1j1, col] = ca1; U[a1i2*m+a1j2, col] = ca1
        U[a2i2*m+a2j1, col] = ca2; U[a2i2*m+a2j2, col] = ca2
        V[b1j1*p+b1k1, col] = cb1
        W[ck1*n+ci2, col] = 1.0
        col += 1

        # M3 = A1[0,:] * (B1[0,1] - B2[0,1])
        # u: ca1 at A1 row0
        # v: cb1 at B1[0,1], -cb2 at B2[0,1]
        # w: C01 (+1)
        U[a1i1*m+a1j1, col] = ca1
        V[b1j1*p+b1k2, col] = cb1; V[b2j1*p+b2k2, col] = -cb2
        W[ck2*n+ci1, col] = 1.0
        col += 1

        # M4 = A2[1,1] * (B2[1,0] - B1[1,0])
        # u: ca2 at A2[1,1]
        # v: cb2 at B2[1,0], -cb1 at B1[1,0]
        # w: C10(+1), C00(-1)
        U[a2i2*m+a2j2, col] = ca2
        V[b2j2*p+b2k1, col] = cb2; V[b1j2*p+b1k1, col] = -cb1
        W[ck1*n+ci2, col] = 1.0; W[ck1*n+ci1, col] = -1.0
        col += 1

        # M5 = (A1[0,:]) * B2[1,1]
        # u: ca1 at A1 row0
        # v: cb2 at B2[1,1]
        # w: C01(+1), C00(-1)
        U[a1i1*m+a1j1, col] = ca1; U[a1i1*m+a1j2, col] = ca1
        V[b2j2*p+b2k2, col] = cb2
        W[ck2*n+ci1, col] = 1.0; W[ck1*n+ci1, col] = -1.0
        col += 1

        # M6 = (A2[1,0]-A1[1,0]) * (B1[0,:])
        # u: ca2 at A2[1,0], -ca1 at A1[1,0]
        # v: cb1 at B1 row0
        # w: C11(+1)
        U[a2i2*m+a2j1, col] = ca2; U[a1i2*m+a1j1, col] = -ca1
        V[b1j1*p+b1k1, col] = cb1; V[b1j1*p+b1k2, col] = cb1
        W[ck2*n+ci2, col] = 1.0
        col += 1

        # M7 = (A1[0,1]-A2[0,1]) * (B2[1,:])
        # u: ca1 at A1[0,1], -ca2 at A2[0,1]
        # v: cb2 at B2 row1
        # w: C00(+1), C11(-1)
        U[a1i1*m+a1j2, col] = ca1; U[a2i1*m+a2j2, col] = -ca2
        V[b2j2*p+b2k1, col] = cb2; V[b2j2*p+b2k2, col] = cb2
        W[ck1*n+ci1, col] = 1.0; W[ck2*n+ci2, col] = -1.0
        col += 1

    # ---- Build 4x4x4 Strassen decomposition (indices 0-3) ----
    # 4x4 blocks (each 2x2):
    # A00: rows 0-1, cols 0-1  -> pos (0,0)
    # A01: rows 0-1, cols 2-3  -> pos (0,2)
    # A10: rows 2-3, cols 0-1  -> pos (2,0)
    # A11: rows 2-3, cols 2-3  -> pos (2,2)
    # B00: rows 0-1, cols 0-1  -> pos (0,0)
    # B01: rows 0-1, cols 2-3  -> pos (0,2)
    # B10: rows 2-3, cols 0-1  -> pos (2,0)
    # B11: rows 2-3, cols 2-3  -> pos (2,2)
    # C00: rows 0-1, cols 0-1  -> pos (0,0)
    # C01: rows 0-1, cols 2-3  -> pos (0,2)
    # C10: rows 2-3, cols 0-1  -> pos (2,0)
    # C11: rows 2-3, cols 2-3  -> pos (2,2)

    A00 = (0, 0); A01 = (0, 2); A10 = (2, 0); A11 = (2, 2)
    B00 = (0, 0); B01 = (0, 2); B10 = (2, 0); B11 = (2, 2)
    C00 = (0, 0); C01 = (0, 2); C10 = (2, 0); C11 = (2, 2)

    # M1 = (A00+A11)(B00+B11) -> contributes to C00 and C11
    # But in Strassen, M1 contributes to C00(+1) and C11(+1)
    # The 2x2 product (A00+A11)*(B00+B11) gives a 2x2 result.
    # We need to split this into contributions to C00 and C11.
    #
    # Actually, in the 4x4 Strassen:
    # C00 = M1 + M4 - M5 + M7  (each M is a 2x2 matrix)
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # So M1 (a 2x2 matrix) adds to both C00 and C11.
    # In the tensor decomposition, each "M" is a 2x2x2 sub-problem.
    # But M1 contributes to C00 with +1 and C11 with +1.
    #
    # The way to handle this: M1 is computed as a 2x2x2 Strassen,
    # and its output is added to C00 and C11.
    # But in the tensor decomposition, each rank-1 term has a specific
    # (i,j,k) output position. So we need to duplicate the 7 terms
    # for each output block? No, that would be 14 terms per M.
    #
    # Actually, the correct approach: the 4x4x4 tensor is the sum of
    # 7 "2x2x2 sub-tensors" where each sub-tensor maps the combined
    # inputs to the combined outputs. Each such sub-tensor has rank 7.
    # Total rank = 7*7 = 49.
    #
    # For M1 = (A00+A11)(B00+B11):
    # This is a 2x2x2 multiplication where:
    # A_combined = A00 + A11 (2x2 matrix)
    # B_combined = B00 + B11 (2x2 matrix)
    # C_combined = result (2x2 matrix) which adds to C00 and C11
    #
    # The 7 rank-1 terms for this sub-problem have:
    # u vectors encoding A00+A11 (coefficients on A00 and A11 elements)
    # v vectors encoding B00+B11
    # w vectors encoding the output, which goes to C00 AND C11
    #
    # But C00 and C11 are at different positions! C00 is at (0,0)-(1,1)
    # and C11 is at (2,2)-(3,3). So the w vector must have entries at
    # both C00 and C11 positions.
    #
    # For M1: w has +1 at C00 positions and +1 at C11 positions.
    # That means w has 4 entries for C00 and 4 entries for C11.
    # But in a rank-1 tensor, w is a single vector. So we can have
    # w with entries at both C00 and C11 positions.
    #
    # Let me implement this properly.

    def strassen_4x4_subterm(a1_pos, a2_pos, b1_pos, b2_pos,
                             c1_pos, c2_pos,
                             ca1, ca2, cb1, cb2,
                             w_coeff_c1, w_coeff_c2):
        """
        One Strassen multiplication for 4x4.
        A = ca1*A1 + ca2*A2 (each 2x2)
        B = cb1*B1 + cb2*B2 (each 2x2)
        Result adds w_coeff_c1 * result to C1 and w_coeff_c2 * result to C2.
        """
        nonlocal col
        # A1: rows a1_pos[0], a1_pos[0]+1; cols a1_pos[1], a1_pos[1]+1
        a1i1, a1j1 = a1_pos
        a1i2, a1j2 = a1i1+1, a1j1+1
        a2i1, a2j1 = a2_pos
        a2i2, a2j2 = a2i1+1, a2j1+1
        b1j1, b1k1 = b1_pos
        b1j2, b1k2 = b1j1+1, b1k1+1
        b2j1, b2k1 = b2_pos
        b2j2, b2k2 = b2j1+1, b2k1+1
        c1i1, c1k1 = c1_pos
        c1i2, c1k2 = c1i1+1, c1k1+1
        c2i1, c2k1 = c2_pos
        c2i2, c2k2 = c2i1+1, c2k1+1

        # M1 = (A1+A2)(B1+B2)
        U[a1i1*m+a1j1, col] = ca1; U[a1i2*m+a1j2, col] = ca1
        U[a2i1*m+a2j1, col] = ca2; U[a2i2*m+a2j2, col] = ca2
        V[b1j1*p+b1k1, col] = cb1; V[b1j2*p+b1k2, col] = cb1
        V[b2j1*p+b2k1, col] = cb2; V[b2j2*p+b2k2, col] = cb2
        W[c1k1*n+c1i1, col] = w_coeff_c1
        W[c1k2*n+c1i2, col] = w_coeff_c1
        W[c2k1*n+c2i1, col] = w_coeff_c2
        W[c2k2*n+c2i2, col] = w_coeff_c2
        col += 1

        # M2 = (A1_row1 + A2_row1) * B1_col0
        U[a1i2*m+a1j1, col] = ca1; U[a1i2*m+a1j2, col] = ca1
        U[a2i2*m+a2j1, col] = ca2; U[a2i2*m+a2j2, col] = ca2
        V[b1j1*p+b1k1, col] = cb1
        W[c1k1*n+c1i2, col] = w_coeff_c1
        W[c2k1*n+c2i2, col] = w_coeff_c2
        col += 1

        # M3 = A1_row0 * (B1_col1 - B2_col1)
        U[a1i1*m+a1j1, col] = ca1
        V[b1j1*p+b1k2, col] = cb1; V[b2j1*p+b2k2, col] = -cb2
        W[c1k2*n+c1i1, col] = w_coeff_c1
        W[c2k2*n+c2i1, col] = w_coeff_c2
        col += 1

        # M4 = A2[1,1] * (B2[1,0] - B1[1,0])
        U[a2i2*m+a2j2, col] = ca2
        V[b2j2*p+b2k1, col] = cb2; V[b1j2*p+b1k1, col] = -cb1
        W[c1k1*n+c1i2, col] = w_coeff_c1; W[c1k1*n+c1i1, col] = -w_coeff_c1
        W[c2k1*n+c2i2, col] = w_coeff_c2; W[c2k1*n+c2i1, col] = -w_coeff_c2
        col += 1

        # M5 = A1_row0 * B2[1,1]
        U[a1i1*m+a1j1, col] = ca1; U[a1i1*m+a1j2, col] = ca1
        V[b2j2*p+b2k2, col] = cb2
        W[c1k2*n+c1i1, col] = w_coeff_c1; W[c1k1*n+c1i1, col] = -w_coeff_c1
        W[c2k2*n+c2i1, col] = w_coeff_c2; W[c2k1*n+c2i1, col] = -w_coeff_c2
        col += 1

        # M6 = (A2[1,0]-A1[1,0]) * B1_row0
        U[a2i2*m+a2j1, col] = ca2; U[a1i2*m+a1j1, col] = -ca1
        V[b1j1*p+b1k1, col] = cb1; V[b1j1*p+b1k2, col] = cb1
        W[c1k2*n+c1i2, col] = w_coeff_c1
        W[c2k2*n+c2i2, col] = w_coeff_c2
        col += 1

        # M7 = (A1[0,1]-A2[0,1]) * B2_row1
        U[a1i1*m+a1j2, col] = ca1; U[a2i1*m+a2j2, col] = -ca2
        V[b2j2*p+b2k1, col] = cb2; V[b2j2*p+b2k2, col] = cb2
        W[c1k1*n+c1i1, col] = w_coeff_c1; W[c1k2*n+c1i2, col] = -w_coeff_c1
        W[c2k1*n+c2i1, col] = w_coeff_c2; W[c2k2*n+c2i2, col] = -w_coeff_c2
        col += 1

    # 4x4 Strassen: 7 top-level multiplications
    # M1 = (A00+A11)(B00+B11) -> C00(+1), C11(+1)
    strassen_4x4_subterm(A00, A11, B00, B11, C00, C11, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    # M2 = (A10+A11)B00 -> C10(+1)
    strassen_4x4_subterm(A10, A11, B00, B00, C10, C10, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0)
    # M3 = A00(B01-B11) -> C01(+1)
    strassen_4x4_subterm(A00, A00, B01, B11, C01, C01, 1.0, 0.0, 1.0, -1.0, 1.0, 0.0)
    # M4 = A11(B10-B00) -> C10(+1), C00(-1)
    strassen_4x4_subterm(A11, A11, B10, B00, C10, C00, 1.0, 0.0, 1.0, -1.0, 1.0, -1.0)
    # M5 = (A00+A01)B11 -> C01(+1), C00(-1)
    strassen_4x4_subterm(A00, A01, B11, B11, C01, C00, 1.0, 1.0, 1.0, 0.0, 1.0, -1.0)
    # M6 = (A10-A00)(B00+B01) -> C11(+1)
    strassen_4x4_subterm(A10, A00, B00, B01, C11, C11, 1.0, -1.0, 1.0, 1.0, 1.0, 0.0)
    # M7 = (A01-A11)(B10+B11) -> C00(+1), C11(-1)
    strassen_4x4_subterm(A01, A11, B10, B11, C00, C11, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0)

    # col should be 49 now
    # ---- Boundary terms (schoolbook) ----

    # C[i,4] += sum_{k=0}^{3} A[i,k]*B[k,4] for i=0..3 (4x4x1)
    for i in range(4):
        for k in range(4):
            U[i*m+k, col] = 1.0
            V[k*p+4, col] = 1.0
            W[4*n+i, col] = 1.0
            col += 1

    # C[4,j] += sum_{k=0}^{3} A[4,k]*B[k,j] for j=0..3 (1x4x4)
    for k in range(4):
        for j in range(4):
            U[4*m+k, col] = 1.0
            V[k*p+j, col] = 1.0
            W[j*n+4, col] = 1.0
            col += 1

    # C[4,4] += sum_{k=0}^{3} A[4,k]*B[k,4] (1x4x1)
    for k in range(4):
        U[4*m+k, col] = 1.0
        V[k*p+4, col] = 1.0
        W[4*n+4, col] = 1.0
        col += 1

    # C[i,4] += A[i,4]*B[4,4] for i=0..3 (4 terms)
    for i in range(4):
        U[i*m+4, col] = 1.0
        V[4*p+4, col] = 1.0
        W[4*n+i, col] = 1.0
        col += 1

    # C[4,j] += A[4,4]*B[4,j] for j=0..3 (4 terms)
    for j in range(4):
        U[4*m+4, col] = 1.0
        V[4*p+j, col] = 1.0
        W[j*n+4, col] = 1.0
        col += 1

    # C[4,4] += A[4,4]*B[4,4] (1 term)
    U[4*m+4, col] = 1.0
    V[4*p+4, col] = 1.0
    W[4*n+4, col] = 1.0
    col += 1

    # C[i,j] += A[i,4]*B[4,j] for i,j=0..3 (16 terms)
    for i in range(4):
        for j in range(4):
            U[i*m+4, col] = 1.0
            V[4*p+j, col] = 1.0
            W[j*n+i, col] = 1.0
            col += 1

    # Total: 49 + 16 + 16 + 4 + 4 + 4 + 1 + 16 = 110
    assert col == R, f"Expected rank {R}, got {col}"

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
