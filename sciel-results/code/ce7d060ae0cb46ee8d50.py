import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Block structure: A is 2x2 blocks of size (1,2); B is 2x2 blocks of size (2,3); C is 2x2 blocks of size (1,3)
    # n_blocks = 2, m_blocks = 2, p_blocks = 2
    # block_a = (1,2), block_b = (2,3), block_c = (1,3)

    # Scalar-level rank-1 tensors for multiplying a 1x2 by a 2x3 (rank 6)
    # A_block is 1x2 (indices a0, a1), B_block is 2x3 (indices b00,b01,b02,b10,b11,b12), C_block is 1x3 (indices c0,c1,c2)
    # C = A_block @ B_block
    # C[0] = a0*b00 + a1*b10
    # C[1] = a0*b01 + a1*b11
    # C[2] = a0*b02 + a1*b12
    # 6 rank-1 terms:
    # Term 0: a0 * b00 * c0
    # Term 1: a1 * b10 * c0
    # Term 2: a0 * b01 * c1
    # Term 3: a1 * b11 * c1
    # Term 4: a0 * b02 * c2
    # Term 5: a1 * b12 * c2

    # Strassen's 7 steps for 2x2 block matrices:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # Each Strassen step involves multiplying a (1x2) block by a (2x3) block.
    # The "A-side" of each step is a linear combination of A-blocks (each 1x2).
    # The "B-side" of each step is a linear combination of B-blocks (each 2x3).
    # The "C-side" contribution is a linear combination of C-blocks (each 1x3).

    # We'll build U, V, W by composing:
    # For each Strassen step s (7 total) and each scalar rank-1 term t (6 total):
    #   U column: linear combination of A entries (via A-side coefficients of step s, times scalar a-coeff of term t)
    #   V column: linear combination of B entries (via B-side coefficients of step s, times scalar b-coeff of term t)
    #   W column: linear combination of C entries (via C-side coefficients of step s, times scalar c-coeff of term t)

    # Define the scalar-level rank-1 tensors for 1x2 * 2x3:
    # Each term: (a_coeffs, b_coeffs, c_coeffs) where a_coeffs is 1x2, b_coeffs is 2x3, c_coeffs is 1x3
    scalar_terms = [
        ([1, 0], [[1, 0, 0], [0, 0, 0]], [1, 0, 0]),  # a0 * b00 * c0
        ([0, 1], [[0, 0, 0], [1, 0, 0]], [1, 0, 0]),  # a1 * b10 * c0
        ([1, 0], [[0, 1, 0], [0, 0, 0]], [0, 1, 0]),  # a0 * b01 * c1
        ([0, 1], [[0, 0, 0], [0, 1, 0]], [0, 1, 0]),  # a1 * b11 * c1
        ([1, 0], [[0, 0, 1], [0, 0, 0]], [0, 0, 1]),  # a0 * b02 * c2
        ([0, 1], [[0, 0, 0], [0, 0, 1]], [0, 0, 1]),  # a1 * b12 * c2
    ]

    # Define Strassen steps: (A_side, B_side, C_side)
    # A_side: 2x2 matrix of coefficients for A-blocks A11, A12, A21, A22 (each 1x2)
    # B_side: 2x2 matrix of coefficients for B-blocks B11, B12, B21, B22 (each 2x3)
    # C_side: 2x2 matrix of coefficients for C-blocks C11, C12, C21, C22 (each 1x3)

    # Block indices: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
    # B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1)
    # C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1)

    def block_coeff_matrix(rows, cols):
        """Create a zero 2x2 block coefficient matrix."""
        return [[0.0]*2 for _ in range(2)]

    # Step 1: M1 = (A11 + A22) * (B11 + B22)
    A1 = block_coeff_matrix(2, 2)
    A1[0][0] = 1.0  # A11
    A1[1][1] = 1.0  # A22
    B1 = block_coeff_matrix(2, 2)
    B1[0][0] = 1.0  # B11
    B1[1][1] = 1.0  # B22
    C1 = block_coeff_matrix(2, 2)
    C1[0][0] = 1.0  # C11
    C1[1][1] = 1.0  # C22

    # Step 2: M2 = (A21 + A22) * B11
    A2 = block_coeff_matrix(2, 2)
    A2[1][0] = 1.0  # A21
    A2[1][1] = 1.0  # A22
    B2 = block_coeff_matrix(2, 2)
    B2[0][0] = 1.0  # B11
    C2 = block_coeff_matrix(2, 2)
    C2[1][0] = 1.0  # C21

    # Step 3: M3 = A11 * (B12 - B22)
    A3 = block_coeff_matrix(2, 2)
    A3[0][0] = 1.0  # A11
    B3 = block_coeff_matrix(2, 2)
    B3[0][1] = 1.0  # B12
    B3[1][1] = -1.0 # B22
    C3 = block_coeff_matrix(2, 2)
    C3[0][1] = 1.0  # C12

    # Step 4: M4 = A22 * (B21 - B11)
    A4 = block_coeff_matrix(2, 2)
    A4[1][1] = 1.0  # A22
    B4 = block_coeff_matrix(2, 2)
    B4[1][0] = 1.0  # B21
    B4[0][0] = -1.0 # B11
    C4 = block_coeff_matrix(2, 2)
    C4[0][0] = 1.0  # C11
    C4[1][0] = 1.0  # C21

    # Step 5: M5 = (A11 + A12) * B22
    A5 = block_coeff_matrix(2, 2)
    A5[0][0] = 1.0  # A11
    A5[0][1] = 1.0  # A12
    B5 = block_coeff_matrix(2, 2)
    B5[1][1] = 1.0  # B22
    C5 = block_coeff_matrix(2, 2)
    C5[0][0] = -1.0 # C11
    C5[0][1] = 1.0  # C12

    # Step 6: M6 = (A21 - A11) * (B11 + B12)
    A6 = block_coeff_matrix(2, 2)
    A6[1][0] = 1.0  # A21
    A6[0][0] = -1.0 # A11
    B6 = block_coeff_matrix(2, 2)
    B6[0][0] = 1.0  # B11
    B6[0][1] = 1.0  # B12
    C6 = block_coeff_matrix(2, 2)
    C6[1][1] = 1.0  # C22

    # Step 7: M7 = (A12 - A22) * (B21 + B22)
    A7 = block_coeff_matrix(2, 2)
    A7[0][1] = 1.0  # A12
    A7[1][1] = -1.0 # A22
    B7 = block_coeff_matrix(2, 2)
    B7[1][0] = 1.0  # B21
    B7[1][1] = 1.0  # B22
    C7 = block_coeff_matrix(2, 2)
    C7[0][0] = 1.0  # C11

    strassen_steps = [
        (A1, B1, C1),
        (A2, B2, C2),
        (A3, B3, C3),
        (A4, B4, C4),
        (A5, B5, C5),
        (A6, B6, C6),
        (A7, B7, C7),
    ]

    rank = 7 * 6  # 42

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0
    for A_side, B_side, C_side in strassen_steps:
        for a_coeffs, b_coeffs, c_coeffs in scalar_terms:
            # a_coeffs: 1x2 -> [a0, a1]
            # b_coeffs: 2x3 -> [[b00,b01,b02],[b10,b11,b12]]
            # c_coeffs: 1x3 -> [c0, c1, c2]

            # For each A-block (i,j) with coefficient A_side[i][j]:
            #   The block is 1x2, so entries are A[i*2 + 0], A[i*2 + 1] (since m=4, each row has 4 entries, blocks of width 2)
            #   Wait: A is 2x4. A[i, j] where i in [0,2), j in [0,4).
            #   Block A11 = A[0, 0:2], A12 = A[0, 2:4], A21 = A[1, 0:2], A22 = A[1, 2:4]
            #   So block (i,j) has entries A[i, j*2] and A[i, j*2+1]
            #   In flattened A (row-major): index = i*4 + j*2 + offset

            for i in range(2):
                for j in range(2):
                    coeff = A_side[i][j]
                    if coeff == 0:
                        continue
                    # Block (i,j) entries: A[i, j*2] and A[i, j*2+1]
                    # Flattened indices: i*m + j*2 and i*m + j*2 + 1 (m=4)
                    a0_idx = i * m + j * 2
                    a1_idx = i * m + j * 2 + 1
                    U[a0_idx, col] += coeff * a_coeffs[0]
                    U[a1_idx, col] += coeff * a_coeffs[1]

            # For each B-block (i,j) with coefficient B_side[i][j]:
            #   Block is 2x3, so entries are B[i*2 + r, j*3 + c] for r in [0,2), c in [0,3)
            #   B is 4x6. B[i, j] where i in [0,4), j in [0,6).
            #   Block B11 = B[0:2, 0:3], B12 = B[0:2, 3:6], B21 = B[2:4, 0:3], B22 = B[2:4, 3:6]
            #   So block (i,j) has entries B[i*2 + r, j*3 + c]
            #   In flattened B (row-major): index = (i*2 + r)*6 + (j*3 + c)

            for i in range(2):
                for j in range(2):
                    coeff = B_side[i][j]
                    if coeff == 0:
                        continue
                    # Block (i,j) entries: B[i*2 + r, j*3 + c]
                    for r in range(2):
                        for c in range(3):
                            b_idx = (i * 2 + r) * p + (j * 3 + c)
                            V[b_idx, col] += coeff * b_coeffs[r][c]

            # For each C-block (i,j) with coefficient C_side[i][j]:
            #   Block is 1x3, so entries are C[i, j*3 + c] for c in [0,3)
            #   C is 2x6. C[i, j] where i in [0,2), j in [0,6).
            #   Block C11 = C[0, 0:3], C12 = C[0, 3:6], C21 = C[1, 0:3], C22 = C[1, 3:6]
            #   So block (i,j) has entries C[i, j*3 + c]
            #   In flattened C (row-major): index = i*6 + j*3 + c
            #   But W uses trace convention: W[k*n + i, r] where k is col of C, i is row of C
            #   So C[i, k] corresponds to W[k*n + i]

            for i in range(2):
                for j in range(2):
                    coeff = C_side[i][j]
                    if coeff == 0:
                        continue
                    # Block (i,j) entries: C[i, j*3 + c]
                    for c in range(3):
                        k = j * 3 + c  # column in C
                        w_idx = k * n + i  # trace convention
                        W[w_idx, col] += coeff * c_coeffs[c]

            col += 1

    # Verification
    if time.time() < DEADLINE - 10:
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_true = A @ B

        A_flat = A.flatten()
        B_flat = B.flatten()

        u_vals = U @ A_flat
        v_vals = V @ B_flat
        prod_vals = u_vals * v_vals

        W_reshaped = W.reshape(p, n, rank).transpose(1, 0, 2)
        C_decomp = np.sum(W_reshaped * prod_vals[None, None, :], axis=2)

        if not np.allclose(C_true, C_decomp):
            raise ValueError("Decomposition verification failed")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result["U"][0]))

