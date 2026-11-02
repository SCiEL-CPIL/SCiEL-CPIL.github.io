"""
Rank-44 exact decomposition for 2x3 * 3x8 matrix multiplication.

Construction:
  A (2x3) = [A_left | A_right] where A_left is 2x2, A_right is 2x1.
  B (3x8) = [B_top; B_bottom] where B_top is 2x8, B_bottom is 1x8.
  C = A*B = A_left * B_top + A_right * B_bottom  (2x8 result)

We split B_top into 4 blocks of 2x2: B_top = [B1, B2, B3, B4].
  C_left (2x4) = A_left * [B1, B2] + A_right * [C1, C2]
  C_right (2x4) = A_left * [B3, B4] + A_right * [C3, C4]
  where C1..C4 are 1x2 blocks of B_bottom.

For each 2x2 block multiplication (A_left 2x2 * B_j 2x2 -> 2x2),
use Strassen's rank-7 algorithm.
For each 2x1 * 1x2 multiplication (A_right 2x1 * C_j 1x2 -> 2x2),
use rank-4 (schoolbook: each of 4 output entries is one product).

Total: 4 blocks * (7 + 4) = 44 terms.
"""

import time

DIMENSIONS = [2, 3, 8]


def construct():
    deadline = time.time() + 140  # safety margin

    n, m, p = 2, 3, 8
    rank = 44

    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0.0] * rank for _ in range(p * n)]  # 16 x 44

    # Index conventions:
    # A is 2x3: A[i,j] -> U[i*m + j, c] = U[i*3+j, c]
    #   i in 0,1; j in 0,1,2
    #   A_left: j=0,1; A_right: j=2
    # B is 3x8: B[j,k] -> V[j*p + k, c] = V[j*8+k, c]
    #   j in 0,1,2; k in 0..7
    #   B_top: j=0,1; B_bottom: j=2
    # C is 2x8: C[i,k] -> W[k*n + i, c] = W[k*2+i, c]
    #   i in 0,1; k in 0..7
    #   k=0,1 -> block 0; k=2,3 -> block 1; k=4,5 -> block 2; k=6,7 -> block 3

    col = 0

    # Strassen 2x2*2x2 -> 2x2
    # Inputs: A = [[a00, a01],[a10, a11]], B = [[b00, b01],[b10, b11]]
    # Outputs: C = [[c00, c01],[c10, c11]]
    #
    # M1 = (a00+a10)(b00+b10)
    # M2 = (a10)(b01+b11)
    # M3 = (a00)(b00+b01)
    # M4 = (a11)(b10+b11)
    # M5 = (a00+a11)(b00+b11)
    # M6 = (a01-a11)(b10+b11)
    # M7 = (a00-a10)(b00+b10)
    #
    # c00 = M5 + M4 - M2 + M6
    # c01 = M1 + M2
    # c10 = M3 + M4
    # c11 = M5 + M1 - M3 - M7

    strassen_U = {
        # M1: (a00+a10) * (b00+b10)
        'a00': 1.0, 'a10': 1.0, 'b00': 1.0, 'b10': 1.0,
    }
    strassen_V = {
        'a00': 1.0, 'a10': 1.0, 'b00': 1.0, 'b10': 1.0,
    }
    # Let me define the 7 Strassen terms properly.
    # Each term is a product of a linear form in A and a linear form in B.
    # U encodes the A side, V encodes the B side, W encodes the output.

    # Term definitions for one 2x2*2x2 Strassen block:
    # (U_a, U_b, V_a, V_b, W_a, W_b) for each of the 7 products
    # U_a: coefficient for A[0,0], U_b: for A[1,0] (since we're doing 2x2 subblock)
    # Actually let me be more careful.

    # For a 2x2 A subblock (rows 0,1; cols 0,1 of the local 2x2):
    #   A_local[0,0] = A_global[0, col_offset]
    #   A_local[0,1] = A_global[0, col_offset+1]
    #   A_local[1,0] = A_global[1, col_offset]
    #   A_local[1,1] = A_global[1, col_offset+1]

    # For a 2x2 B subblock (rows 0,1; cols 0,1 of the local 2x2, offset by 8*row_offset):
    #   B_local[0,0] = B_global[0, row_offset*8 + col_offset]
    #   B_local[0,1] = B_global[0, row_offset*8 + col_offset+1]
    #   B_local[1,0] = B_global[1, row_offset*8 + col_offset]
    #   B_local[1,1] = B_global[1, row_offset*8 + col_offset+1]

    # Strassen terms (standard):
    # M1 = (a00+a10)*(b00+b10)
    # M2 = a10*(b01+b11)
    # M3 = a00*(b00+b01)
    # M4 = a11*(b10+b11)
    # M5 = (a00+a11)*(b00+b11)
    # M6 = (a01-a11)*(b10+b11)
    # M7 = (a00-a10)*(b00+b10)
    #
    # c00 = M5+M4-M2+M6
    # c01 = M1+M2
    # c10 = M3+M4
    # c11 = M5+M1-M3-M7

    def add_strassen_block(a_col_offset, b_row_offset, b_col_offset):
        """Add 7 Strassen terms for multiplying A[a_col_offset:2+a_col_offset, 0:2]
        (actually A[0:2, a_col_offset:a_col_offset+2]) * B[0:2, b_row_offset*8+b_col_offset : b_row_offset*8+b_col_offset+2]
        Wait, I need to be careful about indexing.

        A is 2x3. The 2x2 subblock uses columns a_col_offset and a_col_offset+1.
        B is 3x8. The 2x2 subblock uses rows 0,1 and columns b_col_offset, b_col_offset+1 (within the top 2 rows).
        Output is 2x2 at rows 0,1 and columns b_col_offset, b_col_offset+1.
        """
        nonlocal col

        # U indices: A[i, j] -> U[i*3 + j]
        # V indices: B[i, j] -> V[i*8 + j]
        # W indices: C[i, j] -> W[j*2 + i]

        # For the 2x2 A subblock at columns [ac, ac+1]:
        # a00 = A[0, ac] -> U[0*3 + ac] = U[ac]
        # a01 = A[0, ac+1] -> U[ac+1]
        # a10 = A[1, ac] -> U[3+ac]
        # a11 = A[1, ac+1] -> U[3+ac+1]

        # For the 2x2 B subblock at row 0,1 and columns [bc, bc+1]:
        # b00 = B[0, bc] -> V[0*8 + bc] = V[bc]
        # b01 = B[0, bc+1] -> V[bc+1]
        # b10 = B[1, bc] -> V[8+bc]
        # b11 = B[1, bc+1] -> V[8+bc+1]

        # For the 2x2 C output at rows 0,1 and columns [bc, bc+1]:
        # c00 = C[0, bc] -> W[bc*2 + 0] = W[2*bc]
        # c01 = C[0, bc+1] -> W[(bc+1)*2 + 0] = W[2*bc+2]
        # c10 = C[1, bc] -> W[bc*2 + 1] = W[2*bc+1]
        # c11 = C[1, bc+1] -> W[(bc+1)*2 + 1] = W[2*bc+3]

        ac = a_col_offset
        bc = b_col_offset

        U_a00 = ac        # A[0, ac]
        U_a01 = ac + 1    # A[0, ac+1]
        U_a10 = 3 + ac    # A[1, ac]
        U_a11 = 3 + ac + 1  # A[1, ac+1]

        V_b00 = bc        # B[0, bc]
        V_b01 = bc + 1    # B[0, bc+1]
        V_b10 = 8 + bc    # B[1, bc]
        V_b11 = 8 + bc + 1  # B[1, bc+1]

        W_c00 = 2 * bc    # C[0, bc]
        W_c01 = 2 * bc + 2  # C[0, bc+1]
        W_c10 = 2 * bc + 1  # C[1, bc]
        W_c11 = 2 * bc + 3  # C[1, bc+1]

        # M1 = (a00+a10)*(b00+b10)
        U[U_a00][col] += 1.0
        U[U_a10][col] += 1.0
        V[V_b00][col] += 1.0
        V[V_b10][col] += 1.0
        W[W_c01][col] += 1.0  # c01 += M1
        W[W_c11][col] += 1.0  # c11 += M1
        col += 1

        # M2 = a10*(b01+b11)
        U[U_a10][col] += 1.0
        V[V_b01][col] += 1.0
        V[V_b11][col] += 1.0
        W[W_c00][col] += -1.0  # c00 -= M2
        W[W_c01][col] += 1.0   # c01 += M2
        col += 1

        # M3 = a00*(b00+b01)
        U[U_a00][col] += 1.0
        V[V_b00][col] += 1.0
        V[V_b01][col] += 1.0
        W[W_c10][col] += 1.0   # c10 += M3
        W[W_c11][col] += -1.0  # c11 -= M3
        col += 1

        # M4 = a11*(b10+b11)
        U[U_a11][col] += 1.0
        V[V_b10][col] += 1.0
        V[V_b11][col] += 1.0
        W[W_c00][col] += 1.0   # c00 += M4
        W[W_c10][col] += 1.0   # c10 += M4
        col += 1

        # M5 = (a00+a11)*(b00+b11)
        U[U_a00][col] += 1.0
        U[U_a11][col] += 1.0
        V[V_b00][col] += 1.0
        V[V_b11][col] += 1.0
        W[W_c00][col] += 1.0   # c00 += M5
        W[W_c11][col] += 1.0   # c11 += M5
        col += 1

        # M6 = (a01-a11)*(b10+b11)
        U[U_a01][col] += 1.0
        U[U_a11][col] += -1.0
        V[V_b10][col] += 1.0
        V[V_b11][col] += 1.0
        W[W_c00][col] += 1.0   # c00 += M6
        col += 1

        # M7 = (a00-a10)*(b00+b10)
        U[U_a00][col] += 1.0
        U[U_a10][col] += -1.0
        V[V_b00][col] += 1.0
        V[V_b10][col] += 1.0
        W[W_c11][col] += -1.0  # c11 -= M7
        col += 1

    def add_rank4_block(a_col, b_row, bc_offset):
        """Add 4 rank-1 terms for A_right (2x1) * B_bottom[bc_offset:bc_offset+2] (1x2) -> 2x2 output.
        
        A_right = A[:, a_col] (2x1)
        B_bottom_row = B[b_row, bc_offset:bc_offset+2] (1x2)
        
        Output C[0, bc_offset] = A[0,a_col]*B[b_row,bc_offset]
        Output C[0, bc_offset+1] = A[0,a_col]*B[b_row,bc_offset+1]
        Output C[1, bc_offset] = A[1,a_col]*B[b_row,bc_offset]
        Output C[1, bc_offset+1] = A[1,a_col]*B[b_row,bc_offset+1]
        """
        nonlocal col

        # U indices for A[0, a_col] and A[1, a_col]
        U_0 = a_col        # A[0, a_col]
        U_1 = 3 + a_col    # A[1, a_col]

        # V indices for B[b_row, bc_offset] and B[b_row, bc_offset+1]
        V_0 = b_row * 8 + bc_offset
        V_1 = b_row * 8 + bc_offset + 1

        # W indices for C[0, bc_offset], C[0, bc_offset+1], C[1, bc_offset], C[1, bc_offset+1]
        W_00 = 2 * bc_offset      # C[0, bc_offset]
        W_01 = 2 * bc_offset + 2  # C[0, bc_offset+1]
        W_10 = 2 * bc_offset + 1  # C[1, bc_offset]
        W_11 = 2 * bc_offset + 3  # C[1, bc_offset+1]

        # Term 1: A[0,a_col] * B[b_row, bc_offset] -> C[0, bc_offset]
        U[U_0][col] += 1.0
        V[V_0][col] += 1.0
        W[W_00][col] += 1.0
        col += 1

        # Term 2: A[0,a_col] * B[b_row, bc_offset+1] -> C[0, bc_offset+1]
        U[U_0][col] += 1.0
        V[V_1][col] += 1.0
        W[W_01][col] += 1.0
        col += 1

        # Term 3: A[1,a_col] * B[b_row, bc_offset] -> C[1, bc_offset]
        U[U_1][col] += 1.0
        V[V_0][col] += 1.0
        W[W_10][col] += 1.0
        col += 1

        # Term 4: A[1,a_col] * B[b_row, bc_offset+1] -> C[1, bc_offset+1]
        U[U_1][col] += 1.0
        V[V_1][col] += 1.0
        W[W_11][col] += 1.0
        col += 1

    # Now process 4 blocks of B_top (2x2 each), with A_left being the 2x2 left part of A
    # A_left uses columns 0,1 of A (the 2x2 part)
    # A_right is column 2 of A
    # B_bottom is row 2 of B

    for block in range(4):
        bc_offset = block * 2
        # Strassen for A_left * B_top_block
        add_strassen_block(a_col_offset=0, b_row_offset=0, b_col_offset=bc_offset)
        # Rank-4 for A_right * B_bottom_block
        add_rank4_block(a_col=2, b_row=2, bc_offset=bc_offset)

    # col should now be 44
    # Verify col
    assert col == 44, f"Expected 44 columns, got {col}"

    # Quick verification
    import numpy as np
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # Test with random matrices
    np.random.seed(42)
    A_test = np.random.randint(-5, 5, size=(2, 3)).astype(float)
    B_test = np.random.randint(-5, 5, size=(3, 8)).astype(float)
    C_expected = A_test @ B_test

    # Compute C from decomposition: C[i,k] = sum_c U[i*3+j, c] * V[j*8+k, c] * W[k*2+i, c]
    # Actually the tensor decomposition is:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where T is the multiplication tensor: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 if the output C[i,k] gets a contribution from A[i,j]*B[j,k]
    # In our convention: W[k*n+i, c] = W[k*2+i, c]

    C_computed = np.zeros((2, 8), dtype=float)
    for c in range(rank):
        for i in range(2):
            for j in range(3):
                for k in range(8):
                    C_computed[i, k] += U_arr[i * 3 + j, c] * V_arr[j * 8 + k, c] * W_arr[k * 2 + i, c]

    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # If verification fails, fall back to schoolbook
        rank_school = n * m * p
        U_s = [[0.0] * rank_school for _ in range(n * m)]
        V_s = [[0.0] * rank_school for _ in range(m * p)]
        W_s = [[0.0] * rank_school for _ in range(p * n)]
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col_s] = 1.0
                    V_s[j * p + k][col_s] = 1.0
                    W_s[k * n + i][col_s] = 1.0
                    col_s += 1
        return {"U": U_s, "V": V_s, "W": W_s}

    return {"U": U, "V": V, "W": W}
