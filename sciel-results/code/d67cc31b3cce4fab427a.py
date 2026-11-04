"""Rewritten program: Strassen-based 2x3x7 matrix multiplication, rank 39."""
import time
import random

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 21, 14

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115.0

    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p  # 42
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    # Try Strassen-based rank-39 decomposition
    def strassen_decomp():
        # We decompose 2x3 x 3x7 multiplication.
        # Partition: A = [[A1, A2]] where A1 is 2x2, A2 is 2x1
        #             B = [[B1, B2]] where B1 is 3x6 (split into 2x2 blocks), B2 is 3x1
        # Actually, let's be more careful.
        #
        # A is 2x3: A[i,j] for i in 0,1; j in 0,1,2
        # B is 3x7: B[j,k] for j in 0,1,2; k in 0..6
        # C[i,k] = sum_j A[i,j] * B[j,k]
        #
        # Partition A into A1 (2x2, cols 0,1) and A2 (2x1, col 2)
        # Partition B into B_top (2x7, rows 0,1) and B_bot (1x7, row 2)
        #
        # C = A1 * B_top + A2 * B_bot
        #
        # For A1 * B_top (2x2 * 2x7): partition B_top into 2x2, 2x2, 2x2, 2x1
        # B_top = [B00, B01, B02, B03] where B00 is 2x2, B01 is 2x2, B02 is 2x2, B03 is 2x1
        #
        # A1 * B00: 2x2 * 2x2 -> use Strassen (rank 7)
        # A1 * B01: 2x2 * 2x2 -> use Strassen (rank 7)
        # A1 * B02: 2x2 * 2x2 -> use Strassen (rank 7)
        # A1 * B03: 2x2 * 2x1 -> schoolbook (rank 4)
        # A2 * B_bot: 2x1 * 1x7 -> schoolbook (rank 14)
        #
        # Total: 7 + 7 + 7 + 4 + 14 = 39

        rank = 39
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]

        col_offset = 0

        def add_strassen_2x2x2(u_off, v_off, w_off, c_off):
            """Add Strassen 2x2x2 decomposition starting at column c_off."""
            # Strassen for 2x2 * 2x2:
            # M1 = (A00 + A11)(B00 + B11)
            # M2 = (A10 + A11)B00
            # M3 = A00(B10 - B00)
            # M4 = A11(B01 - B10)
            # M5 = (A00 + A01)B11
            # M6 = (A10 - A00)(B00 + B10)
            # M7 = (A01 - A11)(B01 + B11)
            #
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            #
            # In tensor form, each Mi contributes to U, V, W.
            # U indices for A block: u_off + 0,1 (A00,A01), u_off+2,3 (A10,A11)
            # V indices for B block: v_off + 0,1 (B00,B01), v_off+2,3 (B10,B11)
            # W indices for C block: w_off + 0,1 (C00,C01), w_off+2,3 (C10,C11)

            # M1: (A00+A11)(B00+B11)
            U[u_off + 0][c_off + 0] += 1   # A00
            U[u_off + 3][c_off + 0] += 1   # A11
            V[v_off + 0][c_off + 0] += 1   # B00
            V[v_off + 3][c_off + 0] += 1   # B11
            W[w_off + 0][c_off + 0] += 1   # C00 += M1
            W[w_off + 3][c_off + 0] += 1   # C11 += M1

            # M2: (A10+A11)B00
            U[u_off + 2][c_off + 1] += 1   # A10
            U[u_off + 3][c_off + 1] += 1   # A11
            V[v_off + 0][c_off + 1] += 1   # B00
            W[w_off + 2][c_off + 1] += 1   # C10 += M2
            W[w_off + 3][c_off + 1] -= 1   # C11 -= M2

            # M3: A00(B10-B00)
            U[u_off + 0][c_off + 2] += 1   # A00
            V[v_off + 2][c_off + 2] += 1   # B10
            V[v_off + 0][c_off + 2] -= 1   # B00
            W[w_off + 1][c_off + 2] += 1   # C01 += M3
            W[w_off + 3][c_off + 2] += 1   # C11 += M3

            # M4: A11(B01-B10)
            U[u_off + 3][c_off + 3] += 1   # A11
            V[v_off + 1][c_off + 3] += 1   # B01
            V[v_off + 2][c_off + 3] -= 1   # B10
            W[w_off + 0][c_off + 3] += 1   # C00 += M4
            W[w_off + 2][c_off + 3] += 1   # C10 += M4

            # M5: (A00+A01)B11
            U[u_off + 0][c_off + 4] += 1   # A00
            U[u_off + 1][c_off + 4] += 1   # A01
            V[v_off + 3][c_off + 4] += 1   # B11
            W[w_off + 0][c_off + 4] -= 1   # C00 -= M5
            W[w_off + 1][c_off + 4] += 1   # C01 += M5

            # M6: (A10-A00)(B00+B10)
            U[u_off + 2][c_off + 5] += 1   # A10
            U[u_off + 0][c_off + 5] -= 1   # A00
            V[v_off + 0][c_off + 5] += 1   # B00
            V[v_off + 2][c_off + 5] += 1   # B10
            W[w_off + 3][c_off + 5] += 1   # C11 += M6

            # M7: (A01-A11)(B01+B11)
            U[u_off + 1][c_off + 6] += 1   # A01
            U[u_off + 3][c_off + 6] -= 1   # A11
            V[v_off + 1][c_off + 6] += 1   # B01
            V[v_off + 3][c_off + 6] += 1   # B11
            W[w_off + 0][c_off + 6] += 1   # C00 += M7
            W[w_off + 3][c_off + 6] += 1   # C11 += M7

        def add_schoolbook_2x2x1(u_off, v_off, w_off, c_off):
            """2x2 * 2x1 -> 2x1, schoolbook rank 4."""
            # C0 = A00*B0 + A01*B1
            # C1 = A10*B0 + A11*B1
            # U: A00, A01, A10, A11
            # V: B0, B1
            # W: C0, C1
            U[u_off + 0][c_off + 0] += 1  # A00
            V[v_off + 0][c_off + 0] += 1  # B0
            W[w_off + 0][c_off + 0] += 1  # C0

            U[u_off + 1][c_off + 1] += 1  # A01
            V[v_off + 1][c_off + 1] += 1  # B1
            W[w_off + 0][c_off + 1] += 1  # C0

            U[u_off + 2][c_off + 2] += 1  # A10
            V[v_off + 0][c_off + 2] += 1  # B0
            W[w_off + 1][c_off + 2] += 1  # C1

            U[u_off + 3][c_off + 3] += 1  # A11
            V[v_off + 1][c_off + 3] += 1  # B1
            W[w_off + 1][c_off + 3] += 1  # C1

        def add_schoolbook_2x1x7(u_off, v_off, w_off, c_off):
            """2x1 * 1x7 -> 2x7, schoolbook rank 14."""
            # C[i,k] = A[i,0] * B[0,k]
            for i in range(2):
                for k in range(7):
                    col = c_off + i * 7 + k
                    U[u_off + i][col] += 1   # A[i,0]
                    V[v_off + 0][col] += 1   # B[0,k] (all B entries are in row 0 of the 1x7 block)
                    W[w_off + k][col] += 1   # C[i,k]

        # Now assemble:
        # A block: A1 is rows 0,1 cols 0,1 of A -> U indices 0..3
        #          A2 is rows 0,1 col 2 of A -> U indices 4..5
        # B block: B_top rows 0,1, B_bot row 2
        #          B00: rows 0,1 cols 0,1 -> V indices 0..3
        #          B01: rows 0,1 cols 2,3 -> V indices 4..7
        #          B02: rows 0,1 cols 4,5 -> V indices 8..11
        #          B03: rows 0,1 col 6 -> V indices 12..13
        #          B_bot: row 2, cols 0..6 -> V indices 14..20
        # C block: C[i,k] -> W index k*n + i = k*2 + i
        #          C00: i=0,k=0 -> W[0]
        #          C01: i=0,k=1 -> W[2]... wait, W[k*n+i] = W[k*2+i]
        #          C[0,0]=W[0], C[0,1]=W[2], C[0,2]=W[4], C[0,3]=W[6], C[0,4]=W[8], C[0,5]=W[10], C[0,6]=W[12]
        #          C[1,0]=W[1], C[1,1]=W[3], etc.
        #
        # For the 2x2 C blocks from Strassen:
        #   Block (rows i0,i1 of C, cols k0,k0+1): W indices k0*2+i0, k0*2+i0+1, (k0+1)*2+i0, (k0+1)*2+i0+1
        #   For i in 0,1: w_off = k0*2 + i... but we have two i values.
        #   Actually the Strassen 2x2x2 operates on a 2x2 sub-block of C.
        #   The W indices for C rows {r0, r0+1} and cols {c0, c0+1} are:
        #     C[r0, c0] -> W[c0*n + r0]
        #     C[r0, c0+1] -> W[(c0+1)*n + r0]
        #     C[r0+1, c0] -> W[c0*n + r0+1]
        #     C[r0+1, c0+1] -> W[(c0+1)*n + r0+1]
        #   In the Strassen add function, w_off + 0,1,2,3 correspond to C00,C01,C10,C11
        #   So we need w_off such that:
        #     w_off+0 = c0*n + r0
        #     w_off+1 = (c0+1)*n + r0
        #     w_off+2 = c0*n + r0+1
        #     w_off+3 = (c0+1)*n + r0+1
        #   This is NOT contiguous in general. So we need a more careful mapping.

        # Let me redo this with explicit index mapping.
        # I'll build U, V, W with explicit entries.

        # Reset
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]

        col = 0

        # Helper: add a 2x2x2 Strassen block
        # A_sub: 2x2 submatrix of A with top-left at (a_row, a_col) in A's row/col indices
        # B_sub: 2x2 submatrix of B with top-left at (b_row, b_col) in B's row/col indices
        # C_sub: 2x2 submatrix of C with top-left at (c_row, c_col) in C's row/col indices
        def add_strassen(a_row, a_col, b_row, b_col, c_row, c_col, c_start):
            nonlocal col
            # A indices: (a_row, a_col), (a_row, a_col+1), (a_row+1, a_col), (a_row+1, a_col+1)
            # In U: A[i,j] is at U[i*m + j]
            a00_u = a_row * m + a_col
            a01_u = a_row * m + a_col + 1
            a10_u = (a_row + 1) * m + a_col
            a11_u = (a_row + 1) * m + a_col + 1

            # B indices: (b_row, b_col), (b_row, b_col+1), (b_row+1, b_col), (b_row+1, b_col+1)
            # In V: B[j,k] is at V[j*p + k]
            b00_v = b_row * p + b_col
            b01_v = b_row * p + b_col + 1
            b10_v = (b_row + 1) * p + b_col
            b11_v = (b_row + 1) * p + b_col + 1

            # C indices: (c_row, c_col), (c_row, c_col+1), (c_row+1, c_col), (c_row+1, c_col+1)
            # In W: C[i,k] is at W[k*n + i]
            c00_w = c_col * n + c_row
            c01_w = (c_col + 1) * n + c_row
            c10_w = c_col * n + (c_row + 1)
            c11_w = (c_col + 1) * n + (c_row + 1)

            # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
            c = c_start
            U[a00_u][c] += 1; U[a11_u][c] += 1
            V[b00_v][c] += 1; V[b11_v][c] += 1
            W[c00_w][c] += 1; W[c11_w][c] += 1
            c += 1

            # M2: (A10+A11)B00 -> C10 += M2, C11 -= M2
            U[a10_u][c] += 1; U[a11_u][c] += 1
            V[b00_v][c] += 1
            W[c10_w][c] += 1; W[c11_w][c] -= 1
            c += 1

            # M3: A00(B10-B00) -> C01 += M3, C11 += M3
            U[a00_u][c] += 1
            V[b10_v][c] += 1; V[b00_v][c] -= 1
            W[c01_w][c] += 1; W[c11_w][c] += 1
            c += 1

            # M4: A11(B01-B10) -> C00 += M4, C10 += M4
            U[a11_u][c] += 1
            V[b01_v][c] += 1; V[b10_v][c] -= 1
            W[c00_w][c] += 1; W[c10_w][c] += 1
            c += 1

            # M5: (A00+A01)B11 -> C00 -= M5, C01 += M5
            U[a00_u][c] += 1; U[a01_u][c] += 1
            V[b11_v][c] += 1
            W[c00_w][c] -= 1; W[c01_w][c] += 1
            c += 1

            # M6: (A10-A00)(B00+B10) -> C11 += M6
            U[a10_u][c] += 1; U[a00_u][c] -= 1
            V[b00_v][c] += 1; V[b10_v][c] += 1
            W[c11_w][c] += 1
            c += 1

            # M7: (A01-A11)(B01+B11) -> C00 += M7, C11 += M7
            U[a01_u][c] += 1; U[a11_u][c] -= 1
            V[b01_v][c] += 1; V[b11_v][c] += 1
            W[c00_w][c] += 1; W[c11_w][c] += 1
            c += 1

            col = c

        # Helper: add 2x2 * 2x1 schoolbook
        def add_schoolbook_2x2x1(a_row, a_col, b_row, b_col, c_row, c_col, c_start):
            nonlocal col
            # A: (a_row, a_col), (a_row, a_col+1), (a_row+1, a_col), (a_row+1, a_col+1)
            # B: (b_row, b_col), (b_row+1, b_col)
            # C: (c_row, c_col), (c_row+1, c_col)
            a00_u = a_row * m + a_col
            a01_u = a_row * m + a_col + 1
            a10_u = (a_row + 1) * m + a_col
            a11_u = (a_row + 1) * m + a_col + 1
            b0_v = b_row * p + b_col
            b1_v = (b_row + 1) * p + b_col
            c0_w = c_col * n + c_row
            c1_w = c_col * n + (c_row + 1)

            c = c_start
            # C0 = A00*B0 + A01*B1
            U[a00_u][c] += 1; V[b0_v][c] += 1; W[c0_w][c] += 1
            c += 1
            U[a01_u][c] += 1; V[b1_v][c] += 1; W[c0_w][c] += 1
            c += 1
            # C1 = A10*B0 + A11*B1
            U[a10_u][c] += 1; V[b0_v][c] += 1; W[c1_w][c] += 1
            c += 1
            U[a11_u][c] += 1; V[b1_v][c] += 1; W[c1_w][c] += 1
            c += 1

            col = c

        # Helper: add 2x1 * 1x7 schoolbook
        def add_schoolbook_2x1x7(a_row, a_col, b_row, b_col_start, c_row, c_col_start, c_start):
            nonlocal col
            # A: (a_row, a_col), (a_row+1, a_col)
            # B: (b_row, b_col_start), ..., (b_row, b_col_start+6)
            # C: (c_row, c_col_start), ..., (c_row, c_col_start+6), (c_row+1, c_col_start), ...
            a0_u = a_row * m + a_col
            a1_u = (a_row + 1) * m + a_col
            b_v = b_row * p + b_col_start
            c0_w_base = c_col_start * n + c_row
            c1_w_base = c_col_start * n + (c_row + 1)

            c = c_start
            for i in range(2):
                for k in range(7):
                    if i == 0:
                        U[a0_u][c] += 1
                        W[c0_w_base + k][c] += 1
                    else:
                        U[a1_u][c] += 1
                        W[c1_w_base + k][c] += 1
                    V[b_v + k][c] += 1
                    c += 1

            col = c

        # Now build the full decomposition:
        # A1 * B00 (2x2 * 2x2): A rows 0,1 cols 0,1; B rows 0,1 cols 0,1; C rows 0,1 cols 0,1
        add_strassen(0, 0, 0, 0, 0, 0, 0)  # uses cols 0-6

        # A1 * B01 (2x2 * 2x2): A rows 0,1 cols 0,1; B rows 0,1 cols 2,3; C rows 0,1 cols 2,3
        add_strassen(0, 0, 0, 2, 0, 2, 7)  # uses cols 7-13

        # A1 * B02 (2x2 * 2x2): A rows 0,1 cols 0,1; B rows 0,1 cols 4,5; C rows 0,1 cols 4,5
        add_strassen(0, 0, 0, 4, 0, 4, 14)  # uses cols 14-20

        # A1 * B03 (2x2 * 2x1): A rows 0,1 cols 0,1; B rows 0,1 col 6; C rows 0,1 col 6
        add_schoolbook_2x2x1(0, 0, 0, 6, 0, 6, 21)  # uses cols 21-24

        # A2 * B_bot (2x1 * 1x7): A rows 0,1 col 2; B row 2 cols 0..6; C rows 0,1 cols 0..6
        add_schoolbook_2x1x7(0, 2, 2, 0, 0, 0, 25)  # uses cols 25-38

        result = {"U": U, "V": V, "W": W}

        if verify(U, V, W, rank):
            return result
        return None

    best = None
    best_rank = n * m * p

    # Try Strassen
    if time.time() < deadline - 5:
        s = strassen_decomp()
        if s is not None:
            best = s
            best_rank = 39

    if best is None:
        best = schoolbook()

    return best
# EVOLVE-BLOCK-END
