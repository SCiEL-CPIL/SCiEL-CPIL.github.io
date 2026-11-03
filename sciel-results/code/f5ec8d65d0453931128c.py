"""Winograd Fourier-based decomposition for 2x4x5 matrix multiplication."""
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    # Output C is 2x5, indexed as W[k*n + i, r] where k is column, i is row
    # U is (n*m, rank) = (8, rank), indexed by (i, j) -> i*m + j
    # V is (m*p, rank) = (20, rank), indexed by (j, k) -> j*p + k
    # W is (p*n, rank) = (10, rank), indexed by (k, i) -> k*n + i

    # We use a block decomposition:
    # Split A into two 2x2 blocks: A_L (cols 0,1) and A_R (cols 2,3)
    # Split B into two 2x5 blocks: B_T (rows 0,1) and B_B (rows 2,3)
    # C = A_L * B_T + A_R * B_B
    #
    # Each term is 2x2 * 2x5.
    # For 2x2 * 2x5, we use Winograd with F2 for the 2x2 part and handle the 5 columns.
    # Actually, let's think of this as: for each of the 5 columns of B, we have a 2x2 * 2x1.
    # But that's not efficient.
    #
    # Better: 2x2 * 2x5 can be viewed as 5 separate 2x2 * 2x1 multiplications, but we can
    # use the same Strassen/Wino coefficients for the 2x2 part and apply them to each column.
    # That would be 7 * 5 = 35 per block, total 70. Too many.
    #
    # Alternative: Use the fact that 2x2 * 2x5 = (2x2 * 2x2) + (2x2 * 2x2) + (2x2 * 2x1)
    # Split the 5 columns into 2+2+1.
    # 2x2 * 2x2: rank 7 (Strassen)
    # 2x2 * 2x1: rank 4 (schoolbook)
    # Total per 2x2 * 2x5: 7 + 7 + 4 = 18
    # Two blocks: 36. Same as current.
    #
    # Let's try a different approach: use rank-6 for 2x2 * 2x1 (Winograd F2 * F1).
    # Actually, 2x2 * 2x1 is just a 2x2 matrix times a 2x1 vector.
    # C_0 = A_00*B_0 + A_01*B_1
    # C_1 = A_10*B_0 + A_11*B_1
    # This is 4 rank-1 terms. Can we do it in 3? No, the rank is 4.
    #
    # Let's try to use a combined Winograd approach for 2x2 * 2x5.
    # Use F2 for the 2x2 part (Strassen-like) and handle the 5 columns via polynomial interpolation.
    #
    # Actually, let's just use the standard block approach but with a better 2x2 * 2x2 decomposition.
    # Strassen rank 7 is optimal for 2x2 * 2x2 over reals.
    #
    # Let's try to reduce the 2x2 * 2x1 part. It's 4 terms. Can we share?
    # If we process all 5 columns of B together, we have 2x2 * 2x5.
    # The tensor is [2,2,2,5] essentially (two A dims, two B dims, 5 output dims).
    #
    # Let me just implement the rank-36 solution correctly first, then try to improve.
    # The current code has bugs in the add_schoolbook_2x2_1 function.
    # Let me rewrite it cleanly.

    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = [0]  # Use list to allow modification in nested function

    def add_rank1_term(u_dict, v_dict, w_dict):
        if col[0] >= rank:
            raise ValueError("Rank exceeded")
        c = col[0]
        for idx, val in u_dict.items():
            U[idx, c] = val
        for idx, val in v_dict.items():
            V[idx, c] = val
        for idx, val in w_dict.items():
            W[idx, c] = val
        col[0] += 1

    def add_strassen_2x2(a_i0, a_j0, b_i0, b_j0, c_i0, c_j0):
        """Add Strassen decomposition for A[2x2] * B[2x2] -> C[2x2]
        a_i0, a_j0: top-left corner of A block (i, j)
        b_i0, b_j0: top-left corner of B block (i, j)
        c_i0, c_j0: top-left corner of C block (i, j) where C is indexed as (k, i) -> k*n+i
        """
        def a_idx(i, j):
            return (a_i0 + i) * m + (a_j0 + j)
        def b_idx(i, j):
            return (b_i0 + i) * p + (b_j0 + j)
        def c_idx(k, i):
            return (c_j0 + k) * n + (c_i0 + i)

        # M1 = (A00+A11)(B00+B11)
        u = {a_idx(0,0): 1, a_idx(1,1): 1}
        v = {b_idx(0,0): 1, b_idx(1,1): 1}
        w = {c_idx(0,0): 1, c_idx(1,1): 1}
        add_rank1_term(u, v, w)

        # M2 = (A10+A11)B00
        u = {a_idx(1,0): 1, a_idx(1,1): 1}
        v = {b_idx(0,0): 1}
        w = {c_idx(0,1): 1, c_idx(1,1): -1}
        add_rank1_term(u, v, w)

        # M3 = A00(B01-B11)
        u = {a_idx(0,0): 1}
        v = {b_idx(0,1): 1, b_idx(1,1): -1}
        w = {c_idx(1,0): 1, c_idx(1,1): 1}
        add_rank1_term(u, v, w)

        # M4 = A11(B10-B00)
        u = {a_idx(1,1): 1}
        v = {b_idx(1,0): 1, b_idx(0,0): -1}
        w = {c_idx(0,0): 1, c_idx(0,1): 1}
        add_rank1_term(u, v, w)

        # M5 = (A00+A01)B11
        u = {a_idx(0,0): 1, a_idx(0,1): 1}
        v = {b_idx(1,1): 1}
        w = {c_idx(0,0): -1, c_idx(1,0): 1}
        add_rank1_term(u, v, w)

        # M6 = (A10-A00)(B00+B01)
        u = {a_idx(1,0): 1, a_idx(0,0): -1}
        v = {b_idx(0,0): 1, b_idx(0,1): 1}
        w = {c_idx(1,1): 1}
        add_rank1_term(u, v, w)

        # M7 = (A01-A11)(B10+B11)
        u = {a_idx(0,1): 1, a_idx(1,1): -1}
        v = {b_idx(1,0): 1, b_idx(1,1): 1}
        w = {c_idx(0,0): 1}
        add_rank1_term(u, v, w)

    def add_schoolbook_2x2_1(a_i0, a_j0, b_i0, b_j0, c_i0, c_j0):
        """Add schoolbook for A[2x2] * B[2x1] -> C[2x1]
        B is 2 rows, 1 column. C is 2 rows, 1 column.
        """
        def a_idx(i, j):
            return (a_i0 + i) * m + (a_j0 + j)
        def b_idx(i, j):
            return (b_i0 + i) * p + (b_j0 + j)
        def c_idx(k, i):
            return (c_j0 + k) * n + (c_i0 + i)

        # C_0 = A00*B0 + A01*B1
        # C_1 = A10*B0 + A11*B1
        # 4 rank-1 terms
        add_rank1_term({a_idx(0,0): 1}, {b_idx(0,0): 1}, {c_idx(0,0): 1})
        add_rank1_term({a_idx(0,1): 1}, {b_idx(1,0): 1}, {c_idx(0,0): 1})
        add_rank1_term({a_idx(1,0): 1}, {b_idx(0,0): 1}, {c_idx(0,1): 1})
        add_rank1_term({a_idx(1,1): 1}, {b_idx(1,0): 1}, {c_idx(0,1): 1})

    # Block 1: A_L (i=0,1; j=0,1) * B_T (i=0,1; j=0,4)
    # Split B_T into B_T0 (j=0,1), B_T1 (j=2,3), B_T2 (j=4)
    add_strassen_2x2(0, 0, 0, 0, 0, 0)   # A_L * B_T0
    add_strassen_2x2(0, 0, 0, 2, 0, 2)   # A_L * B_T1
    add_schoolbook_2x2_1(0, 0, 0, 4, 0, 4)  # A_L * B_T2

    # Block 2: A_R (i=0,1; j=2,3) * B_B (i=2,3; j=0,4)
    # Split B_B into B_B0 (j=0,1), B_B1 (j=2,3), B_B2 (j=4)
    add_strassen_2x2(0, 2, 2, 0, 0, 0)   # A_R * B_B0
    add_strassen_2x2(0, 2, 2, 2, 0, 2)   # A_R * B_B1
    add_schoolbook_2x2_1(0, 2, 2, 4, 0, 4)  # A_R * B_B2

    # Verify we used exactly 'rank' columns
    assert col[0] == rank, f"Used {col[0]} columns, expected {rank}"

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
