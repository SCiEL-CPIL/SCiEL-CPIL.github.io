"""Tensor decomposition for 2x5 * 5x6 matrix multiplication."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    start_time = time.time()
    deadline = start_time + 140

    # Build rank-54 decomposition using Strassen on 2x2 sub-blocks
    # Partition: A (2x5) -> A1 (2x2), A2 (2x3)
    #             B (5x6) -> B1 (2x6), B2 (3x6)
    # C = A1*B1 + A2*B2
    #
    # A1*B1: M_{2,2,6} = 3 blocks of M_{2,2,2}, each via Strassen (rank 7) -> rank 21
    # A2*B2: M_{2,3,6} = M_{2,2,6} + M_{2,1,6} -> 21 + 12 = 33
    # Total rank: 21 + 33 = 54

    rank = 54
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)

    col = 0

    # Strassen rank-7 coefficients for 2x2*2x2 -> 2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    # Strassen U coefficients (for A block, 2x2)
    # Each term is (coeff, row, col) where row/col are local indices 0,1
    strassen_U = [
        [(1, 0, 0), (1, 1, 1)],  # M1: A11+A22
        [(1, 1, 0), (1, 1, 1)],  # M2: A21+A22
        [(1, 0, 0)],             # M3: A11
        [(1, 1, 1)],             # M4: A22
        [(1, 0, 0), (1, 0, 1)],  # M5: A11+A12
        [(-1, 0, 0), (1, 1, 0)], # M6: A21-A11
        [(1, 0, 1), (-1, 1, 1)], # M7: A12-A22
    ]

    # Strassen V coefficients (for B block, 2x2)
    strassen_V = [
        [(1, 0, 0), (1, 1, 1)],  # M1: B11+B22
        [(1, 0, 0)],             # M2: B11
        [(-1, 1, 0), (1, 1, 1)], # M3: B21-B22
        [(-1, 0, 0), (1, 0, 1)], # M4: B12-B11
        [(1, 1, 1)],             # M5: B22
        [(1, 0, 0), (1, 0, 1)],  # M6: B11+B12
        [(-1, 1, 0), (1, 1, 1)], # M7: B21+B22
    ]

    # Strassen W coefficients (for C block, 2x2)
    # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
    strassen_W = [
        [(1, 0, 0), (1, 1, 1), (-1, 0, 1), (1, 1, 0)],  # M1: C11+C22-C12+C21
        [(1, 1, 0), (1, 1, 1)],                          # M2: C21+C22
        [(1, 0, 1), (1, 1, 1)],                          # M3: C12+C22
        [(1, 0, 0), (1, 1, 0)],                          # M4: C11+C21
        [(-1, 0, 0), (1, 0, 1)],                         # M5: -C11+C12
        [(1, 1, 1)],                                      # M6: C22
        [(1, 0, 0)],                                      # M7: C11
    ]

    def add_strassen_block(a_row_offset, b_row_offset, c_col_offset, b_col_offset, col_start):
        """Add a Strassen 2x2*2x2->2x2 block.
        a_row_offset: starting row in A (local to the 2x2 A block)
        b_row_offset: starting row in B (local to the 2x2 B block)  
        c_col_offset: starting col in C (local to the 2x2 C block)
        b_col_offset: starting col in B (local to the 2x2 B block)
        """
        global col
        for t in range(7):
            # U part
            for (coeff, r, c) in strassen_U[t]:
                u_idx = a_row_offset + r  # global row in A's 2x2 sub-block
                U[u_idx, col] = (U[u_idx, col] or 0) + coeff
            # V part
            for (coeff, r, c) in strassen_V[t]:
                v_idx = b_row_offset + r * 2 + b_col_offset + c  # row*2+col in B's 2x2
                V[v_idx, col] = (V[v_idx, col] or 0) + coeff
            # W part
            for (coeff, r, c) in strassen_W[t]:
                w_idx = c_col_offset + c * 2 + r  # col*2+row in C's 2x2
                W[w_idx, col] = (W[w_idx, col] or 0) + coeff
            col += 1

    def add_schoolbook_block(a_row_start, a_row_end, b_row_start, b_row_end, c_col_start, c_col_end, col_start):
        """Add schoolbook for A[a_row_start:a_row_end, :] * B[:, b_col_start:b_col_end]
        where A is 2 rows, B has (b_row_end-b_row_start) rows, C has (c_col_end-c_col_start) cols.
        Actually this is for: A_sub (2 x d) * B_sub (d x e) -> C_sub (2 x e)
        a_row_start/end: which rows of A (0 or 1, both)
        b_row_start/end: which rows of B (the d dimension)
        c_col_start/end: which cols of C (the e dimension)
        """
        global col
        for j in range(b_row_start, b_row_end):
            for k in range(c_col_start, c_col_end):
                for i in range(2):
                    u_idx = i * 5 + j  # A[i, j]
                    v_idx = j * 6 + k  # B[j, k]
                    w_idx = k * 2 + i  # C[k, i]
                    U[u_idx, col] = 1
                    V[v_idx, col] = 1
                    W[w_idx, col] = 1
                col += 1

    # Part 1: A1*B1 where A1 = A[0:2, 0:2], B1 = B[0:2, 0:6]
    # This is M_{2,2,6}: 2x2 A times 2x6 B -> 2x6 C
    # Split B1 into three 2x2 blocks: B1[:, 0:2], B1[:, 2:4], B1[:, 4:6]
    # Each gives a 2x2 C block
    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 0, 0, 0, col)  # a_rows 0-1, b_rows 0-1, c_cols 0-1, b_cols 0-1
    # Block 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2, 2:4]
    add_strassen_block(0, 0, 2, 2, col)  # a_rows 0-1, b_rows 0-1, c_cols 2-3, b_cols 2-3
    # Block 3: A[0:2,0:2] * B[0:2,4:6] -> C[0:2, 4:6]
    add_strassen_block(0, 0, 4, 4, col)  # a_rows 0-1, b_rows 0-1, c_cols 4-5, b_cols 4-5

    # Part 2: A2*B2 where A2 = A[0:2, 2:5], B2 = B[2:5, 0:6]
    # This is M_{2,3,6}: 2x3 A times 3x6 B -> 2x6 C
    # Split into M_{2,2,6} (using B rows 2:4) + M_{2,1,6} (using B row 4)
    # M_{2,2,6}: A[0:2,2:4] * B[2:4, 0:6]
    # Split B[2:4, 0:6] into three 2x2 blocks
    # Block 4: A[0:2,2:4] * B[2:4,0:2] -> C[0:2, 0:2]
    add_strassen_block(2, 2, 0, 0, col)  # a_rows 2-3 (j=2,3), b_rows 2-3, c_cols 0-1, b_cols 0-1
    # Block 5: A[0:2,2:4] * B[2:4,2:4] -> C[0:2, 2:4]
    add_strassen_block(2, 2, 2, 2, col)
    # Block 6: A[0:2,2:4] * B[2:4,4:6] -> C[0:2, 4:6]
    add_strassen_block(2, 2, 4, 4, col)

    # M_{2,1,6}: A[0:2,4:5] * B[4:5, 0:6]
    # Schoolbook: 2*1*6 = 12 terms
    add_schoolbook_block(0, 2, 4, 5, 0, 6, col)

    # Convert to lists of ints or halves
    def to_list(mat):
        result = []
        for row in mat:
            r = []
            for val in row:
                if isinstance(val, (int, np.integer)):
                    r.append(int(val))
                elif isinstance(val, float):
                    # Check if it's a half-integer
                    if abs(val * 2 - round(val * 2)) < 1e-9:
                        r.append(int(round(val * 2)) / 2)
                    else:
                        r.append(val)
                else:
                    r.append(val)
            result.append(r)
        return result

    U_list = to_list(U)
    V_list = to_list(V)
    W_list = to_list(W)

    # Verify
    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to schoolbook
    return _schoolbook(n, m, p)


def _verify_decomposition(U, V, W, n, m, p, sample_size=5000):
    """Verify that the decomposition exactly represents matrix multiplication."""
    import random

    nm = n * m
    mp = m * p
    pn = p * n
    rank = len(U[0]) if U else 0

    # Build the set of tensor entries
    tensor_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                tensor_set.add((u, v, w))

    # Check all tensor entries
    for (u, v, w) in tensor_set:
        val = 0
        for r in range(rank):
            val += U[u][r] * V[v][r] * W[w][r]
        if abs(val - 1.0) > 1e-9:
            return False

    # Check a sample of non-tensor entries
    random.seed(42)
    for _ in range(sample_size):
        u = random.randint(0, nm - 1)
        v = random.randint(0, mp - 1)
        w = random.randint(0, pn - 1)
        if (u, v, w) in tensor_set:
            continue
        val = 0
        for r in range(rank):
            val += U[u][r] * V[v][r] * W[w][r]
        if abs(val) > 1e-9:
            return False

    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
