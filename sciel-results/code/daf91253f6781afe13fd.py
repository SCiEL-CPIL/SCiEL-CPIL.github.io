import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 140.0
    n, m, p = 4, 4, 6

    # Rank-32 decomposition: Strassen (rank 28) for 4x4x4 + schoolbook (rank 4) for 4x4x2
    # U: (16, 32), V: (24, 32), W: (24, 32)

    U = np.zeros((n * m, 32))
    V = np.zeros((m * p, 32))
    W = np.zeros((p * n, 32))

    col = 0

    # --- Part 1: Strassen for 4x4x4 (cols 0-3 of B, cols 0-3 of C) ---
    # 7 Strassen terms, each is a product of two 2x2 "super-matrices" (linear combos of 2x2 blocks)
    # Each super-product is done with 4 rank-1 terms (schoolbook 2x2x2)
    # Total: 7 * 4 = 28 rank-1 terms

    # Strassen combinations at the block level:
    # Blocks of A: A00 (r0-1,c0-1), A01 (r0-1,c2-3), A10 (r2-3,c0-1), A11 (r2-3,c2-3)
    # Blocks of B: B00 (r0-1,c0-1), B01 (r0-1,c2-3), B10 (r2-3,c0-1), B11 (r2-3,c2-3)
    # 
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1+M4-M5+M7  (C rows 0-1, cols 0-1)
    # C01 = M3+M5         (C rows 0-1, cols 2-3)
    # C10 = M2+M4         (C rows 2-3, cols 0-1)
    # C11 = M1-M2+M3+M6   (C rows 2-3, cols 2-3)

    # For each Strassen term, A_sub is a 2x2 matrix (linear combo of A blocks) and B_sub is 2x2.
    # The product A_sub * B_sub is 2x2, computed with 4 rank-1 terms.
    # Each rank-1 term: C[i,k] += coeff * A_sub[i,j] * B_sub[j,k]
    # where i,j,k are local 2x2 indices.

    # Let's define the 7 Strassen (A_combo, B_combo) pairs as 2x2 matrices with entries
    # being linear combinations of block entries.

    # A block entries:
    # A00 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # A01 = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]
    # A10 = [[A[2,0], A[2,1]], [A[3,0], A[3,1]]]
    # A11 = [[A[2,2], A[2,3]], [A[3,2], A[3,3]]]

    # B block entries (for 4x4x4 part, cols 0-3):
    # B00 = [[B[0,0], B[0,1]], [B[1,0], B[1,1]]]
    # B01 = [[B[0,2], B[0,3]], [B[1,2], B[1,3]]]
    # B10 = [[B[2,0], B[2,1]], [B[3,0], B[3,1]]]
    # B11 = [[B[2,2], B[2,3]], [B[3,2], B[3,3]]]

    # For each Strassen term, define A_sub and B_sub as 2x2 matrices where each entry
    # is a linear combination of the original A/B entries. Then expand into rank-1 terms.

    # Let me define each Strassen term's A_sub and B_sub explicitly.

    # M1: A_sub = A00+A11, B_sub = B00+B11
    # A_sub[0,0] = A[0,0]+A[2,2], A_sub[0,1] = A[0,1]+A[2,3]
    # A_sub[1,0] = A[1,0]+A[3,2], A_sub[1,1] = A[1,1]+A[3,3]
    # B_sub[0,0] = B[0,0]+B[2,2], B_sub[0,1] = B[0,1]+B[2,3]
    # B_sub[1,0] = B[1,0]+B[3,2], B_sub[1,1] = B[1,1]+B[3,3]

    # For the rank-1 expansion of A_sub * B_sub:
    # C_local[i,k] = sum_j A_sub[i,j] * B_sub[j,k]
    # This is 4 rank-1 terms (j=0,1 and... no, it's 2*2=4 terms for (i,j,k) but grouped)
    # Actually, schoolbook 2x2x2 is 4 terms: for each (j,k) pair, C[:,k] += A[:,j]*B[j,k]
    # So 2*2 = 4 terms.

    # Let me just enumerate all rank-1 terms for each Strassen term.

    def add_strassen_term(A_entries, B_entries, c_row_off, c_col_off, sign):
        """A_entries: dict mapping (i,j) -> list of (coeff, a_row, a_col) for the 2x2 A_sub
           B_entries: dict mapping (j,k) -> list of (coeff, b_row, b_col) for the 2x2 B_sub
           c_row_off, c_col_off: offset in C for the 2x2 result
           sign: overall sign
        """
        nonlocal col
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # A_sub[i,j] * B_sub[j,k]
                    a_list = A_entries.get((i, j), [])
                    b_list = B_entries.get((j, k), [])
                    for (ac, ar, acol) in a_list:
                        for (bc, br, bcol) in b_list:
                            coeff = sign * ac * bc
                            if coeff == 0:
                                continue
                            U[ar * m + acol, col] += coeff
                            V[br * p + bcol, col] += 1.0
                            W[(c_col_off + k) * n + (c_row_off + i), col] += 1.0
        col += 1

    # This approach is getting messy. Let me use a cleaner formulation.

    # I'll build the U, V, W directly by enumerating each rank-1 term.
    # Each rank-1 term is defined by:
    # - A coefficient: a linear combination of A entries
    # - B coefficient: a linear combination of B entries  
    # - C position: where the result goes

    # Let me just use the known rank-32 decomposition from the cell finding.
    # The cell finding says: "Correctly implements the rank 81 decomposition by combining 
    # Strassen (rank 49) for the 4x4x4 part and Schoolbook (rank 32) for the 4x4x2 part"
    # Wait, that's rank 81, not 32. Let me re-read.
    # "Strassen (rank 49) for the 4x4x4 part and Schoolbook (rank 32) for the 4x4x2 part"
    # 49 + 32 = 81. But that's worse than 96? No, 81 < 96.
    # But the known Strassen for 4x4x4 is rank 7*4=28, not 49.
    # Maybe they used a different decomposition.

    # Let me just use the simple schoolbook rank-96 which is definitely correct,
    # and structure it cleanly. This gives score 1.0 (same as seed).
    # To beat it, I need rank < 96.

    # Let me try rank-32: Strassen (28) for 4x4x4 + schoolbook (4) for 4x4x2.
    # For the 4x4x2 part (cols 4-5 of B, cols 4-5 of C):
    # C[i,4] = sum_j A[i,j]*B[j,4], C[i,5] = sum_j A[i,j]*B[j,5]
    # Schoolbook: 4*2 = 8 terms? No, n*p for the tail = 4*2 = 8 rank-1 terms.
    # Wait, schoolbook for 4x4x2: rank = 4*4*2 = 32? No, that's the full tensor.
    # Schoolbook rank for n x m times m x p is n*m*p. For 4x4x2, that's 32.
    # But we can do better: for each output column k, C[:,k] = A * B[:,k], which is a
    # matrix-vector product with rank 4 (one per row of A... no).
    # Actually, C[i,k] = sum_j A[i,j]*B[j,k]. For fixed k, this is 4 terms (one per j).
    # So for 2 columns, 8 terms total. Rank 8 for 4x4x2.

    # So total rank = 28 + 8 = 36.

    # Let me implement rank-36.
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    # --- Part 1: Strassen for 4x4x4 (28 rank-1 terms) ---
    # I'll define each of the 28 terms explicitly.
    # Each term: U has a value at (i,j), V has a value at (j,k), W has a value at (k,i)
    # with appropriate coefficients.

    # Strassen M1 = (A00+A11)(B00+B11), contributes to C00 with sign +1
    # A00+A11:
    #   [A[0,0]+A[2,2], A[0,1]+A[2,3]]
    #   [A[1,0]+A[3,2], A[1,1]+A[3,3]]
    # B00+B11:
    #   [B[0,0]+B[2,2], B[0,1]+B[2,3]]
    #   [B[1,0]+B[3,2], B[1,1]+B[3,3]]
    # Product is 2x2, placed at C[0:2, 0:2]

    # For each (i,j,k) in the 2x2x2 product:
    # C[i,k] += A_sub[i,j] * B_sub[j,k]

    # Let me define a helper that adds a 2x2x2 product with given A and B 2x2 matrices
    # (as numpy arrays) at a given C offset, with a sign.

    def add_2x2x2(A_mat, B_mat, c_row_off, c_col_off, sign):
        nonlocal col
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    a_val = A_mat[i, j]
                    b_val = B_mat[j, k]
                    if a_val == 0 or b_val == 0:
                        continue
                    coeff = sign * a_val * b_val
                    if coeff == 0:
                        continue
                    # But A_mat and B_mat are symbolic (linear combos of A/B entries).
                    # I need to expand them.
                    pass

    # This is too complex to do symbolically. Let me use a different approach:
    # Just use numpy to compute the Strassen decomposition numerically and verify it.

    # Actually, let me just return the schoolbook rank-96 which is correct and verified.
    # This matches the seed score of 1.0.

    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    def to_numeric(mat):
        out = []
        for row in mat:
            new_row = []
            for x in row:
                if abs(x - round(x)) < 1e-10:
                    new_row.append(int(round(x)))
                else:
                    new_row.append(float(x))
            out.append(new_row)
        return out

    return {"U": to_numeric(U), "V": to_numeric(V), "W": to_numeric(W)}
