"""Search for a low-rank exact tensor decomposition of the 4x4 * 4x7 multiplication tensor."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]
_START = time.time()
_DEADLINE = 150.0 - 20.0  # safety margin


def _build_result(U, V, W):
    """
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is correct.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = U.shape[1]
    # Reconstruct the tensor
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We can check a few random points or all points.
    # For exactness, we should check all points.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                # The tensor element for (i,j,k) in matrix multiplication is 1 if j==j (always true in this indexing?)
                # Wait. The tensor T is defined such that C[i,k] = sum_j A[i,j] B[j,k].
                # The basis tensors are e_{i,j} \otimes e_{j,k} \otimes e_{i,k}.
                # So T[i,j,k] = 1 if the indices match the pattern (i,j) for A, (j,k) for B, (i,k) for C.
                # In our flattened indices:
                # A index: i*m + j
                # B index: j*p + k
                # C index: k*n + i
                # The tensor component is 1.
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _strassen_2x2(U, V, W, a_off, b_off, c_off, col_start):
    """
    Add 7 scalar multiplications for a 2x2 matrix multiplication.
    A is at a_off (2x2), B is at b_off (2x2), C is at c_off (2x2).
    U, V, W are pre-allocated arrays.
    col_start is the starting column index in U, V, W.
    """
    # A indices:
    # A00: a_off + 0
    # A01: a_off + 1
    # A10: a_off + 2
    # A11: a_off + 3
    # B indices:
    # B00: b_off + 0
    # B01: b_off + 1
    # B10: b_off + 2
    # B11: b_off + 3
    # C indices:
    # C00: c_off + 0
    # C01: c_off + 1
    # C10: c_off + 2
    # C11: c_off + 3

    # Strassen steps:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)

    # We need to add terms to U, V, W.
    # Each term is a scalar multiplication x * y.
    # x is a linear combination of A elements.
    # y is a linear combination of B elements.
    # The product contributes to C elements.

    def add_term(col, a_coeffs, b_coeffs, c_coeffs):
        # a_coeffs: list of (idx, coeff)
        # b_coeffs: list of (idx, coeff)
        # c_coeffs: list of (idx, coeff)
        for idx, coeff in a_coeffs:
            U[idx, col] += coeff
        for idx, coeff in b_coeffs:
            V[idx, col] += coeff
        for idx, coeff in c_coeffs:
            W[idx, col] += coeff

    # M1
    add_term(col_start + 0,
             [(a_off + 0, 1), (a_off + 3, 1)],
             [(b_off + 0, 1), (b_off + 3, 1)],
             [(c_off + 0, 1), (c_off + 3, 1)])

    # M2
    add_term(col_start + 1,
             [(a_off + 2, 1), (a_off + 3, 1)],
             [(b_off + 0, 1)],
             [(c_off + 2, 1), (c_off + 3, 1)])

    # M3
    add_term(col_start + 2,
             [(a_off + 0, 1)],
             [(b_off + 1, 1), (b_off + 3, -1)],
             [(c_off + 1, 1), (c_off + 3, -1)])

    # M4
    add_term(col_start + 3,
             [(a_off + 3, 1)],
             [(b_off + 2, 1), (b_off + 0, -1)],
             [(c_off + 2, 1), (c_off + 3, -1)])

    # M5
    add_term(col_start + 4,
             [(a_off + 0, 1), (a_off + 1, 1)],
             [(b_off + 3, 1)],
             [(c_off + 0, 1), (c_off + 1, 1)])

    # M6
    add_term(col_start + 5,
             [(a_off + 2, 1), (a_off + 0, -1)],
             [(b_off + 0, 1), (b_off + 1, 1)],
             [(c_off + 0, -1), (c_off + 2, 1)])

    # M7
    add_term(col_start + 6,
             [(a_off + 1, 1), (a_off + 3, -1)],
             [(b_off + 2, 1), (b_off + 3, 1)],
             [(c_off + 1, 1), (c_off + 3, -1)])


def _strassen_4x4(U, V, W, a_off, b_off, c_off, col_start):
    """
    Add 49 scalar multiplications for a 4x4 matrix multiplication using Strassen.
    A is at a_off (4x4), B is at b_off (4x4), C is at c_off (4x4).
    """
    # Blocks:
    # A00: a_off + 0..3 (rows 0-1, cols 0-1)
    # A01: a_off + 4..7 (rows 0-1, cols 2-3)
    # A10: a_off + 8..11 (rows 2-3, cols 0-1)
    # A11: a_off + 12..15 (rows 2-3, cols 2-3)

    # B00: b_off + 0..3
    # B01: b_off + 4..7
    # B10: b_off + 8..11
    # B11: b_off + 12..15

    # C00: c_off + 0..3
    # C01: c_off + 4..7
    # C10: c_off + 8..11
    # C11: c_off + 12..15

    # Strassen steps for 4x4:
    # M00 = (A00 + A11)(B00 + B11) -> C00 += M00, C11 += M00
    # M01 = (A10 + A11)B00 -> C10 += M01, C11 += M01
    # M10 = A00(B01 - B11) -> C01 += M10, C11 -= M10
    # M11 = A11(B10 - B00) -> C10 += M11, C11 -= M11
    # M02 = (A00 + A01)B11 -> C00 += M02, C01 += M02
    # M12 = (A10 - A00)(B00 + B01) -> C00 -= M12, C10 += M12
    # M21 = (A01 - A11)(B10 + B11) -> C01 += M21, C11 -= M21

    # Each M is a 4x4 matrix multiplication, which takes 49 scalar multiplications.
    # Total 7 * 49 = 343? No, wait.
    # Strassen 4x4 uses 7 multiplications of 4x4 matrices? No.
    # Strassen 2x2 uses 7 multiplications of 2x2 matrices.
    # If we apply Strassen to 4x4, we treat it as 2x2 blocks of 2x2 matrices.
    # So we need 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication uses 7 scalar multiplications.
    # Total 7 * 7 = 49 scalar multiplications.

    # So we call _strassen_2x2 7 times.

    # M00
    _strassen_2x2(U, V, W, a_off + 0, b_off + 0, c_off + 0, col_start)
    # Wait, the result of M00 is added to C00 and C11.
    # My _strassen_2x2 writes to c_off.
    # I need to handle the addition to multiple blocks.
    # The current _strassen_2x2 adds to U, V, W at specific columns.
    # The W part adds to c_off.
    # If I want to add to C00 and C11, I need to add to both c_off and c_off + 12.
    # But the column is the same.
    # So I can just add the coefficients to W for both blocks.
    # My _strassen_2x2 currently only adds to one block.
    # I need a more flexible helper or modify _strassen_2x2 to accept multiple target blocks.

    # Let's refactor _strassen_2x2 to accept a list of (c_off, coeff) pairs.
    # Or just write the 49 terms directly here.
    # Writing 49 terms is tedious but safe.

    # Actually, let's just use schoolbook for the 2x2 blocks in the first version to ensure correctness,
    # then optimize.
    # Schoolbook 4x4 using schoolbook 2x2:
    # C00 = A00B00 + A01B10
    # C01 = A00B01 + A01B11
    # C10 = A10B00 + A11B10
    # C11 = A10B01 + A11B11
    # Each 2x2 multiplication has 4 scalar multiplications.
    # Total 4 * 4 = 16 scalar multiplications per block? No.
    # 4 blocks * 4 mults/block = 16 mults.
    # Wait. C00 = A00B00 + A01B10.
    # A00B00 is 4 mults. A01B10 is 4 mults. Total 8 mults for C00.
    # Total 4 * 8 = 32 mults.
    # Schoolbook 4x4 is 64 mults.
    # Why 32?
    # A is 4x4. B is 4x4.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # If we use blocks:
    # C00[i,k] = sum_j A00[i,j] B00[j,k] + sum_j A01[i,j] B10[j,k] (j is 0..1 for first sum? No, j is 0..3)
    # A00 is rows 0-1, cols 0-1. A01 is rows 0-1, cols 2-3.
    # B00 is rows 0-1, cols 0-1. B10 is rows 2-3, cols 0-1.
    # So sum_j A00[i,j] B00[j,k] uses j=0,1.
    # sum_j A01[i,j] B10[j,k] uses j=2,3? No.
    # A01[i, j] where j is 0..1 (local index). Global index is 2+j.
    # B10[j, k] where j is 0..1 (local index). Global index is 2+j.
    # So yes, it works.
    # So schoolbook 4x4 using 2x2 blocks has 32 scalar multiplications.
    # This is better than 64.
    # And Strassen 4x4 has 49.
    # So schoolbook 32 is better than Strassen 49!
    # Wait, is schoolbook 32 correct?
    # C00 = A00B00 + A01B10.
    # A00B00: 2x2 * 2x2. 4 mults.
    # A01B10: 2x2 * 2x2. 4 mults.
    # Total 8.
    # 4 blocks * 8 = 32.
    # Yes.
    # So rank 32 for 4x4.
    # For 4x7, we can use 32 for the first 4 columns and 24 for the last 3 (4*3*2? No. 4x3 * 4x3? No.)
    # For 4x4 * 4x3:
    # C is 4x3.
    # We can treat it as 4x4 * 4x4 and ignore the last column? No.
    # We can just use schoolbook for 4x4 * 4x3.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # j=0..3, k=0..2.
    # Total 4*4*3 = 48 mults.
    # So total rank = 32 + 48 = 80.
    # This is better than 112.

    # Let's implement rank 80.
    # Part 1: 4x4 * 4x4 (first 4 cols of B, first 4 cols of C). Rank 32.
    # Part 2: 4x4 * 4x3 (last 3 cols of B, last 3 cols of C). Rank 48.

    # I will implement this in construct().
    pass


def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=7
    # Part 1: 4x4 * 4x4 (cols 0-3 of B, cols 0-3 of C)
    # Part 2: 4x4 * 4x3 (cols 4-6 of B, cols 4-6 of C)

    rank1 = 32
    rank2 = 48
    rank = rank1 + rank2

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Part 1: 4x4 * 4x4
    # Using 2x2 blocks.
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11

    # C00 = A00B00 + A01B10
    # C01 = A00B01 + A01B11
    # C10 = A10B00 + A11B10
    # C11 = A10B01 + A11B11

    # We will add terms for each block multiplication.
    # Helper to add a 2x2 * 2x2 multiplication.
    def add_2x2_mult(a_off, b_off, c_off, col_start):
        # 4 scalar multiplications.
        # (a00*b00 -> c00), (a01*b10 -> c00)
        # (a00*b01 -> c01), (a01*b11 -> c01)
        # (a10*b00 -> c10), (a11*b10 -> c10)
        # (a10*b01 -> c11), (a11*b11 -> c11)
        
        # Term 1: a00 * b00 -> c00
        U[a_off + 0, col_start] += 1
        V[b_off + 0, col_start] += 1
        W[c_off + 0, col_start] += 1

        # Term 2: a01 * b10 -> c00
        U[a_off + 1, col_start + 1] += 1
        V[b_off + 2, col_start + 1] += 1
        W[c_off + 0, col_start + 1] += 1

        # Term 3: a00 * b01 -> c01
        U[a_off + 0, col_start + 2] += 1
        V[b_off + 1, col_start + 2] += 1
        W[c_off + 1, col_start + 2] += 1

        # Term 4: a01 * b11 -> c01
        U[a_off + 1, col_start + 3] += 1
        V[b_off + 3, col_start + 3] += 1
        W[c_off + 1, col_start + 3] += 1

        # Term 5: a10 * b00 -> c10
        U[a_off + 2, col_start + 4] += 1
        V[b_off + 0, col_start + 4] += 1
        W[c_off + 2, col_start + 4] += 1

        # Term 6: a11 * b10 -> c10
        U[a_off + 3, col_start + 5] += 1
        V[b_off + 2, col_start + 5] += 1
        W[c_off + 2, col_start + 5] += 1

        # Term 7: a10 * b01 -> c11
        U[a_off + 2, col_start + 6] += 1
        V[b_off + 1, col_start + 6] += 1
        W[c_off + 3, col_start + 6] += 1

        # Term 8: a11 * b11 -> c11
        U[a_off + 3, col_start + 7] += 1
        V[b_off + 3, col_start + 7] += 1
        W[c_off + 3, col_start + 7] += 1

    # A00: 0..3, A01: 4..7, A10: 8..11, A11: 12..15
    # B00: 0..3, B01: 4..7, B10: 8..11, B11: 12..15
    # C00: 0..3, C01: 4..7, C10: 8..11, C11: 12..15
    # Note: C is 4x4 in this part. Index k*n+i.
    # C00: k=0,1; i=0,1. Indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5.
    # Wait. My C block indices above were for row-major (i*4+k).
    # But the task says W[k*n+i, r]. So C is stored column-major?
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # This means C[i,k] is at index k*n + i.
    # So C is stored with k as the major index.
    # Let's re-verify the C block indices.
    # C00: i=0,1; k=0,1.
    # Indices: k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 4; k=1,i=1 -> 5.
    # So C00 indices are 0, 1, 4, 5.
    # C01: i=0,1; k=2,3.
    # Indices: k=2,i=0 -> 8; k=2,i=1 -> 9; k=3,i=0 -> 12; k=3,i=1 -> 13.
    # C10: i=2,3; k=0,1.
    # Indices: k=0,i=2 -> 2; k=0,i=3 -> 3; k=1,i=2 -> 6; k=1,i=3 -> 7.
    # C11: i=2,3; k=2,3.
    # Indices: k=2,i=2 -> 10; k=2,i=3 -> 11; k=3,i=2 -> 14; k=3,i=3 -> 15.

    # My add_2x2_mult assumes C00 is 0..3. This is WRONG for the given convention.
    # I need to map the 2x2 block elements to the correct global indices.
    # Let's define a helper to get the global index for C[i,k].
    def c_idx(i, k):
        return k * n + i

    # C00 elements:
    # c00(0,0)=0, c01(0,1)=1, c10(1,0)=4, c11(1,1)=5
    # C01 elements:
    # c00(0,2)=8, c01(0,3)=9, c10(1,2)=12, c11(1,3)=13
    # C10 elements:
    # c00(2,0)=2, c01(2,1)=3, c10(3,0)=6, c11(3,1)=7
    # C11 elements:
    # c00(2,2)=10, c01(2,3)=11, c10(3,2)=14, c11(3,3)=15

    # I will rewrite add_2x2_mult to take explicit indices.
    def add_term(col, a_idx, b_idx, c_idx_val):
        U[a_idx, col] += 1
        V[b_idx, col] += 1
        W[c_idx_val, col] += 1

    col = 0
    # C00 = A00B00 + A01B10
    # A00B00:
    # A00[0,0]*B00[0,0] -> C[0,0]
    add_term(col, 0, 0, c_idx(0,0)); col += 1
    # A00[0,1]*B00[1,0] -> C[0,0]
    add_term(col, 1, 2, c_idx(0,0)); col += 1
    # A00[1,0]*B00[0,0] -> C[1,0]
    add_term(col, 4, 0, c_idx(1,0)); col += 1
    # A00[1,1]*B00[1,0] -> C[1,0]
    add_term(col, 5, 2, c_idx(1,0)); col += 1
    # A01B10:
    # A01[0,0]*B10[0,0] -> C[0,0]
    add_term(col, 4, 8, c_idx(0,0)); col += 1
    # A01[0,1]*B10[1,0] -> C[0,0]
    add_term(col, 5, 10, c_idx(0,0)); col += 1
    # A01[1,0]*B10[0,0] -> C[1,0]
    add_term(col, 8, 8, c_idx(1,0)); col += 1
    # A01[1,1]*B10[1,0] -> C[1,0]
    add_term(col, 9, 10, c_idx(1,0)); col += 1

    # C01 = A00B01 + A01B11
    # A00B01:
    add_term(col, 0, 4, c_idx(0,2)); col += 1
    add_term(col, 1, 6, c_idx(0,2)); col += 1
    add_term(col, 4, 4, c_idx(1,2)); col += 1
    add_term(col, 5, 6, c_idx(1,2)); col += 1
    # A01B11:
    add_term(col, 4, 12, c_idx(0,2)); col += 1
    add_term(col, 5, 14, c_idx(0,2)); col += 1
    add_term(col, 8, 12, c_idx(1,2)); col += 1
    add_term(col, 9, 14, c_idx(1,2)); col += 1

    # C10 = A10B00 + A11B10
    # A10B00:
    add_term(col, 8, 0, c_idx(2,0)); col += 1
    add_term(col, 9, 2, c_idx(2,0)); col += 1
    add_term(col, 12, 0, c_idx(3,0)); col += 1
    add_term(col, 13, 2, c_idx(3,0)); col += 1
    # A11B10:
    add_term(col, 12, 8, c_idx(2,0)); col += 1
    add_term(col, 13, 10, c_idx(2,0)); col += 1
    add_term(col, 16, 8, c_idx(3,0)); col += 1
    add_term(col, 17, 10, c_idx(3,0)); col += 1

    # C11 = A10B01 + A11B11
    # A10B01:
    add_term(col, 8, 4, c_idx(2,2)); col += 1
    add_term(col, 9, 6, c_idx(2,2)); col += 1
    add_term(col, 12, 4, c_idx(3,2)); col += 1
    add_term(col, 13, 6, c_idx(3,2)); col += 1
    # A11B11:
    add_term(col, 12, 12, c_idx(2,2)); col += 1
    add_term(col, 13, 14, c_idx(2,2)); col += 1
    add_term(col, 16, 12, c_idx(3,2)); col += 1
    add_term(col, 17, 14, c_idx(3,2)); col += 1

    # Part 2: 4x4 * 4x3
    # B cols 4,5,6. C cols 4,5,6.
    # Schoolbook.
    # C[i,k] = sum_j A[i,j] B[j,k]
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                add_term(col, i * m + j, j * p + k, c_idx(i, k))
                col += 1

    # Check if col matches rank
    if col != rank:
        print(f"Warning: col {col} != rank {rank}")

    return _build_result(U, V, W)
