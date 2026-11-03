"""Program for matrix multiplication tensor decomposition 3x3x6."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]


def _verify(U, V, W, n=3, m=3, p=6, atol=1e-9, seed=42):
    """Verify that the decomposition is correct by checking on random matrices."""
    np.random.seed(seed)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    rank = len(U[0])
    for r in range(rank):
        a_proj = sum(U[i * m + j][r] * A[i, j] for i in range(n) for j in range(m))
        b_proj = sum(V[j * p + k][r] * B[j, k] for j in range(m) for k in range(p))
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += W[k * n + i][r] * a_proj * b_proj
    return np.allclose(C_computed, C_expected, atol=atol)


def _build_strassen_2x2x2():
    """
    Return rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U[i*2+j], V[j*2+k], W[k*2+i] for i,j,k in {0,1}.
    """
    r = 7
    U = [[0] * r for _ in range(4)]
    V = [[0] * r for _ in range(4)]
    W = [[0] * r for _ in range(4)]

    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = -1

    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1

    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1

    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1

    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[3][4] = 1

    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[1][6] = 1; W[3][6] = 1

    return U, V, W


def _build_rank20_3x3x3():
    """
    Build a rank-20 decomposition for 3x3x3 matrix multiplication.
    Uses Strassen (rank 7) for the 2x2 sub-block plus 13 schoolbook terms
    for the remaining cross terms involving row/col 2.
    """
    r = 20
    U = [[0] * r for _ in range(9)]
    V = [[0] * r for _ in range(9)]
    W = [[0] * r for _ in range(9)]

    # Strassen 7 terms for the 2x2 block (indices 0-6)
    # A indices: 0=A[0,0], 1=A[0,1], 2=A[0,2], 3=A[1,0], 4=A[1,1], 5=A[1,2],
    #           6=A[2,0], 7=A[2,1], 8=A[2,2]
    # B indices: 0=B[0,0], 1=B[0,1], 2=B[0,2], 3=B[1,0], 4=B[1,1], 5=B[1,2],
    #           6=B[2,0], 7=B[2,1], 8=B[2,2]
    # W indices: W[k*3+i]: 0=C[0,0], 1=C[0,1], 2=C[0,2], 3=C[1,0], 4=C[1,1],
    #            5=C[1,2], 6=C[2,0], 7=C[2,1], 8=C[2,2]

    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[3][0] = 1; W[4][0] = -1

    # M2 = (A10+A11)B00
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[3][1] = 1

    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[1][2] = 1

    # M4 = A11(B10-B00)
    U[4][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[3][3] = 1

    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[1][4] = 1; W[4][4] = 1

    # M6 = (A10-A00)(B00+B01)
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[4][5] = 1

    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[4][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[1][6] = 1; W[4][6] = 1

    # Cross terms (indices 7-19), 13 terms:
    # Row 0, Col 2: C[0,2] = A00*B02 + A01*B12 + A02*B22
    U[0][7] = 1; V[2][7] = 1; W[2][7] = 1
    U[1][8] = 1; V[5][8] = 1; W[2][8] = 1
    U[2][9] = 1; V[8][9] = 1; W[2][9] = 1
    # Row 1, Col 2: C[1,2] = A10*B02 + A11*B12 + A12*B22
    U[3][10] = 1; V[2][10] = 1; W[5][10] = 1
    U[4][11] = 1; V[5][11] = 1; W[5][11] = 1
    U[5][12] = 1; V[8][12] = 1; W[5][12] = 1
    # Row 2, Col 0: C[2,0] = A20*B00 + A21*B10 + A22*B20
    U[6][13] = 1; V[0][13] = 1; W[6][13] = 1
    U[7][14] = 1; V[3][14] = 1; W[6][14] = 1
    U[8][15] = 1; V[6][15] = 1; W[6][15] = 1
    # Row 2, Col 1: C[2,1] = A20*B01 + A21*B11 + A22*B21
    U[6][16] = 1; V[1][16] = 1; W[7][16] = 1
    U[7][17] = 1; V[4][17] = 1; W[7][17] = 1
    U[8][18] = 1; V[7][18] = 1; W[7][18] = 1
    # Row 2, Col 2: C[2,2] = A20*B02 + A21*B12 + A22*B22
    U[6][19] = 1; V[2][19] = 1; W[8][19] = 1
    U[7][19] = 1; V[5][19] = 1; W[8][19] = 1
    U[8][19] = 1; V[8][19] = 1; W[8][19] = 1

    # Wait, I used index 19 three times. Let me fix - I need 13 terms (7-19).
    # Let me recount: I have terms 7,8,9,10,11,12,13,14,15,16,17,18,19 = 13 terms.
    # But term 19 has 3 entries. That's fine, it's one rank-1 term with 3 non-zero W entries.
    # Actually no - each rank-1 term should be a product of 3 vectors.
    # Let me redo this more carefully.

    # Actually, let me just use schoolbook for the cross terms properly.
    # Each cross term A[i,j]*B[j,k] is its own rank-1 term.
    # Cross terms needed:
    # C[0,2]: (0,0,2), (0,1,2), (0,2,2) -> 3 terms
    # C[1,2]: (1,0,2), (1,1,2), (1,2,2) -> 3 terms
    # C[2,0]: (2,0,0), (2,1,0), (2,2,0) -> 3 terms
    # C[2,1]: (2,0,1), (2,1,1), (2,2,1) -> 3 terms
    # C[2,2]: (2,0,2), (2,1,2), (2,2,2) -> 3 terms
    # Total: 15 terms. 7 + 15 = 22. Not 20.

    # Let me just use rank 22 to be safe, or try to combine.
    # Actually for the purpose of 3x3x6, let me just use schoolbook directly.

    return None  # Skip this, use direct construction for 3x3x6


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    n, m, p = 3, 3, 6

    # Strategy: Direct schoolbook decomposition for 3x3x6
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Each term A[i,j]*B[j,k] contributes to C[i,k] with coefficient 1
    # Total rank = n*m*p = 54

    # Try to find a lower rank decomposition using structure.
    # Key insight: For 3x3 x 3x6, we can view B as two 3x3 matrices B1 and B2.
    # C = A * [B1 | B2] = [A*B1 | A*B2]
    # If we use Strassen for each 3x3x3 block, we get rank 2*20 = 40 (if rank-20 works)
    # or 2*27 = 54 (schoolbook for each block).
    # But the Strassen decomposition for 3x3x3 needs verification.

    # Let's try a direct approach: use the block structure.
    # B has 6 columns. Split into B_left (cols 0,1,2) and B_right (cols 3,4,5).
    # C_left = A * B_left (3x3)
    # C_right = A * B_right (3x3)

    # For each 3x3x3 block, use a decomposition.
    # Schoolbook for 3x3x3 has rank 27.
    # Total: 2 * 27 = 54.

    # Can we do better? Let's try Strassen-based rank for 3x3x3.
    # The tensor rank of 3x3 matmul is known to be 19 (or 20 depending on source).
    # Let me construct a verified rank-20 decomposition.

    # Actually, let me try a different approach:
    # For 3x3 x 3x6, think of it as: for each column k of B, compute A*b_k.
    # A*b_k[i] = sum_j A[i,j] * b_k[j]
    # This is a bilinear map. The "tensor" has entries T[i,j,k] = 1 for all i,j,k.
    # The rank of this tensor is what we need.

    # A key observation: we can factor out the j-index.
    # For fixed j, we have A[:,j] (a 3-vector) and B[j,:] (a 6-vector).
    # The outer product A[:,j] * B[j,:] contributes to C.
    # C = sum_j A[:,j] * B[j,:]
    # This is a sum of 3 rank-1 (in the outer product sense) terms, but each
    # "term" is a 3x6 matrix, not a scalar. So this doesn't directly give tensor rank.

    # The tensor rank is the minimum r such that we can write:
    # C[i,k] = sum_{t=1}^r (sum_{i',j'} U[i',j',t] * A[i',j']) * (sum_{j'',k'} V[j'',k'',t] * B[j'',k'']) * W[i,k,t]
    # Wait, that's not right either. The standard form is:
    # C[i,k] = sum_t U_i(t) * V_k(t) * W(t)
    # where U_i(t) = sum_j U[i,j,t]*A[i,j], V_k(t) = sum_j V[j,k,t]*B[j,k], W(t) = W[i,k,t]
    # Hmm, no. The correct form for matrix multiplication tensor:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # A rank-r decomposition writes:
    # (A*B)[i,k] = sum_{t=1}^r (sum_{i',j'} U[i',j',t] * A[i',j']) * (sum_{j'',k'} V[j'',k'',t] * B[j'',k'']) * W[i,k,t]
    # For this to equal sum_j A[i,j]*B[j,k], we need the right tensor identity.

    # Let me just use the verified schoolbook approach with rank 54,
    # but try to reduce using Strassen on the 2x2 sub-blocks.

    # Alternative: use the fact that 3x3x6 can be decomposed as:
    # Take the 2x2 top-left block of A and B (using Strassen, rank 7)
    # Plus the remaining terms.

    # For now, let me construct a careful decomposition and verify it.

    # Direct construction: schoolbook rank 54
    rank = n * m * p  # 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Verify
    if not _verify(U, V, W, n, m, p):
        # Should not happen, but just in case
        pass

    # Now try to find a lower rank decomposition.
    # Strategy: Use Strassen for the 2x2 sub-block.
    # Partition A into [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # Partition B into [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x4, B21 is 1x2, B22 is 1x4
    # (since B is 3x6, B11 is 2x2, B12 is 2x4, B21 is 1x2, B22 is 1x4)

    # C11 = A11*B11 + A12*B21  (2x2 result)
    # C12 = A11*B12 + A12*B22  (2x4 result)
    # C21 = A21*B11 + A22*B21  (1x2 result)
    # C22 = A21*B12 + A22*B22  (1x4 result)

    # For C11: A11*B11 uses Strassen (7 mults), A12*B21 is 2x1 * 1x2 = outer product (4 mults)
    #   Total: 11
    # For C12: A11*B12 is 2x2 * 2x4. This can be done with 7*2=14? No.
    #   Actually, 2x2 * 2x4: each of the 4 columns is a 2x2 * 2x1 product.
    #   A 2x2 * 2x1 product needs 4 mults (schoolbook) or can share.
    #   Actually, for 2x2 * 2x4, we can use Strassen-like: treat B12 as 4 vectors of size 2.
    #   Each vector multiplication is 2x2 * 2x1, which needs 4 mults.
    #   But we can share the Strassen intermediates!
    #   For 2x2 * 2x4: compute M1..M7 (Strassen) once, then each output column uses
    #   combinations of these. Actually, Strassen gives 7 mults for the 2x2 product.
    #   For 2x2 * 2x4, we need 2x2 * (2x4 matrix). This is 4 separate 2x2*2x1 products.
    #   Each 2x2*2x1 needs 4 mults (or can be done with the Strassen intermediates).
    #   Hmm, this is getting complicated. Let me just count schoolbook:
    #   2x2 * 2x4: 2*2*4 = 16 mults schoolbook.
    #   A12*B22: 2x1 * 1x4 = 8 mults.
    #   Total C12: 16 + 8 = 24.
    #
    # This is getting too complex. Let me try a different approach.

    # Simplest improvement: use Strassen for the 2x2x2 core and schoolbook for the rest.
    # A = [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]]
    # B = [[b00, b01, b02, b03, b04, b05],
    #      [b10, b11, b12, b13, b14, b15],
    #      [b20, b21, b22, b23, b24, b25]]

    # C = A*B, C is 3x6.
    # C[0,k] = a00*b0k + a01*b1k + a02*b2k
    # C[1,k] = a10*b0k + a11*b1k + a12*b2k
    # C[2,k] = a20*b0k + a21*b1k + a22*b2k

    # The 2x2 sub-block (rows 0,1 of A; cols 0,1 of B) can use Strassen.
    # But B has 6 columns, so the "2x2" block of B is actually 2x2 (cols 0,1).
    # The remaining 4 columns of B need separate handling.

    # Let me try: use Strassen for the top-left 2x2 of A times top-left 2x2 of B.
    # That handles C[0,0], C[0,1], C[1,0], C[1,1] with 7 mults.
    # Then handle the remaining terms with schoolbook.

    # Remaining terms:
    # C[0,0]: a02*b20 (already in Strassen? No, Strassen only uses a00,a01,a10,a11 and b00,b01,b10,b11)
    # Actually Strassen for 2x2x2 uses only the 2x2 submatrices.
    # So after Strassen, we still need:
    # C[0,0]: a02*b20
    # C[0,1]: a02*b21
    # C[0,2]: a00*b02 + a01*b12 + a02*b22
    # C[0,3]: a00*b03 + a01*b13 + a02*b23
    # C[0,4]: a00*b04 + a01*b14 + a02*b24
    # C[0,5]: a00*b05 + a01*b15 + a02*b25
    # C[1,0]: a12*b20
    # C[1,1]: a12*b21
    # C[1,2]: a10*b02 + a11*b12 + a12*b22
    # C[1,3]: a10*b03 + a11*b13 + a12*b23
    # C[1,4]: a10*b04 + a11*b14 + a12*b24
    # C[1,5]: a10*b05 + a11*b15 + a12*b25
    # C[2,0]: a20*b00 + a21*b10 + a22*b20
    # C[2,1]: a20*b01 + a21*b11 + a22*b21
    # C[2,2]: a20*b02 + a21*b12 + a22*b22
    # C[2,3]: a20*b03 + a21*b13 + a22*b23
    # C[2,4]: a20*b04 + a21*b14 + a22*b24
    # C[2,5]: a20*b05 + a21*b15 + a22*b25

    # Count: 2 + 4*3 + 2 + 4*3 + 3*6 = 2 + 12 + 2 + 12 + 18 = 46
    # Plus 7 Strassen = 53. Only saves 1 over schoolbook 54. Not worth the complexity.

    # Let me try a different partition. What if I use Strassen on a 2x2 sub-block
    # that covers more of the matrix?

    # Actually, the best approach for 3x3x6 might be to view it as:
    # C = A * B where B is 3x6.
    # Split B into B = [B1 | B2] where B1, B2 are 3x3.
    # C = [A*B1 | A*B2]
    # Each A*Bk is a 3x3x3 multiplication.
    # If I can do 3x3x3 with rank 20, total is 40.
    # If rank 19, total is 38.

    # Let me try to construct and verify a rank-20 3x3x3 decomposition.

    def _make_3x3x3_decomp():
        """Construct a 3x3x3 matrix multiplication decomposition."""
        # Use Strassen for 2x2 + schoolbook for remaining
        # This gives rank 7 + 15 = 22, or we can try to optimize.
        # For now, let's use schoolbook rank 27 for each block.
        # Total: 2 * 27 = 54. Same as direct schoolbook.

        # Let me try to actually build a rank-20 and verify.
        r = 20
        U = [[0] * r for _ in range(9)]
        V = [[0] * r for _ in range(9)]
        W = [[0] * r for _ in range(9)]

        # Strassen for top-left 2x2 (terms 0-6)
        U[0][0] = 1; U[4][0] = 1; V[0][0] = 1; V[4][0] = 1
        W[0][0] = 1; W[1][0] = 1; W[3][0] = 1; W[4][0] = -1

        U[3][1] = 1; U[4][1] = 1; V[0][1] = 1; W[3][1] = 1

        U[0][2] = 1; V[1][2] = 1; V[4][2] = -1; W[1][2] = 1

        U[4][3] = 1; V[3][3] = 1; V[0][3] = -1; W[3][3] = 1

        U[0][4] = 1; U[1][4] = 1; V[4][4] = 1; W[1][4] = 1; W[4][4] = 1

        U[3][5] = 1; U[0][5] = -1; V[0][5] = 1; V[1][5] = 1; W[4][5] = 1

        U[1][6] = 1; U[4][6] = -1; V[3][6] = 1; V[4][6] = 1; W[1][6] = 1; W[4][6] = 1

        # Now add terms for the remaining 12 entries (not covered by Strassen)
        # Strassen covers C[0,0], C[0,1], C[1,0], C[1,1] using A[0,0],A[0,1],A[1,0],A[1,1]
        # and B[0,0],B[0,1],B[1,0],B[1,1].
        # Remaining contributions:
        # C[0,0]: A[0,2]*B[2,0]
        # C[0,1]: A[0,2]*B[2,1]
        # C[1,0]: A[1,2]*B[2,0]
        # C[1,1]: A[1,2]*B[2,1]
        # C[0,2]: A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
        # C[1,2]: A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
        # C[2,0]: A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
        # C[2,1]: A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
        # C[2,2]: A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]

        # That's 4 + 3 + 3 + 3 + 3 + 3 = 19 more terms. Total 7+19=26. Not 20.

        # I need to combine terms. Let me use a different approach.
        # Use the known rank-19 or rank-20 construction from literature.

        # For now, return None to indicate failure, and fall back to schoolbook.
        return None

    # Since constructing a verified lower-rank 3x3x3 decomposition is complex,
    # let's stick with the schoolbook rank-54 decomposition which is verified.

    # But wait - let me try one more thing. Can I use Strassen on a 2x2x2
    # sub-problem within the 3x3x6?

    # Actually, let me try to directly construct a lower-rank 3x3x6 decomposition
    # by exploiting the structure.

    # Key idea: C[i,k] = sum_j A[i,j] * B[j,k]
    # For fixed i, C[i,:] = A[i,:] * B (a 1x3 times 3x6 = 1x6)
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]

    # The tensor T[i,j,k] = 1 for all i,j,k in {0,1,2} x {0,1,2} x {0,...,5}
    # We want to decompose this tensor.

    # One approach: use the fact that the j-index appears in both A and B.
    # T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[i,k,t]

    # Let me try a rank-40 construction:
    # For each pair (i,j), define a "line" in the tensor.
    # T[i,j,k] for fixed i,j is a vector in k: (1,1,1,1,1,1).
    # We can write this as a sum of rank-1 tensors in the (i,j,k) space.

    # Actually, the simplest improvement: use the fact that for fixed j,
    # the slice T[:,j,:] is a 3x6 all-ones matrix, which has rank 1.
    # So T = sum_j (e_i * e_j * 1_k) for each j... no, that's not right.

    # T[i,j,k] = 1 = sum_j delta_{j,j'} * 1
    # This is just the schoolbook decomposition.

    # Let me try: T[i,j,k] = sum_{a=0}^{2} sum_{b=0}^{5} U[i,j,a,b] * V[j,k,a,b] * W[i,k,a,b]
    # No, that's the same as schoolbook with rank 54.

    # I think the schoolbook rank 54 is hard to beat without a sophisticated
    # tensor decomposition algorithm. Let me just return the schoolbook result.

    return {"U": U, "V": V, "W": W}
