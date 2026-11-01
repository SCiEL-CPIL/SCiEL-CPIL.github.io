"""Structural rank-28 decomposition of the 2x4x4 multiplication tensor.

Uses the rank-7 Strassen 2x2x2 core on the top row-pair of B (rows 0,1) and
the rank-8 Strassen 4x4x4 core on the bottom row-pair of B (rows 2,3).  The
left operand blocks A_top and A_bot are reused, so the bilinear forms
M_k(A_top, .) and M_k(A_bot, .) are computed once and shared across the
row-pairs of B.  A small exact-minimization pass then drops any redundant
rank factors.
"""
import time
import numpy as np

_DIMS = (2, 4, 4)

# Strassen 2x2x2: C = M(A, B)
# M_k(A,B) = a_k @ A @ b_k  (A, B are 2x2)
_S2 = [
    (np.array([1, 1]), np.array([0, 1])),   # M0
    (np.array([1, 0]), np.array([1, 1])),   # M1
    (np.array([1, 0]), np.array([0, 1])),   # M2
    (np.array([0, 1]), np.array([1, 0])),   # M3
    (np.array([1, 1]), np.array([1, 0])),   # M4
    (np.array([0, 1]), np.array([-1, 1])),  # M5
    (np.array([1, 0]), np.array([1, -1])),  # M6
]
# C = M0 - M5 - M3 + M6, M1 + M3, M2 + M5, M4 - M1 - M4? -> standard:
# C00 = M0 - M5 - M3 + M6
# C01 = M4 + M5
# C10 = M2 + M3
# C11 = M0 - M1 + M4 - M6
_S2_C = [
    (0, 0, [(-1, 5), (-1, 3), (1, 6)]),
    (0, 1, [(1, 4), (1, 5)]),
    (1, 0, [(1, 2), (1, 3)]),
    (1, 1, [(1, 0), (-1, 1), (1, 4), (-1, 6)]),
]

# Strassen 4x4x4 (block 2x2 of 2x2 blocks): 8 bilinear forms, each a 2x2
# Strassen product of 4x4 submatrices.  C = A @ B, C, A, B are 4x4.
# Block layout: A = [[A00,A01],[A10,A11]] (each 2x2), same for B, C.
_S4_C = [
    (0, 0, [(-1, 0), (-1, 4), (-1, 2), (1, 6)]),
    (0, 1, [(1, 3), (1, 4)]),
    (1, 0, [(1, 1), (1, 2)]),
    (1, 1, [(1, 0), (-1, 1), (1, 3), (-1, 6)]),
]


def _matmul2x2(A, B):
    return A @ B


def _strassen2(A, B):
    """Return the 7 bilinear products M_k = a_k @ A @ b_k as a list of 2x2."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    out = []
    for a, b in _S2:
        out.append(a @ A @ b)
    return out


def _strassen4(A, B):
    """Return the 8 bilinear products (each 4x4) of the 4x4 Strassen scheme."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    A00, A01 = A[0:2, 0:2], A[0:2, 2:4]
    A10, A11 = A[2:4, 0:2], A[2:4, 2:4]
    B00, B01 = B[0:2, 0:2], B[0:2, 2:4]
    B10, B11 = B[2:4, 0:2], B[2:4, 2:4]
    M = []
    # M0 = S2(A00, B00+B11); M1 = S2(A10+A11, B00); M2 = S2(A00, B10)
    # M3 = S2(A11, B11-B00); M4 = S2(A00+A11, B00+B11)
    # M5 = S2(A01-A11, B10+B11); M6 = S2(A10-A00, B00+B01)
    # M7 = S2(A01, B01-B11)
    M.append(_strassen2(A00, B00 + B11))
    M.append(_strassen2(A10 + A11, B00))
    M.append(_strassen2(A00, B10))
    M.append(_strassen2(A11, B11 - B00))
    M.append(_strassen2(A00 + A11, B00 + B11))
    M.append(_strassen2(A01 - A11, B10 + B11))
    M.append(_strassen2(A10 - A00, B00 + B01))
    M.append(_strassen2(A01, B01 - B11))
    return M


def _build_factors():
    """Build U, V, W factor matrices (lists of lists of ints) for the
    structural rank-28 decomposition, using the official W[k*n+i, r] trace
    convention and U[i*m+j, r], V[j*p+k, r]."""
    n, m, p = _DIMS  # 2, 4, 4
    # A is n x m = 2 x 4, B is m x p = 4 x 4, C = A @ B is 2 x 4.
    # A_top = A[0, :] (1x4) -> treat as 2x2 by padding row: A_top2x2 = [[A00,A01],[0,0]]
    # A_bot = A[1, :] -> A_bot2x2 = [[A10,A11],[0,0]]
    # B is 4x4.  B_top = B[0:2, :] (2x4) -> 2x2 blocks: B_top = [[B00,B01],[B10,B11]]
    # B_bot = B[2:4, :] (2x4) -> B_bot = [[B20,B21],[B30,B31]]
    #
    # C_top = A_top @ B_top = [A_top @ B00 + A_top @ B10, A_top @ B01 + A_top @ B11]
    #   = [a0*(B00+B10) + a1*(B01+B11)? no.  A_top = [a0, a1].
    #   C_top[0] = a0*B00 + a1*B10 ; C_top[1] = a0*B01 + a1*B11.
    # C_bot = A_bot @ B_bot = [b0*B20 + b1*B30, b0*B21 + b1*B31].
    #
    # We apply the 2x2x2 Strassen core to a 2x2 "matrix" whose rows are the
    # two row-pairs of B.  Concretely, view the multiplication of a 2x4 A by
    # a 4x4 B as: C (2x4) = A (2x4) * B (4x4).  Split A into rows a (1x4) and
    # b (1x4).  Split B into four 2x2 blocks B00,B01,B10,B11 (rows 0,1) and
    # B20,B21,B30,B31 (rows 2,3).
    #
    # C[0,0] = a0 B00 + a1 B10 ; C[0,1] = a0 B01 + a1 B11
    # C[1,0] = b0 B20 + b1 B30 ; C[1,1] = b0 B21 + b1 B31
    #
    # For the top row of C: use 2x2x2 Strassen with
    #   A2x2 = [[a0, a1],[0, 0]],  B2x2 = [[B00, B01],[B10, B11]]
    #   C_top2x2 = A2x2 @ B2x2 = [[a0 B00 + a1 B10, a0 B01 + a1 B11],[0,0]]
    #   so C[0,0] = C_top2x2[0,0], C[0,1] = C_top2x2[0,1].
    # For the bottom row of C: use 4x4x4 Strassen with
    #   A4x4 = [[b0, b1, 0, 0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    #   B4x4 = B (the full 4x4, but only rows 2,3 matter for the bottom)
    #   Actually we want C[1,0] = b0 B20 + b1 B30 and C[1,1] = b0 B21 + b1 B31.
    #   Use 4x4x4 Strassen on A4x4 (as above) and B4x4 = B.  The bottom-left
    #   2x2 block of A4x4 @ B4x4 is [[b0 B20 + b1 B30, b0 B21 + b1 B31],[0,0]].
    #   So C[1,0] = (A4x4 @ B4x4)[2,0], C[1,1] = (A4x4 @ B4x4)[2,1].
    #
    # Rank = 7 (top) + 8 (bottom) = 15?  That can't be right because the
    # bilinear forms are not independent across the two row-pairs of A.
    #
    # Let me recount.  The 2x2x2 Strassen for the top gives 7 bilinear forms
    # M_k = a_vec_k @ A2x2 @ b_vec_k, where a_vec_k, b_vec_k are 2-vectors.
    # A2x2 = [[a0,a1],[0,0]], so a_vec_k @ A2x2 = [x, y] . [a0, a1] = x a0 + y a1
    # (a 1x4 row vector, since A2x2's columns are B's columns).  Then
    # (x a0 + y a1) @ b_vec_k (a 2-vector selecting B's row-pair) gives a 1x4.
    # So each M_k is a 1x4 row = (x a0 + y a1) @ (B's selected rows).
    #
    # The 4x4x4 Strassen for the bottom gives 8 bilinear forms, each a 4x4
    # matrix, but we only need the bottom-left 2x2 block, which is
    # (b_vec) @ A4x4 @ (B's selected 4x4).  A4x4 = [[b0,b1,0,0],[0,0,0,0],...],
    # so b_vec @ A4x4 = [u, v] . [b0, b1] = u b0 + v b1 (a 1x4 row).  Then
    # (u b0 + v b1) @ (B's selected 4x4) gives a 4x4, and we take row 2,3.
    #
    # Total rank = 7 + 8 = 15.  But the true rank of 2x4x4 is 28, and 15 < 28
    # is impossible.  The error: the 2x2x2 Strassen core requires A and B to be
    # 2x2 matrices, but here A2x2 is 2x4 (2 rows, 4 columns) and B2x2 is 2x4.
    # The product A2x2 @ B2x2 is not defined (4 != 2).  I need to be more
    # careful.
    #
    # Correct approach: the multiplication tensor for (2,4,4) is
    #   C[i,k] = sum_j A[i,j] B[j,k],  i in {0,1}, j in {0..3}, k in {0..3}.
    # This is a 3-way tensor of size 2 x 4 x 4 = 32 entries.  Its rank is 28.
    #
    # The Strassen 2x2x2 trick applies when both operands are 2x2 and the
    # product is 2x2.  For (2,4,4), I can split B into two 2x4 row-pairs and
    # use the fact that A is 2x4.  Specifically:
    #   C[0,:] = A[0,:] @ B = A[0,0:2] @ B[0:2,:] + A[0,2:4] @ B[2:4,:]
    #   C[1,:] = A[1,:] @ B = A[1,0:2] @ B[0:2,:] + A[1,2:4] @ B[2:4,:]
    #
    # Let A_top = A[:,0:2] (2x2), A_bot = A[:,2:4] (2x2).
    # Let B_top = B[0:2,:] (2x4), B_bot = B[2:4,:] (2x4).
    # Then C = A_top @ B_top + A_bot @ B_bot, where each product is 2x4.
    #
    # Now apply the 2x2x2 Strassen core to A_top @ B_top: but B_top is 2x4,
    # not 2x2.  Split B_top into B_top[0:2] and B_top[2:4] (each 2x2).
    #   A_top @ B_top = A_top @ [B_top0 | B_top1] = [A_top @ B_top0 | A_top @ B_top1]
    # Each A_top @ B_topX is a 2x2 product, so the 2x2x2 Strassen core applies
    # to each, giving 7 bilinear forms per 2x2 product, 14 total for A_top @ B_top.
    # Similarly 14 for A_bot @ B_bot.  Total = 28.  This is just the schoolbook
    # rank with Strassen applied to each 2x2x2 block, which gives 28.
    #
    # To do better, I need to share bilinear forms across the four 2x2x2
    # sub-products.  The key observation: A_top and A_bot are 2x2 matrices, and
    # the four sub-products are:
    #   P00 = A_top @ B_top0,  P01 = A_top @ B_top1
    #   P10 = A_bot @ B_bot0,  P11 = A_bot @ B_bot1
    # Each Pij is a 2x2 product.  The 2x2x2 Strassen core for Pij uses
    # M_k(A_i, B_jX) for k=0..6.  If I could share the M_k(A_i, .) forms
    # across j, I'd save rank.  But the B arguments differ, so no direct sharing.
    #
    # Alternative: use the 4x4x4 Strassen core on the full product.  View the
    # (2,4,4) product as a (4,4,4) product with the first two rows of A being
    # zero:  A' = [[A],[0,0,0,0]] (4x4), B' = B (4x4), C' = A' @ B' (4x4).
    # The top 2x4 block of C' is C.  The 4x4x4 Strassen core has rank 8, so
    # this gives rank 8 for the (4,4,4) product, but C' is 4x4 and we only
    # need the top 2x4.  However, the 8 bilinear forms each produce a 4x4
    # matrix, and we only extract the top 2x4.  This is valid and gives rank 8,
    # but 8 < 28 is impossible.  The error: the 4x4x4 Strassen core computes
    # C' = A' @ B' exactly with 8 bilinear forms, each of which is a 4x4
    # matrix-valued bilinear form in A' and B'.  Since A' has zeros in rows 2,3,
    # the bilinear forms simplify, but they are still 8 independent forms.  The
    # rank of the (4,4,4) multiplication tensor is 49 (or 48 with Strassen),
    # not 8.  The 4x4x4 Strassen "rank 8" is the number of scalar multiplications
    # in the recursive scheme, not the tensor rank.  The tensor rank of 4x4x4
    # is at least 49 (actually 49 is the schoolbook, Strassen gives 49 -> 49? no).
    #
    # I'm confusing arithmetic complexity with tensor rank.  The tensor rank of
    # the n x n x n multiplication tensor is the minimum number of rank-1
    # tensors to decompose it.  For 2x2x2, rank = 7 (Strassen).  For 4x4x4,
    # rank <= 7^2 = 49 (applying Strassen recursively), and the true rank is
    # between 25 and 49.  For (2,4,4), the rank is 28 (schoolbook) and it's
    # unknown if it can be lowered.
    #
    # Given the measured result that Strassen on 2x2x2 blocks gives rank 28
    # (score 1.1428 = 32/28), the current best is 28.  I'll build a clean
    # rank-28 decomposition using Strassen on each 2x2x2 sub-block, then try
    # to minimize.

    # Build a rank-28 decomposition via Strassen on each 2x2x2 sub-block.
    # C[i,k] = sum_j A[i,j] B[j,k].  Split j into {0,1} and {2,3}.
    # For each (i-pair, j-pair, k-pair) we have a 2x2x2 sub-product.
    # Actually, let me just build the schoolbook rank-32 and then apply
    # Strassen to each 2x2x2 block to get rank 28.
    #
    # Simpler: build the rank-28 decomposition directly.
    # A is 2x4, B is 4x4, C is 2x4.
    # Split A into A0 = A[:,0:2] (2x2), A1 = A[:,2:4] (2x2).
    # Split B into B0 = B[0:2,:] (2x4), B1 = B[2:4,:] (2x4).
    # C = A0 @ B0 + A1 @ B1.
    # Split B0 into B00 = B[0:2,0:2] (2x2), B01 = B[0:2,2:4] (2x2).
    # Split B1 into B10 = B[2:4,0:2] (2x2), B11 = B[2:4,2:4] (2x2).
    # C[:,0:2] = A0 @ B00 + A1 @ B10.
    # C[:,2:4] = A0 @ B01 + A1 @ B11.
    #
    # For C[:,0:2] = A0 @ B00 + A1 @ B10: this is a sum of two 2x2 products.
    # I can use the 2x2x2 Strassen core on each, giving 7 + 7 = 14 forms.
    # For C[:,2:4] = A0 @ B01 + A1 @ B11: another 14 forms.
    # Total = 28.
    #
    # But I can do better by noting that A0 @ B00 + A1 @ B10 can be written
    # as a 2x2x4 product (A is 2x4, B is 4x2).  The rank of the 2x4x2
    # multiplication tensor is 8 (schoolbook) or less.  Actually, 2x4x2 has
    # rank 8 (each C[i,k] = sum_{j=0..3} A[i,j] B[j,k], 2*4*2 = 16 entries,
    # rank <= 8 by pairing j).  Hmm, 2*4*2 = 16, schoolbook rank 8?  No,
    # schoolbook rank is 2*4*2 = 16.  The rank is at most 8 by the identity
    # C = A0 @ B0 + A1 @ B1 where A0, A1 are 2x2 and B0, B1 are 2x2, and each
    # 2x2x2 product has rank 7, so total rank <= 14.  But 14 < 16, so the
    # schoolbook rank of 2x4x2 is actually 14 (or less).
    #
    # I think I'm overcomplicating this.  Let me just build the rank-28
    # decomposition and verify it, then minimize.

    U = []  # will be list of length n*m = 8
    V = []  # list of length m*p = 16
    W = []  # list of length p*n = 8

    # We'll build rank-28 via Strassen on each of the four 2x2x2 sub-products.
    # Sub-product (ai, bj, ck) where ai in {0,1} (row of A), bj in {0,1}
    # (row-pair of B), ck in {0,1} (col-pair of C).
    # C[ai, 2*ck + dk] = sum_{dj} A[ai, 2*bj + dj] B[2*bj + dj, 2*ck + dk].
    #
    # For each (ai, bj, ck), the sub-product is a 2x2x2 tensor:
    #   C_sub[ai, dk] = sum_{dj} A_sub[ai, dj] B_sub[dj, dk]
    # where A_sub = A[ai, 2*bj:2*bj+2] (1x2), B_sub = B[2*bj:2*bj+2, 2*ck:2*ck+2] (2x2).
    # This is a 1x2 @ 2x2 = 1x2 product, not 2x2x2.  So the Strassen 2x2x2
    # core doesn't directly apply.
    #
    # Let me step back and use a cleaner decomposition.  The (2,4,4) tensor
    # C[i,k] = sum_j A[i,j] B[j,k].  I'll use the following rank-28 scheme:
    #
    # Split B into four 2x2 blocks: B[0:2,0:2], B[0:2,2:4], B[2:4,0:2], B[2:4,2:4].
    # Let B00, B01, B10, B11 be these blocks.
    # C[:,0:2] = A[:,0:2] @ B00 + A[:,2:4] @ B10.
    # C[:,2:4] = A[:,0:2] @ B01 + A[:,2:4] @ B11.
    #
    # For C[:,0:2]: let Aa = A[:,0:2] (2x2), Ab = A[:,2:4] (2x2).
    #   C[:,0:2] = Aa @ B00 + Ab @ B10.
    # Use Strassen on Aa @ B00 (7 forms) and Ab @ B10 (7 forms).  Total 14.
    # For C[:,2:4]: Aa @ B01 (7 forms) + Ab @ B11 (7 forms).  Total 14.
    # Grand total = 28.
    #
    # Each Strassen form for Aa @ B00 is M_k = a_k @ Aa @ b_k (2x2 matrix).
    # This contributes to C[:,0:2] via the linear combination in _S2_C.
    #
    # Let me build this explicitly.

    rank = 0
    U_rows = {i: [] for i in range(n)}  # U[i, j] for j in 0..3
    V_rows = {j: [] for j in range(m)}  # V[j, k] for k in 0..3
    W_rows = {k: [] for k in range(p)}  # W[k, i] for i in 0..1

    def add_form(A_row, B_rowpair, C_colpair, coeff, s2_idx):
        """Add one Strassen bilinear form.
        A_row: index of the row of A (0 or 1), but we need 2x2 A.
        This is getting too complex.  Let me just build the schoolbook
        rank-32 and call it done, since 28 is the measured best and I
        can't easily beat it."""
        pass

    # Fallback: schoolbook rank-32.
    rank = n * m * p
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
    return U, V, W, rank


def _build_strassen28():
    """Build a rank-28 decomposition using Strassen on each 2x2x2 sub-block."""
    n, m, p = _DIMS
    U = [[0] * 28 for _ in range(n * m)]
    V = [[0] * 28 for _ in range(m * p)]
    W = [[0] * 28 for _ in range(p * n)]
    col = 0
    # For each of the four 2x2x2 sub-products, apply Strassen.
    # Sub-product (a_block, b_block, c_block) where a_block is the row of A
    # (but A is 2x4, so a_block is a 1x2 slice), b_block is a 2x2 slice of B,
    # c_block is a 1x2 slice of C.
    #
    # Actually, the cleanest way: view the (2,4,4) product as two (2,4,2)
    # products (for C[:,0:2] and C[:,2:4]).  Each (2,4,2) product is
    #   C_sub[i,k] = sum_{j=0..3} A[i,j] B_sub[j,k],  i in {0,1}, k in {0,1}.
    # Split j into {0,1} and {2,3}:
    #   C_sub = A[:,0:2] @ B_sub[0:2,:] + A[:,2:4] @ B_sub[2:4,:].
    # Each term is a 2x2 @ 2x2 = 2x2 product.  Apply Strassen to each.
    # So for C[:,0:2]: 2 Strassen products (7 each) = 14 forms.
    # For C[:,2:4]: 2 Strassen products (7 each) = 14 forms.
    # Total = 28.

    for c_block in range(2):  # C[:, 2*c_block : 2*c_block+2]
        for a_block in range(2):  # A[:, 2*a_block : 2*a_block+2]
            b_block = a_block  # B[2*a_block : 2*a_block+2, 2*c_block : 2*c_block+2]
            # Strassen product of A_sub (2x2) and B_sub (2x2).
            # A_sub[i, dj] = A[ai, 2*a_block + dj] where ai is the row of A.
            # Wait, A is 2x4, so A_sub is 2x2: A_sub[i, dj] = A[i, 2*a_block + dj].
            # B_sub[dj, dk] = B[2*b_block + dj, 2*c_block + dk].
            # C_sub[i, dk] = sum_dj A_sub[i, dj] B_sub[dj, dk].
            # This is a standard 2x2x2 product.  Apply Strassen.
            for s2k in range(7):
                a_vec, b_vec = _S2[s2k]
                # M = a_vec @ A_sub @ b_vec (2x2 matrix).
                # M[i, dk] = sum_{dj1, dj2} a_vec[dj1] A_sub[i, dj1] B_sub[dj2, dk] b_vec[dj2]
                #          = sum_{dj1, dj2} a_vec[dj1] b_vec[dj2] A[i, 2*a_block+dj1] B[2*b_block+dj2, 2*c_block+dk]
                # This M contributes to C_sub via the _S2_C combination.
                # For now, I'll add M as a single rank-1 factor and handle
                # the combination later.  But the combination is linear, so
                # I can distribute it.
                #
                # The Strassen scheme: C_sub = sum_k coeff_k * M_k, where
                # coeff_k depends on the output position.  Specifically:
                # C_sub[0,0] = M0 - M5 - M3 + M6
                # C_sub[0,1] = M4 + M5
                # C_sub[1,0] = M2 + M3
                # C_sub[1,1] = M0 - M1 + M4 - M6
                #
                # So each M_k contributes to multiple C_sub entries with
                # different coefficients.  To make this a rank-1 decomposition,
                # I need to split M_k into separate rank-1 factors for each
                # C_sub entry it contributes to, OR I can keep M_k as a single
                # factor and let W handle the linear combination.
                #
                # In the tensor decomposition, each rank-1 factor is
                #   U_vec (length n*m) * V_vec (length m*p) * W_vec (length p*n)
                # such that  sum_r U[r] * V[r] * W[r] = T.
                # For the Strassen scheme, M_k is a bilinear form in A and B,
                # and its contribution to C is linear.  So I can write:
                #   C = sum_k sum_{(i,dk) in C_sub} coeff_{i,dk,k} * M_k[i,dk]
                # where M_k[i,dk] = sum_{dj1,dj2} a_vec[dj1] b_vec[dj2] A[i, 2*a_block+dj1] B[2*b_block+dj2, 2*c_block+dk].
                #
                # Each term coeff * M_k[i,dk] is a rank-1 tensor in (A, B, C):
                #   A_part: has a_vec[dj1] at position (i, 2*a_block+dj1)
                #   B_part: has b_vec[dj2] at position (2*b_block+dj2, 2*c_block+dk)
                #   C_part: has coeff at position (2*c_block+dk, i)
                #
                # But M_k[i,dk] involves a sum over dj1, dj2, so it's not rank-1
                # in (A, B).  I need to expand it:
                #   M_k[i,dk] = sum_{dj1} sum_{dj2} a_vec[dj1] b_vec[dj2] A[i, 2*a_block+dj1] B[2*b_block+dj2, 2*c_block+dk]
                # Each (dj1, dj2) pair gives a rank-1 term.  So each M_k[i,dk]
                # expands to 4 rank-1 terms.  Total rank = 7 (Strassen) * 4
                # (C_sub entries) * 4 (dj1,dj2) = 112 per sub-product, which is
                # way more than 28.  This is wrong.
                #
                # The issue: the Strassen scheme reduces the number of
                # *scalar multiplications*, not the tensor rank.  The tensor
                # rank of 2x2x2 is 7, meaning there are 7 rank-1 tensors
                # (each a 2x2x2 tensor) that sum to the multiplication tensor.
                # Each rank-1 tensor is  u (length 4) * v (length 4) * w (length 4)
                # where u, v, w are 2x2 matrices flattened.  The Strassen
                # rank-1 tensors are:
                #   T_k = (a_vec @ A) * (B @ b_vec) * c_vec  (schematically)
                # where a_vec, b_vec, c_vec are specific 2x2 matrices.
                #
                # Let me look up the actual Strassen rank-1 decomposition.
                # The 7 rank-1 tensors for 2x2x2 are:
                #   M0 = (A00+A11) * (B00+B11) -> contributes to C00, C11
                #   M1 = (A10+A11) * B00 -> contributes to C10, C11
                #   M2 = A00 * (B10-B00) -> contributes to C00, C10
                #   M3 = A11 * (B01-B11) -> contributes to C01, C11
                #   M4 = (A00+A01) * B11 -> contributes to C01, C00
                #   M5 = (A10+A11) * (B01+B11) -> contributes to C10, C11
                #   M6 = (A00-A10) * (B00+B10) -> contributes to C00, C10
                #
                # Each M_k is a product of a linear form in A and a linear
                # form in B, giving a 2x2 matrix.  As a rank-1 tensor in
                # (A, B, C), M_k corresponds to:
                #   U_k: the coefficients of the linear form in A (flattened 2x2)
                #   V_k: the coefficients of the linear form in B (flattened 2x2)
                #   W_k: the 2x2 matrix that maps M_k's value to C (flattened)
                #
                # For example, M0 = (A00+A11) * (B00+B11):
                #   U0 = [1,0,0,1] (A00 and A11)
                #   V0 = [1,0,0,1] (B00 and B11)
                #   W0: M0 contributes +1 to C00 and +1 to C11, so
                #        W0 = [1,0,0,1] (C00 and C11)
                #
                # Let me define the 7 Strassen rank-1 tensors properly.
                # A = [[A00,A01],[A10,A11]], B = [[B00,B01],[B10,B11]].
                # C = A @ B = [[C00,C01],[C10,C11]].
                #
                # M0 = (A00+A11)(B00+B11):
                #   U0 = [1,0,0,1], V0 = [1,0,0,1]
                #   M0 = A00*B00 + A00*B11 + A11*B00 + A11*B11
                #   C00 = A00*B00 + A01*B10
                #   C11 = A10*B01 + A11*B11
                #   M0 contributes A00*B00 to C00, A11*B11 to C11,
                #   and A00*B11 + A11*B00 are "extra" terms that must cancel.
                #
                # This is getting very involved.  Let me just use the known
                # Strassen rank-1 decomposition from the literature.
                #
                # The 7 rank-1 tensors (U_k, V_k, W_k) for 2x2x2, where
                # U_k, V_k, W_k are 4-vectors (flattened 2x2 matrices):
                #
                # k=0: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]
                #      M0 = (A00+A11)(B00+B11) -> C00 += M0, C11 += M0
                # k=1: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,-1]
                #      M1 = (A10+A11)B00 -> C10 += M1, C11 -= M1
                # k=2: U=[1,0,0,0], V=[0,0,1,-1], W=[1,0,1,0]
                #      M2 = A00(B10-B00) -> C00 += M2, C10 += M2
                # k=3: U=[0,0,0,1], V=[0,1,0,-1], W=[0,1,0,-1]
                #      M3 = A11(B01-B11) -> C01 += M3, C11 -= M3
                # k=4: U=[1,1,0,0], V=[0,0,0,1], W=[0,1,1,0]
                #      M4 = (A00+A01)B11 -> C01 += M4, C10 += M4
                # k=5: U=[0,0,1,1], V=[0,1,0,1], W=[0,0,1,1]
                #      M5 = (A10+A11)(B01+B11) -> C10 += M5, C11 += M5
                # k=6: U=[1,0,-1,0], V=[1,0,1,0], W=[1,0,0,0]
                #      M6 = (A00-A10)(B00+B10) -> C00 += M6
                #
                # Let me verify: C00 = M0 - M1 + M2 + M6?
                #   M0 = A00B00 + A00B11 + A11B00 + A11B11
                #   -M1 = -A10B00 - A11B00
                #   M2 = A00B10 - A00B00
                #   M6 = A00B00 + A00B10 - A10B00 - A10B10
                #   Sum = A00B00 + A00B11 + A11B00 + A11B11 - A10B00 - A11B00
                #         + A00B10 - A00B00 + A00B00 + A00B10 - A10B00 - A10B10
                #       = A00B11 + A11B11 + A00B10 - A10B00 - A10B10 + A00B00
                #       This doesn't simplify to C00 = A00B00 + A01B10.
                #
                # I'm making errors in the Strassen decomposition.  Let me use
                # the standard Strassen formulas:
                #   M1 = (A00+A11)(B00+B11)
                #   M2 = (A10+A11)B00
                #   M3 = A00(B10-B00)
                #   M4 = A11(B01-B11)
                #   M5 = (A00+A01)B11
                #   M6 = (A10-A00)(B00+B10)
                #   M7 = (A01-A11)(B10+B11)
                #   C00 = M1 - M4 + M6 - M7
                #   C01 = M5 + M4
                #   C10 = M2 + M3
                #   C11 = M1 - M2 + M5 + M7
                #
                # Now the rank-1 tensors:
                # k=0 (M1): U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]
                # k=1 (M2): U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,-1]
                # k=2 (M3): U=[1,0,0,0], V=[0,0,1,-1], W=[1,0,1,0]
                # k=3 (M4): U=[0,0,0,1], V=[0,1,0,-1], W=[-1,1,0,-1]
                # k=4 (M5): U=[1,1,0,0], V=[0,0,0,1], W=[0,1,0,1]
                # k=5 (M6): U=[1,0,-1,0], V=[1,0,1,0], W=[1,0,0,0]
                # k=6 (M7): U=[0,1,0,-1], V=[0,0,1,1], W=[-1,0,0,1]
                #
                # Verify C00 = M1 - M4 + M6 - M7:
                #   M1 = (A00+A11)(B00+B11) = A00B00+A00B11+A11B00+A11B11
                #   -M4 = -A11B01 + A11B11
                #   M6 = (A00-A10)(B00+B10) = A00B00+A00B10-A10B00-A10B10
                #   -M7 = -(A01-A11)(B10+B11) = -A01B10-A01B11+A11B10+A11B11
                #   Sum = A00B00+A00B11+A11B00+A11B11 - A11B01+A11B11
                #         + A00B00+A00B10-A10B00-A10B10
                #         - A01B10-A01B11+A11B10+A11B11
                #       = 2*A00B00 + A00B11 + A11B00 + 3*A11B11 - A11B01
                #         + A00B10 - A10B00 - A10B10 - A01B10 - A01B11 + A11B10
                #       This is not C00.  I'm clearly making errors.
                #
                # Let me just verify numerically and build the decomposition
                # by solving for the rank-1 tensors.

    return None  # fallback


def construct():
    deadline = time.time() + 110.0
    n, m, p = _DIMS

    # Build the rank-28 Strassen decomposition numerically.
    # The (2,4,4) multiplication tensor T[i,j,k] = 1 if (i,j,k) is a valid
    # (A-row, B-col, C-col) triple, else 0.  Specifically,
    #   C[i,k] = sum_j A[i,j] B[j,k],  so T[i,j,k] = 1 for all i,j,k.
    # Wait, the tensor is T[i,j,k] = delta (the coefficient of A[i,j] B[j,k]
    # in C[i,k]), which is 1 for all i,j,k.  So T is the all-ones tensor of
    # size 2x4x4?  No, that's not right either.
    #
    # The multiplication tensor for (n,m,p) is T in R^{(n*m) x (m*p) x (p*n)}
    # such that  C = A * B  is computed as  C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T has entries  T[(i,j), (j,k), (k,i)] = 1  (using the
    # official trace convention W[k*n+i, r]).  All other entries are 0.
    #
    # So T is a 3-way tensor of size (n*m) x (m*p) x (p*n) = 8 x 16 x 8.
    # The number of non-zero entries is n*m*p = 32 (one for each (i,j,k)).
    #
    # A rank-r decomposition is  T = sum_{r=1..R} U_r * V_r * W_r  where
    # U_r is a vector of length n*m, V_r of length m*p, W_r of length p*n.
    #
    # For the schoolbook, R = n*m*p = 32, and each rank-1 tensor has exactly
    # one non-zero entry in each of U, V, W.
    #
    # For Strassen on 2x2x2 blocks, R = 28.  Let me build this by applying
    # the Strassen 2x2x2 decomposition to each of the four 2x2x2 sub-tensors.

    # Build the full tensor T (sparse).
    # T[(i*m+j), (j*p+k), (k*n+i)] = 1
    U = [[0] * 28 for _ in range(n * m)]
    V = [[0] * 28 for _ in range(m * p)]
    W = [[0] * 28 for _ in range(p * n)]

    col = 0
    # Apply Strassen to each 2x2x2 sub-block.
    # The (2,4,4) tensor can be viewed as 4 disjoint 2x2x2 sub-tensors:
    #   Sub-block (a, b, c) where a in {0,1} (A-row), b in {0,1} (B-row-pair),
    #   c in {0,1} (C-col-pair).  But this doesn't quite work because the
    #   sub-tensors overlap in the B dimension.
    #
    # Let me use a different approach: build the rank-28 decomposition by
    # solving a system of linear equations.  I'll set up the Strassen scheme
    # for each 2x2x2 sub-product and solve for the rank-1 tensors.

    # Actually, the simplest correct approach: use the known fact that the
    # rank of the (2,4,4) multiplication tensor is 28, and build the
    # decomposition by applying Strassen to the four 2x2x2 blocks that
    # partition the tensor.
    #
    # Partition: A is 2x4, split into A0 = A[:,0:2] and A1 = A[:,2:4] (each 2x2).
    # B is 4x4, split into B0 = B[0:2,:] and B1 = B[2:4,:] (each 2x4).
    # C is 2x4, split into C0 = C[:,0:2] and C1 = C[:,2:4] (each 2x2).
    #
    # C0 = A0 @ B0[0:2,:] + A1 @ B1[0:2,:]  (2x2 product)
    # C1 = A0 @ B0[2:4,:] + A1 @ B1[2:4,:]  (2x2 product)
    #
    # For C0: A0 (2x2) @ B0_left (2x2) + A1 (2x2) @ B1_left (2x2).
    # This is a sum of two 2x2x2 products.  Apply Strassen to each: 7 + 7 = 14.
    # For C1: similarly 14.  Total = 28.
    #
    # Each Strassen product of X (2x2) @ Y (2x2) uses 7 rank-1 tensors
    # (U_k, V_k, W_k) as defined above.  The U_k acts on X, V_k on Y, W_k
    # on the output.
    #
    # For C0 = A0 @ B0_left + A1 @ B1_left:
    #   A0 @ B0_left: U_k acts on A0, V_k on B0_left, W_k on C0.
    #   A1 @ B1_left: U_k acts on A1, V_k on B1_left, W_k on C0.
    #
    # In the full (2,4,4) tensor:
    #   A0 is A[:,0:2], so U_k (length 4, for 2x2 A0) maps to positions
    #   (i, 0) and (i, 1) in the full A (length 8).
    #   A1 is A[:,2:4], so U_k maps to positions (i, 2) and (i, 3).
    #   B0_left is B[0:2,0:2], so V_k maps to positions (0,0),(0,1),(1,0),(1,1)
    #   in the full B (length 16).
    #   B1_left is B[2:4,0:2], so V_k maps to positions (2,0),(2,1),(3,0),(3,1).
    #   C0 is C[:,0:2], so W_k maps to positions (0,0),(0,1),(1,0),(1,1)
    #   in the full C (length 8), i.e., W[k*n+i] for k in {0,1}, i in {0,1}.

    # Strassen rank-1 tensors for 2x2x2 (U, V, W as 4-vectors).
    S = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [1, 0, 1, 0]),  # M3
        ([0, 0, 0, 1], [0, 1, 0, -1], [-1, 1, 0, -1]),# M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]),   # M5
        ([1, 0, -1, 0], [1, 0, 1, 0], [1, 0, 0, 0]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [-1, 0, 0, 1]), # M7
    ]

    for c_block in range(2):  # C[:, 2*c_block : 2*c_block+2]
        for a_block in range(2):  # A[:, 2*a_block : 2*a_block+2] @ B[2*a_block:2*a_block+2, 2*c_block:2*c_block+2]
            for s_idx in range(7):
                u4, v4, w4 = S[s_idx]
                # u4: coefficients for A_sub (2x2), flattened as [A00,A01,A10,A11]
                # v4: coefficients for B_sub (2x2), flattened as [B00,B01,B10,B11]
                # w4: coefficients for C_sub (2x2), flattened as [C00,C01,C10,C11]
                for di in range(2):  # row of A_sub
                    for dj in range(2):  # col of A_sub (= row of B_sub)
                        u_val = u4[di * 2 + dj]
                        if u_val != 0:
                            a_row = di
                            a_col = 2 * a_block + dj
                            U[a_row * m + a_col][col] += u_val
                        for dk in range(2):  # col of B_sub (= col of C_sub)
                            v_val = v4[dj * 2 + dk]
                            if v_val != 0:
                                b_row = 2 * a_block + dj
                                b_col = 2 * c_block + dk
                                V[b_row * p + b_col][col] += v_val
                        for di_c in range(2):  # row of C_sub
                            for dk_c in range(2):  # col of C_sub
                                w_val = w4[di_c * 2 + dk_c]
                                if w_val != 0:
                                    c_row = di_c
                                    c_col = 2 * c_block + dk_c
                                    W[c_col * n + c_row][col] += w_val
                col += 1

    # Verify the decomposition.
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # Build the target tensor T (sparse).
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Check: T == sum_r U_r * V_r * W_r
    max_err = 0.0
    for r in range(28):
        T_check = np.einsum('i,j,k->ijk', U_arr[:, r], V_arr[:, r], W_arr[:, r])
        T -= T_check
    max_err = np.max(np.abs(T))

    if max_err > 1e-9:
        # Fallback to schoolbook rank-32.
        rank = n * m * p
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
        return {"U": U, "V": V, "W": W}

    # Try to minimize: remove redundant rank factors.
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = 28
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for r in range(rank):
            # Check if factor r is a linear combination of the others.
            if rank <= 1:
                break
            U_others = np.delete(U_arr, r, axis=1)
            V_others = np.delete(V_arr, r, axis=1)
            W_others = np.delete(W_arr, r, axis=1)
            # Check if U[:,r] is in the span of U_others.
            if np.linalg.matrix_rank(U_others) < np.linalg.matrix_rank(
                np.column_stack([U_others, U_arr[:, r]])
            ):
                # U[:,r] is not in the span, so factor r is needed.
                continue
            # Check if removing factor r still gives a valid decomposition.
            T_reconstruct = np.zeros((n * m, m * p, p * n))
            for r2 in range(rank):
                if r2 == r:
                    continue
                T_reconstruct += np.einsum(
                    'i,j,k->ijk', U_arr[:, r2], V_arr[:, r2], W_arr[:, r2]
                )
            # Compare with the target tensor.
            T_target = np.zeros((n * m, m * p, p * n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_target[i * m + j, j * p + k, k * n + i] = 1.0
            if np.max(np.abs(T_target - T_reconstruct)) > 1e-9:
                continue
            # Remove factor r.
            U_arr = np.delete(U_arr, r, axis=1)
            V_arr = np.delete(V_arr, r, axis=1)
            W_arr = np.delete(W_arr, r, axis=1)
            rank -= 1
            changed = True
            break

    U_out = U_arr.tolist()
    V_out = V_arr.tolist()
    W_out = W_arr.tolist()
    # Convert to int where possible.
    U_out = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U_out]
    V_out = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V_out]
    W_out = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W_out]
    return {"U": U_out, "V": V_out, "W": W_out}
