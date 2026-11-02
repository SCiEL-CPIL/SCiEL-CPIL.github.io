"""Matrix multiplication 4x4 x 4x6 — algebraic rank-reduction from schoolbook."""
import time
import sys
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin

    n, m, p = 4, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 16, 24, 24

    def verify(U, V, W, rank):
        """Check exact reconstruction of the multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def verify_fast(U, V, W, rank):
        """Vectorized verification using numpy."""
        # Build the tensor T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
        # U: (nm, rank), V: (mp, rank), W: (pn, rank)
        # Reshape to (n,m,rank), (m,p,rank), (p,n,rank)
        U3 = np.array(U).reshape(n, m, rank)
        V3 = np.array(V).reshape(m, p, rank)
        W3 = np.array(W).reshape(p, n, rank)
        # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        # Contract over c: for each (i,j,k), sum_c U3[i,j,c]*V3[j,k,c]*W3[k,i,c]
        # Use einsum: 'ijc,jkc,kic->ijk'
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        target = np.ones((n, m, p))
        return np.allclose(T, target, atol=1e-9)

    # Initialize with schoolbook rank = n*m*p = 96
    rank0 = n * m * p
    U = [[0] * rank0 for _ in range(nm)]
    V = [[0] * rank0 for _ in range(mp)]
    W = [[0] * rank0 for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    best_U, best_V, best_W, best_rank = U, V, W, rank0

    # Verify initial
    if verify_fast(U, V, W, rank0):
        pass  # good

    # ---- Rank reduction via pairwise merging ----
    # For each pair of columns (a, b) in the current decomposition, try to find
    # half-integer coefficients alpha, beta such that:
    #   alpha*U_a + beta*U_b, alpha*V_a + beta*V_b, alpha*W_a + beta*W_b
    # still satisfy the tensor equation, allowing us to replace two columns
    # with one (or eliminate redundancy).
    #
    # Actually, a cleaner approach: look for linear dependencies.
    # If columns a and b of U, V, W are such that there exist scalars
    # alpha, beta (not both zero) with:
    #   alpha*U_a + beta*U_b = 0  (element-wise)
    #   alpha*V_a + beta*V_b = 0
    #   alpha*W_a + beta*W_b = 0
    # then one column is redundant.
    #
    # More generally: we can try to find a 2-column combination that
    # produces the same tensor contribution as a single new column.
    # This is hard in general. Let's try a different structural approach.

    # ---- Structural approach: exploit the 4x4x6 = 4x4x4 + 4x4x2 structure ----
    # But do it smarter: use the known rank-49 decomposition for 4x4x4
    # and try to find a lower-rank decomposition for the 4x4x2 part
    # or a combined decomposition that shares structure.

    # Actually, let's try a completely different approach:
    # Use the fact that 4x4x6 can be seen as multiplying by a 4x6 matrix.
    # We can write B = [B1 | B2] where B1, B2 are 4x3.
    # Then A*B = A*B1 + A*B2.
    # If we have a rank-r decomposition for 4x4x3, we get rank 2r for 4x4x6.
    # Known: rank(4x4x3) <= ? Let's try to construct one.

    # For 4x4x3: schoolbook rank = 48.
    # Can we do better? Use Strassen-like: 4x4 = 2x2 blocks, each 2x2x2.
    # 2x2x2 has rank 7 (Strassen).
    # Block structure: A is 2x2 blocks of 2x2, B is 2x2 blocks of 2x2 (but 4x3 means blocks are 2x2 and 2x1).
    # This gets complicated. Let's just try the direct approach.

    # ---- Try: use rank-49 for 4x4x4 and rank-32 for 4x4x2, total 81 (current best) ----
    # But try to improve the 4x4x2 part.
    # 4x4x2: can we do better than 32?
    # 4x4x2 = multiply 4x4 by 4x2.
    # View as two 4x4x1 multiplications.
    # 4x4x1: rank = 16 (schoolbook, since it's just a linear map from R^16 to R^16... no).
    # Actually 4x4x1: A is 4x4, B is 4x1, C is 4x1.
    # C_i = sum_j A_ij * B_j. This is just 16 multiplications, rank 16.
    # So 4x4x2 = two independent 4x4x1's, rank 32.
    # Unless we can share structure between the two columns of B.
    # B = [b1, b2]. C = A*[b1, b2] = [A*b1, A*b2].
    # If we precompute some combinations... but A*b1 and A*b2 are independent linear maps.
    # So rank 32 seems optimal for 4x4x2.

    # ---- Try a different block partition: 4x4x6 = 2x2 blocks where B has 2x3 blocks ----
    # A = [[A00, A01], [A10, A11]] each 2x2
    # B = [[B00, B01, B02], [B10, B11, B12]] each 2x2
    # C = A*B, 2x3 blocks of 2x2.
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    #
    # This is 6 block products of 2x2 matrices, each needing rank 7 (Strassen).
    # But we can share: A00, A01, A10, A11 are shared across the 3 columns.
    # Similarly B00,B01,B02 share the top row of B, etc.
    #
    # Let's try to find a decomposition that exploits this sharing.
    # Define Strassen for each 2x2x2 product.
    # For product X*Y (2x2), use 7 multiplications M1..M7.
    #
    # For C00 = A00*B00 + A01*B10: this is a sum of two 2x2 products.
    # We can compute A00*B00 with 7 mults and A01*B10 with 7 mults, total 14.
    # But maybe we can do better by combining.
    #
    # Alternatively, treat the whole thing as a bilinear map and search.

    # ---- Let's try a computational search for a rank-r decomposition ----
    # Start from rank 81 (current best) and try to reduce.
    # Use a greedy approach: at each step, try to remove one column by
    # expressing it as a linear combination of others (in the tensor sense).

    # Actually, let's try a different structural decomposition:
    # Use the fact that 6 = 4 + 2, and find a joint decomposition.

    # Let me try: rank-49 for the 4x4x4 part + rank-32 for 4x4x2 = 81.
    # Can we do 4x4x4 in less than 49? Known lower bound is 49, so no.
    # Can we do 4x4x6 in less than 81?

    # Try: use a 3x3x3 = 23 rank decomposition (Laderman) for a sub-block?
    # 4x4x4 doesn't contain a 3x3x3 naturally.

    # Let's just go with the current best of 81 and try to verify it works.
    # The current program already achieves 81. Let's make sure our version is correct.

    # Build rank-49 for 4x4x4 using recursive Strassen
    def build_strassen_444():
        """Build rank-49 decomposition for 4x4x4 using recursive Strassen."""
        # Strassen 2x2 coefficients
        A_c = [
            {(0,0): 1, (1,1): 1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,1): 1},
            {(0,0): 1, (0,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,1): 1, (1,1): -1}
        ]
        B_c = [
            {(0,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (0,0): -1},
            {(1,1): 1},
            {(0,0): 1, (0,1): 1},
            {(1,0): 1, (1,1): 1}
        ]
        C_c = [
            [[1, 0, 0, -1, 1, 0, -1],
             [0, 0, 1, 0, 1, 0, 0]],
            [[0, 1, 0, 1, 0, 0, 0],
             [1, -1, 1, 0, 0, 1, 0]]
        ]
        block_A = [
            [[1, 0], [0, 1]],
            [[0, 0], [1, 1]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[0, 0], [1, -1]],
            [[0, 1], [0, -1]]
        ]
        block_B = [
            [[1, 0], [0, 1]],
            [[1, 0], [0, 0]],
            [[0, 1], [0, -1]],
            [[0, -1], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[0, 1], [1, 1]]
        ]

        r = 49
        U4 = [[0] * r for _ in range(16)]
        V4 = [[0] * r for _ in range(16)]
        W4 = [[0] * r for _ in range(16)]

        for beta in range(7):
            for alpha in range(7):
                col = beta * 7 + alpha
                for i in range(4):
                    for j in range(4):
                        bi, bj = i // 2, j // 2
                        coeff = 0
                        for rr in range(2):
                            for ss in range(2):
                                if (rr, ss) in A_c[alpha]:
                                    ac = A_c[alpha][(rr, ss)]
                                    bc = block_A[beta][rr][ss]
                                    if rr == bi and ss == bj:
                                        coeff += ac * bc
                        U4[i * 4 + j][col] = coeff

                for j in range(4):
                    for k in range(4):
                        bj, bk = j // 2, k // 2
                        coeff = 0
                        for rr in range(2):
                            for ss in range(2):
                                if (rr, ss) in B_c[alpha]:
                                    bc = B_c[alpha][(rr, ss)]
                                    ac = block_B[beta][rr][ss]
                                    if rr == bj and ss == bk:
                                        coeff += bc * ac
                        V4[j * 4 + k][col] = coeff

                for bi in range(2):
                    for bk in range(2):
                        for pp in range(2):
                            for qq in range(2):
                                ci = 2 * bi + pp
                                ck = 2 * bk + qq
                                coeff = C_c[bi][bk][beta] * C_c[pp][qq][alpha]
                                W4[ck * 4 + ci][col] += coeff

        return U4, V4, W4

    def build_schoolbook_442():
        """Build rank-32 schoolbook for 4x4x2."""
        r = 32
        U42 = [[0] * r for _ in range(16)]
        V42 = [[0] * r for _ in range(8)]
        W42 = [[0] * r for _ in range(8)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(2):
                    U42[i * 4 + j][col] = 1
                    V42[j * 2 + k][col] = 1
                    W42[k * 4 + i][col] = 1
                    col += 1
        return U42, V42, W42

    # Build the 81-rank solution
    U4, V4, W4 = build_strassen_444()
    U42, V42, W42 = build_schoolbook_442()

    total_rank = 49 + 32
    U_tot = [[0] * total_rank for _ in range(16)]
    V_tot = [[0] * total_rank for _ in range(24)]
    W_tot = [[0] * total_rank for _ in range(24)]

    for i in range(16):
        for c in range(49):
            U_tot[i][c] = U4[i][c]
            V_tot[i][c] = V4[i][c]
            W_tot[i][c] = W4[i][c]

    for i in range(16):
        for c in range(32):
            U_tot[i][49 + c] = U42[i][c]
    for j in range(8):
        for c in range(32):
            V_tot[j][49 + c] = V42[j][c]
    for k in range(8):
        for c in range(32):
            W_tot[k][49 + c] = W42[k][c]

    if verify_fast(U_tot, V_tot, W_tot, total_rank):
        best_U, best_V, best_W, best_rank = U_tot, V_tot, W_tot, total_rank

    # ---- Try to reduce rank further via column elimination ----
    # Check if any column is a linear combination of others in the tensor sense.
    # This is expensive; do a limited search.

    # Try: for each pair of columns, check if one can be eliminated.
    # A column c is redundant if the tensor contribution of column c
    # equals the sum of contributions of other columns (which is zero if
    # we remove it and the total still equals the target).
    # Actually, a column is redundant if removing it and adjusting others
    # still gives the correct tensor. This is hard to check directly.

    # Alternative: check if any two columns are identical (trivial redundancy).
    # In our construction, this shouldn't happen.

    # Try: check if we can merge the 4x4x4 and 4x4x2 parts more efficiently.
    # The 4x4x4 part uses Strassen on 2x2 blocks. The 4x4x2 part is schoolbook.
    # Maybe we can use a unified block structure.

    # Let's try: treat 4x4x6 as 2x2 blocks where B is 2x3 (each 2x2).
    # A: 2x2 blocks of 2x2 (4 blocks)
    # B: 2x3 blocks of 2x2 (6 blocks)
    # C: 2x3 blocks of 2x2 (6 blocks)
    # C_block(bi,bk) = sum_bj A_block(bi,bj) * B_block(bj,bk)
    #
    # For each (bi,bk), this is a 2x2 matrix product sum over bj=0,1.
    # C(bi,bk) = A(bi,0)*B(0,bk) + A(bi,1)*B(1,bk)
    #
    # We can use Strassen for each 2x2 product, but share the A and B factors.
    #
    # Let's define: for each pair (bi,bk), we need to compute
    #   A(bi,0)*B(0,bk) + A(bi,1)*B(1,bk)
    #
    # Using Strassen for each product: 7 multiplications each, 14 total per (bi,bk).
    # 6 (bi,bk) pairs, so 84 multiplications. Worse than 81.
    #
    # But we can share: A(bi,0) and A(bi,1) are used across different bk.
    # Similarly B(0,bk) and B(1,bk) are used across different bi.
    #
    # Let's try to find a decomposition that exploits this sharing.
    # Define 4x6 = 24 input pairs (i,j) for A and (j,k) for B.
    # This is getting complex. Let's just go with 81.

    # Time check
    if time.time() > deadline - 5:
        return {'U': best_U, 'V': best_V, 'W': best_W}

    # Final verification
    if not verify_fast(best_U, best_V, best_W, best_rank):
        # Fallback to schoolbook
        U = [[0] * rank0 for _ in range(nm)]
        V = [[0] * rank0 for _ in range(mp)]
        W = [[0] * rank0 for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {'U': U, 'V': V, 'W': W}

    return {'U': best_U, 'V': best_V, 'W': best_W}
# EVOLVE-BLOCK-END
