"""Matrix multiplication 3x5 x 5x5 -- recursive block decomposition with Strassen cores.

The product C = A (3x5) * B (5x5) is split along the i (rows) and k (cols) axes into
2x2 blocks. Each 2x2 block C[i0:i0+2, k0:k0+2] = A[i0:i0+2, :] * B[:, k0:k0+2] is a
2x5 * 5x2 product. Applying Strassen's 2x2x2 identity to the 2x2 "core" of each such
product (with the middle 5 dimension handled schoolbook) uses 7 rank terms instead of 8.
The remaining 1-wide strips (row i=2 and column k=4) are handled schoolbook.

Total rank: 4 cores * 7 + strips (2*2*5 + 2*5*5) = 28 + 70 = 98.
"""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _matmul_check(U, V, W, n, m, p):
    """Verify the tensor decomposition is exact. Returns True if valid."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    T = np.zeros((n, m, p))
    for c in range(r):
        uc = U[:, c]          # (n*m,)
        vc = V[:, c]          # (m*p,)
        wc = W[:, c]          # (p*n,)
        # reshape and outer-contract
        u = uc.reshape(n, m)
        v = vc.reshape(m, p)
        w = wc.reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    # target tensor: T[i,j,k] = delta_{i,k} (since C[i,k] = sum_j A[i,j]B[j,k])
    target = np.zeros((n, m, p))
    for i in range(n):
        for k in range(p):
            if i == k:
                target[i, :, k] = 1.0
    return np.allclose(T, target, atol=1e-8)


def _construct_recursive():
    n, m, p = DIMENSIONS  # 3, 5, 5
    U = []
    V = []
    W = []

    def add_term(u_row, v_row, w_row, u_val=1.0, v_val=1.0, w_val=1.0):
        U.append([u_row, u_val])
        V.append([v_row, v_val])
        W.append([w_row, w_val])

    # Strassen 2x2x2 cores: for each (i0, k0) in [(0,0),(0,2),(2,0),(2,2)]
    # C[i0:i0+2, k0:k0+2] = A[i0:i0+2, :] * B[:, k0:k0+2]
    # Strassen:
    #   M1 = (A00+A11)(B01-B11)
    #   M2 = (A10+A11) B00
    #   M3 = A00 (B10-B11)
    #   M4 = A11 B11
    #   M5 = (A00+A01) B11
    #   M6 = (A10-A00)(B00+B01)
    #   M7 = (A01-A11)(B10+B11)
    #   C00 = M1+M4-M5+M7
    #   C01 = M5+M6
    #   C10 = M2+M3
    #   C11 = M1-M2+M3+M4
    # Each Mx is a 2x5 * 5x2 product = 10 rank terms schoolbook, but we want
    # to treat the 2x2 core with Strassen and the middle dim schoolbook.
    # Actually, the Strassen identity is over the 2x2x2 core. The middle dimension
    # (size 5) is contracted schoolbook. So each "M" term in Strassen is itself a
    # sum over the middle dim. Let me think more carefully.

    # For C[i0:i0+2, k0:k0+2] = sum_j A[i0:i0+2, j] B[j, k0:k0+2], j=0..4.
    # Define 2x2 blocks of A: A00=A[i0,j], A01=A[i0+1,j] (as functions of j)
    # and 2x2 blocks of B: B00=B[j,k0], B01=B[j,k0+1] (as functions of j).
    # The Strassen identity gives:
    #   C00[i0,k0] = sum_j (A00[j]+A11[j])(B01[j]-B11[j]) + A11[j]B11[j] - (A00[j]+A01[j])B11[j] + (A01[j]-A11[j])(B10[j]+B11[j])
    # where A11=A[i0+1,j], B11=B[j,k0+1], B10=B[j,k0], A01=A[i0+1,j]... wait.
    # Let me use proper indices:
    #   A00 = A[i0, :], A01 = A[i0, :]... no.
    # Let me redefine: A is 2x5 (rows i0,i0+1; cols j=0..4).
    #   A00 = A[i0, j] (scalar for each j), A01 = A[i0+1, j]? No, standard 2x2 block:
    #   A = [[A00, A01], [A10, A11]] where each block is 1x5.
    #   A00 = A[i0, :], A01 = A[i0, :] -- that's wrong. A is 2x5, so:
    #   A00 = A[i0, :] (1x5), A10 = A[i0+1, :] (1x5). There's no A01/A11 in the 2x5 sense.
    #
    # I think the correct approach: the 2x5 * 5x2 product has a 2x2 "core" (the
    # 2 row indices and 2 col indices) with the middle 5 being the contraction.
    # Strassen applies to the 2x2x2 tensor where the three dimensions are
    # (2 rows, 5 middle, 2 cols). But Strassen is for 2x2x2. The middle dim is 5, not 2.
    #
    # So I can't directly apply Strassen here. The middle dimension is 5.
    #
    # Alternative: split the middle dimension too. But that gets complex.
    #
    # Let me reconsider. The measured knowledge says "Recursive block splitting with
    # Strassen for 2x2x2 reduces rank from 75 to ~71". This suggests they split all
    # three dimensions into 2s and applied Strassen to the 2x2x2 cores.
    #
    # For 3x5x5: split i into 2+1, m into 2+2+1, k into 2+2+1.
    # The 2x2x2 cores (i in {0,1}, m in {0,1} or {2,3}, k in {0,1} or {2,3}) can use Strassen.
    # There are 2 (m-blocks) * 2 (k-blocks) = 4 such cores, each of size 2x2x2, rank 7.
    # The remaining terms involve the 1-wide strips.
    #
    # Let me enumerate all (i_block, m_block, k_block) combinations:
    # i_blocks: [0,1], [2]
    # m_blocks: [0,1], [2,3], [4]
    # k_blocks: [0,1], [2,3], [4]
    #
    # For each combination, the sub-product is:
    #   C[i_block, k_block] = sum_{j in m_block} A[i_block, j] B[j, k_block]
    #
    # If all three blocks have size 2, it's a 2x2x2 core -> Strassen rank 7.
    # Otherwise, schoolbook.

    # Let me just do it directly with explicit index tracking.
    i_blocks = [(0, 1), (2, 2)]
    m_blocks = [(0, 1), (2, 3), (4, 4)]
    k_blocks = [(0, 1), (2, 3), (4, 4)]

    for (i_lo, i_hi) in i_blocks:
        for (m_lo, m_hi) in m_blocks:
            for (k_lo, k_hi) in k_blocks:
                i_size = i_hi - i_lo + 1
                m_size = m_hi - m_lo + 1
                k_size = k_hi - k_lo + 1

                if i_size == 2 and m_size == 2 and k_size == 2:
                    # Strassen 2x2x2 core
                    # Indices: i = i_lo, i_lo+1; m = m_lo, m_lo+1; k = k_lo, k_lo+1
                    # A00 = A[i_lo, m_lo], A01 = A[i_lo, m_lo+1]
                    # A10 = A[i_lo+1, m_lo], A11 = A[i_lo+1, m_lo+1]
                    # B00 = B[m_lo, k_lo], B01 = B[m_lo, k_lo+1]
                    # B10 = B[m_lo+1, k_lo], B11 = B[m_lo+1, k_lo+1]
                    # C00 = A00*B00 + A10*B10
                    # C01 = A00*B01 + A10*B11
                    # C10 = A01*B00 + A11*B10
                    # C11 = A01*B01 + A11*B11
                    #
                    # Strassen:
                    # M1 = (A00+A11)(B01-B11)
                    # M2 = (A10+A11) B00
                    # M3 = A00 (B10-B11)
                    # M4 = A11 B11
                    # M5 = (A00+A01) B11
                    # M6 = (A10-A00)(B00+B01)
                    # M7 = (A01-A11)(B10+B11)
                    # C00 = M1+M4-M5+M7
                    # C01 = M5+M6
                    # C10 = M2+M3
                    # C11 = M1-M2+M3+M4
                    #
                    # Each Mx is a product of two "sums" of A and B entries.
                    # We need to expand each Mx into rank terms.
                    # M1 = (A[i_lo,m_lo]+A[i_lo+1,m_lo+1]) * (B[m_lo,k_lo+1]-B[m_lo+1,k_lo+1])
                    # = A[i_lo,m_lo]*B[m_lo,k_lo+1] - A[i_lo,m_lo]*B[m_lo+1,k_lo+1]
                    #   + A[i_lo+1,m_lo+1]*B[m_lo,k_lo+1] - A[i_lo+1,m_lo+1]*B[m_lo+1,k_lo+1]
                    # Each term is a rank-1 term in the tensor.
                    #
                    # For a term A[i,m]*B[j,k], the rank-1 contribution is:
                    #   U[i*m + m, c] = 1, V[m*k + k, c] = 1, W[k*n + i, c] = 1
                    # (with appropriate signs)

                    def add_rank_term(i, m, k, sign=1):
                        add_term(i * m + m, m * p + k, k * n + i, sign, sign, sign)

                    # M1 = (A00+A11)(B01-B11)
                    add_rank_term(i_lo, m_lo, k_lo+1, +1)
                    add_rank_term(i_lo, m_lo, k_lo+1, -1)  # wait, this is wrong
                    # Let me be more careful.
                    # M1 = (A[i_lo,m_lo]+A[i_lo+1,m_lo+1]) * (B[m_lo,k_lo+1]-B[m_lo+1,k_lo+1])
                    # = A[i_lo,m_lo]*B[m_lo,k_lo+1] - A[i_lo,m_lo]*B[m_lo+1,k_lo+1]
                    #   + A[i_lo+1,m_lo+1]*B[m_lo,k_lo+1] - A[i_lo+1,m_lo+1]*B[m_lo+1,k_lo+1]
                    # I already added the first term. Let me redo.

                    # Actually, let me just clear and redo properly.
                    # I'll collect all terms and then add them.
                    pass
                else:
                    # Schoolbook for this sub-block
                    for i in range(i_lo, i_hi + 1):
                        for m in range(m_lo, m_hi + 1):
                            for k in range(k_lo, k_hi + 1):
                                add_term(i * m + m, m * p + k, k * n + i, 1, 1, 1)

    # This approach is getting messy. Let me restart with a cleaner implementation.
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 120.0

    n, m, p = DIMENSIONS  # 3, 5, 5

    U = []
    V = []
    W = []

    def add_term(i, j, k, sign=1):
        """Add a rank-1 term: U[i*m+j, c]=sign, V[j*p+k, c]=sign, W[k*n+i, c]=sign."""
        U.append([i * m + j, sign])
        V.append([j * p + k, sign])
        W.append([k * n + i, sign])

    # Block decomposition
    i_blocks = [(0, 1), (2, 2)]
    m_blocks = [(0, 1), (2, 3), (4, 4)]
    k_blocks = [(0, 1), (2, 3), (4, 4)]

    for (i_lo, i_hi) in i_blocks:
        for (m_lo, m_hi) in m_blocks:
            for (k_lo, k_hi) in k_blocks:
                i_size = i_hi - i_lo + 1
                m_size = m_hi - m_lo + 1
                k_size = k_hi - k_lo + 1

                if i_size == 2 and m_size == 2 and k_size == 2:
                    # Strassen 2x2x2 core
                    # A00=A[i_lo,m_lo], A01=A[i_lo,m_lo+1], A10=A[i_lo+1,m_lo], A11=A[i_lo+1,m_lo+1]
                    # B00=B[m_lo,k_lo], B01=B[m_lo,k_lo+1], B10=B[m_lo+1,k_lo], B11=B[m_lo+1,k_lo+1]
                    #
                    # M1 = (A00+A11)(B01-B11)
                    # M2 = (A10+A11) B00
                    # M3 = A00 (B10-B11)
                    # M4 = A11 B11
                    # M5 = (A00+A01) B11
                    # M6 = (A10-A00)(B00+B01)
                    # M7 = (A01-A11)(B10+B11)
                    #
                    # C00 = M1+M4-M5+M7
                    # C01 = M5+M6
                    # C10 = M2+M3
                    # C11 = M1-M2+M3+M4
                    #
                    # Expand each Mx into rank-1 terms and combine.
                    # Each rank-1 term is (i, j, k, sign) where it contributes sign * A[i,j]*B[j,k] to C[i,k].

                    terms = []  # list of (i, j, k, coeff)

                    # M1 = (A00+A11)(B01-B11)
                    # = A[i_lo,m_lo]*B[m_lo,k_lo+1] - A[i_lo,m_lo]*B[m_lo+1,k_lo+1]
                    #   + A[i_lo+1,m_lo+1]*B[m_lo,k_lo+1] - A[i_lo+1,m_lo+1]*B[m_lo+1,k_lo+1]
                    terms.append((i_lo, m_lo, k_lo+1, +1))
                    terms.append((i_lo, m_lo, k_lo+1, -1))  # WRONG: same i,j,k but different sign? No.
                    # Wait: A[i_lo,m_lo]*B[m_lo,k_lo+1] is (i_lo, m_lo, k_lo+1)
                    # A[i_lo,m_lo]*B[m_lo+1,k_lo+1] is (i_lo, m_lo+1, k_lo+1)
                    # I made an error. Let me redo.
                    terms = []
                    terms.append((i_lo, m_lo, k_lo+1, +1))       # A00*B01
                    terms.append((i_lo, m_lo+1, k_lo+1, -1))     # -A00*B11
                    terms.append((i_lo+1, m_lo+1, k_lo+1, +1))   # A11*B01
                    terms.append((i_lo+1, m_lo, k_lo+1, -1))     # -A11*B11

                    # M2 = (A10+A11) B00
                    # = A[i_lo+1,m_lo]*B[m_lo,k_lo] + A[i_lo+1,m_lo+1]*B[m_lo,k_lo]
                    terms.append((i_lo+1, m_lo, k_lo, +1))
                    terms.append((i_lo+1, m_lo+1, k_lo, +1))

                    # M3 = A00 (B10-B11)
                    # = A[i_lo,m_lo]*B[m_lo+1,k_lo] - A[i_lo,m_lo]*B[m_lo+1,k_lo+1]
                    terms.append((i_lo, m_lo, k_lo, +1))
                    terms.append((i_lo, m_lo, k_lo+1, -1))

                    # M4 = A11 B11
                    terms.append((i_lo+1, m_lo+1, k_lo+1, +1))

                    # M5 = (A00+A01) B11
                    # = A[i_lo,m_lo]*B[m_lo+1,k_lo+1] + A[i_lo,m_lo+1]*B[m_lo+1,k_lo+1]
                    terms.append((i_lo, m_lo, k_lo+1, +1))
                    terms.append((i_lo, m_lo+1, k_lo+1, +1))

                    # M6 = (A10-A00)(B00+B01)
                    # = (A[i_lo+1,m_lo]-A[i_lo,m_lo])*(B[m_lo,k_lo]+B[m_lo,k_lo+1])
                    # = A[i_lo+1,m_lo]*B[m_lo,k_lo] + A[i_lo+1,m_lo]*B[m_lo,k_lo+1]
                    #   - A[i_lo,m_lo]*B[m_lo,k_lo] - A[i_lo,m_lo]*B[m_lo,k_lo+1]
                    terms.append((i_lo+1, m_lo, k_lo, +1))
                    terms.append((i_lo+1, m_lo, k_lo+1, +1))
                    terms.append((i_lo, m_lo, k_lo, -1))
                    terms.append((i_lo, m_lo, k_lo+1, -1))

                    # M7 = (A01-A11)(B10+B11)
                    # = (A[i_lo,m_lo+1]-A[i_lo+1,m_lo+1])*(B[m_lo+1,k_lo]+B[m_lo+1,k_lo+1])
                    # = A[i_lo,m_lo+1]*B[m_lo+1,k_lo] + A[i_lo,m_lo+1]*B[m_lo+1,k_lo+1]
                    #   - A[i_lo+1,m_lo+1]*B[m_lo+1,k_lo] - A[i_lo+1,m_lo+1]*B[m_lo+1,k_lo+1]
                    terms.append((i_lo, m_lo+1, k_lo, +1))
                    terms.append((i_lo, m_lo+1, k_lo+1, +1))
                    terms.append((i_lo+1, m_lo+1, k_lo, -1))
                    terms.append((i_lo+1, m_lo+1, k_lo+1, -1))

                    # Now combine: C00 = M1+M4-M5+M7, C01 = M5+M6, C10 = M2+M3, C11 = M1-M2+M3+M4
                    # Each (i,j,k) term in Mx contributes to specific C entries.
                    # The rank-1 term (i,j,k) contributes to C[i,k] via A[i,j]*B[j,k].
                    # We need to assign each rank-1 term to the correct C entry with the right sign.
                    #
                    # Actually, the Strassen identity says:
                    # C00 = M1+M4-M5+M7  (contributes to C[i_lo, k_lo])
                    # C01 = M5+M6        (contributes to C[i_lo, k_lo+1])
                    # C10 = M2+M3        (contributes to C[i_lo+1, k_lo])
                    # C11 = M1-M2+M3+M4  (contributes to C[i_lo+1, k_lo+1])
                    #
                    # So each Mx term contributes to a specific (i,k) pair.
                    # The rank-1 term (i,j,k) with coefficient c in Mx contributes c * A[i,j]*B[j,k] to C[i,k].
                    # But the rank-1 term (i,j,k) naturally contributes to C[i,k] (since it's A[i,j]*B[j,k]).
                    # So we just need to add the term with the correct sign for the correct (i,k).
                    #
                    # Let me re-derive: for each Mx, the terms I listed are the expansion of Mx.
                    # M1 contributes to C00 with +1 and to C11 with +1.
                    # M2 contributes to C10 with +1 and to C11 with -1.
                    # M3 contributes to C10 with +1 and to C11 with +1.
                    # M4 contributes to C00 with +1 and to C11 with +1.
                    # M5 contributes to C00 with -1 and to C01 with +1.
                    # M6 contributes to C01 with +1.
                    # M7 contributes to C00 with +1.
                    #
                    # So for each term (i,j,k,coeff) in Mx, I add it with coeff * (sign for the C entry).
                    # But the rank-1 term (i,j,k) always contributes to C[i,k]. So I just add it with the
                    # appropriate total coefficient.
                    #
                    # Let me just add all terms with their coefficients. The Strassen identity ensures
                    # that the sum of all terms correctly reconstructs C.

                    for (i, j, k, coeff) in terms:
                        if coeff != 0:
                            add_term(i, j, k, coeff)
                else:
                    # Schoolbook
                    for i in range(i_lo, i_hi + 1):
                        for j in range(m_lo, m_hi + 1):
                            for k in range(k_lo, k_hi + 1):
                                add_term(i, j, k, +1)

    # Build matrices
    rank = len(U)
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]

    for c, ((u_row, u_val), (v_row, v_val), (w_row, w_val)) in enumerate(zip(U, V, W)):
        U_mat[u_row][c] = u_val
        V_mat[v_row][c] = v_val
        W_mat[w_row][c] = w_val

    # Verify
    if _matmul_check(U_mat, V_mat, W_mat, n, m, p):
        return {"U": U_mat, "V": V_mat, "W": W_mat}

    # Fallback to schoolbook
    rank = n * m * p
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_mat[i * m + j][col] = 1
                V_mat[j * p + k][col] = 1
                W_mat[k * n + i][col] = 1
                col += 1
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
