"""
Matrix multiplication 6x6x6 via Strassen decomposition at the block level.

Strategy:
  Partition each 6x6 matrix into a 2x2 grid of 3x3 blocks.
  Use Strassen's 7-multiplication scheme at the block level.
  Each of the 7 block multiplications is a 3x3 x 3x3 -> 3x3 product,
  which we decompose with the schoolbook rank-27 method.
  Total rank = 7 * 27 = 189.
"""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]


def construct():
    deadline = time.time() + 110  # safety margin before 120s limit

    n, m, p = 6, 6, 6
    # Block partition: 2x2 blocks of size 3x3
    b = 3  # block size

    # Strassen's 7 intermediate products for 2x2 block multiplication:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # Result blocks:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # Each Mi is a 3x3 x 3x3 product, decomposed via schoolbook (rank 27).
    # We need to build U, V, W of shapes (36, rank), (36, rank), (36, rank)
    # where the row indices are flattened as per the convention:
    #   U: row index = i*m + j  (i in [0,6), j in [0,6))
    #   V: row index = j*p + k  (j in [0,6), k in [0,6))
    #   W: row index = k*n + i  (k in [0,6), i in [0,6))  -- trace convention W[k*n+i, r]

    # Total rank = 7 * 27 = 189
    rank = 7 * 27
    U = [[0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * rank for _ in range(p * n)]  # 36 x 189

    # Define the 7 Strassen terms with their coefficient in A-side and B-side
    # Each term: (coeffA, row_block_A, col_block_A, coeffB, row_block_B, col_block_B)
    # Block indices: (br, bc) where br in {0,1}, bc in {0,1}
    # Block (br, bc) covers rows [br*b, (br+1)*b) and cols [bc*b, (bc+1)*b)

    # For each of the 7 Strassen products, we need:
    #   A-side: a 3x3 matrix formed by linear combination of A-blocks
    #   B-side: a 3x3 matrix formed by linear combination of B-blocks
    #
    # We'll represent each side as a list of (block_row, block_col, coeff) tuples.

    strassen_terms = [
        # M1 = (A11 + A22)(B11 + B22)
        {
            'A': [(0, 0, 1), (1, 1, 1)],
            'B': [(0, 0, 1), (1, 1, 1)],
            'C': [(0, 0, 1), (1, 1, 1)]  # contributes to C11 and C22
        },
        # M2 = (A21 + A22)B11
        {
            'A': [(1, 0, 1), (1, 1, 1)],
            'B': [(0, 0, 1)],
            'C': [(1, 0, 1)]  # contributes to C21
        },
        # M3 = A11(B12 - B22)
        {
            'A': [(0, 0, 1)],
            'B': [(0, 1, 1), (1, 1, -1)],
            'C': [(0, 1, 1), (1, 1, 1)]  # contributes to C12 and C22
        },
        # M4 = A22(B21 - B11)
        {
            'A': [(1, 1, 1)],
            'B': [(1, 0, 1), (0, 0, -1)],
            'C': [(0, 0, 1), (1, 0, 1)]  # contributes to C11 and C21
        },
        # M5 = (A11 + A12)B22
        {
            'A': [(0, 0, 1), (0, 1, 1)],
            'B': [(1, 1, 1)],
            'C': [(0, 0, -1)]  # contributes to C11 with -1
        },
        # M6 = (A21 - A11)(B11 + B12)
        {
            'A': [(1, 0, 1), (0, 0, -1)],
            'B': [(0, 0, 1), (0, 1, 1)],
            'C': [(1, 1, 1)]  # contributes to C22
        },
        # M7 = (A12 - A22)(B21 + B22)
        {
            'A': [(0, 1, 1), (1, 1, -1)],
            'B': [(1, 0, 1), (1, 1, 1)],
            'C': [(0, 0, 1)]  # contributes to C11
        },
    ]

    # For each Strassen term, decompose the 3x3 x 3x3 product via schoolbook.
    # The schoolbook for 3x3 has rank 27, with entries indexed by (i, j, k) in [0,3)^3.
    # For a product X*Y where X and Y are 3x3:
    #   U entries: for each (i,j), U[i,j, term_offset + i*9+j*3+k] = (coeff from A-side for block containing (i,j))
    #   V entries: for each (j,k), V[j,k, term_offset + i*9+j*3+k] = (coeff from B-side for block containing (j,k))
    #   W entries: for each (k,i), W[k,i, term_offset + i*9+j*3+k] = (coeff from C-side for block containing (k,i))

    # Wait, let me think more carefully.
    # The tensor for 3x3 matrix mult has indices (i, j, k) where:
    #   i = row of A and C (0..2)
    #   j = col of A and row of B (0..2)
    #   k = col of B and C (0..2)
    #
    # In the full 6x6 problem:
    #   Global row i_global in [0,6), col j_global in [0,6) for A
    #   Global row j_global in [0,6), col k_global in [0,6) for B
    #   Global col k_global in [0,6), row i_global in [0,6) for C (W index = k*n + i)
    #
    # Block (br, bc) of A covers global rows [br*3, br*3+3) and global cols [bc*3, bc*3+3).
    # So local (i,j) in block (br,bc) maps to global (br*3+i, bc*3+j).
    #
    # For the U matrix (shape 36 x rank):
    #   U[global_i * 6 + global_j, col] = coefficient
    #
    # For V (shape 36 x rank):
    #   V[global_j * 6 + global_k, col] = coefficient
    #
    # For W (shape 36 x rank):
    #   W[global_k * 6 + global_i, col] = coefficient

    for term_idx, term in enumerate(strassen_terms):
        offset = term_idx * 27
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    col = offset + i * 9 + j * 3 + k

                    # Determine which A-block(s) contribute to A[i,j]
                    # A-side is a linear combination of A-blocks
                    a_coeff = 0
                    for (abr, abc, acoef) in term['A']:
                        # A[i,j] gets contribution from block (abr, abc) with coefficient acoef
                        # The local (i,j) within that block corresponds to global (abr*3+i, abc*3+j)
                        # But wait: the Strassen formula uses the same local indices (i,j) for all blocks.
                        # So A[i,j] = sum over blocks of acoef * A_block[abr,abc][i,j]
                        # This means for the U factor, we need:
                        # U[global_i * 6 + global_j, col] += acoef
                        # where global_i = abr*3 + i, global_j = abc*3 + j
                        a_coeff += acoef
                        U[abr * 3 + i] = U[abr * 3 + i]  # placeholder, we'll do it below

                    # Actually, let me restructure. For each (i,j,k) in the 3x3 schoolbook,
                    # the U factor has a 1 at position corresponding to (global_i, global_j) for each
                    # A-block that contributes. Since the same local (i,j) is used across blocks,
                    # we add acoef to U[(abr*3+i)*6 + (abc*3+j), col].

                    for (abr, abc, acoef) in term['A']:
                        gi = abr * 3 + i
                        gj = abc * 3 + j
                        U[gi * 6 + gj][col] += acoef

                    for (bbr, bbc, bcoef) in term['B']:
                        gj = bbr * 3 + j
                        gk = bbc * 3 + k
                        V[gj * 6 + gk][col] += bcoef

                    for (cbr, cbc, ccoef) in term['C']:
                        gk = cbr * 3 + k
                        gi = cbc * 3 + i
                        W[gk * 6 + gi][col] += ccoef

    # Verify correctness with a random test
    if time.time() < deadline - 5:
        rng = np.random.default_rng(42)
        A_test = rng.integers(-5, 6, size=(6, 6))
        B_test = rng.integers(-5, 6, size=(6, 6))
        C_expected = A_test @ B_test

        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        # C = sum_r U[:,r] * V[:,r] * W[:,r]  (outer product sum)
        # C[i,j] = sum_k A[i,k] B[k,l]... wait, let me think about the tensor contraction.
        # The matrix multiplication tensor T[i,j,k] = 1 if we compute C[i,k] += A[i,j]*B[j,k]
        # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # So C[i,k] = sum_j A[i,j] B[j,k] = sum_{j,r} A[i,j] B[j,k] U[i,j,r] V[j,k,r] W[k,i,r]... 
        # No, the standard decomposition is:
        # C = sum_r (U[:,r] outer product with something)... 
        # Actually: C[i,k] = sum_{j} A[i,j] B[j,k]
        # = sum_{j,r} A[i,j] B[j,k] * delta... 
        # The tensor decomposition says: T_{i,j,k} = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)}
        # where T_{i,j,k} = 1 for the matmul tensor (it's 1 for all valid i,j,k since it's the identity tensor).
        # Wait no. The matmul tensor has T[i,j,k] = 1 for all i,j,k (it's the "all ones" tensor in this indexing).
        # Then C[i,k] = sum_j A[i,j] B[j,k] T[i,j,k] = sum_j A[i,j] B[j,k].
        # With decomposition: C[i,k] = sum_r W[k,i,r] * sum_j U[i,j,r] A[i,j] V[j,k,r] B[j,k]
        # Hmm, that's not quite right either. Let me use the standard form:
        # C[i,k] = sum_{j,r} A[i,j] B[j,k] * (U[i,j,r] * V[j,k,r] * W[k,i,r]) / (W[k,i,r])... 
        # 
        # Standard: if T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r], then
        # C[i,k] = sum_j A[i,j] B[j,k] T[i,j,k]... no.
        #
        # The correct interpretation: the matmul operation is a bilinear map.
        # C = sum_r (U_r^T A) * (V_r^T B) reshaped... 
        # More precisely: C[i,k] = sum_r W[k,i,r] * (sum_j U[i,j,r] A[i,j]) * (sum_j V[j,k,r] B[j,k])
        # Wait that has a sum over j appearing twice. Let me be more careful.
        #
        # The standard tensor decomposition for matmul:
        # T_{ijk} = sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)}
        # where T_{ijk} = 1 (the matmul tensor is all ones in the standard basis).
        # Then C_{ik} = sum_j A_{ij} B_{jk} T_{ijk} = sum_j A_{ij} B_{jk} sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)}
        # = sum_r W_{ki}^{(r)} (sum_j U_{ij}^{(r)} A_{ij}) (sum_j V_{jk}^{(r)} B_{jk})
        # 
        # Hmm, that's not a standard matrix product form. Let me reconsider.
        #
        # Actually the standard form is:
        # C_{ik} = sum_r W_{ki}^{(r)} * (sum_j U_{ij}^{(r)} A_{ij}) * (sum_j V_{jk}^{(r)} B_{jk})
        # But this involves separate sums over j, which is unusual.
        #
        # The more standard bilinear form: C = sum_r alpha_r (A * U_r) (V_r * B) where * denotes some contraction.
        #
        # Let me just verify numerically:
        # C[i,k] = sum_{j,r} A[i,j] B[j,k] U[i,j,r] V[j,k,r] W[k,i,r]
        # (assuming W is constant over j for fixed k,i,r, which it is in our construction)

        C_computed = np.zeros((6, 6))
        for r in range(rank):
            u_col = U_np[:, r]  # 36 vector
            v_col = V_np[:, r]  # 36 vector
            w_val = W_np  # 36 vector, but we need W[k,i,r]
            # Reshape u_col to 6x6 (i,j), v_col to 6x6 (j,k)
            u_mat = u_col.reshape(6, 6)
            v_mat = v_col.reshape(6, 6)
            # w is indexed as [k*n + i] = [k*6 + i], so reshape to (6,6) with (k,i)
            w_mat = W_np[:, r].reshape(6, 6)  # w_mat[k, i]
            # C[i,k] += sum_j u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            # This is element-wise: for each (i,j,k): u_mat[i,j]*v_mat[j,k]*w_mat[k,i]
            # Sum over j.
            for i in range(6):
                for k in range(6):
                    C_computed[i, k] += np.sum(u_mat[i, :] * v_mat[:, k] * w_mat[k, i])

        if not np.allclose(C_computed, C_expected):
            # Fallback: return schoolbook
            rank_sb = 216
            U_sb = [[0] * rank_sb for _ in range(36)]
            V_sb = [[0] * rank_sb for _ in range(36)]
            W_sb = [[0] * rank_sb for _ in range(36)]
            col = 0
            for i in range(6):
                for j in range(6):
                    for k in range(6):
                        U_sb[i * 6 + j][col] = 1
                        V_sb[j * 6 + k][col] = 1
                        W_sb[k * 6 + i][col] = 1
                        col += 1
            return {"U": U_sb, "V": V_sb, "W": W_sb}

    return {"U": U, "V": V, "W": W}
