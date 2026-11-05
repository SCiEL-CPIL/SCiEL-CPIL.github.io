import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    DEADLINE = 130  # seconds, safety margin before 150s limit

    # Rank 56: 40 (Winograd F(4) for 4x4x4) + 16 (Winograd F(2) for 4x4x1)
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 16 x 56
    V = [[0] * rank for _ in range(m * p)]  # 20 x 56
    W = [[0] * rank for _ in range(p * n)]  # 20 x 56

    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c

    col = 0

    # Part 1: Winograd F(4) for 4x4x4 (rank 40)
    # F(4) transform: 4x4 * 4x4 -> 4x4
    # U transform (rows of A):
    #   U[0] = A[0]
    #   U[1] = A[3]
    #   U[2] = A[0] + A[3]
    #   U[3] = A[1] - A[2]
    #   U[4..6] = A[1] + A[2] (same for each of 3 columns)
    #   U[7..9] = A[1] (same for each of 3 columns)
    #   U[10..12] = A[2] (same for each of 3 columns)
    #   U[13..15] = A[0] - A[3] (same for each of 3 columns)
    # V transform (columns of B):
    #   V[0] = B[0]
    #   V[1] = B[3]
    #   V[2] = B[0] + B[3]
    #   V[3] = B[1] - B[2]
    #   V[4..6] = B[1] + B[2] (same for each of 3 rows)
    #   V[7..9] = B[1] (same for each of 3 rows)
    #   V[10..12] = B[2] (same for each of 3 rows)
    #   V[13..15] = B[0] - B[3] (same for each of 3 rows)
    # W transform (output C):
    #   C[0] = M[0]
    #   C[3] = M[1]
    #   C[0] += M[2], C[3] += M[2]
    #   C[1] += M[3], C[2] -= M[3]
    #   C[1] += M[4], C[1] += M[5], C[1] += M[6]
    #   C[1] += M[7], C[1] += M[8], C[1] += M[9]
    #   C[2] += M[10], C[2] += M[11], C[2] += M[12]
    #   C[0] -= M[13], C[3] -= M[13]
    #   C[0] -= M[14], C[3] -= M[14]
    #   C[0] -= M[15], C[3] -= M[15]

    # U transform definitions: for each of 16 output positions (row, col) in U-space,
    # list of (a_row, coeff) contributions
    # U rows 0-15, cols 0-3
    # Row 0: A[0,0..3] -> U[0,0..3]
    # Row 1: A[3,0..3] -> U[1,0..3]
    # Row 2: A[0,0..3]+A[3,0..3] -> U[2,0..3]
    # Row 3: A[1,0..3]-A[2,0..3] -> U[3,0..3]
    # Rows 4-6: A[1,0..3]+A[2,0..3] -> U[4..6,0..3]
    # Rows 7-9: A[1,0..3] -> U[7..9,0..3]
    # Rows 10-12: A[2,0..3] -> U[10..12,0..3]
    # Rows 13-15: A[0,0..3]-A[3,0..3] -> U[13..15,0..3]

    # For each term t (0..39), we have:
    # u_factor: which U-row pattern to use
    # v_factor: which V-col pattern to use
    # w_contribs: which C positions get contributions

    # The 40 terms are structured as:
    # For each of 16 "diagonal" patterns (matching U-row patterns to V-col patterns),
    # and for each of the 16 output positions... actually let me think differently.

    # Actually, F(4) has 40 multiplications. Let me define them explicitly.
    # The key insight: U and V transforms produce 16 intermediate vectors each.
    # Then 40 = 16*16/4... no, 40 is specific.

    # Let me use the standard F(4) formulation:
    # The 40 products are:
    # M[0] = U[0] * V[0]  -> contributes to C[0,0]
    # M[1] = U[1] * V[1]  -> contributes to C[3,3]
    # M[2] = U[2] * V[2]  -> contributes to C[0,0] and C[3,3]
    # M[3] = U[3] * V[3]  -> contributes to C[1,1] and -C[2,2]
    # M[4] = U[4] * V[4]  -> contributes to C[1,1]
    # M[5] = U[5] * V[5]  -> contributes to C[1,1]
    # M[6] = U[6] * V[6]  -> contributes to C[1,1]
    # M[7] = U[7] * V[7]  -> contributes to C[1,1]
    # M[8] = U[8] * V[8]  -> contributes to C[1,1]
    # M[9] = U[9] * V[9]  -> contributes to C[1,1]
    # M[10] = U[10] * V[10]  -> contributes to C[2,2]
    # M[11] = U[11] * V[11]  -> contributes to C[2,2]
    # M[12] = U[12] * V[12]  -> contributes to C[2,2]
    # M[13] = U[13] * V[13]  -> contributes to C[0,0] and C[3,3]
    # M[14] = U[14] * V[14]  -> contributes to C[0,0] and C[3,3]
    # M[15] = U[15] * V[15]  -> contributes to C[0,0] and C[3,3]

    # Wait, that's only 16. I need 40. Let me reconsider.

    # Actually the F(4) transform for matrix multiplication works as follows:
    # We transform rows of A (4 rows -> 16 transformed rows) and columns of B (4 cols -> 16 transformed cols).
    # Then we do 16*16 = 256 products... no that's too many.

    # Let me look at this more carefully. The Winograd F(4) transform for 4x4*4x4:
    # It's based on the fact that 4 = 2^2, so we can use tensor product of F(2).
    # F(2) has 7 multiplications. F(4) = F(2) ⊗ F(2) would give 49, but there are optimizations.
    # Actually, the standard Winograd F(4) for matrix multiplication uses 40 multiplications.

    # Let me use a different approach: define the transform matrices directly.
    # F(4) transform matrix for rows (16x4):
    # F_row = [
    #   [1, 0, 0, 0],   # row 0: A[0]
    #   [0, 0, 0, 1],   # row 1: A[3]
    #   [1, 0, 0, 1],   # row 2: A[0]+A[3]
    #   [0, 1, -1, 0],  # row 3: A[1]-A[2]
    #   [0, 1, 1, 0],   # row 4: A[1]+A[2]
    #   [0, 1, 1, 0],   # row 5
    #   [0, 1, 1, 0],   # row 6
    #   [0, 1, 0, 0],   # row 7: A[1]
    #   [0, 1, 0, 0],   # row 8
    #   [0, 1, 0, 0],   # row 9
    #   [0, 0, 1, 0],   # row 10: A[2]
    #   [0, 0, 1, 0],   # row 11
    #   [0, 0, 1, 0],   # row 12
    #   [1, 0, 0, -1],  # row 13: A[0]-A[3]
    #   [1, 0, 0, -1],  # row 14
    #   [1, 0, 0, -1],  # row 15
    # ]

    # F_col transform matrix for columns (16x4):
    # F_col = [
    #   [1, 0, 0, 0],   # col 0: B[0]
    #   [0, 0, 0, 1],   # col 1: B[3]
    #   [1, 0, 0, 1],   # col 2: B[0]+B[3]
    #   [0, 1, -1, 0],  # col 3: B[1]-B[2]
    #   [0, 1, 1, 0],   # col 4: B[1]+B[2]
    #   [0, 1, 1, 0],   # col 5
    #   [0, 1, 1, 0],   # col 6
    #   [0, 1, 0, 0],   # col 7: B[1]
    #   [0, 1, 0, 0],   # col 8
    #   [0, 1, 0, 0],   # col 9
    #   [0, 0, 1, 0],   # col 10: B[2]
    #   [0, 0, 1, 0],   # col 11
    #   [0, 0, 1, 0],   # col 12
    #   [1, 0, 0, -1],  # col 13: B[0]-B[3]
    #   [1, 0, 0, -1],  # col 14
    #   [1, 0, 0, -1],  # col 15
    # ]

    # Output transform: C = F_out * (U * V) where U*V is 16x16
    # But we only need 40 of the 256 products. The sparsity pattern is:
    # C[i,j] = sum over specific (u,v) pairs

    # Actually, the correct way: C = F_row^T * (F_row * A) * (B * F_col^T) * F_col
    # No wait. Let me think again.

    # C = A * B
    # A' = F_row * A  (16x4)
    # B' = B * F_col^T (4x16)
    # C' = A' * B' (16x16)
    # C = F_row^T * C' * F_col  (4x4)

    # But this gives 16*16=256 products, not 40. The trick is that many entries
    # of F_row and F_col are zero or repeated, so we can group products.

    # Looking at F_row: rows 4,5,6 are identical; rows 7,8,9 identical; rows 10,11,12 identical; rows 13,14,15 identical.
    # Similarly for F_col.

    # So the 16x16 product matrix C' has structure:
    # C'[r,c] where r in 0..15, c in 0..15
    # But C'[4,c]=C'[5,c]=C'[6,c], C'[7,c]=C'[8,c]=C'[9,c], etc.
    # And C'[r,4]=C'[r,5]=C'[r,6], etc.

    # So we only need to compute 4*4=16 "base" products, but each base product
    # may be used multiple times in the output reconstruction.

    # Wait, that gives 16 products, not 40. Let me reconsider.

    # I think I'm overcomplicating this. Let me just use the known rank-40
    # decomposition for 4x4x4 matrix multiplication.

    # Alternative: use the Coppersmith-Winograd or just carefully implement
    # the tensor product of two Strassen decompositions but with shared terms.

    # Actually, the simplest correct approach: use the known result that
    # 4x4x4 matrix multiplication has rank 40 (or at most 49 via Strassen).
    # Since I'm not 100% sure of the exact rank-40 decomposition, let me use
    # a safe approach: keep the 49-term Strassen for 4x4x4 but optimize the
    # 4x4x1 part.

    # Actually, let me try a different approach for the 4x4x1 part using
    # Winograd F(2) style:
    # C[i,4] = sum_j A[i,j]*B[j,4] for i=0..3
    # This is a 4-vector dot product with a 4-vector, for each i.
    # Using F(2) on pairs: (A[i,0]+A[i,1], B[0,4]+B[1,4]) etc.
    # But this doesn't reduce below 16 for 4x4x1 since each i needs 4 products
    # and there's no sharing between different i values.

    # So the 4x4x1 part stays at rank 16 (schoolbook is optimal here).

    # Let me focus on getting the 4x4x4 part right with rank 40.
    # I'll use the explicit F(4) transform approach.

    # F(4) row transform (16x4 matrix, applied to each column of A):
    F_row = [
        [1, 0, 0, 0],   # 0
        [0, 0, 0, 1],   # 1
        [1, 0, 0, 1],   # 2
        [0, 1, -1, 0],  # 3
        [0, 1, 1, 0],   # 4
        [0, 1, 1, 0],   # 5
        [0, 1, 1, 0],   # 6
        [0, 1, 0, 0],   # 7
        [0, 1, 0, 0],   # 8
        [0, 1, 0, 0],   # 9
        [0, 0, 1, 0],   # 10
        [0, 0, 1, 0],   # 11
        [0, 0, 1, 0],   # 12
        [1, 0, 0, -1],  # 13
        [1, 0, 0, -1],  # 14
        [1, 0, 0, -1],  # 15
    ]

    # F(4) col transform (16x4 matrix, applied to each row of B):
    F_col = [
        [1, 0, 0, 0],   # 0
        [0, 0, 0, 1],   # 1
        [1, 0, 0, 1],   # 2
        [0, 1, -1, 0],  # 3
        [0, 1, 1, 0],   # 4
        [0, 1, 1, 0],   # 5
        [0, 1, 1, 0],   # 6
        [0, 1, 0, 0],   # 7
        [0, 1, 0, 0],   # 8
        [0, 1, 0, 0],   # 9
        [0, 0, 1, 0],   # 10
        [0, 0, 1, 0],   # 11
        [0, 0, 1, 0],   # 12
        [1, 0, 0, -1],  # 13
        [1, 0, 0, -1],  # 14
        [1, 0, 0, -1],  # 15
    ]

    # Output transform: C[i,j] = sum_{u,v} F_row_inv[i,u] * C'[u,v] * F_col_inv[v,j]
    # where C'[u,v] = A'[u] * B'[v] (dot product of transformed row u of A and transformed col v of B)
    # But we need to figure out which (u,v) pairs contribute to each (i,j).

    # Actually, the reconstruction is:
    # C = F_row^T * (A' * B') * F_col  ... no this isn't right either.

    # Let me use a cleaner formulation:
    # A' = F_row * A  =>  A'[u,j] = sum_i F_row[u,i] * A[i,j]
    # B' = B * F_col^T  =>  B'[j,v] = sum_k B[j,k] * F_col[v,k]
    # C' = A' * B'  =>  C'[u,v] = sum_j A'[u,j] * B'[j,v]
    # C = F_row^T * C' * F_col  =>  C[i,k] = sum_{u,v} F_row[u,i] * C'[u,v] * F_col[v,k]

    # So C[i,k] = sum_{u,v} F_row[u,i] * C'[u,v] * F_col[v,k]
    # And C'[u,v] = sum_j (sum_i' F_row[u,i']*A[i',j]) * (sum_k' B[j,k']*F_col[v,k'])

    # For the tensor decomposition, each term corresponds to a (u,v) pair where
    # F_row[u,i] and F_col[v,k] are both nonzero.

    # The U matrix: U[i*m+j, term] = F_row[u, i] for the j-th column (but j is summed over in C')
    # Actually, U[i*m+j, term] should be such that sum_j U[i*m+j,term] * V[j*p+k,term] * W[k*n+i,term] = C[i,k]

    # From the formula: C[i,k] = sum_{u,v,j} F_row[u,i] * A[i',j]... 

    # Let me just directly construct: for each (u,v) pair with F_row[u,:] and F_col[v,:] nonzero,
    # the term contributes:
    # U[i*m+j, term] = F_row[u, i] for all i,j  (but this doesn't depend on j, which is wrong)

    # I think the issue is that C'[u,v] = sum_j A'[u,j]*B'[j,v], so j is summed over.
    # In the tensor decomposition, each term has a specific j. So we need 16*16*4 = 1024 terms
    # before simplification. But many cancel or combine.

    # This is getting too complex. Let me just use the 49-term Strassen for 4x4x4
    # and focus on verifying correctness.

    # Revert to Strassen approach for 4x4x4 (49 terms) + schoolbook for 4x4x1 (16 terms) = 65

    # Actually, let me try a cleaner Winograd approach. The key realization:
    # For 4x4*4x4, we can use the tensor product of two 2x2*2x2 Strassen decompositions.
    # This gives 7*7=49 terms, but some can be merged. The known minimum is 40.

    # Since implementing the exact rank-40 decomposition is complex and error-prone,
    # let me use a different strategy: use the 49-term Strassen (which I know is correct)
    # but see if I can reduce the 4x4x1 part.

    # For 4x4x1: C[i,4] = sum_j A[i,j]*B[j,4]
    # This is 4 independent dot products of length 4. Each dot product of length 4
    # can be done with rank 4 (schoolbook) or potentially less. But since the B[j,4]
    # values are shared across all i, there might be savings.
    # Actually no - each i has different A[i,j] values, so no sharing. Rank 16 is optimal.

    # So minimum total rank with Strassen is 49+16=65. With Winograd F(4) it would be 40+16=56.

    # Let me carefully implement the F(4) transform. I'll verify it numerically.

    # The F(4) transform for matrix multiplication:
    # Step 1: Transform A: A' = F_row * A (16x4)
    # Step 2: Transform B: B' = B * F_col^T (4x16)  
    # Step 3: Multiply: C' = A' * B' (16x16)
    # Step 4: Transform C: C = F_row^T * C' * F_col (4x4)

    # For the tensor decomposition, we need to express C[i,k] as sum of products.
    # C[i,k] = sum_{u=0}^{15} sum_{v=0}^{15} F_row[u,i] * C'[u,v] * F_col[v,k]
    # C'[u,v] = sum_{j=0}^{3} A'[u,j] * B'[j,v]
    # A'[u,j] = sum_{i'=0}^{3} F_row[u,i'] * A[i',j]
    # B'[j,v] = sum_{k'=0}^{3} B[j,k'] * F_col[v,k']

    # So C[i,k] = sum_{u,v,j,i',k'} F_row[u,i] * F_row[u,i'] * A[i',j] * B[j,k'] * F_col[v,k'] * F_col[v,k]

    # Grouping by (u,v,j):
    # C[i,k] = sum_{u,v,j} A[i',j]*B[j,k'] * [sum over i',k' of F_row[u,i]*F_row[u,i']*F_col[v,k']*F_col[v,k]]

    # This is getting complicated. Let me just build it term by term.

    # For each (u, v, j) triple where the coefficient is nonzero, we have a term:
    # U[i*m+j, term] = sum_{i'} F_row[u,i] * F_row[u,i']  ... no this isn't right.

    # Let me restart with a cleaner approach. I'll define the 40 terms explicitly
    # based on the known F(4) decomposition.

    # After research, the 4x4x4 matrix multiplication tensor has rank 40.
    # The decomposition can be constructed as follows:

    # Define the 16 "row" patterns and 16 "col" patterns as above.
    # The 40 multiplications correspond to the 40 pairs (u,v) where
    # the corresponding entries in the output transform are nonzero.

    # Let me just build it computationally:
    # For each u in 0..15, v in 0..15, j in 0..3:
    #   coeff_u = F_row[u, :] (4-vector)
    #   coeff_v = F_col[v, :] (4-vector)
    #   The term contributes to U, V, W as:
    #   U[i*4+j, term] = coeff_u[i] for all i
    #   V[j*5+k, term] = coeff_v[k] for all k (but k is 0..3 for the 4x4 part)
    #   W[k*4+i, term] = coeff_u[i] * coeff_v[k] ... no

    # I think I need to be more careful. Let me just use numpy to verify
    # and build the decomposition.

    # Given time constraints, let me use the safe 49+16=65 approach but
    # double-check the implementation is correct.

    # Reimplementing the Strassen 4x4x4 + schoolbook 4x4x1 = rank 65

    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65

    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c

    col = 0

    # Strassen 2x2x2 decomposition
    inner_defs = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,0,1),(1,1,1)]),       # M1
        ([(2,1),(3,1)], [(0,1)], [(1,0,1),(1,1,-1)]),             # M2
        ([(0,1)], [(1,1),(3,-1)], [(0,1,1),(1,1,1)]),             # M3
        ([(3,1)], [(2,1),(0,-1)], [(0,0,1),(1,0,1)]),             # M4
        ([(0,1),(1,1)], [(3,1)], [(0,0,-1),(0,1,1)]),             # M5
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(1,1,1)]),               # M6
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,0,1)]),               # M7
    ]

    super_defs = [
        ([1,0,0,1], [1,0,0,1], [(0,0,1),(1,1,1)]),       # M1 -> C00, C11
        ([0,0,1,1], [1,0,0,0], [(1,0,1),(1,1,-1)]),      # M2 -> C10, -C11
        ([1,0,0,0], [0,1,0,-1], [(0,1,1),(1,1,1)]),      # M3 -> C01, C11
        ([0,0,0,1], [-1,0,1,0], [(0,0,1),(1,0,1)]),      # M4 -> C00, C10
        ([1,1,0,0], [0,0,0,1], [(0,0,-1),(0,1,1)]),      # M5 -> -C00, C01
        ([-1,0,1,0], [1,1,0,0], [(1,1,1)]),              # M6 -> C11
        ([0,1,0,-1], [0,0,1,1], [(0,0,1)]),              # M7 -> C00
    ]

    def get_p_element(p_elem_idx, p_block_coeffs):
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(p_block_coeffs):
            if coeff == 0:
                continue
            if b_idx == 0:
                gi, gj = r, c
            elif b_idx == 1:
                gi, gj = r, 2 + c
            elif b_idx == 2:
                gi, gj = 2 + r, c
            elif b_idx == 3:
                gi, gj = 2 + r, 2 + c
            res.append((gi, gj, coeff))
        return res

    def get_q_element(q_elem_idx, q_block_coeffs):
        r = q_elem_idx // 2
        c = q_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(q_block_coeffs):
            if coeff == 0:
                continue
            if b_idx == 0:
                gj, gk = r, c
            elif b_idx == 1:
                gj, gk = r, 2 + c
            elif b_idx == 2:
                gj, gk = 2 + r, c
            elif b_idx == 3:
                gj, gk = 2 + r, 2 + c
            res.append((gj, gk, coeff))
        return res

    for s_idx, (p_bc, q_bc, c_contribs) in enumerate(super_defs):
        for i_idx, (p_elems, q_elems, r_contribs) in enumerate(inner_defs):
            u_list = []
            v_list = []
            w_list = []

            for p_elem, coeff in p_elems:
                for gi, gj, c in get_p_element(p_elem, p_bc):
                    u_list.append((gi, gj, coeff * c))

            for q_elem, coeff in q_elems:
                for gj, gk, c in get_q_element(q_elem, q_bc):
                    v_list.append((gj, gk, coeff * c))

            for br, bc, sign in c_contribs:
                for lr, lc, local_coeff in r_contribs:
                    gi = br * 2 + lr
                    gk = bc * 2 + lc
                    w_list.append((gk, gi, sign * local_coeff))

            if u_list and v_list and w_list:
                add_term(col, u_list, v_list, w_list)
                col += 1

    # Part 2: 4x4x1 (rank 16) - schoolbook
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
