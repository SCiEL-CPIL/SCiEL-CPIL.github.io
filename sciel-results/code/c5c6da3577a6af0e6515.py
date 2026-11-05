"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]
import time

# EVOLVE-BLOCK-START
def construct():
    # We use a block decomposition:
    # A (4x4), B (4x5) -> C (4x5)
    # Split B into B1 (4x4) and B2 (4x1).
    # C1 = A * B1 (4x4), C2 = A * B2 (4x1).
    #
    # For the 4x4x4 part, we use Strassen's algorithm (rank 49).
    # For the 4x4x1 part, we use schoolbook (rank 16).
    # Total rank = 49 + 16 = 65.
    #
    # The evaluator expects:
    #   U: shape (n*m, rank) = (16, rank)
    #   V: shape (m*p, rank) = (20, rank)
    #   W: shape (p*n, rank) = (20, rank)
    #   with W[k*n + i, r] convention.
    #
    # The decomposition satisfies:
    #   C[i, k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]  ... wait, no.
    #
    # Standard tensor decomposition for matrix multiplication:
    #   C[i, k] = sum_{j, r} A[i, j] * B[j, k] * (coefficient)
    #   The tensor is T[i, j, k] = delta_{i,j,k} meaning T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k].
    #   Actually, the multiplication tensor T is defined by T[i, j, k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
    #   In a rank-r decomposition: T[i,j,k] = sum_{r=1}^r U[i,j,r] * V[j,k,r] * W[i,k,r]
    #   But the interface uses flattened indices:
    #     U has shape (n*m, rank) indexed by (i*m + j, r)
    #     V has shape (m*p, rank) indexed by (j*p + k, r)
    #     W has shape (p*n, rank) indexed by (k*n + i, r)
    #   So the decomposition is:
    #     T[(i,j), (j,k), (k,i)] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #   And the product C[i,k] = sum_j A[i,j] * B[j,k] is recovered by:
    #     C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    #   Wait, that's not quite right either. Let me re-derive.
    #
    # The bilinear form for matrix multiplication:
    #   C[i,k] = sum_j A[i,j] * B[j,k]
    # This can be written as a sum of rank-1 terms:
    #   C[i,k] = sum_r (sum_j A[i,j] * u_{i,j}^{(r)}) * (sum_j v_{j,k}^{(r)} * B[j,k]) * w_{i,k}^{(r)}
    # where the product of the three factors gives the contribution.
    #
    # Actually, the standard way: each rank-1 term corresponds to a "gate" that:
    #   - reads a linear combination of A entries: a_r = sum_j A[i,j] * U[i*m+j, r]  (but U depends on i,j)
    #   - reads a linear combination of B entries: b_r = sum_j B[j,k] * V[j*p+k, r]
    #   - writes to C[i,k]: c_r = W[k*n+i, r] * a_r * b_r
    #
    # So C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    #
    # For this to equal sum_j A[i,j] * B[j,k], we need:
    #   sum_r W[k*n+i, r] * U[i*m+j', r] * V[j'*p+k, r] = delta_{j', j} ... no.
    #
    # Let me be more careful. The condition is:
    #   For all i, j, k: sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1
    #   For all i, j, k with j != j' (i.e., the "cross" terms): 
    #     Actually, the full condition is that the tensor T[i,j,k] = delta_{i,k} * delta_{j,j} ... no.
    #
    # The matrix multiplication tensor is: T[i, j, k] = 1 if and only if the entry A[i,j] * B[j,k] contributes to C[i,k].
    # In the flattened indexing: T[(i,j), (j,k), (i,k)] where the first index is (i,j), second is (j,k), third is (i,k).
    # But note that j must be the same in the first two indices. So T is zero unless the j in (i,j) equals the j in (j,k).
    #
    # The rank-r decomposition requires:
    #   T[(i,j), (j',k), (i',k')] = sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
    #   = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    #
    # So for the 4x4x4 block (i in 0..3, j in 0..3, k in 0..3):
    #   We need: sum_r U_44[i*m+j, r] * V_44[j*p+k, r] * W_44[k*n+i, r] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    #   where U_44, V_44, W_44 are the parts used for the 4x4x4 subproblem.
    #
    # For the 4x4x1 block (i in 0..3, j in 0..3, k in 4):
    #   We need: sum_r U_41[i*m+j, r] * V_41[j*p+4, r] * W_41[4*n+i, r] = delta_{i,i'} * delta_{j,j'}
    #   (k is fixed at 4, so no k' variation)
    #
    # Let's implement Strassen for 4x4x4 and schoolbook for 4x4x1.
    #
    # Strassen 4x4x4 decomposition:
    # Split 4x4 into four 2x2 blocks. Use Strassen for 2x2x2 (7 terms), and recursively.
    # 4x4x4 rank = 4 * 7 = 28? No, that's not right. Strassen for 2x2 is rank 7.
    # For 4x4, we split into 4 quadrants of 2x2. The 4x4 product requires:
    #   C11 = A11*B11 + A12*B21
    #   C12 = A11*B12 + A12*B22
    #   C21 = A21*B11 + A22*B21
    #   C22 = A21*B12 + A22*B22
    # Using Strassen on the 2x2 blocks:
    #   M1 = (A11+A22)*(B11+B22)
    #   M2 = (A21+A22)*B11
    #   M3 = A11*(B12-B22)
    #   M4 = A22*(B21-B11)
    #   M5 = (A11+A12)*B22
    #   M6 = (A21-A11)*(B11+B12)
    #   M7 = (A12-A22)*(B21+B22)
    #   C11 = M1 + M4 - M5 + M7
    #   C12 = M3 + M5
    #   C21 = M2 + M4
    #   C22 = M1 - M2 + M3 + M6
    # This gives 7 multiplications of 2x2 matrices. Each 2x2 multiplication has rank 4 (schoolbook) or can be further decomposed.
    # If we use schoolbook for the 2x2 multiplications, total rank = 7 * 4 = 28.
    # But we can use Strassen recursively: 2x2 can be done in rank 7, so 4x4 in rank 7*7 = 49? No.
    # Actually, each of the 7 products is a 2x2 x 2x2 = 2x2 multiplication. If each 2x2 multiplication uses rank 4 (schoolbook), total = 7*4 = 28.
    # If we use Strassen for the 2x2 (rank 7), total = 7*7 = 49.
    # So 4x4x4 can be done in rank 28 (using Strassen at the 4x4 level and schoolbook at the 2x2 level).
    #
    # Let me implement rank-28 for 4x4x4.

    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    total_rank = 28 + 16  # 28 for 4x4x4, 16 for 4x4x1
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    # --- 4x4x4 part: indices i=0..3, j=0..3, k=0..3 ---
    # We'll build the Strassen-based decomposition.
    # The 4x4 matrices are split into 2x2 blocks:
    #   A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    #   Similarly for B.
    #
    # Strassen terms (each is a 2x2 multiplication):
    #   M1 = (A11+A22)*(B11+B22)
    #   M2 = (A21+A22)*B11
    #   M3 = A11*(B12-B22)
    #   M4 = A22*(B21-B11)
    #   M5 = (A11+A12)*B22
    #   M6 = (A21-A11)*(B11+B12)
    #   M7 = (A12-A22)*(B21+B22)
    #
    # For each 2x2 multiplication, we use schoolbook (4 terms).
    # Each 2x2 multiplication X*Y = Z where X, Y, Z are 2x2.
    #   Z[0,0] = X[0,0]*Y[0,0] + X[0,1]*Y[1,0]
    #   Z[0,1] = X[0,0]*Y[0,1] + X[0,1]*Y[1,1]
    #   Z[1,0] = X[1,0]*Y[0,0] + X[1,1]*Y[1,0]
    #   Z[1,1] = X[1,0]*Y[0,1] + X[1,1]*Y[1,1]
    #
    # The linear combinations for X and Y involve entries from the 4x4 matrices.
    # We need to track which 4x4 entries are involved.

    # Let's define the 2x2 block indices in terms of 4x4 indices.
    # A11: i in {0,1}, j in {0,1}
    # A12: i in {0,1}, j in {2,3}
    # A21: i in {2,3}, j in {0,1}
    # A22: i in {2,3}, j in {2,3}
    # Similarly for B.

    # For each Strassen term, we have a 2x2 multiplication.
    # Let's define a helper to add a 2x2 schoolbook multiplication.

    def add_2x2_mult(col_offset, X_coeffs, Y_coeffs, Z_coeffs):
        """
        X_coeffs: dict mapping (x_i, x_j) to coefficient, where x_i, x_j are 4x4 indices for A.
        Y_coeffs: dict mapping (y_i, y_j) to coefficient, where y_i, y_j are 4x4 indices for B.
        Z_coeffs: dict mapping (z_i, z_j) to coefficient, where z_i, z_j are 4x4 indices for C.
        
        The 2x2 multiplication: X*Y = Z, where X, Y, Z are 2x2 matrices embedded in 4x4.
        We add 4 rank-1 terms (schoolbook for 2x2).
        """
        for dx_i in range(2):
            for dx_j in range(2):
                for dy_i in range(2):
                    for dy_j in range(2):
                        # This is getting complex. Let me think differently.
                        pass
        return col_offset

    # Actually, let me just directly construct the U, V, W for the 4x4x4 part using Strassen.
    # I'll enumerate the 28 terms.

    # Strassen 4x4x4: 7 groups of 4 schoolbook 2x2 terms = 28 terms.
    # For each Strassen term t (0..6), we have a 2x2 multiplication.
    # The 2x2 multiplication has 4 schoolbook terms.
    #
    # Let me define the Strassen terms more carefully.
    #
    # Term 0: M1 = (A11+A22)*(B11+B22)
    #   X = A11 + A22, Y = B11 + B22
    #   X[0,0] = A[0,0], X[0,1] = A[0,1], X[1,0] = A[1,0], X[1,1] = A[1,1]
    #            + A[2,2], A[2,3], A[3,2], A[3,3]
    #   So X[0,0] = A[0,0] + A[2,2], X[0,1] = A[0,1] + A[2,3], X[1,0] = A[1,0] + A[3,2], X[1,1] = A[1,1] + A[3,3]
    #   Y[0,0] = B[0,0] + B[2,2], Y[0,1] = B[0,1] + B[2,3], Y[1,0] = B[1,0] + B[3,2], Y[1,1] = B[1,1] + B[3,3]
    #   Z = C11 - M4 + M5 - M7 ... no, Z contributes to C11, C12, C21, C22 with coefficients.
    #   Actually, M1 contributes to C11 and C22.
    #   C11 += M1, C22 += M1
    #   So Z[0,0] contributes to C[0,0] with coeff 1, Z[0,1] to C[0,1] with coeff 0 (M1 doesn't contribute to C12 or C21)
    #   Wait, M1 is a 2x2 matrix. M1[0,0] contributes to C[0,0], M1[0,1] to C[0,1], M1[1,0] to C[1,0], M1[1,1] to C[1,1].
    #   But M1 contributes to C11 and C22, not C12 or C21.
    #   C11 = M1 + M4 - M5 + M7, so M1[0,0] -> C[0,0], M1[1,1] -> C[1,1] (within the C11 block)
    #   C22 = M1 - M2 + M3 + M6, so M1[0,0] -> C[2,2], M1[1,1] -> C[3,3] (within the C22 block)
    #   M1[0,1] and M1[1,0] don't contribute to anything? That can't be right.
    #
    # I think I'm overcomplicating this. Let me just use a known rank-28 decomposition for 4x4x4.
    #
    # Actually, the simplest approach: use schoolbook for 4x4x4 (rank 16) and schoolbook for 4x4x1 (rank 16), total 32.
    # But that's worse than 65? No, 32 < 65. Wait, schoolbook 4x4x4 is rank 16, and 4x4x1 is rank 16, total 32.
    # But the baseline is 80. So 32 is much better.
    #
    # Wait, I was confused. Let me re-read the task.
    # n=4, m=4, p=5.
    # Schoolbook: rank = n*m*p = 4*4*5 = 80.
    # If I split into 4x4x4 (rank 16 schoolbook) + 4x4x1 (rank 16 schoolbook) = 32.
    # That's already much better than 80.
    #
    # Can I do better? For 4x4x4, Strassen gives rank 28 (if using schoolbook for 2x2) or 49 (if using Strassen for 2x2).
    # So 28 + 16 = 44, which is better than 32? No, 44 > 32.
    # So schoolbook for both is better: 32.
    #
    # Can I do even better? What about a mixed approach?
    # For 4x4x4, the optimal rank is unknown, but it's at most 28 (Strassen) and at least 16 (schoolbook).
    # Actually, schoolbook is 16, Strassen is 28. So schoolbook is better for 4x4x4.
    #
    # So the best I can do with this block approach is 16 + 16 = 32.
    #
    # But wait, can I do better than 32? What if I don't split?
    # The 4x4x5 problem might have a rank lower than 32.
    #
    # Let me just go with schoolbook for both blocks: rank 32.

    total_rank = 32
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0
    # 4x4x4 part: i=0..3, j=0..3, k=0..3
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # 4x4x1 part: i=0..3, j=0..3, k=4
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
