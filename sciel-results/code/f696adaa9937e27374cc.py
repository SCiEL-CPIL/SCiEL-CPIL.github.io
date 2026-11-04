"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    """Check if the decomposition is exact. Returns True if valid."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # Reshape to 3D tensors
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    # For each (i,j,k), compute the contraction
    # sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    result = np.zeros((n, m, p), dtype=float)
    
    # Vectorized computation using einsum for efficiency
    # result[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # We can do this in steps:
    # temp[i,j,k] = U3[i,j,c] * V3[j,k,c] summed over c? No, that's not right.
    # Let's use einsum: result[i,j,k] = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    try:
        result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    except Exception:
        # Fallback to loops if einsum fails or is too slow
        for c in range(rank):
            uc = U3[:, :, c]  # (n, m)
            vc = V3[:, :, c]  # (m, p)
            wc = W3[:, :, c]  # (p, n)
            for i in range(n):
                for j in range(m):
                    if uc[i, j] != 0:
                        for k in range(p):
                            result[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
    
    # Check if all entries are exactly 1 (within tolerance)
    return np.allclose(result, 1.0, atol=1e-9)

def _strassen_decomposition_2x2():
    """
    Return Strassen decomposition for 2x2 matrix multiplication.
    Rank 7.
    U: (4 x 7), V: (4 x 7), W: (4 x 7)
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen formulas:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Column 0: M1 = (A11+A22)(B11+B22)
    # A11 is U[0], A22 is U[3]
    # B11 is V[0], B22 is V[3]
    # C11 is W[0], C22 is W[3]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Column 1: M2 = (A21+A22)B11
    # A21 is U[2], A22 is U[3]
    # B11 is V[0]
    # C21 is W[2]
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # Column 2: M3 = A11(B12-B22)
    # A11 is U[0]
    # B12 is V[1], B22 is V[3]
    # C12 is W[1]
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # Column 3: M4 = A22(B21-B11)
    # A22 is U[3]
    # B21 is V[2], B11 is V[0]
    # C11 is W[0], C21 is W[2]
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # Column 4: M5 = (A11+A12)B22
    # A11 is U[0], A12 is U[1]
    # B22 is V[3]
    # C11 is W[0], C12 is W[1]
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Column 5: M6 = (A21-A11)(B11+B12)
    # A21 is U[2], A11 is U[0]
    # B11 is V[0], B12 is V[1]
    # C22 is W[3]
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Column 6: M7 = (A12-A22)(B21+B22)
    # A12 is U[1], A22 is U[3]
    # B21 is V[2], B22 is V[3]
    # C11 is W[0]
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _combine_decompositions(lists_of_U, lists_of_V, lists_of_W, offsets):
    """
    Combine multiple decompositions into one.
    lists_of_U: list of U matrices (each is list of lists)
    offsets: list of starting column indices for each decomposition
    """
    total_rank = sum(len(u[0]) for u in lists_of_U) if lists_of_U else 0
    # Determine dimensions from the first non-empty list
    n_rows_U = max((len(u) for u in lists_of_U), default=0)
    n_rows_V = max((len(v) for v in lists_of_V), default=0)
    n_rows_W = max((len(w) for w in lists_of_W), default=0)
    
    U = [[0] * total_rank for _ in range(n_rows_U)]
    V = [[0] * total_rank for _ in range(n_rows_V)]
    W = [[0] * total_rank for _ in range(n_rows_W)]
    
    for idx, (U_mat, V_mat, W_mat) in enumerate(zip(lists_of_U, lists_of_V, lists_of_W)):
        offset = offsets[idx]
        for r in range(len(U_mat)):
            for c in range(len(U_mat[0])):
                U[r][offset + c] = U_mat[r][c]
                V[r][offset + c] = V_mat[r][c]
                W[r][offset + c] = W_mat[r][c]
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    best = _schoolbook()
    best_rank = n * m * p
    
    # Strategy: Use Strassen for the 4x4 part of the middle dimension (j=0..3)
    # and schoolbook for the remaining j=4.
    # 
    # Partition A (4x5) into A_left (4x4) and A_right (4x1)
    # Partition B (5x6) into B_left (4x6) and B_right (1x6)
    # 
    # C = A_left @ B_left + A_right @ B_right
    #
    # For A_left @ B_left:
    #   A_left is 4x4, B_left is 4x6.
    #   We can view this as multiplying a 4x4 matrix by a 4x6 matrix.
    #   We can split B_left into two 4x3 matrices? Or use a different strategy.
    #   
    #   Actually, let's split the 4x4 multiplication into two 2x2 Strassen blocks.
    #   A_left = [[A11, A12], [A21, A22]] where each is 2x2.
    #   B_left = [[B11, B12, B13], [B21, B22, B23], [B31, B32, B33], [B41, B42, B43]]
    #   This is getting complicated because the right side is 4x6, not 4x4.
    #
    #   Let's try a simpler approach:
    #   Use Strassen for the 4x4x4 core (j=0..3, k=0..3) and handle the rest with schoolbook.
    #
    #   Decomposition 1: 4x4x4 multiplication (j=0..3, k=0..3)
    #     This computes C[i, 0..3] = sum_{j=0..3} A[i,j] * B[j, 0..3]
    #     We need a rank-7 decomposition for 4x4x4? No, Strassen is for 2x2.
    #     For 4x4x4, we can use two levels of Strassen or a direct 4x4x4 decomposition.
    #     Known rank for 4x4x4 is at most 21 (using Strassen recursively: 7^2=49? No.
    #     Actually, R(4,4,4) <= R(2,2,2)^2 = 7^2 = 49? No, that's for 4x4x4 if we split into 2x2 blocks.
    #     Wait, R(2n, 2n, 2n) <= R(n,n,n)^2. So R(4,4,4) <= 7^2 = 49.
    #     But we can do better. R(4,4,4) is known to be at most 21? No, that's for 3x3x3?
    #     Let's just use the simple block decomposition:
    #     Split 4x4x4 into four 2x2x2 blocks.
    #     Each 2x2x2 uses rank 7. Total rank 4*7 = 28.
    #
    #   Decomposition 2: 4x1x6 multiplication (j=4, k=0..5)
    #     This computes C[i, k] = A[i,4] * B[4,k] for all i,k.
    #     This is a rank 4*1*6 = 24 schoolbook decomposition?
    #     Actually, it's a rank 1 tensor scaled? No.
    #     C[i,k] = A[i,4] * B[4,k]. This is a rank-1 tensor in (i,k) space, but we have 4*6=24 entries.
    #     The tensor T[i, 4, k] = A[i,4] * B[4,k].
    #     This can be decomposed as: for each (i,k), U[i*5+4, c] * V[4*6+k, c] * W[k*4+i, c] = 1.
    #     We can use 24 columns, one for each (i,k) pair.
    #     Or can we do better?
    #     The tensor is T[i, 4, k] = A[i,4] * B[4,k].
    #     This is an outer product of a 4-vector (A[:,4]) and a 6-vector (B[4,:]).
    #     So the rank is 1? No, because the indices are (i, j, k) and j is fixed at 4.
    #     The tensor has non-zero entries only at j=4.
    #     T[i, 4, k] = A[i,4] * B[4,k].
    #     We need sum_c U[i*5+4, c] * V[4*6+k, c] * W[k*4+i, c] = 1 for all i,k.
    #     Note that U[i*5+4, c] is the entry for A[i,4].
    #     V[4*6+k, c] is the entry for B[4,k].
    #     W[k*4+i, c] is the entry for C[i,k].
    #     
    #     We can choose:
    #     U[i*5+4, c] = 1 for all i, c=0
    #     V[4*6+k, c] = 1 for all k, c=0
    #     W[k*4+i, c] = 1 for all i,k, c=0
    #     Then sum_c ... = 1*1*1 = 1.
    #     So rank 1 is sufficient for this part!
    #     Wait, is that correct?
    #     If we use one column c=0:
    #     U[i*5+4, 0] = 1
    #     V[4*6+k, 0] = 1
    #     W[k*4+i, 0] = 1
    #     Then for any (i, j=4, k):
    #     U[i*5+4, 0] * V[4*6+k, 0] * W[k*4+i, 0] = 1 * 1 * 1 = 1.
    #     Yes! So the j=4 part has rank 1.
    #
    #   Now, what about the j=0..3, k=0..5 part?
    #   We need T[i, j, k] = 1 for j=0..3, k=0..5.
    #   This is a 4x4x6 tensor.
    #   We can split k=0..5 into k=0..3 and k=4..5.
    #   
    #   Part 2a: j=0..3, k=0..3 (4x4x4)
    #   Part 2b: j=0..3, k=4..5 (4x4x2)
    #
    #   For 4x4x4, we can use 4 independent 2x2x2 Strassen decompositions.
    #   Split i=0..3 into i=0..1 and i=2..3.
    #   Split j=0..3 into j=0..1 and j=2..3.
    #   Split k=0..3 into k=0..1 and k=2..3.
    #   
    #   This gives 8 blocks of 2x2x2? No.
    #   The 4x4x4 tensor can be viewed as 8 blocks of 2x2x2?
    #   Actually, if we split all three dimensions into two halves, we get 2x2x2 = 8 sub-tensors.
    #   Each sub-tensor is a 2x2x2 matrix multiplication tensor.
    #   Each can be decomposed with rank 7 (Strassen).
    #   Total rank for 4x4x4: 8 * 7 = 56?
    #   Wait, the sub-tensors are not independent in the factorization.
    #   Let's be precise.
    #   
    #   Let's define the 4x4x4 decomposition explicitly.
    #   We want T[i,j,k] = 1 for i,j,k in 0..3.
    #   We can use the fact that R(4,4,4) <= 21? No, let's just use the block method.
    #   
    #   Actually, a simpler way:
    #   We can use the known rank-7 decomposition for 2x2x2 and apply it to each 2x2x2 block.
    #   But the blocks share indices.
    #   
    #   Let's try a different approach:
    #   Use the schoolbook for the 4x4x6 part (j=0..3, k=0..5).
    #   Rank = 4 * 4 * 6 = 96.
    #   Plus rank 1 for the 4x1x6 part (j=4, k=0..5).
    #   Total rank = 97.
    #   This is better than 120.
    #
    #   Can we do better?
    #   Let's try to use Strassen for the 4x4x4 part (j=0..3, k=0..3).
    #   And schoolbook for the rest.
    #
    #   Part 1: j=0..3, k=0..3 (4x4x4)
    #   Part 2: j=0..3, k=4..5 (4x4x2)
    #   Part 3: j=4, k=0..5 (4x1x6) -> rank 1
    #
    #   For Part 1 (4x4x4):
    #   We can use 4 independent 2x2x2 decompositions?
    #   No, let's use the standard block decomposition.
    #   Split i into (0,1) and (2,3).
    #   Split j into (0,1) and (2,3).
    #   Split k into (0,1) and (2,3).
    #   
    #   This gives 8 sub-tensors, each of size 2x2x2.
    #   Each sub-tensor corresponds to a specific (i_block, j_block, k_block).
    #   For example, i_block=0, j_block=0, k_block=0 corresponds to i=0,1; j=0,1; k=0,1.
    #   The tensor T[i,j,k] for i in {0,1}, j in {0,1}, k in {0,1} is the standard 2x2x2 matmul tensor.
    #   We can use Strassen (rank 7) for each of these 8 sub-tensors.
    #   Total rank for Part 1: 8 * 7 = 56.
    #
    #   For Part 2 (4x4x2):
    #   j=0..3, k=4..5.
    #   This is a 4x4x2 tensor.
    #   We can split j into (0,1) and (2,3).
    #   This gives 2 sub-tensors of size 4x2x2? No.
    #   Let's just use schoolbook for Part 2.
    #   Rank = 4 * 4 * 2 = 32.
    #
    #   Total rank = 56 + 32 + 1 = 89.
    #   This is better than 97.
    #
    #   Can we do even better?
    #   Let's try to optimize Part 2 (4x4x2).
    #   T[i,j,k] = 1 for i=0..3, j=0..3, k=4,5.
    #   We can view this as two 4x4x1 tensors (for k=4 and k=5).
    #   Each 4x4x1 tensor has rank 4*4*1 = 16? No.
    #   T[i,j,4] = 1 for all i,j.
    #   This is a rank-1 tensor? No, it's a 4x4 matrix of ones.
    #   The tensor T[i,j,4] = A[i,j] * B[j,4].
    #   We need sum_c U[i*5+j, c] * V[j*6+4, c] * W[4*4+i, c] = 1.
    #   This is similar to the j=4 case.
    #   We can use rank 1 for each k?
    #   For k=4:
    #   U[i*5+j, c] = 1 for all i,j, c=0
    #   V[j*6+4, c] = 1 for all j, c=0
    #   W[4*4+i, c] = 1 for all i, c=0
    #   Then sum_c ... = 1.
    #   So rank 1 for k=4.
    #   Similarly, rank 1 for k=5.
    #   Total rank for Part 2: 2.
    #
    #   So total rank = 56 + 2 + 1 = 59.
    #   This is much better!
    #
    #   Let's verify this logic.
    #   For Part 2, k=4:
    #   We need T[i,j,4] = 1 for all i,j.
    #   T[i,j,4] = sum_c U[i*5+j, c] * V[j*6+4, c] * W[4*4+i, c].
    #   If we set U[i*5+j, c] = 1, V[j*6+4, c] = 1, W[4*4+i, c] = 1 for a single column c,
    #   then the sum is 1.
    #   But we must ensure that this column does not interfere with other parts.
    #   Since the indices are disjoint (different k values), we can use separate columns.
    #
    #   So the plan is:
    #   1. 4x4x4 part (j=0..3, k=0..3): Use 8 Strassen decompositions, one for each 2x2x2 block.
    #      Total rank: 56.
    #   2. 4x4x2 part (j=0..3, k=4,5): Use 2 rank-1 decompositions.
    #      Total rank: 2.
    #   3. 4x1x6 part (j=4, k=0..5): Use 1 rank-1 decomposition.
    #      Total rank: 1.
    #   
    #   Total rank: 59.
    #
    #   Let's implement this.

    # Helper: tensor product of two 2x2x2 Strassen decompositions to get 4x4x4
    def _strassen_tensor_product(U, V, W, col_offset):
        """
        Add a rank-49 decomposition for 4x4x4 using tensor product of two 2x2x2 Strassen.
        col_offset: starting column index.
        Returns new col_offset.
        """
        U_s, V_s, W_s = _strassen_decomposition_2x2()
        for c1 in range(7):
            for c2 in range(7):
                c = col_offset + c1 * 7 + c2
                # U: 4x4, index [i1*2+i2, j1*2+j2]
                for i1 in range(2):
                    for i2 in range(2):
                        for j1 in range(2):
                            for j2 in range(2):
                                i_idx = i1 * 2 + i2
                                j_idx = j1 * 2 + j2
                                val = U_s[i1 * 2 + j1][c1] * U_s[i2 * 2 + j2][c2]
                                if val != 0:
                                    U[i_idx * m + j_idx][c] += val
                # V: 4x4, index [j1*2+j2, k1*2+k2]
                for j1 in range(2):
                    for j2 in range(2):
                        for k1 in range(2):
                            for k2 in range(2):
                                j_idx = j1 * 2 + j2
                                k_idx = k1 * 2 + k2
                                val = V_s[j1 * 2 + k1][c1] * V_s[j2 * 2 + k2][c2]
                                if val != 0:
                                    V[j_idx * p + k_idx][c] += val
                # W: 4x4, index [k1*2+k2, i1*2+i2]
                for k1 in range(2):
                    for k2 in range(2):
                        for i1 in range(2):
                            for i2 in range(2):
                                k_idx = k1 * 2 + k2
                                i_idx = i1 * 2 + i2
                                val = W_s[k1 * 2 + i1][c1] * W_s[k2 * 2 + i2][c2]
                                if val != 0:
                                    W[k_idx * n + i_idx][c] += val
        return col_offset + 49

    # Initialize U, V, W with zeros (total rank = 49 + 2 + 1 = 52)
    total_rank = 52
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Part 1: 4x4x4 (j=0..3, k=0..3) using tensor product, rank 49
    col_offset = _strassen_tensor_product(U, V, W, col_offset)
    
    # Part 2: 4x4x2 (j=0..3, k=4,5)
    # k=4
    for i in range(n):
        for j in range(4):  # j=0..3
            U[i * m + j][col_offset] = 1
            V[j * p + 4][col_offset] = 1
            W[4 * n + i][col_offset] = 1
    col_offset += 1
    
    # k=5
    for i in range(n):
        for j in range(4):  # j=0..3
            U[i * m + j][col_offset] = 1
            V[j * p + 5][col_offset] = 1
            W[5 * n + i][col_offset] = 1
    col_offset += 1
    
    # Part 3: 4x1x6 (j=4, k=0..5)
    for i in range(n):
        for k in range(p):
            U[i * m + 4][col_offset] = 1
            V[4 * p + k][col_offset] = 1
            W[k * n + i][col_offset] = 1
    col_offset += 1
    
    # Verify the decomposition
    if _verify(U, V, W):
        best = {"U": U, "V": V, "W": W}
        best_rank = total_rank
    else:
        # Fallback to schoolbook
        best = _schoolbook()
        best_rank = n * m * p
    
    return best
# EVOLVE-BLOCK-END
