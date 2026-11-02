"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """
    Build the 2x2 Strassen decomposition.
    Returns (U, V, W) where:
      U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_strassen_4x4():
    """
    Build 4x4 Strassen decomposition. Rank 49.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    strassen_defs = [
        ([((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)]),
        ([((1,0), 1), ((1,1), 1)],
         [((0,0), 1)],
         [((1,0), 1), ((1,1), -1)]),
        ([((0,0), 1)],
         [((0,1), 1), ((1,1), -1)],
         [((0,1), 1), ((1,1), 1)]),
        ([((1,1), 1)],
         [((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((1,0), 1)]),
        ([((0,0), 1), ((0,1), 1)],
         [((1,1), 1)],
         [((0,0), -1), ((0,1), 1)]),
        ([((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((0,1), 1)],
         [((1,1), 1)]),
        ([((0,1), 1), ((1,1), -1)],
         [((1,0), 1), ((1,1), 1)],
         [((0,0), 1)]),
    ]
    
    for s, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_defs):
        for t in range(7):
            col = s * 7 + t
            
            for la_i in range(2):
                for la_j in range(2):
                    local_a_row = la_i * 2 + la_j
                    coeff_a = U2[local_a_row][t]
                    if coeff_a != 0:
                        for (bi, bj), sigma in a_blocks:
                            global_i = bi * 2 + la_i
                            global_j = bj * 2 + la_j
                            u_row = global_i * 4 + global_j
                            U[u_row][col] += coeff_a * sigma
            
            for lb_j in range(2):
                for lb_k in range(2):
                    local_b_row = lb_j * 2 + lb_k
                    coeff_b = V2[local_b_row][t]
                    if coeff_b != 0:
                        for (bj, bk), sigma in b_blocks:
                            global_j = bj * 2 + lb_j
                            global_k = bk * 2 + lb_k
                            v_row = global_j * 4 + global_k
                            V[v_row][col] += coeff_b * sigma
            
            for lc_k in range(2):
                for lc_i in range(2):
                    local_c_row = lc_k * 2 + lc_i
                    coeff_c = W2[local_c_row][t]
                    if coeff_c != 0:
                        for (bi, bk), sigma in c_blocks:
                            global_k = bk * 2 + lc_k
                            global_i = bi * 2 + lc_i
                            w_row = global_k * 4 + global_i
                            W[w_row][col] += coeff_c * sigma
    
    return U, V, W

def _build_strassen_4x4x2():
    """
    Build 4x4x2 decomposition using 4 Strassen 2x2x2 blocks.
    A is 4x4, B is 4x2, C is 4x2.
    Rank = 28.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 28
    U = [[0]*rank for _ in range(16)] # 4x4 = 16 rows
    V = [[0]*rank for _ in range(8)]  # 4x2 = 8 rows
    W = [[0]*rank for _ in range(8)]  # 4x2 = 8 rows (p*n = 2*4)
    
    def add_strassen_block(a_i_start, a_j_start, b_j_start, c_i_start, c_k_start, base_rank):
        for t in range(7):
            col = base_rank + t
            
            # A part
            for li in range(2):
                for lj in range(2):
                    local_a_row = li * 2 + lj
                    coeff_a = U2[local_a_row][t]
                    if coeff_a != 0:
                        global_i = a_i_start + li
                        global_j = a_j_start + lj
                        u_row = global_i * 4 + global_j # m=4
                        U[u_row][col] += coeff_a
                        
            # B part
            for lj in range(2):
                for lk in range(2):
                    local_b_row = lj * 2 + lk
                    coeff_b = V2[local_b_row][t]
                    if coeff_b != 0:
                        global_j = b_j_start + lj
                        global_k = c_k_start + lk # B's col is C's col
                        v_row = global_j * 2 + global_k # p=2
                        V[v_row][col] += coeff_b
                        
            # C part
            for lk in range(2):
                for li in range(2):
                    local_c_row = lk * 2 + li
                    coeff_c = W2[local_c_row][t]
                    if coeff_c != 0:
                        global_k = c_k_start + lk
                        global_i = c_i_start + li
                        w_row = global_k * 4 + global_i # n=4
                        W[w_row][col] += coeff_c

    # 1. A11 * B11 -> C11
    add_strassen_block(0, 0, 0, 0, 0, 0)
    
    # 2. A12 * B21 -> C11
    add_strassen_block(0, 2, 2, 0, 0, 7)
    
    # 3. A21 * B11 -> C21
    add_strassen_block(2, 0, 0, 2, 0, 14)
    
    # 4. A22 * B21 -> C21
    add_strassen_block(2, 2, 2, 2, 0, 21)
    
    return U, V, W

def _build_strassen_2x2x3():
    """
    Build 2x2x3 decomposition.
    A is 2x2, B is 2x3, C is 2x3.
    Split B and C into 2x2 and 2x1 parts.
    Rank = 7 + 4 = 11.
    """
    U_strassen, V_strassen, W_strassen = _build_strassen_2x2()
    
    rank = 11
    U = [[0]*rank for _ in range(4)] # 2x2 = 4 rows
    V = [[0]*rank for _ in range(6)] # 2x3 = 6 rows
    W = [[0]*rank for _ in range(6)] # 2x3 = 6 rows
    
    # Part 1: 2x2x2 Strassen for cols 0,1
    for t in range(7):
        col = t
        # U
        for i in range(4):
            U[i][col] = U_strassen[i][t]
        # V
        for j in range(4):
            V[j][col] = V_strassen[j][t]
        # W
        for k in range(4):
            W[k][col] = W_strassen[k][t]
    
    # Part 2: 2x2x1 Schoolbook for col 2
    # Rank 4
    # C[:,2] = A * B[:,2]
    # C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2]
    # Terms:
    # 1. A[0,0]*B[0,2] -> C[0,2]
    # 2. A[0,1]*B[1,2] -> C[0,2]
    # 3. A[1,0]*B[0,2] -> C[1,2]
    # 4. A[1,1]*B[1,2] -> C[1,2]
    
    base = 7
    for t in range(4):
        col = base + t
        # U (A indices)
        if t == 0: U[0][col] = 1  # A[0,0]
        elif t == 1: U[1][col] = 1  # A[0,1]
        elif t == 2: U[2][col] = 1  # A[1,0]
        elif t == 3: U[3][col] = 1  # A[1,1]
        
        # V (B indices)
        # B is 2x3. Row j, Col k. Index j*3+k.
        # We need B[0,2] and B[1,2]. Indices 0*3+2=2, 1*3+2=5.
        if t == 0: V[2][col] = 1  # B[0,2]
        elif t == 1: V[5][col] = 1  # B[1,2]
        elif t == 2: V[2][col] = 1  # B[0,2]
        elif t == 3: V[5][col] = 1  # B[1,2]
        
        # W (C indices)
        # C is 2x3. Row i, Col k. Index i*3+k.
        # We need C[0,2] and C[1,2]. Indices 0*3+2=2, 1*3+2=5.
        if t == 0: W[2][col] = 1  # C[0,2]
        elif t == 1: W[2][col] = 1  # C[0,2]
        elif t == 2: W[5][col] = 1  # C[1,2]
        elif t == 3: W[5][col] = 1  # C[1,2]
        
    return U, V, W

def _build_strassen_4x3x6():
    """
    Build 4x3x6 decomposition.
    A is 4x3, B is 3x6, C is 4x6.
    Split 4 into 2+2. 3 into 2+1. 6 into 3+3.
    (2x2x3) + (2x2x3) + (2x2x3) + (2x2x3)
    Rank = 4 * 11 = 44.
    """
    U223, V223, W223 = _build_strassen_2x2x3()
    
    rank = 44
    U = [[0]*rank for _ in range(12)] # 4x3 = 12 rows
    V = [[0]*rank for _ in range(18)] # 3x6 = 18 rows
    W = [[0]*rank for _ in range(24)] # 4x6 = 24 rows
    
    # Blocks:
    # A11 (rows 0-1, cols 0-1) * B11 (rows 0-1, cols 0-5) -> C11 (rows 0-1, cols 0-5)
    # A12 (rows 0-1, cols 2)   * B21 (row 2, cols 0-5)     -> C11
    # A21 (rows 2-3, cols 0-1) * B11 (rows 0-1, cols 0-5) -> C21 (rows 2-3, cols 0-5)
    # A22 (rows 2-3, cols 2)   * B21 (row 2, cols 0-5)     -> C21
    
    # Wait, B is 3x6.
    # B11 is 2x6 (rows 0-1).
    # B21 is 1x6 (row 2).
    
    # The 2x2x3 decomposition handles A(2x2) * B(2x3).
    # Here B is 2x6.
    # So we can split B(2x6) into B_left(2x3) and B_right(2x3).
    # C(2x6) into C_left(2x3) and C_right(2x3).
    
    # So for each 2x2 block of A, we have two 2x2x3 problems.
    
    # Total blocks:
    # A11 (2x2) * B11 (2x6) -> C11 (2x6)
    #   Split B11 into B11_L (2x3), B11_R (2x3).
    #   Split C11 into C11_L (2x3), C11_R (2x3).
    #   Problem 1: A11 * B11_L -> C11_L
    #   Problem 2: A11 * B11_R -> C11_R
    
    # A12 (2x1) * B21 (1x6) -> C12 (2x6)
    #   This is not 2x2x3.
    #   A12 is 2x1. B21 is 1x6.
    #   C12[i, k] = A12[i, 0] * B21[0, k].
    #   This is an outer product. Rank 2*6 = 12?
    #   Actually, C12 = A12 * B21.
    #   A12 has 2 entries. B21 has 6 entries.
    #   C12 has 12 entries.
    #   Rank is at most 2*6=12, but likely 2 or 6?
    #   Since it's rank 1 matrix (outer product of two vectors), rank is 1?
    #   No, it's a sum of products.
    #   C12[i,k] = sum_j A12[i,j] B21[j,k].
    #   Here j goes from 0 to 0 (since 1 col).
    #   So C12 = A12 (2x1) * B21 (1x6).
    #   This is a rank 1 matrix.
    #   But we need a tensor decomposition.
    #   U has shape (2*1, R) = (2, R).
    #   V has shape (1*6, R) = (6, R).
    #   W has shape (6*2, R) = (12, R).
    #   T[i, j, k] = U[i, r] V[j, r] W[k, r] ? No.
    #   T[i, j, k] = U[i, r] V[j, r] W[k, r] is for 2x2x2.
    #   Here dimensions are 2x1x6.
    #   T[i, j, k] where i in 0..1, j in 0..0, k in 0..5.
    #   U[i, r], V[j, r], W[k, r].
    #   T[i, 0, k] = sum_r U[i, r] V[0, r] W[k, r].
    #   We need T[i, 0, k] = A12[i, 0] B21[0, k].
    #   This is separable!
    #   T[i, 0, k] = u_i v_k.
    #   We can set R=1.
    #   U[i, 0] = A12[i, 0].
    #   V[0, 0] = 1.
    #   W[k, 0] = B21[0, k].
    #   So Rank 1 for this block?
    #   Wait, the tensor is linear in A and B.
    #   The decomposition must be linear in A and B.
    #   So U[i, r] must be coefficients of A entries.
    #   V[j, r] must be coefficients of B entries.
    #   W[k, r] must be coefficients of C entries? No, W is the output coefficients.
    #   In the tensor T_{ijk}, i is A index, j is B index, k is C index.
    #   T_{ijk} = U[i, r] V[j, r] W[k, r].
    #   For A12 * B21:
    #   i is row of A12 (0..1). j is col of A12 (0). k is col of B21 (0..5).
    #   Wait, B21 is 1x6. So j index for B is just 0 (row), and k is col?
    #   No, in the global tensor, B is 3x6.
    #   Block A12 is rows 0-1, col 2 of A.
    #   Block B21 is row 2, cols 0-5 of B.
    #   Block C12 is rows 0-1, cols 0-5 of C.
    #   So global indices:
    #   i: 0,1. j (A col): 2. k (B row): 2. l (B col): 0..5.
    #   Wait, standard tensor is T[i, j, k] where C[i, k] = sum_j A[i, j] B[j, k].
    #   So for block:
    #   i: 0,1 (global rows 0,1).
    #   j: 2 (global col 2).
    #   k: 2 (global row 2).
    #   Wait, B[j, k] where j is row, k is col.
    #   So B[2, l] where l is 0..5.
    #   So the indices involved are:
    #   A[i, 2] for i in 0,1.
    #   B[2, l] for l in 0..5.
    #   C[i, l] for i in 0,1, l in 0..5.
    #   T[i, 2, 2, l] = sum_j A[i, j] B[j, l] -> here j is fixed at 2?
    #   No, the sum is over the middle dimension.
    #   In the full multiplication, C[i, l] = sum_{j=0}^{4} A[i, j] B[j, l].
    #   For this block, we are isolating j=2.
    #   So C[i, l] += A[i, 2] B[2, l].
    #   This term is A[i, 2] * B[2, l].
    #   This is rank 1 in the tensor of this block.
    #   U[i, r] should pick A[i, 2].
    #   V[2, l, r] should pick B[2, l].
    #   W[l, i, r] should pick C[i, l].
    #   We can do this with R=1 per (i,l) pair? No, shared.
    #   U[i, 0] = 1 for i=0,1. (Picks A[i, 2])
    #   V[2*6+l, 0] = 1 for l=0..5. (Picks B[2, l])
    #   W[l*4+i, 0] = 1 for l=0..5, i=0,1. (Picks C[i, l])
    #   But we need to multiply A[i, 2] and B[2, l].
    #   If U[i, 0] = 1, V[2*6+l, 0] = 1, W[l*4+i, 0] = 1,
    #   then sum_r U[i, r] V[2*6+l, r] W[l*4+i, r] = 1*1*1 = 1.
    #   This gives 1 for all (i, l).
    #   But we need A[i, 2] * B[2, l].
    #   So U and V must contain the values? No, they are constant matrices.
    #   The values A and B are multiplied externally?
    #   No, the decomposition is of the tensor T_{ijk} = delta_{ik}? No.
    #   The tensor is T_{i, j, k} = 1 if j is the middle index connecting i and k.
    #   Actually, the matrix multiplication tensor T_{i, j, k} is 1 if the indices match in the sum.
    #   C_{i, k} = sum_j A_{i, j} B_{j, k}.
    #   The tensor is T_{i, j, k} such that C_{i, k} = sum_j T_{i, j, k} A_{i, j} B_{j, k}.
    #   So T_{i, j, k} = 1 if the indices are consistent?
    #   No, T_{i, j, k} is 1 for all i, j, k?
    #   No, the sum is over j.
    #   C_{i, k} = sum_j A_{i, j} B_{j, k}.
    #   This is a bilinear form.
    #   The tensor T has components T_{i, j, k} = 1.
    #   Wait, if T_{i, j, k} = 1, then sum_j 1 * A_{i, j} B_{j, k} = sum_j A_{i, j} B_{j, k}.
    #   Yes.
    #   So the tensor is the all-ones tensor.
    #   We are decomposing the all-ones tensor T_{i, j, k} = 1 into rank R.
    #   U_{i, r} V_{j, r} W_{k, r}.
    #   So for the block A12 * B21:
    #   Indices: i in {0, 1}, j in {2}, k in {0, ..., 5} (Wait, k is col of C, which corresponds to col of B).
    #   Wait, B is 3x6. So B's columns are 0..5.
    #   So k in 0..5.
    #   We need sum_r U[i, r] V[2, r] W[k, r] = 1 for all i in {0, 1}, k in 0..5.
    #   Wait, V index is j=2. But V is flattened.
    #   V has shape (m*p, R) = (3*6, R) = (18, R).
    #   The index for B[2, k] is 2*6 + k = 12 + k.
    #   So V[12+k, r].
    #   We need sum_r U[i, r] V[12+k, r] W[k*4+i, r] = 1.
    #   Note W index is k*n + i = 6*4 + i? No, p*n = 6*4 = 24.
    #   W index is k*4 + i.
    #   We can set R=1.
    #   U[i, 0] = 1 for i=0, 1.
    #   V[12+k, 0] = 1 for k=0..5.
    #   W[k*4+i, 0] = 1 for k=0..5, i=0, 1.
    #   Then sum = 1*1*1 = 1.
    #   This works! Rank 1 for this block.
    
    # So for A12 * B21, we need 1 rank unit.
    # Similarly for A22 * B21, we need 1 rank unit.
    
    # Let's re-evaluate the 4x3x6 decomposition.
    # Blocks:
    # 1. A11 (2x2) * B11 (2x6) -> C11 (2x6). Rank 22 (two 2x2x3 blocks).
    # 2. A12 (2x1) * B21 (1x6) -> C12 (2x6). Rank 1.
    # 3. A21 (2x2) * B11 (2x6) -> C21 (2x6). Rank 22.
    # 4. A22 (2x1) * B21 (1x6) -> C22 (2x6). Rank 1.
    
    # Total Rank = 22 + 1 + 22 + 1 = 46.
    # Previous estimate was 44. The difference is due to the 2x1x6 blocks.
    # 2x1x6 rank is 1, not 4?
    # Yes, because T_{i, j, k} = 1 is rank 1 if j is fixed and i, k vary independently?
    # T_{i, j, k} = 1.
    # If j is fixed, T_{i, j0, k} = 1.
    # This is the outer product of vector u (1s) and vector w (1s).
    # u_i w_k.
    # Yes, rank 1.
    
    # So 4x3x6 rank is 46.
    
    # Let's build this.
    
    def add_2x2x3_block(a_i_start, a_j_start, b_j_start, c_i_start, c_k_start, base_rank):
        for t in range(11):
            col = base_rank + t
            # U
            for li in range(2):
                for lj in range(2):
                    local_a_row = li * 2 + lj
                    coeff = U223[local_a_row][t]
                    if coeff != 0:
                        gi = a_i_start + li
                        gj = a_j_start + lj
                        u_row = gi * 3 + gj
                        U[u_row][col] += coeff
                        
            # V
            for lj in range(2):
                for lk in range(3):
                    local_b_row = lj * 3 + lk
                    coeff = V223[local_b_row][t]
                    if coeff != 0:
                        gj = b_j_start + lj
                        gk = c_k_start + lk
                        v_row = gj * 6 + gk
                        V[v_row][col] += coeff
                        
            # W
            for lk in range(3):
                for li in range(2):
                    local_c_row = lk * 2 + li
                    coeff = W223[local_c_row][t]
                    if coeff != 0:
                        gk = c_k_start + lk
                        gi = c_i_start + li
                        w_row = gk * 4 + gi
                        W[w_row][col] += coeff
                        
    def add_2x1x6_block(a_i_start, a_j_start, b_j_start, c_i_start, c_k_start, base_rank):
        col = base_rank
        # U
        for li in range(2):
            gi = a_i_start + li
            gj = a_j_start
            u_row = gi * 3 + gj
            U[u_row][col] = 1
        # V
        for lk in range(6):
            gj = b_j_start
            gk = c_k_start + lk
            v_row = gj * 6 + gk
            V[v_row][col] = 1
        # W
        for lk in range(6):
            for li in range(2):
                gk = c_k_start + lk
                gi = c_i_start + li
                w_row = gk * 4 + gi
                W[w_row][col] = 1

    # 1. A11 * B11 (Left) -> C11 (Left)
    add_2x2x3_block(0, 0, 0, 0, 0, 0)
    # 2. A11 * B11 (Right) -> C11 (Right)
    add_2x2x3_block(0, 0, 0, 0, 3, 11)
    # 3. A12 * B21 -> C12
    add_2x1x6_block(0, 2, 2, 0, 0, 22)
    # 4. A21 * B11 (Left) -> C21 (Left)
    add_2x2x3_block(2, 0, 0, 2, 0, 23)
    # 5. A21 * B11 (Right) -> C21 (Right)
    add_2x2x3_block(2, 0, 0, 2, 3, 34)
    # 6. A22 * B21 -> C22
    add_2x1x6_block(2, 2, 2, 2, 0, 45)
    
    return U, V, W

def _build_strassen_2x2x2_rank6():
    """
    Try to build a rank 6 decomposition for 2x2x2?
    No, Strassen is optimal for 2x2x2.
    """
    pass

def _build_strassen_4x2x6():
    """
    Build 4x2x6 decomposition.
    A is 4x2, B is 2x6, C is 4x6.
    Split 4 into 2+2. 2 is 2. 6 into 3+3.
    (2x2x3) + (2x2x3) + (2x2x3) + (2x2x3)
    Rank = 4 * 11 = 44.
    """
    U223, V223, W223 = _build_strassen_2x2x3()
    
    rank = 44
    U = [[0]*rank for _ in range(8)] # 4x2 = 8 rows
    V = [[0]*rank for _ in range(12)] # 2x6 = 12 rows
    W = [[0]*rank for _ in range(24)] # 4x6 = 24 rows
    
    def add_2x2x3_block(a_i_start, a_j_start, b_j_start, c_i_start, c_k_start, base_rank):
        for t in range(11):
            col = base_rank + t
            # U
            for li in range(2):
                for lj in range(2):
                    local_a_row = li * 2 + lj
                    coeff = U223[local_a_row][t]
                    if coeff != 0:
                        gi = a_i_start + li
                        gj = a_j_start + lj
                        u_row = gi * 2 + gj
                        U[u_row][col] += coeff
                        
            # V
            for lj in range(2):
                for lk in range(3):
                    local_b_row = lj * 3 + lk
                    coeff = V223[local_b_row][t]
                    if coeff != 0:
                        gj = b_j_start + lj
                        gk = c_k_start + lk
                        v_row = gj * 6 + gk
                        V[v_row][col] += coeff
                        
            # W
            for lk in range(3):
                for li in range(2):
                    local_c_row = lk * 2 + li
                    coeff = W223[local_c_row][t]
                    if coeff != 0:
                        gk = c_k_start + lk
                        gi = c_i_start + li
                        w_row = gk * 4 + gi
                        W[w_row][col] += coeff

    # 1. A11 * B11 (Left) -> C11 (Left)
    add_2x2x3_block(0, 0, 0, 0, 0, 0)
    # 2. A11 * B11 (Right) -> C11 (Right)
    add_2x2x3_block(0, 0, 0, 0, 3, 11)
    # 3. A21 * B11 (Left) -> C21 (Left)
    add_2x2x3_block(2, 0, 0, 2, 0, 22)
    # 4. A21 * B11 (Right) -> C21 (Right)
    add_2x2x3_block(2, 0, 0, 2, 3, 33)
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """
    Build schoolbook decomposition for n x m * m x p -> n x p.
    Returns U, V, W with rank n*m*p.
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _combine_decompositions(decomps, n, m, p):
    """
    Combine a list of decompositions into one.
    """
    total_rank = sum(len(d[0][0]) for d in decomps)
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    current_rank = 0
    for U_sub, V_sub, W_sub, (i_s, i_e), (j_s, j_e), (k_s, k_e) in decomps:
        n_sub = i_e - i_s
        m_sub = j_e - j_s
        p_sub = k_e - k_s
        rank_sub = len(U_sub[0])
        
        for r_sub in range(n_sub * m_sub):
            i_loc = r_sub // m_sub
            j_loc = r_sub % m_sub
            i_glob = i_s + i_loc
            j_glob = j_s + j_loc
            r_glob = i_glob * m + j_glob
            
            for c_sub in range(rank_sub):
                U[r_glob][current_rank + c_sub] = U_sub[r_sub][c_sub]
        
        for r_sub in range(m_sub * p_sub):
            j_loc = r_sub // p_sub
            k_loc = r_sub % p_sub
            j_glob = j_s + j_loc
            k_glob = k_s + k_loc
            r_glob = j_glob * p + k_glob
            
            for c_sub in range(rank_sub):
                V[r_glob][current_rank + c_sub] = V_sub[r_sub][c_sub]
        
        for r_sub in range(p_sub * n_sub):
            k_loc = r_sub // n_sub
            i_loc = r_sub % n_sub
            k_glob = k_s + k_loc
            i_glob = i_s + i_loc
            r_glob = k_glob * n + i_glob
            
            for c_sub in range(rank_sub):
                W[r_glob][current_rank + c_sub] = W_sub[r_sub][c_sub]
        
        current_rank += rank_sub
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    try:
        # Option 1: 4x4x4 Strassen (49) + 4x1x6 Schoolbook (24) + 4x4x2 Strassen (28)
        # Total Rank: 101
        
        # Option 2: 4x3x6 (46) + 4x2x6 (44)
        # Total Rank: 90
        
        # Option 3: 4x2x6 (44) + 4x2x6 (44) + 4x1x6 (24)
        # Total Rank: 112 (Worse)
        
        # Let's try Option 2.
        U1, V1, W1 = _build_strassen_4x3x6()
        decomp1 = (U1, V1, W1, (0, 4), (0, 3), (0, 6))
        
        U2, V2, W2 = _build_strassen_4x2x6()
        decomp2 = (U2, V2, W2, (0, 4), (3, 5), (0, 6))
        
        U, V, W = _combine_decompositions([decomp1, decomp2], n, m, p)
        
        return {"U": U, "V": V, "W": W}
        
    except Exception:
        # Fallback to Option 1 if Option 2 fails
        try:
            U1, V1, W1 = _build_strassen_4x4()
            decomp1 = (U1, V1, W1, (0, 4), (0, 4), (0, 4))
            
            U2, V2, W2 = _build_schoolbook(4, 1, 6)
            decomp2 = (U2, V2, W2, (0, 4), (4, 5), (0, 6))
            
            U3, V3, W3 = _build_strassen_4x4x2()
            decomp3 = (U3, V3, W3, (0, 4), (0, 4), (4, 6))
            
            U, V, W = _combine_decompositions([decomp1, decomp2, decomp3], n, m, p)
            return {"U": U, "V": V, "W": W}
        except Exception:
            # Fallback to schoolbook
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

# EVOLVE-BLOCK-END
