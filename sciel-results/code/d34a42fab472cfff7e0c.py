"""Improved program for matrix multiplication 4,4,6 rank-minimal search."""
import numpy as np

DIMENSIONS = [4, 4, 6]


def _strassen_2x2():
    """
    Returns U, V, W for 2x2 * 2x2 -> 2x2 with rank 7.
    Indices:
    U: (4, 7)
    V: (4, 7)
    W: (4, 7)
    T[a, b, c] = 1 if (i,j,k) is valid.
    a = i*2+j, b = j*2+k, c = k*2+i
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Strassen's 7 products:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Mapping of (i,j,k) to indices a=i*m+j, b=j*p+k, c=k*n+i
    # i,j,k in {0,1}
    # a: 00=0, 01=1, 10=2, 11=3
    # b: 00=0, 01=1, 10=2, 11=3
    # c: 00=0, 01=1, 10=2, 11=3
    
    # M1: (A00 + A11)(B00 + B11)
    # A00: i=0, j=0 -> a=0
    # A11: i=1, j=1 -> a=3
    # B00: j=0, k=0 -> b=0
    # B11: j=1, k=1 -> b=3
    # U[:, 0] has 1 at a=0, a=3
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # W: c indices for M1 contribution
    # C00 += M1 -> c=0 (k=0, i=0)
    # C11 += M1 -> c=3 (k=1, i=1)
    # C01 -= M1 -> c=1 (k=0, i=1) ? No, C01 = A00B01 + A01B11. M1 is (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # Strassen formulas:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's populate W based on these formulas.
    # W[c, r] is the coefficient for product r in output c.
    
    # r=0 (M1)
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11
    
    # r=1 (M2): (A10 + A11)B00
    # A10: i=1, j=0 -> a=2
    # A11: i=1, j=1 -> a=3
    # B00: j=0, k=0 -> b=0
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C10 += M2 -> c=2 (k=0, i=1)
    # C11 -= M2 -> c=3
    W[2, 1] = 1
    W[3, 1] = -1
    
    # r=2 (M3): A00(B10 - B11)
    # A00: i=0, j=0 -> a=0
    # B10: j=1, k=0 -> b=2
    # B11: j=1, k=1 -> b=3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    # C01 += M3 -> c=1 (k=0, i=1)
    # C11 += M3 -> c=3
    W[1, 2] = 1
    W[3, 2] = 1
    
    # r=3 (M4): A11(B01 - B00)
    # A11: i=1, j=1 -> a=3
    # B01: j=0, k=1 -> b=1
    # B00: j=0, k=0 -> b=0
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    # C00 += M4 -> c=0
    # C10 += M4 -> c=2
    W[0, 3] = 1
    W[2, 3] = 1
    
    # r=4 (M5): (A00 + A01)B11
    # A00: i=0, j=0 -> a=0
    # A01: i=0, j=1 -> a=1
    # B11: j=1, k=1 -> b=3
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C00 -= M5 -> c=0
    # C01 += M5 -> c=1
    W[0, 4] = -1
    W[1, 4] = 1
    
    # r=5 (M6): (A10 - A00)(B00 + B01)
    # A10: i=1, j=0 -> a=2
    # A00: i=0, j=0 -> a=0
    # B00: j=0, k=0 -> b=0
    # B01: j=0, k=1 -> b=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C11 += M6 -> c=3
    W[3, 5] = 1
    
    # r=6 (M7): (A01 - A11)(B10 + B11)
    # A01: i=0, j=1 -> a=1
    # A11: i=1, j=1 -> a=3
    # B10: j=1, k=0 -> b=2
    # B11: j=1, k=1 -> b=3
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C00 += M7 -> c=0
    W[0, 6] = 1
    
    return U, V, W


def _construct_4x4x4():
    """
    Construct U, V, W for 4x4 * 4x4 -> 4x4 using Strassen's algorithm on 2x2 blocks.
    Rank = 7 * 7 = 49.
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    U_2, V_2, W_2 = _strassen_2x2()
    
    # We have 7 block multiplications.
    # Each block multiplication corresponds to a (2x2) * (2x2) -> (2x2) problem.
    # The indices for the 4x4 case are composed of block indices and local indices.
    # Block indices: bi, bj in {0,1} for A, bj, bk in {0,1} for B, bk, bi in {0,1} for C.
    # Local indices: li, lj in {0,1} for A, lj, lk in {0,1} for B, lk, li in {0,1} for C.
    
    # Global indices:
    # a = (bi*2 + li)*4 + (bj*2 + lj) ? No.
    # A is 4x4. Row i = bi*2 + li. Col j = bj*2 + lj.
    # a = i*4 + j = (bi*2 + li)*4 + bj*2 + lj.
    
    # b = j*4 + k = (bj*2 + lj)*4 + bk*2 + lk.
    
    # c = k*4 + i = (bk*2 + lk)*4 + bi*2 + li.
    
    # For each block multiplication r_block in 0..6:
    #   For each local multiplication r_local in 0..6:
    #     r = r_block * 7 + r_local
    #     The U, V, W entries are determined by U_2, V_2, W_2.
    
    for r_block in range(7):
        for r_local in range(7):
            r = r_block * 7 + r_local
            
            # U_2 has non-zeros at specific local (i, j) pairs for this r_block?
            # No, U_2[r_local] is a vector of length 4 (2x2).
            # It has non-zeros at local indices corresponding to the Strassen formula for r_block.
            # Wait, U_2 is the U matrix for the 2x2 problem.
            # U_2[local_a, r_local] is the coefficient.
            # But U_2 depends on r_block?
            # No, the Strassen algorithm is the same for all 7 products.
            # The coefficients in U_2, V_2, W_2 are fixed for the 2x2 case.
            # However, the *combination* of blocks is what matters.
            # Actually, the Strassen algorithm computes 7 products M1..M7.
            # Each Mk is a product of two 2x2 matrices.
            # So for each Mk, we use the rank-7 decomposition of 2x2 matmul.
            
            # The U matrix for the 4x4 case:
            # U[a, r] = U_2[local_a, r_local] if the block (bi, bj) is involved in Mk?
            # No, the Strassen algorithm defines linear combinations of the A blocks.
            # E.g., M1 = (A00 + A11)(B00 + B11).
            # This involves A blocks 00 and 11.
            # So for r_block=0 (M1), U should have non-zeros for A00 and A11.
            # My _strassen_2x2() returns U for a *single* 2x2 multiplication.
            # It doesn't encode which blocks are combined.
            # I need to construct the U, V, W for the 4x4 case directly using the block structure.
            
            # Let's restart the construction logic for 4x4.
            pass
    
    # Correct construction:
    # The 7 products are:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # For each Mk, we perform a 2x2 matrix multiplication.
    # Let Uk, Vk, Wk be the rank-7 decomposition for the 2x2 multiplication.
    # Then for the 4x4 multiplication, the contribution to U is:
    # For each term in Mk, e.g., A00, we add Uk to the rows corresponding to A00.
    
    # Let's define the block mappings.
    # Blocks of A: A00, A01, A10, A11.
    # Block Aij has rows i*2..i*2+1, cols j*2..j*2+1.
    # Global row index for local row l in block i: 2*i + l.
    # Global col index for local col j in block j: 2*j + j.
    # Global index a = (2*i + l)*4 + (2*j + l_j).
    
    # We will iterate over the 7 products.
    for k in range(7):
        # Determine which A blocks and B blocks are involved in Mk, and their signs.
        # Mk = sum_{ij} coeff_A_ij * A_ij  *  sum_{jk} coeff_B_jk * B_jk
        # Actually, Mk is a product of two linear combinations.
        # L_A = sum_{ij} c_A_ij A_ij
        # L_B = sum_{jk} c_B_jk B_jk
        # Mk = L_A * L_B
        
        # We need the coefficients c_A_ij and c_B_jk for each k.
        # Based on Strassen's formulas:
        if k == 0: # M1
            c_A = {(0,0): 1, (1,1): 1}
            c_B = {(0,0): 1, (1,1): 1}
        elif k == 1: # M2
            c_A = {(1,0): 1, (1,1): 1}
            c_B = {(0,0): 1}
        elif k == 2: # M3
            c_A = {(0,0): 1}
            c_B = {(1,0): 1, (1,1): -1}
        elif k == 3: # M4
            c_A = {(1,1): 1}
            c_B = {(0,1): 1, (0,0): -1}
        elif k == 4: # M5
            c_A = {(0,0): 1, (0,1): 1}
            c_B = {(1,1): 1}
        elif k == 5: # M6
            c_A = {(1,0): 1, (0,0): -1}
            c_B = {(0,0): 1, (0,1): 1}
        else: # M7
            c_A = {(0,1): 1, (1,1): -1}
            c_B = {(1,0): 1, (1,1): 1}
            
        # For each local rank component r_local in 0..6:
        # The 2x2 multiplication L_A * L_B uses the rank-7 decomposition.
        # U_2[local_a, r_local] is the coefficient for A_local[local_i, local_j].
        # But L_A is a sum of blocks.
        # So U[global_a, r] gets contribution from each block A_ij involved.
        
        for r_local in range(7):
            r = k * 7 + r_local
            
            # Get U_2, V_2, W_2 for this r_local
            # U_2 is 4x7. U_2[:, r_local] is a vector of length 4.
            # local_a = local_i * 2 + local_j
            
            # For each A block (bi, bj) with coeff c_A:
            for (bi, bj), coeff_A in c_A.items():
                # For each local index in U_2[:, r_local]
                for local_a in range(4):
                    val = U_2[local_a, r_local]
                    if val == 0: continue
                    # local_a = li * 2 + lj
                    li = local_a // 2
                    lj = local_a % 2
                    # Global indices for A block (bi, bj)
                    gi = 2 * bi + li
                    gj = 2 * bj + lj
                    global_a = gi * 4 + gj
                    U[global_a, r] += coeff_A * val
            
            # For each B block (bj, bk) with coeff c_B:
            for (bj, bk), coeff_B in c_B.items():
                for local_b in range(4):
                    val = V_2[local_b, r_local]
                    if val == 0: continue
                    # local_b = lj * 2 + lk
                    lj = local_b // 2
                    lk = local_b % 2
                    # Global indices for B block (bj, bk)
                    gj = 2 * bj + lj
                    gk = 2 * bk + lk
                    global_b = gj * 4 + gk
                    V[global_b, r] += coeff_B * val
            
            # For W, the output is C.
            # C = sum_k Mk * (linear combination of C blocks)
            # The W_2 matrix gives the coefficients for the 2x2 output.
            # But we need to map these to the 4x4 output blocks.
            # The Strassen algorithm reconstructs C blocks from M1..M7.
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # So for a specific Mk, it contributes to specific C blocks.
            # Let's define the mapping from k to C blocks and signs.
            c_C = {}
            if k == 0: # M1
                c_C = {(0,0): 1, (1,1): 1}
            elif k == 1: # M2
                c_C = {(1,0): 1, (1,1): -1}
            elif k == 2: # M3
                c_C = {(0,1): 1, (1,1): 1}
            elif k == 3: # M4
                c_C = {(0,0): 1, (1,0): 1}
            elif k == 4: # M5
                c_C = {(0,0): -1, (0,1): 1}
            elif k == 5: # M6
                c_C = {(1,1): 1}
            else: # M7
                c_C = {(0,0): 1}
            
            for (bk, bi), coeff_C in c_C.items():
                for local_c in range(4):
                    val = W_2[local_c, r_local]
                    if val == 0: continue
                    # local_c = lk * 2 + li
                    lk = local_c // 2
                    li = local_c % 2
                    # Global indices for C block (bk, bi)
                    gk = 2 * bk + lk
                    gi = 2 * bi + li
                    global_c = gk * 4 + gi
                    W[global_c, r] += coeff_C * val
    
    return U, V, W, rank


def _construct_4x4x2():
    """
    Construct U, V, W for 4x4 * 4x2 -> 4x2 using Schoolbook.
    Rank = 32.
    """
    n, m, p = 4, 4, 2
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W, rank


def construct():
    n, m, p = DIMENSIONS
    
    # Part 1: 4x4 * 4x4 (columns 0-3)
    U1, V1, W1, r1 = _construct_4x4x4()
    
    # Part 2: 4x4 * 4x2 (columns 4-5)
    U2, V2, W2, r2 = _construct_4x4x2()
    
    # Combine
    # U1 is (16, 49), U2 is (16, 32)
    # We need to map the columns of B and C correctly.
    # B is 4x6.
    # B1 is 4x4 (cols 0-3). B2 is 4x2 (cols 4-5).
    # V1 is (16, 49) for B1. Indices j*p+k where p=4.
    # V2 is (8, 32) for B2. Indices j*2+k where p=2.
    # We need to embed V2 into a (24, 32) matrix? No, V is (m*p, rank) = (24, rank).
    # For Part 1, V1 is (16, 49). We need to place it in the first 16 rows of a (24, 49) matrix?
    # No, the rows of V correspond to (j, k).
    # For Part 1, k is 0..3. So rows 0..15 of V (if p=4).
    # But the total p is 6. So rows are j*6+k.
    # For Part 1 (k=0..3), the row index in the full V is j*6 + k.
    # For Part 2 (k=0..1 in local, 4..5 in global), the row index is j*6 + (4+k).
    
    # Let's construct the full U, V, W.
    total_rank = r1 + r2
    U = np.zeros((16, total_rank), dtype=int)
    V = np.zeros((24, total_rank), dtype=int)
    W = np.zeros((24, total_rank), dtype=int) # p*n = 6*4 = 24
    
    # Fill U
    U[:, :r1] = U1
    U[:, r1:] = U2
    
    # Fill V
    # V1: rows j*4+k. Map to j*6+k.
    for j in range(4):
        for k in range(4):
            V[j*6 + k, :r1] = V1[j*4 + k, :]
    
    # V2: rows j*2+k (local). Map to j*6 + (4+k).
    for j in range(4):
        for k in range(2):
            V[j*6 + 4 + k, r1:] = V2[j*2 + k, :]
    
    # Fill W
    # W1: rows k*4+i. Map to k*4+i. (k is 0..3)
    # But W index is k*n+i. n=4.
    # For Part 1, k=0..3.
    for k in range(4):
        for i in range(4):
            W[k*4 + i, :r1] = W1[k*4 + i, :]
    
    # W2: rows k*4+i (local k=0..1). Map to (4+k)*4 + i.
    for k in range(2):
        for i in range(4):
            W[(4+k)*4 + i, r1:] = W2[k*4 + i, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
