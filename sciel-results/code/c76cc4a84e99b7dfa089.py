"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time

def _make_2x3x9_rank_51():
    # Returns U (6x51), V (27x51), W (18x51) as lists of lists.
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=2, m=3, p=9.
    n, m, p = 2, 3, 9
    rank = 51
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    c = 0

    # --- Part 1: A_top * B_top for j in {0,1}, k in {0..7} (rank 22) ---
    # For j in {0,1}, we treat B_{j,:} as a 2x4 matrix [[b_{j0},b_{j1}], [b_{j2},b_{j3}]] etc.
    # Actually, let's partition k into two groups of 4: {0,1,2,3} and {4,5,6,7}.
    # For each group, we do Strassen on the 2x2 block of B for rows k={2a,2a+1} and k={2a+2,2a+3}?
    # No, Strassen works on 2x2 * 2x2.
    # Let's partition k into 4 groups of 2: {0,1}, {2,3}, {4,5}, {6,7}.
    # For each group g in 0..3, we compute C[:, 2g:2g+2] = A_top * B_{top, 2g:2g+2}.
    # A_top is 2x2 (rows i=0,1; cols j=0,1).
    # B_{top, 2g:2g+2} is 2x2 (rows j=0,1; cols k=2g, 2g+1).
    # Strassen gives 7 terms. 4 groups * 7 = 28 terms.
    # But we can do better. Let's use Strassen for the entire 2x8 multiplication? No, standard Strassen is for powers of 2.
    # Let's stick to the plan:
    # For k in {0,1,2,3} (group 0): 7 terms.
    # For k in {4,5,6,7} (group 1): 7 terms.
    # Total 14 terms for A_top * B_top (where B_top has cols j=0,1 and k=0..7).
    # Wait, A_top has 2 rows and 2 cols (j=0,1). B_top has 2 rows (j=0,1) and 8 cols (k=0..7).
    # So we are multiplying (2x2) by (2x8).
    # We can split the 8 cols into four 2-col blocks.
    # Each block multiplication is (2x2) * (2x2) -> (2x2).
    # Rank 7 per block. 4 blocks -> 28 terms.
    
    # Let's re-evaluate the block strategy.
    # Total 2x3x9.
    # Split j into {0,1} and {2}.
    # Split A into A_L (2x2, cols 0,1) and A_R (2x1, col 2).
    # Split B into B_L (2x9, rows 0,1) and B_R (1x9, row 2).
    
    # Term 1: A_L * B_L (2x2 * 2x9 -> 2x9). Rank?
    #   Split k in B_L into {0..3} and {4..8}? Or {0,1}, {2,3}, {4,5}, {6,7}, {8}.
    #   Using 2x2 Strassen for the first 8 columns: 4 blocks * 7 = 28.
    #   Column 8 is 2x2 * 2x1. Rank 4.
    #   Total 32.
    
    # Term 2: A_L * B_R (2x2 * 1x9 -> 2x9).
    #   This is just scaling. a_{i0} * b_{2k} and a_{i1} * b_{2k}.
    #   Actually, A_L * B_R = sum_{j in {0,1}} a_{ij} b_{jk}.
    #   Since B_R is 1x9 (row index j=2 is not in A_L? No, A_L has columns j=0,1. B_R has row j=2. They don't contract directly?
    #   Wait, matrix multiplication C_{ik} = sum_j A_{ij} B_{jk}.
    #   A_L has non-zeros at j=0,1. B_R has non-zeros at j=2.
    #   So A_L * B_R is zero!
    #   Similarly, A_R * B_L is zero!
    #   
    #   So the only non-zero cross terms are:
    #   A_L * B_L (j in {0,1}, rows in B_L are 0,1).
    #   A_R * B_R (j=2, row in B_R is 2).
    
    # So the matrix multiplication tensor is a direct sum of (2x2x9) and (2x1x9)?
    # No, the indices are coupled.
    # C_{ik} = a_{i0}b_{0k} + a_{i1}b_{1k} + a_{i2}b_{2k}.
    # The first two terms involve a_{i0}, a_{i1} and b_{0k}, b_{1k}.
    # The third term involves a_{i2} and b_{2k}.
    # These two parts are independent because the sets of A entries and B entries are disjoint.
    # So Rank(2x3x9) = Rank(2x2x9) + Rank(2x1x9).
    
    # Rank(2x1x9) = 2*1*9 = 18. (Schoolbook: a_{i2} * b_{2k} for each i,k).
    # Rank(2x2x9) = ?
    #   Split k into 4 groups of 2: {0,1}, {2,3}, {4,5}, {6,7} and one group of 1: {8}.
    #   For k in {0..7}, we can use Strassen.
    #   2x2x8 = 4 * (2x2x2). Rank 4*7 = 28.
    #   For k=8, it's 2x2x1. Rank 4.
    #   Total 32.
    
    # So total rank = 32 + 18 = 50.
    
    # Can we do better for 2x2x9?
    #   Maybe use a better algorithm for 2x2x9?
    #   Or maybe my assumption about independence is wrong?
    #   No, the tensor is a direct sum.
    #   
    #   Let's check Rank(2x2x2). It is 7.
    #   Rank(2x2x1) is 4.
    #   
    #   Is there a lower rank for 2x2x9?
    #   Maybe we can combine the last column k=8 with the previous ones?
    #   2x2x9 is equivalent to computing 2 vectors of length 9.
    #   
    #   Let's implement rank 50.
    
    # --- Part A: 2x2x9 (rank 32) ---
    # Terms for k in {0..7} (4 blocks of 2):
    for g in range(4):
        k0 = 2 * g
        k1 = 2 * g + 1
        # Strassen for (2x2) * (2x2)
        # A = [[a00, a01], [a10, a11]]
        # B = [[b_{0,k0}, b_{0,k1}], [b_{1,k0}, b_{1,k1}]]
        # M1 = (a00+a11) * (b_{0,k0}+b_{1,k1})
        # M2 = (a10+a11) * b_{0,k0}
        # M3 = a00 * (b_{1,k1}-b_{0,k1})
        # M4 = a11 * (b_{0,k0}-b_{1,k0})
        # M5 = (a00+a01) * b_{1,k1}
        # M6 = (a10-a00) * (b_{0,k0}+b_{0,k1})
        # M7 = (a01-a11) * (b_{1,k0}+b_{1,k1})
        
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # U entries (indices i*m+j). i in {0,1}, j in {0,1}.
        # V entries (indices j*p+k). j in {0,1}, k in {k0, k1}.
        # W entries (indices k*n+i). k in {k0, k1}, i in {0,1}.
        
        def add_term(U_entries, V_entries, W_entries, c_idx):
            for (u_idx, u_val) in U_entries:
                U[u_idx][c_idx] += u_val
            for (v_idx, v_val) in V_entries:
                V[v_idx][c_idx] += v_val
            for (w_idx, w_val) in W_entries:
                W[w_idx][c_idx] += w_val
        
        # M1
        add_term(
            [(0*m+0, 1), (1*m+1, 1)],  # a00, a11
            [(0*p+k0, 1), (0*p+k1, 1), (1*p+k0, 1), (1*p+k1, 1)], # b00, b01, b10, b11
            [(k0*n+0, 1), (k0*n+1, 1), (k1*n+0, 1), (k1*n+1, 1)], # c00, c01, c10, c11
            c
        )
        c += 1
        
        # M2
        add_term(
            [(1*m+0, 1), (1*m+1, 1)], # a10, a11
            [(0*p+k0, 1)], # b00
            [(k0*n+0, 1), (k0*n+1, 1)], # c10, c11? No.
            # C10 = M2+M4. C11 = M1-M2+M3+M6.
            # So M2 contributes to c10 and c11?
            # Wait.
            # C = [[C00, C01], [C10, C11]]
            # C10 = M2 + M4.
            # C11 = M1 - M2 + M3 + M6.
            # So M2 adds to C10 (coeff 1) and C11 (coeff -1).
            # Let's redo W entries.
            # W[k*n+i, c].
            # C10 is i=1, k=k0. Index k0*n+1.
            # C11 is i=1, k=k1. Index k1*n+1.
            [(k0*n+1, 1), (k1*n+1, -1)],
            c
        )
        c += 1
        
        # M3
        add_term(
            [(0*m+0, 1)], # a00
            [(1*p+k1, 1), (0*p+k1, -1)], # b11 - b01
            [(k1*n+0, 1)], # C01 = M3+M5. M3 adds to C01.
            # Wait, C01 = M3 + M5.
            # C11 = ... + M3 ...
            # So M3 adds to C01 (coeff 1) and C11 (coeff 1).
            [(k1*n+0, 1), (k1*n+1, 1)],
            c
        )
        c += 1
        
        # M4
        add_term(
            [(1*m+1, 1)], # a11
            [(0*p+k0, 1), (1*p+k0, -1)], # b00 - b10
            [(k0*n+0, -1), (k0*n+1, 1)], # C00 = M1-M4+... (coeff -1). C10 = M2+M4 (coeff 1).
            c
        )
        c += 1
        
        # M5
        add_term(
            [(0*m+0, 1), (0*m+1, 1)], # a00, a01
            [(1*p+k1, 1)], # b11
            [(k0*n+0, 1), (k1*n+0, 1)], # C00 = ...+M5... (coeff 1). C01 = M3+M5 (coeff 1).
            c
        )
        c += 1
        
        # M6
        add_term(
            [(1*m+0, 1), (0*m+0, -1)], # a10 - a00
            [(0*p+k0, 1), (0*p+k1, 1)], # b00 + b01
            [(k1*n+1, 1)], # C11 = ...+M6... (coeff 1).
            c
        )
        c += 1
        
        # M7
        add_term(
            [(0*m+1, 1), (1*m+1, -1)], # a01 - a11
            [(1*p+k0, 1), (1*p+k1, 1)], # b10 + b11
            [(k0*n+0, -1)], # C00 = M1-M4+M5-M7 (coeff -1).
            c
        )
        c += 1

    # Terms for k=8 (2x2x1, rank 4)
    k8 = 8
    # C_{0,8} = a00*b08 + a01*b18
    # C_{1,8} = a10*b08 + a11*b18
    # Term 1: a00 * b08 -> C08
    add_term([(0*m+0, 1)], [(0*p+k8, 1)], [(k8*n+0, 1)], c)
    c += 1
    # Term 2: a01 * b18 -> C08
    add_term([(0*m+1, 1)], [(1*p+k8, 1)], [(k8*n+0, 1)], c)
    c += 1
    # Term 3: a10 * b08 -> C18
    add_term([(1*m+0, 1)], [(0*p+k8, 1)], [(k8*n+1, 1)], c)
    c += 1
    # Term 4: a11 * b18 -> C18
    add_term([(1*m+1, 1)], [(1*p+k8, 1)], [(k8*n+1, 1)], c)
    c += 1

    # --- Part B: 2x1x9 (rank 18) ---
    # C_{i,k} += a_{i2} * b_{2k}
    for i in range(2):
        for k in range(9):
            add_term(
                [(i*m+2, 1)], # a_{i2}
                [(2*p+k, 1)], # b_{2k}
                [(k*n+i, 1)], # C_{ik}
                c
            )
            c += 1

    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    try:
        U, V, W = _make_2x3x9_rank_51()
        if time.time() > deadline:
            raise TimeoutError()
        return {"U": U, "V": V, "W": W}
    except Exception:
        # Fallback to schoolbook if something goes wrong
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
# EVOLVE-BLOCK-END
