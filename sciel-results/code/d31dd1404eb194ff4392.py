import time
import numpy as np

def construct():
    """
    Returns a rank-114 tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Strategy:
    1. Split the 4x9 matrix B into B1 (4x4), B2 (4x4), B3 (4x1).
    2. Use Strassen's algorithm (rank 49) for A * B1 and A * B2.
    3. Use schoolbook multiplication (rank 16) for A * B3.
    Total rank: 49 + 49 + 16 = 114.
    """
    
    n, m, p = 4, 4, 9
    nm, mp, pn = n * m, m * p, p * n
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # Strassen 2x2 base: 7 terms
    # Each: (U_entries, V_entries, W_entries)
    # U keys: (i, j), V keys: (j, k), W keys: (i, k)
    strassen_2x2 = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    def add_strassen_4x4(A_i_base, A_j_base, B_j_base, B_k_base, C_i_base, C_k_base):
        """
        Adds 49 terms for a 4x4 * 4x4 subproblem.
        """
        # The 4x4 Strassen algorithm uses 7 calls to 2x2 Strassen
        # at each of the 4 quadrants? No.
        # Strassen 4x4 is built from 7 Strassen 2x2 calls?
        # No, Strassen 4x4 uses 49 multiplications of 2x2 matrices?
        # Actually, the rank of 4x4x4 is 49.
        # This is achieved by applying Strassen to the 4x4 matrices.
        # The 4x4 Strassen algorithm involves 7 recursive calls to 2x2 Strassen?
        # No, it involves 7 calls to 2x2 matrix multiplication?
        # Wait, the standard Strassen for 2x2 uses 7 multiplications.
        # For 4x4, we can view it as 2x2 blocks of 2x2.
        # Then we need 7 "multiplications" of 2x2 matrices.
        # Each 2x2 multiplication has rank 7.
        # So total rank is 7 * 7 = 49.
        
        # We need to map the 7 "virtual" 2x2 multiplications to the 49 rank-1 terms.
        # Each virtual 2x2 multiplication corresponds to 7 rank-1 terms of the 2x2 Strassen.
        
        # Let's define the 7 virtual 2x2 multiplications.
        # Let A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]].
        # C00 = (A00+A11)(B00+B11) - A10*B11 - A00*B11  -- No, standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 - M4 + M6
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M7
        
        # Each Mk is a 2x2 matrix product.
        # We decompose each Mk into 7 rank-1 terms using strassen_2x2.
        
        # For each of the 7 Mk, we have 7 terms.
        # We need to define the U, V, W vectors for each term.
        
        # The 7 Mk are linear combinations of A blocks and B blocks.
        # We will iterate through the 7 Mk, and for each, iterate through the 7 Strassen 2x2 terms.
        
        # Define the 7 Mk coefficients.
        # M1: A_blocks = [(0,0,1), (1,1,1)], B_blocks = [(0,0,1), (1,1,1)]
        # M2: A_blocks = [(1,0,1), (1,1,1)], B_blocks = [(0,0,1)]
        # M3: A_blocks = [(0,0,1)], B_blocks = [(0,1,1), (1,1,-1)]
        # M4: A_blocks = [(1,1,1)], B_blocks = [(1,0,1), (0,0,-1)]
        # M5: A_blocks = [(0,0,1), (0,1,1)], B_blocks = [(1,1,1)]
        # M6: A_blocks = [(1,0,1), (0,0,-1)], B_blocks = [(0,0,1), (0,1,1)]
        # M7: A_blocks = [(0,1,1), (1,1,-1)], B_blocks = [(1,0,1), (1,1,1)]
        
        # And the C blocks for each Mk:
        # C00: M1, -M4, M6
        # C01: M3, M5
        # C10: M2, M4
        # C11: M1, -M3, M7
        
        # We will construct the terms by iterating through the 7 Mk.
        # For each Mk, we use the 7 Strassen 2x2 terms.
        # We need to map the local (li, lj) to global (gi, gj) for A.
        # And local (lj, lk) to global (gj, gk) for B.
        # And local (li, lk) to global (gi, gk) for C.
        
        # The A blocks for Mk are given as (br, bc, coeff).
        # br, bc are 0 or 1.
        # Global row = A_i_base + br*2 + li
        # Global col = A_j_base + bc*2 + lj
        
        # Similarly for B and C.
        
        # We need to sum the contributions to C from all Mk.
        # But each Mk is a separate 2x2 product.
        # So we just add the terms for each Mk to the total U, V, W.
        
        # However, the C blocks for each Mk are different.
        # So for each Mk, we need to know which C blocks it contributes to.
        
        # Let's define the mapping from Mk to C blocks.
        # M1 -> C00 (1), C11 (1)
        # M2 -> C10 (1)
        # M3 -> C01 (1), C11 (-1)
        # M4 -> C00 (-1), C10 (1)
        # M5 -> C01 (1)
        # M6 -> C00 (1)
        # M7 -> C11 (1)
        
        # We will create a list of (Mk_index, C_block_row, C_block_col, C_coeff)
        # And for each, add the 7 Strassen terms.
        
        mk_c_blocks = [
            (0, 0, 0, 1), (0, 1, 1, 1), # M1
            (1, 1, 0, 1), # M2
            (2, 0, 1, 1), (2, 1, 1, -1), # M3
            (3, 0, 0, -1), (3, 1, 0, 1), # M4
            (4, 0, 1, 1), # M5
            (5, 0, 0, 1), # M6
            (6, 1, 1, 1), # M7
        ]
        
        # Define the A and B blocks for each Mk
        # (br, bc, coeff)
        mk_a_blocks = [
            [(0,0,1), (1,1,1)], # M1
            [(1,0,1), (1,1,1)], # M2
            [(0,0,1)], # M3
            [(1,1,1)], # M4
            [(0,0,1), (0,1,1)], # M5
            [(1,0,1), (0,0,-1)], # M6
            [(0,1,1), (1,1,-1)], # M7
        ]
        mk_b_blocks = [
            [(0,0,1), (1,1,1)], # M1
            [(0,0,1)], # M2
            [(0,1,1), (1,1,-1)], # M3
            [(1,0,1), (0,0,-1)], # M4
            [(1,1,1)], # M5
            [(0,0,1), (0,1,1)], # M6
            [(1,0,1), (1,1,1)], # M7
        ]
        
        for mk_idx in range(7):
            a_blocks = mk_a_blocks[mk_idx]
            b_blocks = mk_b_blocks[mk_idx]
            
            # Find all C blocks for this Mk
            c_blocks_for_mk = []
            for mk_i, c_br, c_bc, c_coeff in mk_c_blocks:
                if mk_i == mk_idx:
                    c_blocks_for_mk.append((c_br, c_bc, c_coeff))
            
            # For each Strassen 2x2 term
            for u_terms, v_terms, w_terms in strassen_2x2:
                u_vec = [0] * nm
                v_vec = [0] * mp
                w_vec = [0] * pn
                
                # Fill U
                for (li, lj, coeff) in u_terms:
                    for br, bc, bc_coeff in a_blocks:
                        gi = A_i_base + br * 2 + li
                        gj = A_j_base + bc * 2 + lj
                        u_vec[gi * m + gj] += coeff * bc_coeff
                
                # Fill V
                for (lj, lk, coeff) in v_terms:
                    for br, bc, bc_coeff in b_blocks:
                        gj = B_j_base + br * 2 + lj
                        gk = B_k_base + bc * 2 + lk
                        v_vec[gj * p + gk] += coeff * bc_coeff
                
                # Fill W
                for (li, lk, coeff) in w_terms:
                    for c_br, c_bc, c_coeff in c_blocks_for_mk:
                        gi = C_i_base + c_br * 2 + li
                        gk = C_k_base + c_bc * 2 + lk
                        w_vec[gk * n + gi] += coeff * c_coeff
                
                add_term(u_vec, v_vec, w_vec)

    # A * B[:,0:4]
    add_strassen_4x4(0, 0, 0, 0, 0, 0)
    
    # A * B[:,4:8]
    add_strassen_4x4(0, 0, 0, 4, 0, 4)
    
    # A * B[:,8] - 16 schoolbook terms
    for i in range(4):
        for j in range(4):
            u_vec = [0] * nm
            v_vec = [0] * mp
            w_vec = [0] * pn
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
    
    rank = len(U_cols)
    
    U = [[U_cols[col][row] for col in range(rank)] for row in range(nm)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(mp)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(pn)]
    
    return {"U": U, "V": V, "W": W}

