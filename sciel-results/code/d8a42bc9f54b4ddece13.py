"""Matrix multiplication 4x4x5 rank-minimal tensor decomposition."""
import numpy as np
import time

def construct():
    t0 = time.time()
    limit = 140.0
    
    # Dimensions
    n, m, p = 4, 4, 5
    # Target rank: 49 (double-Strassen for 4x4x4) + 16 (schoolbook for 4x4x1) = 65
    total_rank = 65
    
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col = 0
    
    # Strassen 2x2x2: 7 bilinear forms
    # Each form: (P, Q, C_map) where P is 2x2 linear comb of A-blocks,
    # Q is 2x2 linear comb of B-blocks, C_map gives which 2x2 blocks of C get contributions
    def strassen_2x2():
        """Return 7 Strassen bilinear forms for 2x2x2 matrix multiplication.
        Each form is (a_mask, b_mask, c_coeffs) where:
        - a_mask: list of (row, col, coeff) in the 2x2 A input
        - b_mask: list of (row, col, coeff) in the 2x2 B input  
        - c_coeffs: list of (row, col, coeff) in the 2x2 C output
        """
        forms = []
        # M0 = (A00+A11)*(B00+B11) -> C00+=M0, C11+=M0
        forms.append(
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)])
        )
        # M1 = (A10+A11)*B00 -> C10+=M1, C11-=M1
        forms.append(
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)])
        )
        # M2 = A00*(B01-B11) -> C01+=M2, C11+=M2
        forms.append(
            ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)])
        )
        # M3 = A11*(B10-B00) -> C00+=M3, C10+=M3
        forms.append(
            ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)])
        )
        # M4 = (A00+A01)*B11 -> C00-=M4, C01+=M4
        forms.append(
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)])
        )
        # M5 = (A10-A00)*(B00+B01) -> C11+=M5
        forms.append(
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)])
        )
        # M6 = (A01-A11)*(B10+B11) -> C00+=M6
        forms.append(
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)])
        )
        return forms
    
    s2x2 = strassen_2x2()
    
    # For 4x4x4: partition into 2x2 blocks
    # A blocks: AA(0,0), AB(0,1), BA(1,0), BB(1,1)
    # Each block is 2x2
    # Level 1: 7 Strassen combinations of 2x2 blocks
    # Level 2: within each, 7 Strassen combinations of 2x2 scalars
    
    # Define 4x4 block indices: block (br, bc) covers rows 2*br to 2*br+1, cols 2*bc to 2*bc+1
    def block_row(br, r):
        return 2 * br + r
    
    def block_col(bc, c):
        return 2 * bc + c
    
    # Level 1 Strassen for 4x4 (operating on 2x2 blocks)
    # p_blocks: list of (block_row, block_col, coeff)
    # q_blocks: list of (block_row, block_col, coeff)  
    # c_blocks: list of (block_row, block_col, coeff)
    l1_forms = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    # Generate 49 terms for 4x4x4 part
    for l1_idx in range(7):
        if time.time() - t0 > limit:
            break
        p_blks, q_blks, c_blks = l1_forms[l1_idx]
        for l2_idx in range(7):
            if time.time() - t0 > limit:
                break
            a_mask, b_mask, c_mask = s2x2[l2_idx]
            
            # Build U contributions: (global_row, global_col, coeff)
            for (abr, abc, ac) in p_blks:
                for (ar, ac2, acoeff) in a_mask:
                    gr = block_row(abr, ar)
                    gc = block_col(abc, ac2)
                    U[gr * m + gc, col] += ac * acoeff
            
            # Build V contributions: (global_row, global_col, coeff)
            for (qbr, qbc, qc) in q_blks:
                for (qr, qc2, qcoeff) in b_mask:
                    gr = block_row(qbr, qr)
                    gc = block_col(qbc, qc2)
                    V[gr * p + gc, col] += qc * qcoeff
            
            # Build W contributions: (global_col_of_C, global_row_of_C, coeff)
            # W index: k*n + i where k is col of C, i is row of C
            for (cbr, cbc, cc) in c_blks:
                for (cr, cc2, wcoeff) in c_mask:
                    gi = block_row(cbr, cr)
                    gj = block_col(cbc, cc2)
                    W[gj * n + gi, col] += cc * wcoeff
            
            col += 1
    
    # Part 2: 4x4 * 4x1 (column 4 of B) using schoolbook: rank 16
    for i in range(n):
        if time.time() - t0 > limit:
            break
        for j in range(m):
            U[i * m + j, col] = 1.0
            V[j * p + 4, col] = 1.0
            W[4 * n + i, col] = 1.0
            col += 1
    
    assert col == total_rank, f"Rank mismatch: {col} != {total_rank}"
    
    # Convert to lists for the expected interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
