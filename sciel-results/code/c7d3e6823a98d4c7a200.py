"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    # Strategy: Block decomposition
    # Split 4x5x6 into:
    # 1. 4x1x6 (Last column of A, Last row of B) -> Rank 24
    # 2. 4x4x6 (First 4 cols of A, First 4 rows of B) -> Rank 84
    #    Split 4x4x6 into four 2x2x6 blocks
    #    Split each 2x2x6 into three 2x2x2 Strassen blocks
    
    total_rank = 24 + 84
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # --- Part 1: 4x1x6 (Schoolbook) ---
    # A indices: i in 0..3, j=4
    # B indices: j=4, k in 0..5
    # W indices: k in 0..5, i in 0..3
    for i in range(4):
        for k in range(6):
            # U index: i * m + j = i * 5 + 4
            # V index: j * p + k = 4 * 6 + k = 24 + k
            # W index: k * n + i = k * 4 + i
            U[i * 5 + 4][col_offset] = 1
            V[24 + k][col_offset] = 1
            W[k * 4 + i][col_offset] = 1
            col_offset += 1
            
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # --- Part 2: 4x4x6 ---
    # A indices: i in 0..3, j in 0..3
    # B indices: j in 0..3, k in 0..5
    # W indices: k in 0..5, i in 0..3
    
    # We decompose this into 4 blocks of 2x2x6.
    # Block (ri, ci) where ri, ci in {0, 1}
    # ri determines which 2 rows of A (and thus W)
    # ci determines which 2 cols of A and which 2 rows of B
    
    # Strassen 2x2x2 helper
    # local_a: 2x2, local_b: 2x2
    # Returns list of (u_indices, v_indices, w_indices, coeff)
    # But we need to add to global U, V, W
    
    def add_strassen_2x2x2(a_row_base, a_col_base, b_row_base, b_col_base, w_row_base, w_col_base):
        """
        Adds a 2x2x2 Strassen decomposition.
        a_row_base: starting row index in A (global)
        a_col_base: starting col index in A (global)
        b_row_base: starting row index in B (global)
        b_col_base: starting col index in B (global)
        w_row_base: starting row index in W (global, which is k)
        w_col_base: starting col index in W (global, which is i)
        """
        nonlocal col_offset
        
        # Strassen coefficients for 2x2x2
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)b00
        # m3 = a00(b01-b11)
        # m4 = a11(b10-b00)
        # m5 = (a00+a01)b11
        # m6 = (a10-a00)(b00+b01)
        # m7 = (a01-a11)(b10+b11)
        
        # We will add 7 terms.
        
        # Term 1: (a00+a11) * (b00+b11)
        # U: a00, a11
        # V: b00, b11
        # W: c00, c11 (with signs)
        
        # c00 += m1
        # c11 += m1
        # c01 -= m1? No.
        # c00 = m1 + m4 - m5 + m7
        # c01 = m3 + m5
        # c10 = m2 + m4
        # c11 = m1 - m2 + m3 + m6
        
        # Let's just add the 7 multiplications and then accumulate into W.
        # But U and V are fixed per multiplication.
        
        # Mult 1: (a00+a11) * (b00+b11)
        U[a_row_base * 5 + a_col_base][col_offset] = 1
        U[(a_row_base+1) * 5 + (a_col_base+1)][col_offset] = 1
        V[b_row_base * 6 + b_col_base][col_offset] = 1
        V[(b_row_base+1) * 6 + (b_col_base+1)][col_offset] = 1
        # W contributions:
        # c00 (w_row_base, w_col_base) += 1
        # c11 (w_row_base+1, w_col_base+1) += 1
        W[w_row_base * 4 + w_col_base][col_offset] = 1
        W[(w_row_base+1) * 4 + (w_col_base+1)][col_offset] = 1
        col_offset += 1
        
        # Mult 2: (a10+a11) * b00
        U[(a_row_base+1) * 5 + a_col_base][col_offset] = 1
        U[(a_row_base+1) * 5 + (a_col_base+1)][col_offset] = 1
        V[b_row_base * 6 + b_col_base][col_offset] = 1
        # W: c10 += 1
        W[(w_row_base+1) * 4 + w_col_base][col_offset] = 1
        col_offset += 1
        
        # Mult 3: a00 * (b01-b11)
        U[a_row_base * 5 + a_col_base][col_offset] = 1
        V[b_row_base * 6 + (b_col_base+1)][col_offset] = 1
        V[(b_row_base+1) * 6 + (b_col_base+1)][col_offset] = -1
        # W: c01 += 1
        W[w_row_base * 4 + (w_col_base+1)][col_offset] = 1
        col_offset += 1
        
        # Mult 4: a11 * (b10-b00)
        U[(a_row_base+1) * 5 + (a_col_base+1)][col_offset] = 1
        V[(b_row_base+1) * 6 + b_col_base][col_offset] = 1
        V[b_row_base * 6 + b_col_base][col_offset] = -1
        # W: c00 += 1, c10 += 1
        W[w_row_base * 4 + w_col_base][col_offset] = 1
        W[(w_row_base+1) * 4 + w_col_base][col_offset] = 1
        col_offset += 1
        
        # Mult 5: (a00+a01) * b11
        U[a_row_base * 5 + a_col_base][col_offset] = 1
        U[a_row_base * 5 + (a_col_base+1)][col_offset] = 1
        V[(b_row_base+1) * 6 + (b_col_base+1)][col_offset] = 1
        # W: c00 -= 1, c01 += 1
        W[w_row_base * 4 + w_col_base][col_offset] = -1
        W[w_row_base * 4 + (w_col_base+1)][col_offset] = 1
        col_offset += 1
        
        # Mult 6: (a10-a00) * (b00+b01)
        U[(a_row_base+1) * 5 + a_col_base][col_offset] = 1
        U[a_row_base * 5 + a_col_base][col_offset] = -1
        V[b_row_base * 6 + b_col_base][col_offset] = 1
        V[b_row_base * 6 + (b_col_base+1)][col_offset] = 1
        # W: c11 += 1
        W[(w_row_base+1) * 4 + (w_col_base+1)][col_offset] = 1
        col_offset += 1
        
        # Mult 7: (a01-a11) * (b10+b11)
        U[a_row_base * 5 + (a_col_base+1)][col_offset] = 1
        U[(a_row_base+1) * 5 + (a_col_base+1)][col_offset] = -1
        V[(b_row_base+1) * 6 + b_col_base][col_offset] = 1
        V[(b_row_base+1) * 6 + (b_col_base+1)][col_offset] = 1
        # W: c00 += 1
        W[w_row_base * 4 + w_col_base][col_offset] = 1
        col_offset += 1

    # Now loop over the 4 blocks of 2x2x6
    for ri in range(2):
        for ci in range(2):
            # A block rows: 2*ri, 2*ri+1
            # A block cols: 2*ci, 2*ci+1
            # B block rows: 2*ci, 2*ci+1
            # B block cols: 0, 1, 2, 3, 4, 5 (all 6 cols)
            
            # W block rows: 0..5 (all 6 cols)
            # W block cols: 2*ri, 2*ri+1
            
            # We split B cols (0..5) into three 2-col blocks
            for bk in range(3):
                # B col base: 2*bk
                # W col base: 2*bk
                add_strassen_2x2x2(
                    a_row_base=2*ri,
                    a_col_base=2*ci,
                    b_row_base=2*ci,
                    b_col_base=2*bk,
                    w_row_base=2*bk,
                    w_col_base=2*ri
                )
                
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
