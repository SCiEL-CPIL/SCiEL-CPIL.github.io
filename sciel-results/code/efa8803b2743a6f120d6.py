import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Hybrid decomposition:
    # Split inner dimension m=4 into two blocks of size 2.
    # For each block, use Strassen for first 2 cols of p=5, schoolbook for remaining 3.
    # Rank per block: 7 (Strassen) + 2*2*3 (Schoolbook) = 19.
    # Total rank: 19 + 19 = 38.
    
    rank = 38
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Strassen coefficients for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # We need to map local i,j,k to global.
    # Block 1: i in {0,1}, j in {0,1}
    # Block 2: i in {0,1}, j in {2,3}
    
    for block_offset in [0, 2]:
        j_offset = block_offset
        # Strassen part for k in {0, 1}
        # Local k 0 -> Global k 0
        # Local k 1 -> Global k 1
        
        # P1: (a+d)(e+h)
        # U: a=1, d=1. V: e=1, h=1. W: C11+=1, C22+=1.
        U[i*4+j][col_idx] = 0 # placeholder, need to fill correctly
        
        # It is easier to generate the terms explicitly.
        
        # Helper to add a term
        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col_idx
            for (ii, jj), val in u_coeffs.items():
                U[ii*4+jj][col_idx] += val
            for (jj, kk), val in v_coeffs.items():
                V[jj*5+kk][col_idx] += val
            for (kk, ii), val in w_coeffs.items():
                W[kk*2+ii][col_idx] += val
            col_idx += 1
        
        # Strassen Terms for Block (i_offset=0, j_offset=j_offset)
        # Note: i is always 0,1 in both blocks.
        
        # Term 1: P1 = (A00+A11) * (B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1
        add_term(
            {(0+j_offset//2*0, 0+j_offset): 1, (1+j_offset//2*0, 1+j_offset): 1}, # A indices
            # Wait, j_offset is 0 or 2.
            # A is 2x4. Rows 0,1. Cols 0,1 for block 1. Cols 2,3 for block 2.
            # So for block 0: j=0,1. For block 1: j=2,3.
            # A indices: (0, j_offset), (1, j_offset+1).
            {(0, j_offset): 1, (1, j_offset+1): 1},
            {(j_offset, 0): 1, (j_offset+1, 1): 1},
            {(0, 0): 1, (1, 1): 1}
        )
        
        # Term 2: P2 = (A10+A11) * B00
        add_term(
            {(1, j_offset): 1, (1, j_offset+1): 1},
            {(j_offset, 0): 1},
            {(0, 1): 1}
        )
        
        # Term 3: P3 = A00 * (B01-B11)
        add_term(
            {(0, j_offset): 1},
            {(j_offset, 1): 1, (j_offset+1, 1): -1},
            {(1, 0): 1}
        )
        
        # Term 4: P4 = A11 * (B10-B00)
        add_term(
            {(1, j_offset+1): 1},
            {(j_offset+1, 0): 1, (j_offset, 0): -1},
            {(0, 1): 1, (1, 1): 1}
        )
        
        # Term 5: P5 = (A00+A01) * B11
        add_term(
            {(0, j_offset): 1, (0, j_offset+1): 1},
            {(j_offset+1, 1): 1},
            {(0, 1): -1, (1, 1): 1}
        )
        
        # Term 6: P6 = (A10-A00) * (B00+B01)
        add_term(
            {(1, j_offset): 1, (0, j_offset): -1},
            {(j_offset, 0): 1, (j_offset, 1): 1},
            {(1, 1): 1}
        )
        
        # Term 7: P7 = (A01-A11) * (B10+B11)
        add_term(
            {(0, j_offset+1): 1, (1, j_offset+1): -1},
            {(j_offset+1, 0): 1, (j_offset+1, 1): 1},
            {(0, 1): -1, (0, 0): 1}
        )
        
        # Schoolbook part for k in {2, 3, 4}
        for k in [2, 3, 4]:
            for i_local in [0, 1]:
                for j_local in [0, 1]:
                    j_global = j_offset + j_local
                    i_global = i_local
                    add_term(
                        {(i_global, j_global): 1},
                        {(j_global, k): 1},
                        {(k, i_global): 1}
                    )
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
