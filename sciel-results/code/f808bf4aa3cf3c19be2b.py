import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # We use a block decomposition of the inner dimension m=5 into 2+1+2.
    # Part 1: A[:, 0:2] (2x2) * B[0:2, :] (2x6) -> 2x2x6 (Rank 21)
    # Part 2: A[:, 2:3] (2x1) * B[2:3, :] (1x6) -> 2x1x6 (Rank 12)
    # Part 3: A[:, 3:5] (2x2) * B[3:5, :] (2x6) -> 2x2x6 (Rank 21)
    # Total Rank: 21 + 12 + 21 = 54
    
    total_rank = 54
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    def add_strassen_222(offset, a_start, b_start, c_start):
        # Adds a 2x2x2 Strassen decomposition starting at factor 'offset'
        # a_start: row index in A (0 or 1) -> actually A is 2x5, so rows 0,1. Cols are 0-4.
        # The U index is i*m + j. i in 0..1, j in 0..4.
        # a_start is the column index in A for the 2x2 block (j=0,1 or j=3,4)
        # b_start is the row index in B for the 2x2 block (k=0,1 or k=3,4)
        # c_start is the column index in C for the 2x2 block (l=0,1, etc)
        
        # A indices:
        # a00 = i=0, j=a_start
        # a01 = i=0, j=a_start+1
        # a10 = i=1, j=a_start
        # a11 = i=1, j=a_start+1
        u_a00 = 0 * m + a_start
        u_a01 = 0 * m + (a_start + 1)
        u_a10 = 1 * m + a_start
        u_a11 = 1 * m + (a_start + 1)
        
        # B indices:
        # b00 = k=b_start, l=c_start
        # b01 = k=b_start, l=c_start+1
        # b10 = k=b_start+1, l=c_start
        # b11 = k=b_start+1, l=c_start+1
        v_b00 = b_start * p + c_start
        v_b01 = b_start * p + (c_start + 1)
        v_b10 = (b_start + 1) * p + c_start
        v_b11 = (b_start + 1) * p + (c_start + 1)
        
        # W indices:
        # c00 = i=0, l=c_start -> k=c_start, i=0 -> c_start*n + 0
        # c01 = i=0, l=c_start+1 -> c_start+1, i=0 -> (c_start+1)*n + 0
        # c10 = i=1, l=c_start -> c_start, i=1 -> c_start*n + 1
        # c11 = i=1, l=c_start+1 -> c_start+1, i=1 -> (c_start+1)*n + 1
        w_c00 = c_start * n + 0
        w_c01 = (c_start + 1) * n + 0
        w_c10 = c_start * n + 1
        w_c11 = (c_start + 1) * n + 1
        
        # Strassen Factors
        # M1 = (a00+a11)(b00+b11)
        U[u_a00][offset] = 1
        U[u_a11][offset] = 1
        V[v_b00][offset] = 1
        V[v_b11][offset] = 1
        W[w_c00][offset] = 1
        W[w_c01][offset] = 0
        W[w_c10][offset] = 0
        W[w_c11][offset] = 1
        # c00 += M1, c11 += M1
        
        # M2 = (a10+a11)b00
        U[u_a10][offset+1] = 1
        U[u_a11][offset+1] = 1
        V[v_b00][offset+1] = 1
        W[w_c10][offset+1] = 1
        W[w_c11][offset+1] = -1
        # c10 += M2, c11 -= M2

        # M3 = a00(b10-b11)
        U[u_a00][offset+2] = 1
        V[v_b10][offset+2] = 1
        V[v_b11][offset+2] = -1
        W[w_c01][offset+2] = 1
        W[w_c11][offset+2] = 1
        # c01 += M3, c11 += M3

        # M4 = a11(b00-b01)
        U[u_a11][offset+3] = 1
        V[v_b00][offset+3] = 1
        V[v_b01][offset+3] = -1
        W[w_c00][offset+3] = 1
        W[w_c10][offset+3] = 1
        # c00 += M4, c10 += M4

        # M5 = (a00+a01)b11
        U[u_a00][offset+4] = 1
        U[u_a01][offset+4] = 1
        V[v_b11][offset+4] = 1
        W[w_c00][offset+4] = -1
        W[w_c01][offset+4] = 1
        # c00 -= M5, c01 += M5

        # M6 = (a10-a00)(b00+b01)
        U[u_a10][offset+5] = 1
        U[u_a00][offset+5] = -1
        V[v_b00][offset+5] = 1
        V[v_b01][offset+5] = 1
        W[w_c11][offset+5] = 1
        # c11 += M6

        # M7 = (a01-a11)(b10+b11)
        U[u_a01][offset+6] = 1
        U[u_a11][offset+6] = -1
        V[v_b10][offset+6] = 1
        V[v_b11][offset+6] = 1
        W[w_c00][offset+6] = 1
        # c00 += M7
        
        # Wait, checking standard Strassen:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # My mappings:
        # M1: C00+, C11+
        # M2: C10+, C11-  -> C21, C22. (C21 is w_c10, C22 is w_c11)
        # M3: C01+, C11+  -> C12, C22.
        # M4: C00+, C10+  -> C11, C21.
        # M5: C00-, C01+  -> C11, C12.
        # M6: C11+        -> C22.
        # M7: C00+        -> C11.
        
        # Let's re-verify M7.
        # Standard: C11 = M1 + M4 - M5 + M7.
        # My M7 adds to C00. Correct.
        
        # Let's re-verify M6.
        # Standard: C22 = M1 - M2 + M3 + M6.
        # My M6 adds to C11 (which is C22). Correct.
        
        pass

    def add_outer_216(offset, a_col, b_row):
        # Adds 2x1x6 decomposition
        # A is 2x1 at col a_col. B is 1x6 at row b_row.
        # Rank 12.
        # c_i,l = a_i * b_l
        # For each l in 0..5:
        #   m1 = a0 * b_l
        #   m2 = a1 * b_l
        #   c0,l += m1
        #   c1,l += m2
        
        u_a0 = 0 * m + a_col
        u_a1 = 1 * m + a_col
        v_b_row = b_row * p
        
        for l in range(6):
            v_b = v_b_row + l
            w_c0 = l * n + 0
            w_c1 = l * n + 1
            
            U[u_a0][offset] = 1
            V[v_b][offset] = 1
            W[w_c0][offset] = 1
            offset += 1
            
            U[u_a1][offset] = 1
            V[v_b][offset] = 1
            W[w_c1][offset] = 1
            offset += 1

    current_offset = 0
    
    # Part 1: A[:, 0:2] * B[0:2, :]
    # B[0:2, :] is 2x6. We split into 3 blocks of 2x2.
    # Block 1: B[0:2, 0:2] -> C[:, 0:2]
    add_strassen_222(current_offset, a_start=0, b_start=0, c_start=0)
    current_offset += 7
    
    # Block 2: B[0:2, 2:4] -> C[:, 2:4]
    add_strassen_222(current_offset, a_start=0, b_start=0, c_start=2)
    current_offset += 7
    
    # Block 3: B[0:2, 4:6] -> C[:, 4:6]
    add_strassen_222(current_offset, a_start=0, b_start=0, c_start=4)
    current_offset += 7
    
    # Part 2: A[:, 2:3] * B[2:3, :]
    # A is 2x1 at col 2. B is 1x6 at row 2.
    add_outer_216(current_offset, a_col=2, b_row=2)
    current_offset += 12
    
    # Part 3: A[:, 3:5] * B[3:5, :]
    # B[3:5, :] is 2x6. Split into 3 blocks of 2x2.
    # Block 1: B[3:5, 0:2] -> C[:, 0:2]
    add_strassen_222(current_offset, a_start=3, b_start=3, c_start=0)
    current_offset += 7
    
    # Block 2: B[3:5, 2:4] -> C[:, 2:4]
    add_strassen_222(current_offset, a_start=3, b_start=3, c_start=2)
    current_offset += 7
    
    # Block 3: B[3:5, 4:6] -> C[:, 4:6]
    add_strassen_222(current_offset, a_start=3, b_start=3, c_start=4)
    current_offset += 7
    
    return {"U": U, "V": V, "W": W}
