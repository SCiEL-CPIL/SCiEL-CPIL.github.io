"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose into 5 blocks of p-size 2.
    # Each block uses Strassen (rank 7) for the first 2 columns of A and first 2 rows of B.
    # And schoolbook (rank 4) for the 3rd column of A and 3rd row of B.
    # Total rank per block = 11. Total rank = 55.
    
    total_rank = 55
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    col = 0
    
    # Strassen terms definition for 2x2x2
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B10-B11)
    # M4: A11(B01-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # Coefficients for C
    # C00 = M1 - M4 + M5 + M7
    # C01 = M4 + M6
    # C10 = M2 + M3
    # C11 = M1 - M2 + M3 + M6
    
    # We will add these 7 terms for each block.
    # A_sub indices in U:
    # A00: 0*3+0 = 0
    # A01: 0*3+1 = 1
    # A10: 1*3+0 = 3
    # A11: 1*3+1 = 4
    
    # For block t (0..4):
    # B_sub rows 0,1, cols 2t, 2t+1 in V:
    # B00: 0*10 + 2t
    # B01: 0*10 + 2t + 1
    # B10: 1*10 + 2t
    # B11: 1*10 + 2t + 1
    
    # W indices for C00, C01, C10, C11 of block t:
    # C00: k=2t, i=0 -> 2t*2 + 0
    # C01: k=2t+1, i=0 -> (2t+1)*2 + 0
    # C10: k=2t, i=1 -> 2t*2 + 1
    # C11: k=2t+1, i=1 -> (2t+1)*2 + 1
    
    strassen_coeffs = [
        # term, u_indices, v_indices, w_indices
        # M1
        ([(0, 1), (4, 1)], [(0, 1), (1, 1)], [(0, 1), (1, 1), (2, -1), (3, -1)]),
        # M2
        ([(3, 1), (4, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3
        ([(0, 1)], [(1, 1), (1, -1)], [(2, 1), (3, 1)]),
        # M4
        ([(4, 1)], [(0, 1), (0, -1)], [(1, 1), (3, -1)]),
        # M5
        ([(0, 1), (1, 1)], [(1, 1)], [(0, 1), (1, 1)]),
        # M6
        ([(3, 1), (0, -1)], [(0, 1), (0, 1)], [(1, 1), (3, 1)]),
        # M7
        ([(1, 1), (4, -1)], [(1, 1), (1, 1)], [(0, 1), (3, 1)]),
    ]
    
    # Wait, the W indices in the tuple above are relative to the 2x2 block C.
    # Block C has indices (0,0), (0,1), (1,0), (1,1).
    # Global W index = k_global * n + i_global.
    
    for t in range(5):
        # Strassen Part
        for u_list, v_list, w_list in strassen_coeffs:
            for (u_idx, u_val) in u_list:
                U[u_idx][col] = u_val
            for (v_row, v_val) in v_list:
                # v_row is 0 or 1 relative to B_sub
                # v_col is 2t or 2t+1 relative to B_sub
                # We need to map the v_list indices properly.
                # My v_list above was just values. I need indices.
                pass # This structure is too messy to fill manually in loop.
                
        # Let's rewrite the Strassen addition explicitly for clarity and correctness.
        
        # M1
        U[0][col] += 1; U[4][col] += 1
        V[0*p + 2*t][col] += 1; V[1*p + 2*t+1][col] += 1 # Wrong, B00 and B11?
        # Strassen M1 is (A00+A11)(B00+B11).
        # B00 is B[0, 2t]. B11 is B[1, 2t+1].
        # So V indices: 0*p + 2*t and 1*p + 2*t+1.
        # Wait, B11 is B[1, 1] in sub. So row 1, col 2t+1.
        # Yes.
        V[0*p + 2*t][col] = 1
        V[1*p + 2*t+1][col] = 1
        # W: C00, C11 get +1, C01, C10 get -1?
        # C00 = M1 ...
        # C11 = M1 ...
        # C01, C10 do not have M1 directly in standard form?
        # Standard:
        # C00 = M1 - M4 + M5 + M7
        # C01 = M4 + M6
        # C10 = M2 + M3
        # C11 = M1 - M2 + M3 + M6
        # So M1 goes to C00 (+1) and C11 (+1).
        W[2*t*n + 0][col] = 1
        W[(2*t+1)*n + 1][col] = 1
        col += 1
        
        # M2: (A10+A11)B00
        U[3][col] = 1; U[4][col] = 1
        V[0*p + 2*t][col] = 1
        W[2*t*n + 1][col] = 1 # C10
        W[(2*t+1)*n + 1][col] = -1 # C11
        col += 1
        
        # M3: A00(B10-B11)
        U[0][col] = 1
        V[1*p + 2*t][col] = 1
        V[1*p + 2*t+1][col] = -1
        W[2*t*n + 1][col] = 1 # C10
        W[(2*t+1)*n + 1][col] = 1 # C11
        col += 1
        
        # M4: A11(B01-B00)
        U[4][col] = 1
        V[0*p + 2*t+1][col] = 1
        V[0*p + 2*t][col] = -1
        W[2*t*n + 0][col] = -1 # C00
        W[2*t*n + 1][col] = 0 # C10 (no)
        # C01 = M4 + M6. So M4 goes to C01 (+1).
        W[(2*t+1)*n + 0][col] = 1 # C01
        col += 1
        
        # M5: (A00+A01)B11
        U[0][col] = 1; U[1][col] = 1
        V[1*p + 2*t+1][col] = 1
        W[2*t*n + 0][col] = 1 # C00
        W[(2*t+1)*n + 0][col] = 1 # C01
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[3][col] = 1; U[0][col] = -1
        V[0*p + 2*t][col] = 1
        V[0*p + 2*t+1][col] = 1
        W[(2*t+1)*n + 0][col] = 1 # C01
        W[(2*t+1)*n + 1][col] = 1 # C11
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[1][col] = 1; U[4][col] = -1
        V[1*p + 2*t][col] = 1
        V[1*p + 2*t+1][col] = 1
        W[2*t*n + 0][col] = 1 # C00
        W[(2*t+1)*n + 1][col] = 1 # C11
        col += 1
        
        # Schoolbook Part for A[:, 2] and B[2, :]
        # A02: idx 0*3+2 = 2
        # A12: idx 1*3+2 = 5
        # B20: idx 2*p + 2*t
        # B21: idx 2*p + 2*t+1
        
        # Term: A02 * B20 -> C00 (global C0, 2t)
        U[2][col] = 1
        V[2*p + 2*t][col] = 1
        W[2*t*n + 0][col] = 1
        col += 1
        
        # Term: A02 * B21 -> C01 (global C0, 2t+1)
        U[2][col] = 1
        V[2*p + 2*t+1][col] = 1
        W[(2*t+1)*n + 0][col] = 1
        col += 1
        
        # Term: A12 * B20 -> C10 (global C1, 2t)
        U[5][col] = 1
        V[2*p + 2*t][col] = 1
        W[2*t*n + 1][col] = 1
        col += 1
        
        # Term: A12 * B21 -> C11 (global C1, 2t+1)
        U[5][col] = 1
        V[2*p + 2*t+1][col] = 1
        W[(2*t+1)*n + 1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
