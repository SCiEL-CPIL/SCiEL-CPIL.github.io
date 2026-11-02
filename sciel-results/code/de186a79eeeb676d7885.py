import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Total rank will be 66
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        for i, val in enumerate(u_vec):
            if val != 0:
                U[i][col] += val
        for i, val in enumerate(v_vec):
            if val != 0:
                V[i][col] += val
        for i, val in enumerate(w_vec):
            if val != 0:
                W[i][col] += val
        col += 1

    # Strassen terms for 2x2 multiplication
    # A = [[a, b], [c, d]] -> indices 0,1,2,3
    # B = [[e, f], [g, h]] -> indices 0,1,2,3
    # C = [[c11, c12], [c21, c22]] -> indices 0,1,2,3
    
    # M1 = a * (h - f)
    # u: [1, 0, 0, 0]
    # v: [0, -1, 0, 1]  (coeff of e,f,g,h) -> e:0, f:-1, g:0, h:1
    # w: [1, 0, 0, 1]   (c11:1, c12:0, c21:0, c22:1) -> c11 += M1, c22 += M1
    # Check: c11 = M1 + M4 + M5 - M7?
    # Standard Strassen:
    # p1 = a(h-f)
    # p2 = (a+b)g
    # p3 = (c+d)h
    # p4 = d(e-g)
    # p5 = (a+c)(e+h)
    # p6 = (b-d)(f+h)
    # p7 = (c-a)(e+f)
    # c11 = p1 + p4 + p5 - p7
    # c12 = p2 + p5
    # c21 = p3 + p7
    # c22 = p1 + p3 - p2 + p6
    
    # Let's verify p1 = a(h-f).
    # u = [1,0,0,0]
    # v = [0,-1,0,1]
    # w = [1,0,0,1]
    
    # p2 = (a+b)g
    # u = [1,1,0,0]
    # v = [0,0,1,0]
    # w = [0,1,0,0]  (c12 += p2)
    
    # p3 = (c+d)h
    # u = [0,0,1,1]
    # v = [0,0,0,1]
    # w = [0,0,1,1]  (c21 += p3, c22 += p3)
    
    # p4 = d(e-g)
    # u = [0,0,0,1]
    # v = [1,0,-1,0]
    # w = [1,0,0,0]  (c11 += p4)
    
    # p5 = (a+c)(e+h)
    # u = [1,0,1,0]
    # v = [1,0,0,1]
    # w = [1,1,0,0]  (c11 += p5, c12 += p5)
    
    # p6 = (b-d)(f+h)
    # u = [0,1,0,-1]
    # v = [0,1,0,1]
    # w = [0,0,0,1]  (c22 += p6)
    
    # p7 = (c-a)(e+f)
    # u = [0,0,1,-1]
    # v = [1,1,0,0]
    # w = [0,0,1,-1] (c21 += p7, c11 -= p7) -> w = [-1, 0, 1, 0]?
    # c11 -= p7 -> w[0] = -1
    # c21 += p7 -> w[2] = 1
    # w = [-1, 0, 1, 0]

    strassen_terms = [
        ([1,0,0,0], [0,-1,0,1], [1,0,0,1]),
        ([1,1,0,0], [0,0,1,0], [0,1,0,0]),
        ([0,0,1,1], [0,0,0,1], [0,0,1,1]),
        ([0,0,0,1], [1,0,-1,0], [1,0,0,0]),
        ([1,0,1,0], [1,0,0,1], [1,1,0,0]),
        ([0,1,0,-1], [0,1,0,1], [0,0,0,1]),
        ([0,0,1,-1], [1,1,0,0], [-1,0,1,0])
    ]
    
    # We need to map these to global indices.
    # Global U (A) size: n*m = 12.
    # Global V (B) size: m*p = 24.
    # Global W (C) size: p*n = 18.
    
    # Decomposition strategy:
    # Split m=4 into m1=2 (cols 0,1) and m2=2 (cols 2,3).
    # For each m_block:
    #   Split n=3 into n1=2 (rows 0,1) and n2=1 (row 2).
    #   For n1=2:
    #     Use Strassen (7 terms) for each of the 3 blocks of p=6 (size 2).
    #     Total 21 terms.
    #   For n2=1:
    #     Use schoolbook (12 terms) for the 1x2x6 multiplication.
    #     Total 12 terms.
    #   Total per m_block = 33.
    #   Total = 66.

    # Helper to add a 2x2 Strassen block
    def add_strassen_2x2_block(a_row_offset, b_row_offset, c_row_offset, k_offset):
        nonlocal col
        # a_row_offset: row in A (0,1,2)
        # b_row_offset: row in B (0,1,2,3) -> corresponds to col in A
        # c_row_offset: row in C (0,1,2)
        # k_offset: col in B/C (0,1,2,3,4,5)
        
        # Local A indices (2x2):
        # A[0,0] -> global U[a_row_offset * m + b_row_offset]
        # A[0,1] -> global U[a_row_offset * m + b_row_offset + 1]
        # A[1,0] -> global U[(a_row_offset+1) * m + b_row_offset]
        # A[1,1] -> global U[(a_row_offset+1) * m + b_row_offset + 1]
        
        # Local B indices (2x2):
        # B[0,0] -> global V[b_row_offset * p + k_offset]
        # B[0,1] -> global V[b_row_offset * p + k_offset + 1]
        # B[1,0] -> global V[(b_row_offset+1) * p + k_offset]
        # B[1,1] -> global V[(b_row_offset+1) * p + k_offset + 1]
        
        # Local C indices (2x2):
        # C[0,0] -> global W[k_offset * n + c_row_offset]
        # C[0,1] -> global W[(k_offset+1) * n + c_row_offset]
        # C[1,0] -> global W[k_offset * n + c_row_offset + 1]
        # C[1,1] -> global W[(k_offset+1) * n + c_row_offset + 1]
        
        for u_loc, v_loc, w_loc in strassen_terms:
            u_g = [0]*12
            v_g = [0]*24
            w_g = [0]*18
            
            # Map u
            u_g[a_row_offset * m + b_row_offset] = u_loc[0]
            u_g[a_row_offset * m + b_row_offset + 1] = u_loc[1]
            u_g[(a_row_offset+1) * m + b_row_offset] = u_loc[2]
            u_g[(a_row_offset+1) * m + b_row_offset + 1] = u_loc[3]
            
            # Map v
            v_g[b_row_offset * p + k_offset] = v_loc[0]
            v_g[b_row_offset * p + k_offset + 1] = v_loc[1]
            v_g[(b_row_offset+1) * p + k_offset] = v_loc[2]
            v_g[(b_row_offset+1) * p + k_offset + 1] = v_loc[3]
            
            # Map w
            w_g[k_offset * n + c_row_offset] = w_loc[0]
            w_g[(k_offset+1) * n + c_row_offset] = w_loc[1]
            w_g[k_offset * n + c_row_offset + 1] = w_loc[2]
            w_g[(k_offset+1) * n + c_row_offset + 1] = w_loc[3]
            
            add_term(u_g, v_g, w_g)

    # Helper to add a 1x2x6 schoolbook block
    def add_1x2x6_block(a_row_offset, b_row_offset, c_row_offset):
        nonlocal col
        # A is 1x2 (row a_row_offset, cols b_row_offset, b_row_offset+1)
        # B is 2x6 (rows b_row_offset, b_row_offset+1, cols 0..5)
        # C is 1x6 (row c_row_offset, cols 0..5)
        
        for j in range(2):
            for k in range(6):
                u_g = [0]*12
                v_g = [0]*24
                w_g = [0]*18
                
                # A index: a_row_offset * m + b_row_offset + j
                u_g[a_row_offset * m + b_row_offset + j] = 1
                
                # B index: (b_row_offset + j) * p + k
                v_g[(b_row_offset + j) * p + k] = 1
                
                # C index: k * n + c_row_offset
                w_g[k * n + c_row_offset] = 1
                
                add_term(u_g, v_g, w_g)

    # Block 1: m in {0, 1}
    # A rows 0,1 (2x2 part)
    for k_block in range(3):
        k_offset = k_block * 2
        add_strassen_2x2_block(0, 0, 0, k_offset)
        add_strassen_2x2_block(1, 0, 1, k_offset)
        
    # A row 2 (1x2 part)
    add_1x2x6_block(2, 0, 2)

    # Block 2: m in {2, 3}
    # A rows 0,1 (2x2 part)
    for k_block in range(3):
        k_offset = k_block * 2
        add_strassen_2x2_block(0, 2, 0, k_offset)
        add_strassen_2x2_block(1, 2, 1, k_offset)
        
    # A row 2 (1x2 part)
    add_1x2x6_block(2, 2, 2)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
