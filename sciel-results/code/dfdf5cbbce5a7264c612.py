import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=6, m=6, p=6
    
    # We use 3x3 blocks of size 2x2
    # Block indices: a, b, c in {0, 1, 2}
    # Element indices: i, j, k in {0, ..., 5}
    
    # Strassen 2x2 decomposition coefficients
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We represent the 2x2 matrices as vectors of length 4 (indices 00, 01, 10, 11)
    # A: [A00, A01, A10, A11]
    # B: [B00, B01, B10, B11]
    # C: [C00, C01, C10, C11]
    
    # Strassen terms (u, v, w) where u, v, w are vectors of length 4
    # u is for A, v is for B, w is for C
    # Each term is (u_vec, v_vec, w_vec)
    
    strassen_terms = []
    
    # M1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,-1] (for C00) + w=[0,0,0,1] (for C11)
    # Actually, it's better to list the contributions to each C element.
    # Let's define the 7 multiplications M1..M7 and their u, v.
    # Then define the linear combination for C00..C11.
    
    # M1: u = [1, 0, 0, 1], v = [1, 0, 0, 1]
    # M2: u = [0, 0, 1, 1], v = [1, 0, 0, 0]
    # M3: u = [1, 0, 0, 0], v = [0, 1, 0, -1]
    # M4: u = [0, 0, 0, 1], v = [0, 0, 1, -1]
    # M5: u = [1, 1, 0, 0], v = [0, 0, 0, 1]
    # M6: u = [0, 0, 1, -1], v = [1, 1, 0, 0]
    # M7: u = [0, 1, 0, -1], v = [0, 0, 1, 1]
    
    m_defs = [
        ([1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1]), # M3
        ([0, 0, 0, 1], [0, 0, 1, -1]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1]), # M5
        ([0, 0, 1, -1], [1, 1, 0, 0]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1])  # M7
    ]
    
    # C combinations:
    # C00 = 1*M1 + 0*M2 + 0*M3 + 1*M4 - 1*M5 + 0*M6 + 1*M7
    # C01 = 0*M1 + 0*M2 + 1*M3 + 0*M4 + 1*M5 + 0*M6 + 0*M7
    # C10 = 0*M1 + 1*M2 + 0*M3 + 1*M4 + 0*M5 + 0*M6 + 0*M7
    # C11 = 1*M1 - 1*M2 + 1*M3 + 0*M4 + 0*M5 + 1*M6 + 0*M7
    
    c_combos = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    rank = 27 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for a in range(3):
        for b in range(3):
            for c in range(3):
                # Block A[a,c], B[c,b], C[a,b]
                # A block indices: rows 2a, 2a+1; cols 2c, 2c+1
                # B block indices: rows 2c, 2c+1; cols 2b, 2b+1
                # C block indices: rows 2a, 2a+1; cols 2b, 2b+1
                
                for m_idx in range(7):
                    u_vec, v_vec = m_defs[m_idx]
                    # u_vec corresponds to A00, A01, A10, A11
                    # v_vec corresponds to B00, B01, B10, B11
                    
                    # Add to U
                    # A00 is at (2a, 2c) -> index (2a)*6 + 2c
                    # A01 is at (2a, 2c+1) -> index (2a)*6 + 2c+1
                    # A10 is at (2a+1, 2c) -> index (2a+1)*6 + 2c
                    # A11 is at (2a+1, 2c+1) -> index (2a+1)*6 + 2c+1
                    
                    u_indices = [
                        (2*a) * 6 + 2*c,
                        (2*a) * 6 + 2*c + 1,
                        (2*a + 1) * 6 + 2*c,
                        (2*a + 1) * 6 + 2*c + 1
                    ]
                    
                    for k in range(4):
                        if u_vec[k] != 0:
                            U[u_indices[k]][col] += u_vec[k]
                            
                    # Add to V
                    # B00 is at (2c, 2b) -> index (2c)*6 + 2b
                    # B01 is at (2c, 2b+1) -> index (2c)*6 + 2b+1
                    # B10 is at (2c+1, 2b) -> index (2c+1)*6 + 2b
                    # B11 is at (2c+1, 2b+1) -> index (2c+1)*6 + 2b+1
                    
                    v_indices = [
                        (2*c) * 6 + 2*b,
                        (2*c) * 6 + 2*b + 1,
                        (2*c + 1) * 6 + 2*b,
                        (2*c + 1) * 6 + 2*b + 1
                    ]
                    
                    for k in range(4):
                        if v_vec[k] != 0:
                            V[v_indices[k]][col] += v_vec[k]
                            
                    # Add to W
                    # C00, C01, C10, C11 contributions
                    # c_combos[m_idx] gives the coefficient for this M term in C00..C11
                    # Wait, c_combos is indexed by C element (0..3) and M term (0..6)?
                    # My c_combos definition: c_combos[i][j] is coeff of M_j in C_i.
                    # So for this M term (m_idx), the contribution to C_k is c_combos[k][m_idx].
                    
                    for k in range(4):
                        coeff = c_combos[k][m_idx]
                        if coeff != 0:
                            # C_k is one of C00, C01, C10, C11
                            # C00: row 2a, col 2b -> index (2b)*6 + 2a ?
                            # Recall W index: k*n + i where k is col, i is row?
                            # In schoolbook: W[k*n + i]. k is from V (col of B), i is from U (row of A).
                            # So W index for C_{row, col} is col*6 + row.
                            
                            if k == 0: # C00
                                w_idx = (2*b) * 6 + 2*a
                            elif k == 1: # C01
                                w_idx = (2*b + 1) * 6 + 2*a
                            elif k == 2: # C10
                                w_idx = (2*b) * 6 + (2*a + 1)
                            else: # C11
                                w_idx = (2*b + 1) * 6 + (2*a + 1)
                                
                            W[w_idx][col] += coeff
                            
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
