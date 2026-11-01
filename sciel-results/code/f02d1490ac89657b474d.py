"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 21  # 3 blocks * 7 Strassen terms
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen coefficients for a single 2x2x2 block
    # U entries: (index, val)
    # V entries: (index, val)
    # W entries: (index, val)
    
    # Indices for A (2x2):
    # A00: 0, A01: 1, A10: 2, A11: 3
    
    # For a block t, B indices are:
    # B_t_00: 2*t
    # B_t_01: 2*t + 1
    # B_t_10: 6 + 2*t
    # B_t_11: 6 + 2*t + 1
    
    # For a block t, C indices in W (k*n + i, n=2):
    # C_t_00: col 2*t, row 0 -> 4*t
    # C_t_01: col 2*t+1, row 0 -> 4*t + 2
    # C_t_10: col 2*t, row 1 -> 4*t + 1
    # C_t_11: col 2*t+1, row 1 -> 4*t + 3
    
    # Strassen terms definition:
    # s=1: M1 = (A00+A11)*(B00+B11)
    # s=2: M2 = (A10+A11)*B01
    # s=3: M3 = A00*B10
    # s=4: M4 = A11*(B10+B11)
    # s=5: M5 = (A00+A01)*B11
    # s=6: M6 = (A10-A00)*(B00+B10)
    # s=7: M7 = (A01-A11)*(B01+B11)
    
    # Output reconstruction:
    # C00 = M1 + M4 - M7 - M5
    # C01 = M5 + M7
    # C10 = M3 + M6
    # C11 = M1 - M3 + M2 + M6
    
    strassen_terms = [
        {
            'u': [(0, 1), (3, 1)],
            'v': [0, 1], # relative to B00, B11
            'w': [0, 1], # relative to C00, C11
            'w_vals': [1, 1]
        },
        {
            'u': [(2, 1), (3, 1)],
            'v': [1], # relative to B01
            'w': [3], # relative to C11
            'w_vals': [1]
        },
        {
            'u': [(0, 1)],
            'v': [2], # relative to B10
            'w': [1, 3], # relative to C10, C11
            'w_vals': [1, -1]
        },
        {
            'u': [(3, 1)],
            'v': [2, 3], # relative to B10, B11
            'w': [0], # relative to C00
            'w_vals': [1]
        },
        {
            'u': [(0, 1), (1, 1)],
            'v': [3], # relative to B11
            'w': [2, 0], # relative to C01, C00
            'w_vals': [1, -1]
        },
        {
            'u': [(2, 1), (0, -1)],
            'v': [0, 2], # relative to B00, B10
            'w': [1, 3], # relative to C10, C11
            'w_vals': [1, 1]
        },
        {
            'u': [(1, 1), (3, -1)],
            'v': [1, 3], # relative to B01, B11
            'w': [2, 0], # relative to C01, C00
            'w_vals': [1, -1]
        }
    ]
    
    for t in range(3):
        for s_idx in range(7):
            col = t * 7 + s_idx
            term = strassen_terms[s_idx]
            
            # Fill U
            for idx, val in term['u']:
                U[idx][col] = val
            
            # Fill V
            # Map relative B indices to absolute
            # 0->B_t_00, 1->B_t_01, 2->B_t_10, 3->B_t_11
            for rel_idx in term['v']:
                if rel_idx == 0:
                    abs_idx = 2 * t
                elif rel_idx == 1:
                    abs_idx = 2 * t + 1
                elif rel_idx == 2:
                    abs_idx = 6 + 2 * t
                elif rel_idx == 3:
                    abs_idx = 6 + 2 * t + 1
                else:
                    continue
                V[abs_idx][col] = 1
            
            # Fill W
            # Map relative C indices to absolute
            # 0->C_t_00, 1->C_t_10, 2->C_t_01, 3->C_t_11
            for rel_idx, val in zip(term['w'], term['w_vals']):
                if rel_idx == 0:
                    abs_idx = 4 * t
                elif rel_idx == 1:
                    abs_idx = 4 * t + 1
                elif rel_idx == 2:
                    abs_idx = 4 * t + 2
                elif rel_idx == 3:
                    abs_idx = 4 * t + 3
                else:
                    continue
                W[abs_idx][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
