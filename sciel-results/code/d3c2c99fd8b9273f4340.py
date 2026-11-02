"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = A11 B11
    # M2 = (A11 + A12) B22
    # M3 = (A21 + A22) B11
    # M4 = A22 (B12 - B22)
    # M5 = (A11 + A22) (B11 + B22)
    # M6 = (A12 - A22) (B21 + B22)
    # M7 = (A11 - A21) (B11 + B12)
    
    # C11 = M1 - M4 + M5 - M7
    # C12 = M2 + M3 - M5 + M6
    # C21 = M1 - M2 + M4 - M7
    # C22 = M3 - M1 + M5 - M6
    
    # Define the 7 terms as (u_coeffs, v_coeffs)
    # u_coeffs: list of (local_row, local_col, coeff)
    # v_coeffs: list of (local_row, local_col, coeff)
    
    terms = [
        # M1
        {
            'u': [(0, 0, 1)],
            'v': [(0, 0, 1)],
            'w': [(0, 0, 1), (0, 1, 0), (1, 0, 1), (1, 1, 0)] # C11+=1, C21+=1? No.
            # C11 gets +1, C12 gets 0, C21 gets +1? No.
            # C11 = M1... C21 = M1...
            # So W contributions: C11(0,0)+1, C21(1,0)+1
        },
        # M2
        {
            'u': [(0, 0, 1), (0, 1, 1)],
            'v': [(1, 1, 1)],
            'w': [(0, 1, 1), (1, 0, -1)] # C12+=1, C21+=-1
        },
        # M3
        {
            'u': [(1, 0, 1), (1, 1, 1)],
            'v': [(0, 0, 1)],
            'w': [(0, 1, 1), (1, 1, 1)] # C12+=1, C22+=1
        },
        # M4
        {
            'u': [(1, 1, 1)],
            'v': [(0, 1, 1), (1, 1, -1)],
            'w': [(0, 0, -1), (1, 0, 1)] # C11+=-1, C21+=1
        },
        # M5
        {
            'u': [(0, 0, 1), (1, 1, 1)],
            'v': [(0, 0, 1), (1, 1, 1)],
            'w': [(0, 0, 1), (0, 1, -1), (1, 0, 0), (1, 1, 1)] # C11+=1, C12+=-1, C22+=1
        },
        # M6
        {
            'u': [(0, 1, 1), (1, 1, -1)],
            'v': [(1, 0, 1), (1, 1, 1)],
            'w': [(0, 1, 1), (1, 1, -1)] # C12+=1, C22+=-1
        },
        # M7
        {
            'u': [(0, 0, 1), (1, 0, -1)],
            'v': [(0, 0, 1), (0, 1, 1)],
            'w': [(0, 0, -1), (1, 0, -1)] # C11+=-1, C21+=-1
        }
    ]
    
    # Note: The 'w' keys above are (local_row, local_col, coeff) for C.
    # C11 is (0,0), C12 is (0,1), C21 is (1,0), C22 is (1,1).
    # Let's re-verify the W coefficients for each term.
    # C11 = M1 - M4 + M5 - M7
    # C12 = M2 + M3 - M5 + M6
    # C21 = M1 - M2 + M4 - M7
    # C22 = M3 - M1 + M5 - M6
    
    # M1: C11+1, C21+1
    # M2: C12+1, C21-1
    # M3: C12+1, C22+1
    # M4: C11-1, C21+1
    # M5: C11+1, C12-1, C22+1
    # M6: C12+1, C22-1
    # M7: C11-1, C21-1
    
    # Correcting the 'w' lists in terms:
    terms[0]['w'] = [(0, 0, 1), (1, 0, 1)]
    terms[1]['w'] = [(0, 1, 1), (1, 0, -1)]
    terms[2]['w'] = [(0, 1, 1), (1, 1, 1)]
    terms[3]['w'] = [(0, 0, -1), (1, 0, 1)]
    terms[4]['w'] = [(0, 0, 1), (0, 1, -1), (1, 1, 1)]
    terms[5]['w'] = [(0, 1, 1), (1, 1, -1)]
    terms[6]['w'] = [(0, 0, -1), (1, 0, -1)]
    
    col = 0
    for a_col_base in [0, 2]:
        for b_row_base in [0, 2]:
            for b_col_base in [0, 2]:
                c_col_base = b_col_base
                for t in terms:
                    # Update U
                    for lr, lc, c in t['u']:
                        u_row = lr * m + (a_col_base + lc)
                        U[u_row][col] += c
                    # Update V
                    for lr, lc, c in t['v']:
                        v_row = (b_row_base + lr) * p + (b_col_base + lc)
                        V[v_row][col] += c
                    # Update W
                    for lr, lc, c in t['w']:
                        w_row = (c_col_base + lc) * n + lr
                        W[w_row][col] += c
                    col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
