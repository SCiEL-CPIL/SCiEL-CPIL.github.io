import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Laderman's algorithm for 3x3 block multiplication
    # Blocks are 2x2 matrices.
    # A blocks: A[i,k] for i,k in 0..2
    # B blocks: B[k,j] for k,j in 0..2
    # C blocks: C[i,j] for i,j in 0..2
    
    # 19 Laderman terms
    # Each term: (a_terms, b_terms)
    # a_terms: list of (i, k, coeff)
    # b_terms: list of (k, j, coeff)
    
    laderman_terms = [
        # P1
        ([(0,0,1), (2,2,1)], [(0,0,1), (2,2,1)]),
        # P2
        ([(0,1,1), (2,2,1)], [(0,0,1)]),
        # P3
        ([(0,0,1), (2,2,-1)], [(2,2,1)]),
        # P4
        ([(1,0,1), (1,2,1)], [(0,0,1), (1,1,1)]),
        # P5
        ([(1,0,1), (1,2,-1)], [(1,1,1)]),
        # P6
        ([(1,1,1), (2,2,1)], [(0,0,1), (1,1,-1)]),
        # P7
        ([(1,1,1), (2,2,-1)], [(1,1,1), (2,2,1)]),
        # P8
        ([(1,2,1), (2,2,1)], [(1,1,1), (2,2,-1)]),
        # P9
        ([(1,2,1), (2,2,-1)], [(0,0,1), (1,1,1)]),
        # P10
        ([(0,0,1), (1,1,1)], [(1,1,1), (2,2,1)]),
        # P11
        ([(0,0,1), (1,1,-1)], [(1,1,1), (2,2,-1)]),
        # P12
        ([(0,1,1), (1,1,1)], [(2,2,1)]),
        # P13
        ([(0,1,1), (1,1,-1)], [(0,0,1)]),
        # P14
        ([(2,0,1), (2,2,1)], [(0,1,1), (1,1,1)]),
        # P15
        ([(2,0,1), (2,2,-1)], [(1,1,1)]),
        # P16
        ([(2,1,1), (2,2,1)], [(0,1,1), (1,1,-1)]),
        # P17
        ([(2,1,1), (2,2,-1)], [(1,1,1), (2,2,1)]),
        # P18
        ([(0,2,1), (2,2,1)], [(1,2,1), (2,2,1)]),
        # P19
        ([(0,2,1), (2,2,-1)], [(1,2,1), (2,2,-1)]),
    ]
    
    # Reconstruction coefficients for C[i,j] from P_l
    # c_contrib[l][i][j] = coeff
    # Note: Indices are 0-based
    c_contrib = [
        # P1
        {(0,0): 1, (1,1): 1, (2,2): 0}, # Placeholder, need correct ones
    ] * 19
    
    # Correct Reconstruction from Laderman:
    # C00 = P1 - P2 + P3 + P6 - P7 + P10 - P11 - P14 + P15 - P16 + P17
    # C01 = P8 - P9 + P12 - P13 + P16 - P17 + P18 - P19
    # C02 = P4 - P5 + P12 - P13 + P18 - P19
    # C10 = P4 - P5 + P6 - P7 + P14 - P15 + P16 - P17
    # C11 = P1 - P2 + P3 - P4 + P5 - P6 + P7 - P8 + P9 + P10 - P11 - P12 + P13 - P14 + P15 - P16 + P17 + P18 - P19
    # C12 = P8 - P9 + P12 - P13 + P18 - P19
    # C20 = P4 - P5 + P14 - P15 + P18 - P19
    # C21 = P8 - P9 + P14 - P15 + P16 - P17
    # C22 = P1 - P2 + P3 + P6 - P7 + P10 - P11 - P14 + P15 - P16 + P17
    
    # Let's map these to a structure: recon[i][j] = list of (l, coeff)
    recon = [
        [
            [(0, 1), (1, -1), (2, 1), (5, 1), (6, -1), (9, 1), (10, -1), (13, -1), (14, 1), (15, -1), (16, 1)], # C00
            [(7, 1), (8, -1), (11, 1), (12, -1), (15, 1), (16, -1), (17, 1), (18, -1)], # C01
            [(3, 1), (4, -1), (11, 1), (12, -1), (17, 1), (18, -1)], # C02
        ],
        [
            [(3, 1), (4, -1), (5, 1), (6, -1), (13, 1), (14, -1), (15, 1), (16, -1)], # C10
            [(0, 1), (1, -1), (2, 1), (3, -1), (4, 1), (5, -1), (6, 1), (7, -1), (8, 1), (9, 1), (10, -1), (11, -1), (12, 1), (13, -1), (14, 1), (15, -1), (16, 1), (17, 1), (18, -1)], # C11
            [(7, 1), (8, -1), (11, 1), (12, -1), (17, 1), (18, -1)], # C12
        ],
        [
            [(3, 1), (4, -1), (13, 1), (14, -1), (17, 1), (18, -1)], # C20
            [(7, 1), (8, -1), (13, 1), (14, -1), (15, 1), (16, -1)], # C21
            [(0, 1), (1, -1), (2, 1), (5, 1), (6, -1), (9, 1), (10, -1), (13, -1), (14, 1), (15, -1), (16, 1)], # C22
        ]
    ]
    
    # Strassen terms for 2x2 multiplication
    # X, Y are 2x2. Z = XY.
    # 7 products M1..M7.
    # Each M_t is defined by linear forms in X and Y.
    # X entries: X00, X01, X10, X11
    # Y entries: Y00, Y01, Y10, Y11
    
    strassen_terms = [
        # M1 = (X00+X11)(Y00+Y11)
        {
            'x': [(0,0,1), (1,1,1)],
            'y': [(0,0,1), (1,1,1)],
            'z': [(0,0,1), (1,1,1)] # Z00 += M1, Z11 += M1
        },
        # M2 = (X10+X11)Y00
        {
            'x': [(1,0,1), (1,1,1)],
            'y': [(0,0,1)],
            'z': [(1,0,1), (1,1,-1)] # Z10 += M2, Z11 -= M2
        },
        # M3 = X00(Y01-Y11)
        {
            'x': [(0,0,1)],
            'y': [(0,1,1), (1,1,-1)],
            'z': [(0,1,1), (1,1,1)] # Z01 += M3, Z11 += M3
        },
        # M4 = X11(Y10-Y00)
        {
            'x': [(1,1,1)],
            'y': [(1,0,1), (0,0,-1)],
            'z': [(0,0,1), (1,0,1)] # Z00 += M4, Z10 += M4
        },
        # M5 = (X00+X01)Y11
        {
            'x': [(0,0,1), (0,1,1)],
            'y': [(1,1,1)],
            'z': [(0,0,-1), (0,1,1)] # Z00 -= M5, Z01 += M5
        },
        # M6 = (X10-X00)(Y00+Y01)
        {
            'x': [(1,0,1), (0,0,-1)],
            'y': [(0,0,1), (0,1,1)],
            'z': [(1,1,1)] # Z11 += M6
        },
        # M7 = (X01-X11)(Y10+Y11)
        {
            'x': [(0,1,1), (1,1,-1)],
            'y': [(1,0,1), (1,1,1)],
            'z': [(0,0,1)] # Z00 += M7
        }
    ]
    
    rank = 19 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for l in range(19):
        a_terms, b_terms = laderman_terms[l]
        # a_terms: list of (i, k, coeff)
        # b_terms: list of (k, j, coeff)
        
        for s in range(7):
            st = strassen_terms[s]
            
            # Update U
            # For each X entry (a,b) in st['x'], add coeff * (sum of a_terms for that entry)
            # X_ab = sum_{(i,k,c)} c * A[i,k][a,b]
            for (xa, xb, cx) in st['x']:
                for (i, k, ca) in a_terms:
                    # A[i,k] is at rows 2i..2i+1, cols 2k..2k+1
                    # Entry [xa, xb] is at global row 2i+xa, col 2k+xb
                    u_row = (2*i + xa) * m + (2*k + xb)
                    U[u_row][col] += ca * cx
            
            # Update V
            # For each Y entry (c,d) in st['y'], add coeff * (sum of b_terms for that entry)
            # Y_cd = sum_{(k,j,c)} c * B[k,j][c,d]
            for (yc, yd, cy) in st['y']:
                for (k, j, cb) in b_terms:
                    # B[k,j] is at rows 2k..2k+1, cols 2j..2j+1
                    # Entry [yc, yd] is at global row 2k+yc, col 2j+yd
                    v_row = (2*k + yc) * p + (2*j + yd)
                    V[v_row][col] += cb * cy
            
            # Update W
            # For each Z entry (e,f) in st['z'], add coeff * (sum of recon for C[i,j][e,f])
            # Z_ef contributes to C[i,j][e,f] with coeff from recon[i][j][l]
            for (ze, zf, cz) in st['z']:
                for i in range(3):
                    for j in range(3):
                        # Find coeff for P_l in C[i,j]
                        coeff_l = 0
                        for (idx, c) in recon[i][j]:
                            if idx == l:
                                coeff_l = c
                                break
                        if coeff_l != 0:
                            # C[i,j] is at rows 2i..2i+1, cols 2j..2j+1
                            # Entry [ze, zf] is at global row 2i+ze, col 2j+zf
                            # W index: col * n + row
                            w_row = (2*j + zf) * n + (2*i + ze)
                            W[w_row][col] += coeff_l * cz
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
