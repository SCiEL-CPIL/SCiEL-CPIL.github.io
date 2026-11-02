"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    rank = 56
    U = [[0] * rank for _ in range(n * m)] # 8 x 56
    V = [[0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0] * rank for _ in range(p * n)] # 16 x 56
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # u coeffs for A (a,b,c,d) -> indices 0,1,2,3 in local 2x2
    # Global A indices for A_L (cols 0,1): 0,1,4,5
    # Global A indices for A_R (cols 2,3): 2,3,6,7
    u_L = [
        [1, 0, 0, 1], # M1: a+d
        [0, 0, 1, 1], # M2: c+d
        [1, 0, 0, 0], # M3: a
        [0, 0, 0, 1], # M4: d
        [1, 1, 0, 0], # M5: a+b
        [-1, 0, 1, 0],# M6: c-a
        [0, 1, 0, -1] # M7: b-d
    ]
    
    u_R = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    # Map local (0,1,2,3) to global U indices
    # A_L: 0->0, 1->1, 2->4, 3->5
    # A_R: 0->2, 1->3, 2->6, 3->7
    u_map_L = [0, 1, 4, 5]
    u_map_R = [2, 3, 6, 7]
    
    # v coeffs for B (e,f,g,h) -> indices 0,1,2,3 in local 2x2
    # v1: e+h
    # v2: e
    # v3: f-h
    # v4: g-e
    # v5: h
    # v6: e+f
    # v7: g+h
    v_coeffs = [
        [1, 0, 0, 1], # M1: e+h
        [1, 0, 0, 0], # M2: e
        [0, 1, 0, -1],# M3: f-h
        [0, 0, 1, -1],# M4: g-e
        [0, 0, 0, 1], # M5: h
        [1, 1, 0, 0], # M6: e+f
        [0, 0, 1, 1]  # M7: g+h
    ]
    
    # w coeffs for C (C00, C01, C10, C11)
    w_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    col = 0
    for j in range(4): # 4 blocks of width 2
        # B_1j: rows 0,1; cols 2j, 2j+1
        # Global V indices for B_1j:
        # e -> 2j, f -> 2j+1, g -> 8+2j, h -> 8+2j+1
        v_map_1 = [2*j, 2*j+1, 8+2*j, 8+2*j+1]
        
        # B_2j: rows 2,3; cols 2j, 2j+1
        # Global V indices for B_2j:
        # e -> 16+2j, f -> 16+2j+1, g -> 24+2j, h -> 24+2j+1
        v_map_2 = [16+2*j, 16+2*j+1, 24+2*j, 24+2*j+1]
        
        for r in range(7):
            # Term from A_L * B_1j
            # U
            for l in range(4):
                if u_L[r][l] != 0:
                    idx = u_map_L[l]
                    U[idx][col] += u_L[r][l]
            # V
            for l in range(4):
                if v_coeffs[r][l] != 0:
                    idx = v_map_1[l]
                    V[idx][col] += v_coeffs[r][l]
            # W
            # Output block j: k = 2j, 2j+1. i = 0, 1.
            # W index: k*n + i = k*2 + i
            # C00 -> k=2j, i=0 -> 4j
            # C01 -> k=2j, i=1 -> 4j+1
            # C10 -> k=2j+1, i=0 -> 4j+2
            # C11 -> k=2j+1, i=1 -> 4j+3
            w_map = [4*j, 4*j+1, 4*j+2, 4*j+3]
            for l in range(4):
                if w_coeffs[r][l] != 0:
                    idx = w_map[l]
                    W[idx][col] += w_coeffs[r][l]
            col += 1
            
            # Term from A_R * B_2j
            # U
            for l in range(4):
                if u_R[r][l] != 0:
                    idx = u_map_R[l]
                    U[idx][col] += u_R[r][l]
            # V
            for l in range(4):
                if v_coeffs[r][l] != 0:
                    idx = v_map_2[l]
                    V[idx][col] += v_coeffs[r][l]
            # W (same as above)
            for l in range(4):
                if w_coeffs[r][l] != 0:
                    idx = w_map[l]
                    W[idx][col] += w_coeffs[r][l]
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
