"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition using Strassen for 2x2x2 blocks
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2x2
    # U: coeffs for A (2x2)
    # V: coeffs for B (2x2)
    # W: coeffs for C (2x2)
    # Indices: i, j for A; j, k for B; i, k for C
    # r=1..7
    
    # A coeffs (i, j)
    U_strassen = [
        {(0,0): 1, (1,1): 1},      # M1
        {(1,0): 1, (1,1): 1},      # M2
        {(0,0): 1},                # M3
        {(1,1): 1},                # M4
        {(0,0): 1, (0,1): 1},      # M5
        {(1,0): 1, (0,0): -1},     # M6
        {(0,1): 1, (1,1): -1}      # M7
    ]
    
    # B coeffs (j, k)
    V_strassen = [
        {(0,0): 1, (1,1): 1},      # M1
        {(0,0): 1},                # M2
        {(0,1): 1, (1,1): -1},     # M3
        {(1,0): 1, (0,0): -1},     # M4
        {(1,1): 1},                # M5
        {(0,0): 1, (0,1): 1},      # M6
        {(1,0): 1, (1,1): 1}       # M7
    ]
    
    # C coeffs (i, k)
    W_strassen = [
        {(0,0): 1, (1,1): -1},     # M1
        {(1,0): 1, (1,1): -1},     # M2
        {(0,1): 1, (1,1): 1},      # M3
        {(0,0): 1, (1,0): 1},      # M4
        {(0,0): -1, (0,1): 1},     # M5
        {(1,1): 1},                # M6
        {(0,0): 1}                 # M7
    ]
    
    # Loop over the two J blocks (splitting m=4 into two 2s)
    for b in range(2):
        j_start = b * 2
        # Loop over the three K blocks (splitting p=6 into three 2s)
        for c in range(3):
            k_start = c * 2
            # Loop over the 7 Strassen terms
            for r in range(7):
                R = b * 21 + c * 7 + r
                
                # Fill U
                for (i_loc, j_loc), val in U_strassen[r].items():
                    i_glob = i_loc
                    j_glob = j_start + j_loc
                    u_idx = i_glob * m + j_glob
                    U[u_idx][R] = val
                    
                # Fill V
                for (j_loc, k_loc), val in V_strassen[r].items():
                    j_glob = j_start + j_loc
                    k_glob = k_start + k_loc
                    v_idx = j_glob * p + k_glob
                    V[v_idx][R] = val
                    
                # Fill W
                for (i_loc, k_loc), val in W_strassen[r].items():
                    i_glob = i_loc
                    k_glob = k_start + k_loc
                    w_idx = k_glob * n + i_glob
                    W[w_idx][R] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
