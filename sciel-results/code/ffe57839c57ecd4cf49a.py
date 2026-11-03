"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    # Rank is 4 * 7 = 28
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for U, V, W
    # Indices 0..6 correspond to M1..M7
    # U coeffs: (i_loc, j_loc, coeff)
    U_coeffs = [
        [(0, 0, 1), (1, 1, 1)],          # M1
        [(1, 0, 1), (1, 1, 1)],          # M2
        [(0, 0, 1)],                     # M3
        [(1, 1, 1)],                     # M4
        [(0, 0, 1), (0, 1, 1)],          # M5
        [(1, 0, 1), (0, 0, -1)],         # M6
        [(0, 1, 1), (1, 1, -1)]          # M7
    ]
    
    # V coeffs: (j_loc, k_loc, coeff)
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)],          # M1
        [(0, 0, 1)],                     # M2
        [(0, 1, 1), (1, 1, -1)],         # M3
        [(1, 0, 1), (0, 0, -1)],         # M4
        [(1, 1, 1)],                     # M5
        [(0, 0, 1), (0, 1, 1)],          # M6
        [(1, 0, 1), (1, 1, 1)]           # M7
    ]
    
    # W coeffs: (k_loc, i_loc, coeff)
    # Note: W index in output is k*n + i. Local k, i are 0..1
    W_coeffs = [
        [(0, 0, 1), (1, 1, 1)],          # M1 -> C00, C11
        [(1, 0, 1), (1, 1, -1)],         # M2 -> C10, -C11
        [(0, 1, 1), (1, 1, 1)],          # M3 -> C01, C11
        [(0, 0, 1), (1, 0, 1)],          # M4 -> C00, C10
        [(0, 0, -1), (0, 1, 1)],         # M5 -> -C00, C01
        [(1, 1, 1)],                     # M6 -> C11
        [(0, 0, 1)]                      # M7 -> C00
    ]
    
    # Blocks: (i_offset, j_offset, k_offset)
    # Block 0: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (k=0:2, i=0:2)
    # Block 1: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2] (k=0:2, i=0:2)
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[2:4, 0:2] (k=2:4, i=0:2)
    # Block 3: A[0:2, 2:4] * B[2:4, 2:4] -> C[2:4, 0:2] (k=2:4, i=0:2)
    
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    col = 0
    for i_off, j_off, k_off in blocks:
        for s in range(7):
            # U
            for i_loc, j_loc, coeff in U_coeffs[s]:
                i_glob = i_off + i_loc
                j_glob = j_off + j_loc
                u_idx = i_glob * m + j_glob
                U[u_idx][col] += coeff
                
            # V
            for j_loc, k_loc, coeff in V_coeffs[s]:
                j_glob = j_off + j_loc
                k_glob = k_off + k_loc
                v_idx = j_glob * p + k_glob
                V[v_idx][col] += coeff
                
            # W
            for k_loc, i_loc, coeff in W_coeffs[s]:
                k_glob = k_off + k_loc
                i_glob = i_off + i_loc
                w_idx = k_glob * n + i_glob
                W[w_idx][col] += coeff
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
