"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using Strassen's algorithm for 4 independent 2x2 blocks.
    rank = 28
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2 * 2x2 -> 2x2
    # U (A) combinations:
    U_defs = [
        [(0, 1), (3, 1)],   # A00 + A11
        [(2, 1), (3, 1)],   # A10 + A11
        [(0, 1)],           # A00
        [(3, 1)],           # A11
        [(0, 1), (1, 1)],   # A00 + A01
        [(2, 1), (0, -1)],  # A10 - A00
        [(1, 1), (3, -1)]   # A01 - A11
    ]
    
    # V (B) combinations:
    V_defs = [
        [(0, 1), (3, 1)],   # B00 + B11
        [(0, 1)],           # B00
        [(1, 1), (3, -1)],  # B01 - B11
        [(2, 1), (0, -1)],  # B10 - B00
        [(3, 1)],           # B11
        [(0, 1), (1, 1)],   # B00 + B01
        [(2, 1), (3, 1)]    # B10 + B11
    ]
    
    # W (C) combinations:
    W_defs = [
        [(0, 1), (3, 1)],   # C00 + C11
        [(1, 1), (3, -1)],  # C10 - C11
        [(2, 1), (3, 1)],   # C01 + C11
        [(0, 1), (1, 1)],   # C00 + C10
        [(0, -1), (2, 1)],  # -C00 + C01
        [(3, 1)],           # C11
        [(0, 1)]            # C00
    ]

    for q in range(4):
        for s in range(7):
            r = q * 7 + s
            
            # Fill U
            for idx, val in U_defs[s]:
                U[idx, r] = val
                
            # Fill V
            for local_idx, val in V_defs[s]:
                j = local_idx // 2
                k_local = local_idx % 2
                k = 2 * q + k_local
                v_idx = j * p + k
                V[v_idx, r] = val
                
            # Fill W
            for local_idx, val in W_defs[s]:
                k_local = local_idx // 2
                i = local_idx % 2
                k = 2 * q + k_local
                w_idx = k * n + i
                W[w_idx, r] = val

    # Convert to lists of lists for return
    U_list = [list(row) for row in U]
    V_list = [list(row) for row in V]
    W_list = [list(row) for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
