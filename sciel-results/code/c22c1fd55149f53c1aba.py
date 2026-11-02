"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Rank is 28 (4 blocks of 2x2x2 Strassen)
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen coefficients for 2x2x2
    # u_coeffs[r][i][j]
    u_coeffs = np.array([
        [[1, 0], [0, 1]], # r0 (P1)
        [[0, 0], [1, 1]], # r1 (P2)
        [[1, 0], [0, 0]], # r2 (P3)
        [[0, 0], [0, 1]], # r3 (P4)
        [[1, 1], [0, 0]], # r4 (P5)
        [[-1, 0], [1, 0]],# r5 (P6)
        [[0, 1], [0, -1]] # r6 (P7)
    ])
    
    # v_coeffs[r][j][k]
    v_coeffs = np.array([
        [[1, 0], [0, 1]], # r0
        [[1, 0], [0, 0]], # r1
        [[0, 1], [0, -1]],# r2
        [[-1, 0], [1, 0]],# r3
        [[0, 0], [0, 1]], # r4
        [[1, 1], [0, 0]], # r5
        [[0, 0], [1, 1]]  # r6
    ])
    
    # w_coeffs[r][i][k]
    w_coeffs = np.array([
        [[1, 0], [0, 1]], # r0
        [[0, 0], [1, -1]],# r1
        [[0, 1], [0, 1]], # r2
        [[1, 0], [1, 0]], # r3
        [[-1, 1], [0, 0]],# r4
        [[0, 0], [0, 1]], # r5
        [[1, 0], [0, 0]]  # r6
    ])

    num_blocks = 4
    
    for block in range(num_blocks):
        for r in range(7):
            col = block * 7 + r
            
            # Fill U
            for i in range(2):
                for j in range(2):
                    val = u_coeffs[r][i][j]
                    if val != 0:
                        U[i * m + j, col] = val
                        
            # Fill V
            for j in range(2):
                for k_local in range(2):
                    val = v_coeffs[r][j][k_local]
                    if val != 0:
                        k_global = block * 2 + k_local
                        V[j * p + k_global, col] = val
                        
            # Fill W
            for i in range(2):
                for k_local in range(2):
                    val = w_coeffs[r][i][k_local]
                    if val != 0:
                        k_global = block * 2 + k_local
                        W[k_global * n + i, col] = val
                        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
