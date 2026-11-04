"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(u_idx, v_idx, w_idx, start_col):
        # u_idx: [x00, x01, x10, x11]
        # v_idx: [y00, y01, y10, y11]
        # w_idx: [z00, z01, z10, z11]
        
        # Strassen terms U, V coefficients
        # Order: M1, M2, M3, M4, M5, M6, M7
        U_coeffs = [
            [1, 0, 1, 0],  # M1: x00+x10
            [1, 0, 1, 0],  # M2: x00+x10
            [1, 0, 0, 0],  # M3: x00
            [0, 0, 0, 1],  # M4: x11
            [1, 1, 0, 0],  # M5: x00+x01
            [-1, 0, 0, 1], # M6: x11-x00
            [0, 1, 0, -1]  # M7: x01-x11
        ]
        V_coeffs = [
            [1, 1, 0, 0],  # M1: y00+y01
            [0, 0, 0, 1],  # M2: y11
            [1, 0, -1, 0], # M3: y00-y10
            [0, 1, 1, 0],  # M4: y01+y10
            [0, 0, 1, 0],  # M5: y10
            [1, 0, 1, 0],  # M6: y00+y10
            [0, 0, 1, 1]   # M7: y10+y11
        ]
        W_coeffs = [
            [1, -1, 0, 1, 0, 1, 0],  # z00: M1 - M2 + M4 + M6
            [0, 1, 1, 0, 0, 0, 0],   # z01: M2 + M3
            [0, 0, 0, 0, 1, 0, 1],   # z10: M5 + M7
            [1, 0, -1, 0, 1, 0, 1]   # z11: M1 - M3 + M5 + M7
        ]
        
        for t in range(7):
            col = start_col + t
            for i in range(4):
                U[u_idx[i]][col] += U_coeffs[t][i]
                V[v_idx[i]][col] += V_coeffs[t][i]
            for i in range(4):
                W[w_idx[i]][col] += W_coeffs[i][t]

    # Block 1: A1 B11 -> C_left
    # A1: cols 0,1. Indices: 0,1,4,5
    # B11: rows 0,1, cols 0,1. Indices: 0,1,4,5
    # C_left: C00, C01, C10, C11. Indices: 0,2,1,3
    add_strassen_block([0, 1, 4, 5], [0, 1, 4, 5], [0, 2, 1, 3], 0)

    # Block 2: A2 B21 -> C_left
    # A2: cols 2,3. Indices: 2,3,6,7
    # B21: rows 2,3, cols 0,1. Indices: 8,9,12,13
    # C_left: C00, C01, C10, C11. Indices: 0,2,1,3
    add_strassen_block([2, 3, 6, 7], [8, 9, 12, 13], [0, 2, 1, 3], 7)

    # Block 3: A1 B12 -> C_right
    # A1: cols 0,1. Indices: 0,1,4,5
    # B12: rows 0,1, cols 2,3. Indices: 2,3,6,7
    # C_right: C02, C03, C12, C13. Indices: 4,6,5,7
    add_strassen_block([0, 1, 4, 5], [2, 3, 6, 7], [4, 6, 5, 7], 14)

    # Block 4: A2 B22 -> C_right
    # A2: cols 2,3. Indices: 2,3,6,7
    # B22: rows 2,3, cols 2,3. Indices: 10,11,14,15
    # C_right: C02, C03, C12, C13. Indices: 4,6,5,7
    add_strassen_block([2, 3, 6, 7], [10, 11, 14, 15], [4, 6, 5, 7], 21)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
