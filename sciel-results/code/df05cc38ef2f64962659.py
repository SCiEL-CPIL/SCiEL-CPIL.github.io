import time
import sys

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    # A: n x m (2x4)
    # B: m x p (4x4)
    # C: n x p (2x4)
    
    # We use 4 block multiplications of 2x2 matrices.
    # Each 2x2 multiplication uses Strassen's algorithm (rank 7).
    # Total rank = 4 * 7 = 28.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)] # 8 x 28
    V = [[0] * rank for _ in range(m * p)] # 16 x 28
    W = [[0] * rank for _ in range(p * n)] # 8 x 28
    
    # Define Strassen coefficients for one 2x2 multiplication
    # X: 2x2, Y: 2x2, Z: 2x2
    # P1 = (x00+x11)(y00+y11)
    # P2 = (x10+x11)y00
    # P3 = x00(y10-y11)
    # P4 = x11(y01-y00)
    # P5 = (x00+x01)y11
    # P6 = (x10-x00)(y00+y10)
    # P7 = (x01-x11)(y11+y01)
    
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    
    # Local indices for X, Y, Z (0,1,2,3)
    # X: 00, 01, 10, 11
    # Y: 00, 01, 10, 11
    # Z: 00, 01, 10, 11
    
    strassen_terms = [
        # (x_coeffs, y_coeffs, z_coeffs)
        # P1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1)]),
        # P2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),
        # P3
        ([(0, 1)], [(2, 1), (3, -1)], [(1, 1)]),
        # P4
        ([(3, 1)], [(1, 1), (0, -1)], [(0, 1), (2, 1)]),
        # P5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # P6
        ([(2, 1), (0, -1)], [(0, 1), (2, 1)], [(3, 1)]),
        # P7
        ([(1, 1), (3, -1)], [(3, 1), (1, 1)], [(0, 1), (3, 1)])
    ]
    
    # We have 4 blocks:
    # Block 1: A_L * B_LL -> C_L_part1
    # Block 2: A_R * B_RL -> C_L_part2
    # Block 3: A_L * B_LR -> C_R_part1
    # Block 4: A_R * B_RR -> C_R_part2
    
    # A indices:
    # A_L: 0, 1, 4, 5 (00, 01, 10, 11)
    # A_R: 2, 3, 6, 7 (00, 01, 10, 11)
    
    # B indices:
    # B_LL: 0, 1, 4, 5
    # B_RL: 8, 9, 12, 13
    # B_LR: 2, 3, 6, 7
    # B_RR: 10, 11, 14, 15
    
    # C indices (k*n + i):
    # C_L:
    # C00 (i=0, k=0): 0
    # C01 (i=0, k=1): 2
    # C10 (i=1, k=0): 1
    # C11 (i=1, k=1): 3
    # C_R:
    # C02 (i=0, k=2): 4
    # C03 (i=0, k=3): 6
    # C12 (i=1, k=2): 5
    # C13 (i=1, k=3): 7
    
    blocks = [
        {
            "A_idx": [0, 1, 4, 5],
            "B_idx": [0, 1, 4, 5],
            "C_idx": [0, 2, 1, 3] # Z00, Z01, Z10, Z11
        },
        {
            "A_idx": [2, 3, 6, 7],
            "B_idx": [8, 9, 12, 13],
            "C_idx": [0, 2, 1, 3] # Added to C_L
        },
        {
            "A_idx": [0, 1, 4, 5],
            "B_idx": [2, 3, 6, 7],
            "C_idx": [4, 6, 5, 7] # Z00, Z01, Z10, Z11 for C_R
        },
        {
            "A_idx": [2, 3, 6, 7],
            "B_idx": [10, 11, 14, 15],
            "C_idx": [4, 6, 5, 7] # Added to C_R
        }
    ]
    
    col = 0
    for block in blocks:
        a_idx = block["A_idx"]
        b_idx = block["B_idx"]
        c_idx = block["C_idx"]
        
        for x_coeffs, y_coeffs, z_coeffs in strassen_terms:
            for (local_x, val) in x_coeffs:
                global_a = a_idx[local_x]
                U[global_a][col] += val
            
            for (local_y, val) in y_coeffs:
                global_b = b_idx[local_y]
                V[global_b][col] += val
            
            for (local_z, val) in z_coeffs:
                global_c = c_idx[local_z]
                W[global_c][col] += val
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
