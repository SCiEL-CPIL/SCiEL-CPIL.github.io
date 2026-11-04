import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use 4 blocks of 2x2 matrix multiplication
    # Each 2x2 mult uses Strassen (rank 7)
    # Total rank 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y10-Y11)
    # M4 = X11(Y01-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y10)
    # M7 = (X01-X11)(Y11+Y01)
    
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # We define the 7 Strassen steps for a generic 2x2 block
    # X indices: (0,0), (0,1), (1,0), (1,1)
    # Y indices: (0,0), (0,1), (1,0), (1,1)
    # Z indices: (0,0), (0,1), (1,0), (1,1)
    
    strassen_steps = [
        # (X coeffs, Y coeffs, Z coeffs)
        # M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]),
        # M3
        ([(0,0,1)], [(1,0,1), (1,1,-1)], [(0,1,1)]),
        # M4
        ([(1,1,1)], [(0,1,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # M5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # M6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)], [(1,1,1)]),
        # M7
        ([(0,1,1), (1,1,-1)], [(1,1,1), (0,1,1)], [(0,0,1), (1,1,-1)]),
    ]
    
    # Map blocks to global indices
    # Block 0: A[:, 0:2] x B[0:2, 0:2] -> C[:, 0:2]
    # Block 1: A[:, 0:2] x B[0:2, 2:4] -> C[:, 2:4]
    # Block 2: A[:, 2:4] x B[2:4, 0:2] -> C[:, 0:2]
    # Block 3: A[:, 2:4] x B[2:4, 2:4] -> C[:, 2:4]
    
    # A global index: i * m + j
    # B global index: j * p + k
    # C global index: k * n + i
    
    blocks = [
        # (A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_row_offset, C_col_offset)
        # A is 2x4, so rows 0-1, cols 0-3
        # B is 4x4, so rows 0-3, cols 0-3
        # C is 2x4, so rows 0-1, cols 0-3
        (0, 0, 0, 0, 0, 0), # Block 0
        (0, 0, 0, 2, 0, 2), # Block 1
        (0, 2, 2, 0, 0, 0), # Block 2
        (0, 2, 2, 2, 0, 2), # Block 3
    ]
    
    col = 0
    for b in blocks:
        a_r, a_c, b_r, b_c, c_r, c_c = b
        for step in strassen_steps:
            x_coeffs, y_coeffs, z_coeffs = step
            for (xi, xj, xv) in x_coeffs:
                # Global A index
                gi = a_r + xi
                gj = a_c + xj
                U[gi * m + gj][col] += xv
            
            for (yi, yj, yv) in y_coeffs:
                # Global B index
                gi = b_r + yi
                gj = b_c + yj
                V[gi * p + gj][col] += yv
            
            for (zi, zj, zv) in z_coeffs:
                # Global C index
                # C index is (i, k) mapped to k * n + i
                gi = c_r + zi
                gj = c_c + zj
                W[gj * n + gi][col] += zv
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
