import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We use a block decomposition with Strassen's algorithm for 2x2 blocks.
    # Total rank = 84.
    rank = 84
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # m1 = (A00 + A11)(B00 + B11)
    # m2 = (A10 + A11)B00
    # m3 = A00(B01 - B11)
    # m4 = A11(B10 - B00)
    # m5 = (A00 + A01)B11
    # m6 = (A10 - A00)(B00 + B01)
    # m7 = (A01 - A11)(B10 + B11)
    
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # Define the 12 units (rx, cx, ry, cy, rz, cz)
    # rx, cx: row, col offset of A block in A (4x4)
    # ry, cy: row, col offset of B block in B (4x6)
    # rz, cz: row, col offset of C block in C (4x6)
    
    units = [
        # C_L (cols 0,1)
        # C_L1 (rows 0,1, cols 0,1)
        (0, 0, 0, 0, 0, 0), # A11 * B11
        (0, 2, 2, 0, 0, 0), # A12 * B21
        # C_L2 (rows 2,3, cols 0,1)
        (2, 0, 0, 0, 2, 0), # A21 * B11
        (2, 2, 2, 0, 2, 0), # A22 * B21
        
        # C_R (cols 2,3,4,5)
        # C_R11 (rows 0,1, cols 2,3)
        (0, 0, 0, 2, 0, 2), # A11 * B11
        (0, 2, 2, 2, 0, 2), # A12 * B21
        # C_R12 (rows 0,1, cols 4,5)
        (0, 0, 0, 4, 0, 4), # A11 * B12
        (0, 2, 2, 4, 0, 4), # A12 * B22
        # C_R21 (rows 2,3, cols 2,3)
        (2, 0, 0, 2, 2, 2), # A21 * B11
        (2, 2, 2, 2, 2, 2), # A22 * B21
        # C_R22 (rows 2,3, cols 4,5)
        (2, 0, 0, 4, 2, 4), # A21 * B12
        (2, 2, 2, 4, 2, 4), # A22 * B22
    ]
    
    col = 0
    for rx, cx, ry, cy, rz, cz in units:
        for k in range(7):
            if time.time() > deadline - 5:
                break
            # Populate U, V, W for this Strassen product m_k
            
            # A indices (rows 0..15)
            # a_00: rx, cx
            # a_01: rx, cx+1
            # a_10: rx+1, cx
            # a_11: rx+1, cx+1
            
            u_00 = rx * m + cx
            u_01 = rx * m + cx + 1
            u_10 = (rx + 1) * m + cx
            u_11 = (rx + 1) * m + cx + 1
            
            # B indices (rows 0..23)
            # b_00: ry, cy
            # b_01: ry, cy+1
            # b_10: ry+1, cy
            # b_11: ry+1, cy+1
            
            v_00 = ry * p + cy
            v_01 = ry * p + cy + 1
            v_10 = (ry + 1) * p + cy
            v_11 = (ry + 1) * p + cy + 1
            
            # C indices (rows 0..23)
            # W index: k*n + i
            # c_00: rz, cz -> cz*n + rz
            # c_01: rz, cz+1 -> (cz+1)*n + rz
            # c_10: rz+1, cz -> cz*n + (rz+1)
            # c_11: rz+1, cz+1 -> (cz+1)*n + (rz+1)
            
            w_00 = cz * n + rz
            w_01 = (cz + 1) * n + rz
            w_10 = cz * n + (rz + 1)
            w_11 = (cz + 1) * n + (rz + 1)
            
            if k == 0: # m1
                U[u_00][col] = 1
                U[u_11][col] = 1
                V[v_00][col] = 1
                V[v_11][col] = 1
                W[w_00][col] = 1
                W[w_11][col] = 1
            elif k == 1: # m2
                U[u_10][col] = 1
                U[u_11][col] = 1
                V[v_00][col] = 1
                W[w_10][col] = 1
                W[w_11][col] = -1
            elif k == 2: # m3
                U[u_00][col] = 1
                V[v_01][col] = 1
                V[v_11][col] = -1
                W[w_01][col] = 1
                W[w_11][col] = 1
            elif k == 3: # m4
                U[u_11][col] = 1
                V[v_10][col] = 1
                V[v_00][col] = -1
                W[w_00][col] = 1
                W[w_10][col] = 1
            elif k == 4: # m5
                U[u_00][col] = 1
                U[u_01][col] = 1
                V[v_11][col] = 1
                W[w_00][col] = -1
                W[w_01][col] = 1
            elif k == 5: # m6
                U[u_10][col] = 1
                U[u_00][col] = -1
                V[v_00][col] = 1
                V[v_01][col] = 1
                W[w_11][col] = 1
            elif k == 6: # m7
                U[u_01][col] = 1
                U[u_11][col] = -1
                V[v_10][col] = 1
                V[v_11][col] = 1
                W[w_00][col] = 1
            
            col += 1
        if time.time() > deadline - 5:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
