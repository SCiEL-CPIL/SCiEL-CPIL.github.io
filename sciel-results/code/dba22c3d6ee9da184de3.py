import numpy as np

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We aim for rank 21.
    # This implementation uses a programmatic construction of the Strassen-like blocks
    # but organizes the code differently (using NumPy for clarity and verification).
    
    rank = 21
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Define the 7 Strassen vectors for 2x2x2
    # u1..u7 for A
    # v1..v7 for B
    # w1..w7 for C
    # Indices: A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3 (local)
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3 (local)
    
    U_strassen = [
        [1, 0, 0, 1],  # 0
        [0, 0, 1, 1],  # 1
        [1, 0, 0, 0],  # 2
        [0, 0, 0, 1],  # 3
        [1, 1, 0, 0],  # 4
        [-1, 0, 1, 0], # 5
        [0, 1, 0, -1]  # 6
    ]
    
    # V and W for Strassen 2x2x2
    # V vectors (for B local 2x2)
    V_strassen = [
        [1, 1, 0, 0],  # 0: b00+b11? No, Strassen P1 uses (b00+b11)
        # Let's stick to the standard Strassen definitions:
        # P1 = (a00+a11)(b00+b11)
        # P2 = (a10+a11)b00
        # P3 = a00(b01-b11)
        # P4 = a11(b10-b00)
        # P5 = (a00+a01)b11
        # P6 = (a10-a00)(b00+b01)
        # P7 = (a01-a11)(b10+b11)
        
        # V vectors (coeffs for b00, b01, b10, b11)
        [1, 0, 0, 1],  # 0: b00+b11
        [1, 0, 0, 0],  # 1: b00
        [0, 1, 0, -1], # 2: b01-b11
        [0, 0, 1, -1], # 3: b10-b00
        [0, 0, 0, 1],  # 4: b11
        [1, 1, 0, 0],  # 5: b00+b01
        [0, 0, 1, 1]   # 6: b10+b11
    ]
    
    # W vectors (coeffs for c00, c01, c10, c11)
    W_strassen = [
        [1, 0, 0, 1],  # 0: c00+c11
        [0, 0, 1, -1], # 1: c10-c11
        [0, 1, 0, 1],  # 2: c01+c11
        [1, 1, 0, 0],  # 3: c00+c10
        [-1, 1, 0, 0], # 4: -c00+c01
        [0, 0, 0, 1],  # 5: c11
        [1, 0, 0, 0]   # 6: c00
    ]
    
    # Verify one block
    # For block r (0, 1, 2), the local B columns are 2r, 2r+1.
    # Local B indices: 0->2r, 1->2r+1 (col), 0->row 0, 1->row 1.
    # V index: j*p + k.
    # W index: k*n + i.
    
    for r in range(3):
        base_col = 2 * r
        for s in range(7):
            col = r * 7 + s
            
            # Fill U
            U[:, col] = U_strassen[s]
            
            # Fill V
            # V_strassen[s] has 4 entries: v00, v01, v10, v11 (local)
            # v00 corresponds to B[0, 2r] -> index 0*p + 2r
            # v01 corresponds to B[0, 2r+1] -> index 0*p + 2r+1
            # v10 corresponds to B[1, 2r] -> index 1*p + 2r
            # v11 corresponds to B[1, 2r+1] -> index 1*p + 2r+1
            
            v_vec = V_strassen[s]
            V[0 * p + base_col][col] = v_vec[0]
            V[0 * p + base_col + 1][col] = v_vec[1]
            V[1 * p + base_col][col] = v_vec[2]
            V[1 * p + base_col + 1][col] = v_vec[3]
            
            # Fill W
            # W_strassen[s] has 4 entries: w00, w01, w10, w11 (local)
            # w00 corresponds to C[2r, 0] -> index 2r*n + 0
            # w01 corresponds to C[2r, 1] -> index 2r*n + 1
            # w10 corresponds to C[2r+1, 0] -> index (2r+1)*n + 0
            # w11 corresponds to C[2r+1, 1] -> index (2r+1)*n + 1
            
            w_vec = W_strassen[s]
            W[base_col * n + 0][col] = w_vec[0]
            W[base_col * n + 1][col] = w_vec[1]
            W[(base_col + 1) * n + 0][col] = w_vec[2]
            W[(base_col + 1) * n + 1][col] = w_vec[3]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
