import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using 4 Strassen blocks
    
    # Strassen forms for 2x2 matrix multiplication
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # Z = XY
    
    # We define the 7 products M1..M7
    # Each product is a pair (u_vec, v_vec) such that M = u_vec . X . v_vec? 
    # No, Strassen products are bilinear: M = u_X(X) * v_Y(Y).
    # u_X is a linear form on X. v_Y is a linear form on Y.
    
    # Indices for 2x2 X: 0,1,2,3 corresponding to x00, x01, x10, x11
    # Indices for 2x2 Y: 0,1,2,3 corresponding to y00, y01, y10, y11
    # Indices for 2x2 Z: 0,1,2,3 corresponding to z00, z01, z10, z11
    
    # Strassen M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y10-y11)
    # M4 = x11(y01-y00)
    # M5 = (x00+x10)y11
    # M6 = (x11-x00)(y00+y10)
    # M7 = (x01-x11)(y11+y01)
    
    # Z reconstruction:
    # z00 = M1 + M4 - M5 - M7
    # z01 = M3 + M7
    # z10 = M2 + M6
    # z11 = M1 - M2 + M3 + M5
    
    # We need to map these to the global indices.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)] # 8 x 28
    V = [[0] * rank for _ in range(m * p)] # 16 x 28
    W = [[0] * rank for _ in range(p * n)] # 8 x 28
    
    # Define the 7 Strassen steps
    # Each step is a dict with 'u' (linear form on X), 'v' (linear form on Y), 'w' (coeffs for Z)
    # u is a vector of length 4 (local indices 0..3)
    # v is a vector of length 4
    # w is a vector of length 4
    
    steps = []
    
    # M1
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1] # z00+=1, z11+=1
    steps.append((u1, v1, w1))
    
    # M2
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, -1] # z10+=1, z11-=-1
    steps.append((u2, v2, w2))
    
    # M3
    u3 = [1, 0, 0, 0]
    v3 = [0, 0, 1, -1]
    w3 = [0, 1, -1, 1] # z01+=1, z10-=-1, z11+=1
    steps.append((u3, v3, w3))
    
    # M4
    u4 = [0, 0, 0, 1]
    v4 = [0, 1, -1, 0]
    w4 = [1, 0, 0, 0] # z00+=1
    steps.append((u4, v4, w4))
    
    # M5
    u5 = [1, 0, 1, 0]
    v5 = [0, 0, 0, 1]
    w5 = [-1, 0, 0, 1] # z00-=-1, z11+=1
    steps.append((u5, v5, w5))
    
    # M6
    u6 = [0, 0, 0, 1] # x11
    # Wait, M6 = (x11-x00)(y00+y10)
    u6 = [-1, 0, 0, 1]
    v6 = [1, 0, 1, 0]
    w6 = [0, 0, 1, 0] # z10+=1
    steps.append((u6, v6, w6))
    
    # M7
    # M7 = (x01-x11)(y11+y01)
    u7 = [0, 1, 0, -1]
    v7 = [0, 1, 0, 1]
    w7 = [-1, 1, 0, 0] # z00-=-1, z01+=1
    steps.append((u7, v7, w7))
    
    # Now assign these to the 4 blocks
    # Block 1: A_L (U idx 0,1,4,5) * B_11 (V idx 0,1,4,5) -> C_L (W idx 0,1,2,3)
    # Block 2: A_R (U idx 2,3,6,7) * B_21 (V idx 8,9,12,13) -> C_L (W idx 0,1,2,3)
    # Block 3: A_L (U idx 0,1,4,5) * B_12 (V idx 2,3,6,7) -> C_R (W idx 4,5,6,7)
    # Block 4: A_R (U idx 2,3,6,7) * B_22 (V idx 10,11,14,15) -> C_R (W idx 4,5,6,7)
    
    # Local to Global Index Mappings
    # A_L: x00->0, x01->1, x10->4, x11->5
    map_AL = {0: 0, 1: 1, 2: 4, 3: 5}
    # A_R: x00->2, x01->3, x10->6, x11->7
    map_AR = {0: 2, 1: 3, 2: 6, 3: 7}
    
    # B_11: y00->0, y01->1, y10->4, y11->5
    map_B11 = {0: 0, 1: 1, 2: 4, 3: 5}
    # B_21: y00->8, y01->9, y10->12, y11->13
    map_B21 = {0: 8, 1: 9, 2: 12, 3: 13}
    # B_12: y00->2, y01->3, y10->6, y11->7
    map_B12 = {0: 2, 1: 3, 2: 6, 3: 7}
    # B_22: y00->10, y01->11, y10->14, y11->15
    map_B22 = {0: 10, 1: 11, 2: 14, 3: 15}
    
    # C_L: z00->0, z01->1, z10->2, z11->3
    map_CL = {0: 0, 1: 1, 2: 2, 3: 3}
    # C_R: z00->4, z01->5, z10->6, z11->7
    map_CR = {0: 4, 1: 5, 2: 6, 3: 7}
    
    col = 0
    blocks = [
        (map_AL, map_B11, map_CL),
        (map_AR, map_B21, map_CL),
        (map_AL, map_B12, map_CR),
        (map_AR, map_B22, map_CR)
    ]
    
    for map_u, map_v, map_w in blocks:
        for u, v, w in steps:
            for local_idx, val in enumerate(u):
                if val != 0:
                    U[map_u[local_idx]][col] += val
            for local_idx, val in enumerate(v):
                if val != 0:
                    V[map_v[local_idx]][col] += val
            for local_idx, val in enumerate(w):
                if val != 0:
                    W[map_w[local_idx]][col] += val
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
