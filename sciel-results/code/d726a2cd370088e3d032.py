import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 50 decomposition
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get indices
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    # Strassen coefficients for 2x2x2
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y10+y11)
    # M4 = x11*y11
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y01)
    # M7 = (x01-x11)(y10+y11)
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # We define the 7 products as (coeffs_X, coeffs_Y)
    # coeffs_X is dict {(i,j): coeff}
    # coeffs_Y is dict {(i,j): coeff}
    strassen_products = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),           # M2
        ({(0,0): 1}, {(1,0): 1, (1,1): 1}),           # M3
        ({(1,1): 1}, {(1,1): 1}),                     # M4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),           # M5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),# M6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}) # M7
    ]
    
    # Output combinations for Z
    # Z00: M1(1) + M4(1) - M5(1) + M7(1)
    # Z01: M3(1) + M5(1)
    # Z10: M2(1) + M4(1)
    # Z11: M1(1) - M2(1) + M3(1) + M6(1)
    strassen_outputs = [
        {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}, # Placeholder, will define below
    ]
    # Let's just map output (i,k) to sum of M_s * coeff
    # Z_00 (i=0, k_local=0): 1*M1 + 1*M4 - 1*M5 + 1*M7
    # Z_01 (i=0, k_local=1): 1*M3 + 1*M5
    # Z_10 (i=1, k_local=0): 1*M2 + 1*M4
    # Z_11 (i=1, k_local=1): 1*M1 - 1*M2 + 1*M3 + 1*M6
    
    output_coeffs = {
        (0, 0): {0: 1, 3: 1, 4: -1, 6: 1}, # M1, M4, M5, M7 (indices 0-6)
        (0, 1): {2: 1, 4: 1},               # M3, M5
        (1, 0): {1: 1, 3: 1},               # M2, M4
        (1, 1): {0: 1, 1: -1, 2: 1, 5: 1}   # M1, M2, M3, M6
    }

    col = 0
    
    # Process Block 1 (Left) and Block 2 (Right)
    for block_id in range(2):
        # Block 0: j in {0,1}, Block 1: j in {2,3}
        j_offset = block_id * 2
        
        # 3 Strassen sub-blocks for output cols 0-5
        for t in range(3):
            k_offset = t * 2
            # 7 Strassen products
            for s in range(7):
                cx, cy = strassen_products[s]
                
                # Fill U (A part)
                for (i, j_local), coeff in cx.items():
                    j_global = j_offset + j_local
                    U[u_idx(i, j_global)][col] = coeff
                
                # Fill V (B part)
                for (j_local, k_local), coeff in cy.items():
                    j_global = j_offset + j_local
                    k_global = k_offset + k_local
                    V[v_idx(j_global, k_global)][col] = coeff
                
                # Fill W (C part)
                # For each output (i, k_global), add contribution
                for (i, k_local), coeffs_dict in output_coeffs.items():
                    if s in coeffs_dict:
                        w_coeff = coeffs_dict[s]
                        k_global = k_offset + k_local
                        W[w_idx(k_global, i)][col] += w_coeff
                        
                col += 1
        
        # 1 Schoolbook sub-block for output col 6
        k_offset = 6
        # 4 products: x00*y06, x01*y16, x10*y06, x11*y16
        schoolbook_products = [
            ({(0,0): 1}, {(0,0): 1}), # x00 y06
            ({(0,1): 1}, {(1,0): 1}), # x01 y16
            ({(1,0): 1}, {(0,0): 1}), # x10 y06
            ({(1,1): 1}, {(1,0): 1})  # x11 y16
        ]
        
        # Outputs:
        # Z_06 (i=0, k=6): x00*y06 + x01*y16 -> 1*P0 + 1*P1
        # Z_16 (i=1, k=6): x10*y06 + x11*y16 -> 1*P2 + 1*P3
        
        for s in range(4):
            cx, cy = schoolbook_products[s]
            
            for (i, j_local), coeff in cx.items():
                j_global = j_offset + j_local
                U[u_idx(i, j_global)][col] = coeff
            
            for (j_local, k_local), coeff in cy.items():
                j_global = j_offset + j_local
                k_global = k_offset + k_local
                V[v_idx(j_global, k_global)][col] = coeff
            
            # Fill W
            if s == 0:
                W[w_idx(k_offset, 0)][col] += 1
            elif s == 1:
                W[w_idx(k_offset, 0)][col] += 1
            elif s == 2:
                W[w_idx(k_offset, 1)][col] += 1
            elif s == 3:
                W[w_idx(k_offset, 1)][col] += 1
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
