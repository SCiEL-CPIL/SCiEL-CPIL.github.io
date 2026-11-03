import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 42 using Strassen for 2x2x2 subproblems
    # Decomposition:
    # 1. Split m=4 into two m'=2 blocks (s=0,1)
    # 2. Split p=6 into three p'=2 blocks (l=0,1,2)
    # 3. Each 2x2x2 block uses Strassen (rank 7)
    # Total rank = 2 * 3 * 7 = 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (U), B (V), C (W)
    # q=0..6 correspond to M1..M7
    # U coeffs: [Y00, Y01, Y10, Y11]
    # V coeffs: [Z00, Z01, Z10, Z11]
    # W coeffs: [X00, X01, X10, X11]
    
    strassen_U = [
        [1, 0, 0, 1], # M1: Y00, Y11
        [0, 0, 1, 1], # M2: Y10, Y11
        [1, 0, 0, 0], # M3: Y00
        [0, 0, 0, 1], # M4: Y11
        [1, 1, 0, 0], # M5: Y00, Y01
        [-1, 0, 1, 0], # M6: Y10, Y00 (-1)
        [0, 1, 0, -1]  # M7: Y01, Y11 (-1)
    ]
    
    strassen_V = [
        [1, 0, 0, 1], # M1: Z00, Z11
        [1, 0, 0, 0], # M2: Z00
        [0, 1, 0, -1], # M3: Z01, Z11 (-1)
        [1, -1, 0, 0], # M4: Z00, Z01 (-1)
        [0, 0, 0, 1], # M5: Z11
        [1, 1, 0, 0], # M6: Z00, Z01
        [0, 0, 1, 1]  # M7: Z10, Z11
    ]
    
    strassen_W = [
        [1, 0, 0, 1], # M1: X00, X11
        [0, 0, 1, -1], # M2: X10, X11 (-1)
        [0, 1, 0, 1], # M3: X01, X11
        [1, 0, 1, 0], # M4: X00, X10
        [-1, 1, 0, 0], # M5: X00 (-1), X01
        [0, 0, 0, 1], # M6: X11
        [1, 0, 0, 0]  # M7: X00
    ]
    
    # Local to Global index mappings
    # A (U): i in {0,1}, j_loc in {0,1} -> global j = 2*s + j_loc
    # U row index = i * m + j = i * 4 + 2*s + j_loc
    # B (V): j_loc in {0,1}, k_loc in {0,1} -> global j = 2*s + j_loc, global k = 2*l + k_loc
    # V row index = j * p + k = (2*s + j_loc) * 6 + (2*l + k_loc)
    # C (W): i in {0,1}, k_loc in {0,1} -> global k = 2*l + k_loc
    # W row index = k * n + i = (2*l + k_loc) * 2 + i
    
    col = 0
    for s in range(2): # 0, 1
        for l in range(3): # 0, 1, 2
            for q in range(7): # 0..6
                # U updates
                for i in range(2):
                    for j_loc in range(2):
                        coeff = strassen_U[q][j_loc] # Wait, strassen_U is [Y00, Y01, Y10, Y11]
                        # My strassen_U list is indexed by q, then by variable index 0..3
                        # Variable 0: Y00 (i=0, j_loc=0)
                        # Variable 1: Y01 (i=0, j_loc=1)
                        # Variable 2: Y10 (i=1, j_loc=0)
                        # Variable 3: Y11 (i=1, j_loc=1)
                        
                        # Need to map variable index to i, j_loc
                        if j_loc == 0:
                             if i == 0: var_idx = 0
                             else: var_idx = 2
                        else:
                             if i == 0: var_idx = 1
                             else: var_idx = 3
                        
                        val = strassen_U[q][var_idx]
                        if val != 0:
                            u_row = i * m + (2 * s + j_loc)
                            U[u_row][col] += val
                
                # V updates
                for j_loc in range(2):
                    for k_loc in range(2):
                        # Map variable index
                        if j_loc == 0:
                             if k_loc == 0: var_idx = 0 # Z00
                             else: var_idx = 1 # Z01
                        else:
                             if k_loc == 0: var_idx = 2 # Z10
                             else: var_idx = 3 # Z11
                             
                        val = strassen_V[q][var_idx]
                        if val != 0:
                            v_row = (2 * s + j_loc) * p + (2 * l + k_loc)
                            V[v_row][col] += val
                
                # W updates
                for i in range(2):
                    for k_loc in range(2):
                        # Map variable index
                        if k_loc == 0:
                             if i == 0: var_idx = 0 # X00
                             else: var_idx = 2 # X10
                        else:
                             if i == 0: var_idx = 1 # X01
                             else: var_idx = 3 # X11
                             
                        val = strassen_W[q][var_idx]
                        if val != 0:
                            w_row = (2 * l + k_loc) * n + i
                            W[w_row][col] += val
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
