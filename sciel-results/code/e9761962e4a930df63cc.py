"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    A_dim = n * m  # 8
    B_dim = m * p  # 32
    C_dim = p * n  # 16
    
    rank = 56
    U = np.zeros((A_dim, rank), dtype=int)
    V = np.zeros((B_dim, rank), dtype=int)
    W = np.zeros((C_dim, rank), dtype=int)
    
    col = 0
    
    # Define Strassen terms
    # Each term is a tuple of (coeffs_A, coeffs_B, coeffs_C)
    # coeffs_A is dict or list of (index, coeff)
    # We use local 2x2 indices (r, c) -> idx = r*2 + c
    
    strassen_terms = [
        # M1
        {
            'A': {0: 1, 3: 1}, # a00 + a11
            'B': {0: 1, 3: 1}, # b00 + b11
            'C': {0: 1, 3: 1}  # c00 + c11
        },
        # M2
        {
            'A': {2: 1, 3: 1}, # a10 + a11
            'B': {0: 1},       # b00
            'C': {2: 1, 3: -1} # c10 - c11
        },
        # M3
        {
            'A': {0: 1},       # a00
            'B': {1: 1, 3: -1},# b01 - b11
            'C': {1: 1, 3: 1}  # c01 + c11
        },
        # M4
        {
            'A': {3: 1},       # a11
            'B': {2: 1, 0: -1},# b10 - b00
            'C': {0: 1, 2: 1}  # c00 + c10
        },
        # M5
        {
            'A': {0: 1, 1: 1}, # a00 + a01
            'B': {3: 1},       # b11
            'C': {0: -1, 1: 1} # -c00 + c01
        },
        # M6
        {
            'A': {2: 1, 0: -1},# a10 - a00
            'B': {0: 1, 1: 1}, # b00 + b01
            'C': {3: 1}        # c11
        },
        # M7
        {
            'A': {1: 1, 3: -1},# a01 - a11
            'B': {2: 1, 3: 1}, # b10 + b11
            'C': {0: 1}        # c00
        }
    ]
    
    for part in range(2):
        for block_m in range(4):
            for term in strassen_terms:
                # Map local indices to global indices
                
                # A mapping
                # part 0: j in {0,1}
                # part 1: j in {2,3}
                # local r, c -> global i, j
                # global u = i * m + j
                
                # B mapping
                # part 0: j in {0,1}
                # part 1: j in {2,3}
                # local r, c -> global j, k
                # global v = j * p + k
                # k = block_m * 2 + local_c
                
                # C mapping
                # local r, c -> global i, k
                # global w = k * n + i
                # k = block_m * 2 + local_c
                
                for local_idx, coeff_A in term['A'].items():
                    r, c = divmod(local_idx, 2)
                    global_i = r
                    global_j = (2 if part == 1 else 0) + c
                    global_u = global_i * m + global_j
                    U[global_u, col] += coeff_A
                
                for local_idx, coeff_B in term['B'].items():
                    r, c = divmod(local_idx, 2)
                    global_j = (2 if part == 1 else 0) + r
                    global_k = block_m * 2 + c
                    global_v = global_j * p + global_k
                    V[global_v, col] += coeff_B
                
                for local_idx, coeff_C in term['C'].items():
                    r, c = divmod(local_idx, 2)
                    global_i = r
                    global_k = block_m * 2 + c
                    global_w = global_k * n + global_i
                    W[global_w, col] += coeff_C
                
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
