"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-21 minimal decomposition for M_{2,2,6}.
    
    The matrix multiplication tensor M_{2,2,6} is the direct sum of three
    M_{2,2,2} tensors. Since the rank of M_{2,2,2} is 7 (Strassen), the
    rank of M_{2,2,6} is 3 * 7 = 21. This is the theoretical minimum.
    """
    n, m, p = DIMENSIONS  # 2, 2, 6
    rank = 21
    
    # U: (n*m, rank) = (4, 21)
    # V: (m*p, rank) = (12, 21)
    # W: (p*n, rank) = (12, 21)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for M_{2,2,2}
    # Each entry is a tuple: (U_coeffs, V_local_coeffs, W_coeffs)
    # U_coeffs: dict {A_idx: val} where A_idx in {0,1,2,3}
    # V_local_coeffs: dict {B_local_idx: val} where B_local_idx in {0,1,2,3}
    # W_coeffs: dict {(C_row, C_col_local): val}
    
    # Standard Strassen 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    strassen_products = [
        # 1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
        # 2
        ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
        # 3
        ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
        # 4
        ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
        # 5
        ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
        # 6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
        # 7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
    ]
    
    r = 0
    for b in range(3):
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen_products[t]
            
            # Fill U (same for all blocks, but distinct columns in the rank dimension)
            for idx, val in u_coeffs.items():
                U[idx][r] = val
            
            # Fill V
            # Map local B index (0-3) to global V index
            # B is 2x6. Global index = row * 6 + col.
            # Local 0: B[0, 2b] -> 0*6 + 2b = 2b
            # Local 1: B[0, 2b+1] -> 0*6 + 2b+1 = 2b+1
            # Local 2: B[1, 2b] -> 1*6 + 2b = 6+2b
            # Local 3: B[1, 2b+1] -> 1*6 + 2b+1 = 7+2b
            for l, val in v_coeffs.items():
                if l == 0:
                    v_idx = 2 * b
                elif l == 1:
                    v_idx = 2 * b + 1
                elif l == 2:
                    v_idx = 6 + 2 * b
                elif l == 3:
                    v_idx = 7 + 2 * b
                V[v_idx][r] = val
            
            # Fill W
            # W index is k*n + i.
            # k_global = 2b + k_local
            # i is row index of C (0 or 1)
            for (i, k_local), val in w_coeffs.items():
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W[w_idx][r] = val
            
            r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
