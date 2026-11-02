"""
Optimized rank-21 tensor decomposition for 2x2 x 2x6 matrix multiplication.
Uses Strassen's algorithm applied to three 2x2 blocks.
"""

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a rank-21 exact tensor decomposition for multiplying a 2x2 matrix
    by a 2x6 matrix.
    
    The decomposition is based on applying Strassen's algorithm independently
    to three 2x2 blocks of the right-hand matrix.
    """
    n, m, p = DIMENSIONS
    rank = 21
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # Shape (4, 21)
    V = [[0] * rank for _ in range(m * p)]  # Shape (12, 21)
    W = [[0] * rank for _ in range(p * n)]  # Shape (12, 21)
    
    # Strassen coefficients for A (U matrix)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # M1..M7 correspond to k=0..6
    U_coeffs = [
        {0: 1, 3: 1},       # M1: A00 + A11
        {2: 1, 3: 1},       # M2: A10 + A11
        {0: 1},             # M3: A00
        {3: 1},             # M4: A11
        {0: 1, 1: 1},       # M5: A00 + A01
        {2: 1, 0: -1},      # M6: A10 - A00
        {1: 1, 3: -1}       # M7: A01 - A11
    ]
    
    # Strassen coefficients for B (V matrix)
    # Local indices: j in {0,1}, k_local in {0,1}
    # Global k = 2*r + k_local
    # V index = j * p + global_k
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)],      # M1: B00 + B11
        [(0, 0, 1)],                 # M2: B00
        [(0, 1, 1), (1, 1, -1)],     # M3: B01 - B11
        [(1, 0, 1), (0, 0, -1)],     # M4: B10 - B00
        [(1, 1, 1)],                 # M5: B11
        [(0, 0, 1), (0, 1, 1)],      # M6: B00 + B01
        [(1, 0, 1), (1, 1, 1)]       # M7: B10 + B11
    ]
    
    # Strassen coefficients for C (W matrix)
    # Local indices: i in {0,1}, j_local in {0,1}
    # Global k = 2*r + j_local
    # W index = global_k * n + i
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # W_map[k] contains list of (i, j_local, coeff) for the k-th Strassen factor
    # i is row index of C (0 or 1)
    # j_local is column index within the block (0 or 1)
    W_map = [
        [(0, 0, 1), (1, 1, 1)],   # k=0 (M1) -> C00(1), C11(1)
        [(1, 0, 1), (1, 1, -1)],  # k=1 (M2) -> C10(1), C11(-1)
        [(0, 1, 1), (1, 1, 1)],   # k=2 (M3) -> C01(1), C11(1)
        [(0, 0, 1), (1, 0, 1)],   # k=3 (M4) -> C00(1), C10(1)
        [(0, 0, -1), (0, 1, 1)],  # k=4 (M5) -> C00(-1), C01(1)
        [(1, 1, 1)],              # k=5 (M6) -> C11(1)
        [(0, 0, 1)]               # k=6 (M7) -> C00(1)
    ]
    
    for r in range(3):
        for k in range(7):
            col = r * 7 + k
            
            # Fill U
            for idx, val in U_coeffs[k].items():
                U[idx][col] = val
                
            # Fill V
            for j_local, k_local, val in V_coeffs[k]:
                global_k = 2 * r + k_local
                v_idx = j_local * p + global_k
                V[v_idx][col] = val
                
            # Fill W
            for i, j_local, val in W_map[k]:
                global_k = 2 * r + j_local
                w_idx = global_k * n + i
                W[w_idx][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
