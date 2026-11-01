"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use Strassen's algorithm for each 2x2 block of B.
    # There are 4 such blocks (since p=8, m=2, so 8/2 = 4 blocks).
    # Each block uses rank 7. Total rank = 4 * 7 = 28.
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for U (A indices)
    # A indices: 00->0, 01->1, 10->2, 11->3
    # M1: A00, A11
    # M2: A10, A11
    # M3: A00
    # M4: A11
    # M5: A00, A01
    # M6: A10, -A00
    # M7: A01, -A11
    
    U_coeffs = [
        {0: 1, 3: 1},       # M1
        {2: 1, 3: 1},       # M2
        {0: 1},             # M3
        {3: 1},             # M4
        {0: 1, 1: 1},       # M5
        {2: 1, 0: -1},      # M6
        {1: 1, 3: -1}       # M7
    ]
    
    # Strassen coefficients for V (B indices, local to block)
    # Local B indices: 00->0, 01->1, 10->2, 11->3
    # M1: B00, B11
    # M2: B00
    # M3: B01, -B11
    # M4: B10, -B00
    # M5: B11
    # M6: B00, B01
    # M7: B10, B11
    
    V_coeffs = [
        {0: 1, 3: 1},       # M1
        {0: 1},             # M2
        {1: 1, 3: -1},      # M3
        {2: 1, 0: -1},      # M4
        {3: 1},             # M5
        {0: 1, 1: 1},       # M6
        {2: 1, 3: 1}        # M7
    ]
    
    # Strassen coefficients for W (C indices, local to block)
    # Local C indices: 00->0, 10->1, 01->2, 11->3
    # M1: C00, C11
    # M2: C10, -C11
    # M3: C01, C11
    # M4: C00, C10
    # M5: -C00, C01
    # M6: C11
    # M7: C00
    
    W_coeffs = [
        {0: 1, 3: 1},       # M1
        {1: 1, 3: -1},      # M2
        {2: 1, 3: 1},       # M3
        {0: 1, 1: 1},       # M4
        {0: -1, 2: 1},      # M5
        {3: 1},             # M6
        {0: 1}              # M7
    ]

    for q in range(4):
        # q is the block index (0, 1, 2, 3)
        # Columns in B for this block: 2*q, 2*q+1
        # Rows in C for this block: 2*q, 2*q+1
        
        for s in range(7):
            r = q * 7 + s
            
            # Fill U
            for u_idx, val in U_coeffs[s].items():
                U[u_idx][r] = val
                
            # Fill V
            # Local B index 0 -> j=0, k=2*q
            # Local B index 1 -> j=0, k=2*q+1
            # Local B index 2 -> j=1, k=2*q
            # Local B index 3 -> j=1, k=2*q+1
            for v_local_idx, val in V_coeffs[s].items():
                j = v_local_idx // 2
                k_local = v_local_idx % 2
                k = 2 * q + k_local
                v_idx = j * p + k
                V[v_idx][r] = val
                
            # Fill W
            # Local C index 0 -> i=0, k=2*q
            # Local C index 1 -> i=1, k=2*q
            # Local C index 2 -> i=0, k=2*q+1
            # Local C index 3 -> i=1, k=2*q+1
            for w_local_idx, val in W_coeffs[s].items():
                i = w_local_idx % 2
                k_local = w_local_idx // 2
                k = 2 * q + k_local
                w_idx = k * n + i
                W[w_idx][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
