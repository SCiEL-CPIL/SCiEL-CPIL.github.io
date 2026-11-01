"""Rewritten program for 2x2x8 matrix multiplication using Strassen's algorithm on blocks."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using Strassen's algorithm for each 2x2 block of B.
    # p=8, m=2 => 4 blocks of size 2x2.
    # Each block requires rank 7.
    # Total rank = 4 * 7 = 28.
    rank = 28
    
    # Initialize U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for U (A indices: 00, 01, 10, 11 -> 0, 1, 2, 3)
    U_coeffs = [
        {0: 1, 3: 1},       # M1: A00 + A11
        {2: 1, 3: 1},       # M2: A10 + A11
        {0: 1},             # M3: A00
        {3: 1},             # M4: A11
        {0: 1, 1: 1},       # M5: A00 + A01
        {2: 1, 0: -1},      # M6: A10 - A00
        {1: 1, 3: -1}       # M7: A01 - A11
    ]
    
    # Strassen coefficients for V (B indices, local to block: 00, 01, 10, 11 -> 0, 1, 2, 3)
    V_coeffs = [
        {0: 1, 3: 1},       # M1: B00 + B11
        {0: 1},             # M2: B00
        {1: 1, 3: -1},      # M3: B01 - B11
        {2: 1, 0: -1},      # M4: B10 - B00
        {3: 1},             # M5: B11
        {0: 1, 1: 1},       # M6: B00 + B01
        {2: 1, 3: 1}        # M7: B10 + B11
    ]
    
    # Strassen coefficients for W (C indices, local to block: 00, 10, 01, 11 -> 0, 1, 2, 3)
    # Note: W index is k*n + i. Local C index 0->C00, 1->C10, 2->C01, 3->C11.
    W_coeffs = [
        {0: 1, 3: 1},       # M1: C00 + C11
        {1: 1, 3: -1},      # M2: C10 - C11
        {2: 1, 3: 1},       # M3: C01 + C11
        {0: 1, 1: 1},       # M4: C00 + C10
        {0: -1, 2: 1},      # M5: -C00 + C01
        {3: 1},             # M6: C11
        {0: 1}              # M7: C00
    ]

    # There are 4 blocks in B (since p=8, m=2, 8/2=4)
    for q in range(4):
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
