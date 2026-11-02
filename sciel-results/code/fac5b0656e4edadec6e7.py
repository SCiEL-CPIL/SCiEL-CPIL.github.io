"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (2x2)
    # u indices: 0:A00, 1:A01, 2:A10, 3:A11
    A_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    # Strassen coefficients for B (2x2)
    # Local indices: 0:B00, 1:B01, 2:B10, 3:B11
    B_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 0, 1, -1],# M3
        [-1, 1, 0, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # Strassen coefficients for C (2x2)
    # Local indices: 0:C00, 1:C01, 2:C10, 3:C11
    C_coeffs = [
        [1, 0, 1, 0], # M1 contributes to C00, C11? 
        # Let's re-verify C mapping from M's
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 - M6
        # M1: C00=1, C11=1
        # M2: C10=1
        # M3: C01=1, C11=1
        # M4: C00=1, C10=1
        # M5: C00=-1, C01=1
        # M6: C11=-1
        # M7: C00=1, C01=-1
        [1, 0, 0, 1], # M1
        [0, 0, 1, 0], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, -1],# M6
        [1, -1, 0, 0] # M7
    ]
    
    # Block 1: k in {0, 1}
    # B indices in V: 0:B00, 1:B01, 4:B10, 5:B11
    # C indices in W: 0:C00, 1:C01, 2:C10, 3:C11
    for t in range(7):
        col = t
        # U
        for i in range(4):
            U[i][col] = A_coeffs[t][i]
        # V
        # Map local B index 0->0, 1->1, 2->4, 3->5
        v_map = [0, 1, 4, 5]
        for local_j, val in enumerate(B_coeffs[t]):
            if val != 0:
                V[v_map[local_j]][col] = val
        # W
        # Map local C index 0->0, 1->1, 2->2, 3->3
        w_map = [0, 1, 2, 3]
        for local_k, val in enumerate(C_coeffs[t]):
            if val != 0:
                W[w_map[local_k]][col] = val

    # Block 2: k in {2, 3}
    # B indices in V: 2:B02, 3:B03, 6:B12, 7:B13
    # C indices in W: 4:C02, 5:C03, 6:C12, 7:C13
    for t in range(7):
        col = t + 7
        # U
        for i in range(4):
            U[i][col] = A_coeffs[t][i]
        # V
        # Map local B index 0->2, 1->3, 2->6, 3->7
        v_map = [2, 3, 6, 7]
        for local_j, val in enumerate(B_coeffs[t]):
            if val != 0:
                V[v_map[local_j]][col] = val
        # W
        # Map local C index 0->4, 1->5, 2->6, 3->7
        w_map = [4, 5, 6, 7]
        for local_k, val in enumerate(C_coeffs[t]):
            if val != 0:
                W[w_map[local_k]][col] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
