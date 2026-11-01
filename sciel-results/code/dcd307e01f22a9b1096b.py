"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (U) and B (V) for each of the 7 products
    # A indices: 0:00, 1:01, 2:10, 3:11
    # B local indices: 0:00, 1:01, 2:10, 3:11
    # We map local B indices to global V indices based on block b
    
    # t: (U_coeffs, V_local_coeffs, W_coeffs)
    # U_coeffs: dict {idx: val}
    # V_local_coeffs: dict {local_idx: val}
    # W_coeffs: dict {(i, k_local): val} where k_local is 0 or 1
    
    strassen = [
        # t=1: (A00+A11)(B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
        # t=2: (A10+A11)B00
        ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
        # t=3: A00(B01-B11)
        ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
        # t=4: A11(B10-B00)
        ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
        # t=5: (A00+A01)B11
        ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
        # t=6: (A10-A00)(B00+B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
        # t=7: (A01-A11)(B10+B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
    ]
    
    r = 0
    for b in range(3):
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen[t]
            
            # Fill U
            for idx, val in u_coeffs.items():
                U[idx][r] = val
                
            # Fill V
            # Local B index l corresponds to global V index:
            # l=0 (B00) -> j=0, k=2b -> 0*p + 2b = 2b
            # l=1 (B01) -> j=0, k=2b+1 -> 0*p + 2b+1 = 2b+1
            # l=2 (B10) -> j=1, k=2b -> 1*p + 2b = 6+2b
            # l=3 (B11) -> j=1, k=2b+1 -> 1*p + 2b+1 = 7+2b
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
            # W index is k*n + i
            # k_global = 2b + k_local
            for (i, k_local), val in w_coeffs.items():
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W[w_idx][r] = val
                
            r += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
