"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-c)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    # Indices for U (A matrix 2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for V (B matrix 2x10): j*10 + k
    # Indices for W (C matrix 10x2): k*2 + i
    
    # Define the 7 terms for one block
    # Each term is a tuple of (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict {index: val}
    # V_coeffs: dict {index_offset: val} (relative to block start in k)
    # W_coeffs: dict {index_offset: val} (relative to block start in k)
    
    terms = [
        # P1
        ({0: 1, 3: 1}, {0: 1, 11: 1}, {0: 1, 5: 1}), 
        # P2
        ({2: 1, 3: 1}, {0: 1}, {1: 1, 5: -1}), 
        # P3
        ({0: 1}, {1: 1, 11: -1}, {4: 1, 5: 1}), 
        # P4
        ({3: 1, 2: -1}, {10: 1}, {0: 1, 1: 1}), 
        # P5
        ({0: 1, 1: 1}, {11: 1}, {0: -1, 4: 1}), 
        # P6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {5: 1}), 
        # P7
        ({1: 1, 3: -1}, {10: 1, 11: 1}, {0: 1})
    ]
    
    for q in range(5):
        base_col = q * 7
        k_start = 2 * q
        
        for s in range(7):
            col = base_col + s
            u_coeffs, v_coeffs, w_coeffs = terms[s]
            
            for idx, val in u_coeffs.items():
                U[idx][col] = val
                
            for idx_offset, val in v_coeffs.items():
                # idx_offset is relative to j*10 + k_start
                # We need to map the local B indices to global V indices
                # Local B indices used in Strassen:
                # e: (0, 0) -> global (0, k_start) -> V[0*10 + k_start]
                # f: (0, 1) -> global (0, k_start+1) -> V[0*10 + k_start+1]
                # g: (1, 0) -> global (1, k_start) -> V[1*10 + k_start]
                # h: (1, 1) -> global (1, k_start+1) -> V[1*10 + k_start+1]
                
                # My v_coeffs keys were:
                # 0: e (0,0)
                # 1: f (0,1)
                # 10: g (1,0)
                # 11: h (1,1)
                
                # Let's verify the mapping in the list above.
                # P1: e (0), h (11). Correct.
                # P2: e (0). Correct.
                # P3: f (1), h (11). Correct.
                # P4: g (10). Correct.
                # P5: h (11). Correct.
                # P6: e (0), f (1). Correct.
                # P7: g (10), h (11). Correct.
                
                # The keys are effectively:
                # 0 -> j=0, k=k_start
                # 1 -> j=0, k=k_start+1
                # 10 -> j=1, k=k_start
                # 11 -> j=1, k=k_start+1
                
                # So global V index = (j * 10) + (k_start + k_offset)
                # From key k:
                # if k < 10: j=0, k_off = k
                # else: j=1, k_off = k - 10
                
                j = 0
                k_off = idx_offset
                if idx_offset >= 10:
                    j = 1
                    k_off = idx_offset - 10
                    
                global_v_idx = j * 10 + (k_start + k_off)
                V[global_v_idx][col] = val
                
            for idx_offset, val in w_coeffs.items():
                # W indices are k*2 + i
                # Local C indices in Strassen:
                # c11: i=0, k=0 -> W[0]
                # c12: i=0, k=1 -> W[1]
                # c21: i=1, k=0 -> W[2]
                # c22: i=1, k=1 -> W[3]
                
                # My w_coeffs keys were:
                # 0: c11 (k=0, i=0)
                # 1: c21 (k=0, i=1) -- Wait, let's check P2 W coeffs.
                # P2: c21 (+1), c22 (-1).
                # c21 is i=1, k=0. Global W index: (k_start + 0)*2 + 1.
                # c22 is i=1, k=1. Global W index: (k_start + 1)*2 + 1.
                
                # Let's re-verify W key mapping in the list.
                # P1: c11 (0), c22 (5).
                # c11: k=0, i=0. Key 0.
                # c22: k=1, i=1. Key 5?
                # If key = k*2 + i:
                # c11: 0*2+0=0.
                # c12: 1*2+0=2.
                # c21: 0*2+1=1.
                # c22: 1*2+1=3.
                
                # My previous list used:
                # P1: {0: 1, 5: 1}. If 5 is c22, then mapping is not k*2+i directly?
                # Let's check P1 again.
                # c11 = P1... -> W[0,0] = 1.
                # c22 = P1... -> W[1,1] = 1.
                # If keys are 0 and 5, and 5 corresponds to c22...
                # Maybe I used a different local indexing?
                
                # Let's stick to standard local indices:
                # 0: (k=0, i=0)
                # 1: (k=0, i=1)
                # 2: (k=1, i=0)
                # 3: (k=1, i=1)
                
                # Re-evaluating W coeffs with this standard mapping:
                # P1: c11(0), c22(3). -> {0: 1, 3: 1}
                # P2: c21(1), c22(3). -> {1: 1, 3: -1}
                # P3: c12(2), c22(3). -> {2: 1, 3: 1}
                # P4: c11(0), c21(1). -> {0: 1, 1: 1}
                # P5: c11(0), c12(2). -> {0: -1, 2: 1}
                # P6: c22(3). -> {3: 1}
                # P7: c11(0). -> {0: 1}
                
                # I will regenerate the terms list with these correct keys.
                
                pass # Placeholder, will define terms correctly below

    # Redefining terms with correct W keys (0:c11, 1:c21, 2:c12, 3:c22)
    terms = [
        # P1: (a+d)(e+h) -> c11(+), c22(+)
        ({0: 1, 3: 1}, {0: 1, 11: 1}, {0: 1, 3: 1}), 
        # P2: (c+d)e -> c21(+), c22(-)
        ({2: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}), 
        # P3: a(f-h) -> c12(+), c22(+)
        ({0: 1}, {1: 1, 11: -1}, {2: 1, 3: 1}), 
        # P4: d(g-c) -> c11(+), c21(+)
        ({3: 1, 2: -1}, {10: 1}, {0: 1, 1: 1}), 
        # P5: (a+b)h -> c11(-), c12(+)
        ({0: 1, 1: 1}, {11: 1}, {0: -1, 2: 1}), 
        # P6: (c-a)(e+f) -> c22(+)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}), 
        # P7: (b-d)(g+h) -> c11(+)
        ({1: 1, 3: -1}, {10: 1, 11: 1}, {0: 1})
    ]
    
    for q in range(5):
        base_col = q * 7
        k_start = 2 * q
        
        for s in range(7):
            col = base_col + s
            u_coeffs, v_coeffs, w_coeffs = terms[s]
            
            for idx, val in u_coeffs.items():
                U[idx][col] = val
                
            for idx_offset, val in v_coeffs.items():
                j = 0
                k_off = idx_offset
                if idx_offset >= 10:
                    j = 1
                    k_off = idx_offset - 10
                    
                global_v_idx = j * 10 + (k_start + k_off)
                V[global_v_idx][col] = val
                
            for idx_offset, val in w_coeffs.items():
                # idx_offset is 0,1,2,3 corresponding to (k=0,i=0), (k=0,i=1), (k=1,i=0), (k=1,i=1)
                k_off = idx_offset // 2
                i_off = idx_offset % 2
                
                global_w_idx = (k_start + k_off) * 2 + i_off
                W[global_w_idx][col] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
