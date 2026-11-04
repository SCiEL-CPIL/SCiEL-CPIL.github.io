import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # M1 = a00(b00+b11)
    # M2 = (a00+a10)b11
    # M3 = (a11+a01)b00
    # M4 = a11(b00-b11)
    # M5 = (a00+a11)(b10+b11)
    # M6 = (a10+a11)(b00-b10)
    # M7 = (a00+a01)(b11-b01)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M4 + M6
    
    # We define the coefficients for each term
    # term_idx: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict {(i,j): coeff}
    # V_coeffs: dict {(j,k): coeff}
    # W_coeffs: dict {(k,i): coeff}
    
    strassen_terms = [
        # M1
        ({(0,0): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(0,0): 1, (1,0): 1}, {(1,1): 1}, {(1,0): 1, (1,1): 1}), # C10 gets M2, C11 gets -M2
        # Wait, W coeffs must reflect the sign in the reconstruction.
        # Let's store the contribution to C_{ik} directly.
        
        # Let's re-structure.
        # For each term r, we have:
        # U_part: linear form on A
        # V_part: linear form on B
        # W_part: linear form on C (coeffs for C_{ik})
        
        # M1: A: a00, B: b00+b11, C: c00 + c11
        # M2: A: a00+a10, B: b11, C: c10 - c11  (since C10=M2, C11=-M2)
        # M3: A: a11+a01, B: b00, C: c01
        # M4: A: a11, B: b00-b11, C: c00 + c11
        # M5: A: a00+a11, B: b10+b11, C: c01 - c00
        # M6: A: a10+a11, B: b00-b10, C: c10 + c11
        # M7: A: a00+a01, B: b11-b01, C: c00
    ]
    
    # Let's verify the W coefficients:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M4 + M6
    
    # So for M1: C00 += 1, C11 += 1. W: {(0,0):1, (1,1):1}
    # For M2: C10 += 1, C11 += -1. W: {(1,0):1, (1,1):-1}
    # For M3: C01 += 1. W: {(0,1):1}
    # For M4: C00 += 1, C11 += 1. W: {(0,0):1, (1,1):1}
    # For M5: C01 += 1, C00 += -1. W: {(0,1):1, (0,0):-1}
    # For M6: C10 += 1, C11 += 1. W: {(1,0):1, (1,1):1}
    # For M7: C00 += 1. W: {(0,0):1}
    
    # U coeffs (on A_{ij}):
    # M1: a00 -> {(0,0):1}
    # M2: a00+a10 -> {(0,0):1, (1,0):1}
    # M3: a11+a01 -> {(1,1):1, (0,1):1}
    # M4: a11 -> {(1,1):1}
    # M5: a00+a11 -> {(0,0):1, (1,1):1}
    # M6: a10+a11 -> {(1,0):1, (1,1):1}
    # M7: a00+a01 -> {(0,0):1, (0,1):1}
    
    # V coeffs (on B_{jk}):
    # M1: b00+b11 -> {(0,0):1, (1,1):1}
    # M2: b11 -> {(1,1):1}
    # M3: b00 -> {(0,0):1}
    # M4: b00-b11 -> {(0,0):1, (1,1):-1}
    # M5: b10+b11 -> {(1,0):1, (1,1):1}
    # M6: b00-b10 -> {(0,0):1, (1,0):-1}
    # M7: b11-b01 -> {(1,1):1, (0,1):-1}
    
    blocks = [
        (0, 0, 0), # i0, j0, k0
        (0, 0, 2),
        (0, 2, 0),
        (0, 2, 2)
    ]
    
    term_data = [
        # (U_dict, V_dict, W_dict)
        ({(0,0):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(0,0):1, (1,0):1}, {(1,1):1}, {(1,0):1, (1,1):-1}),
        ({(1,1):1, (0,1):1}, {(0,0):1}, {(0,1):1}),
        ({(1,1):1}, {(0,0):1, (1,1):-1}, {(0,0):1, (1,1):1}),
        ({(0,0):1, (1,1):1}, {(1,0):1, (1,1):1}, {(0,1):1, (0,0):-1}),
        ({(1,0):1, (1,1):1}, {(0,0):1, (1,0):-1}, {(1,0):1, (1,1):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1, (0,1):-1}, {(0,0):1})
    ]
    
    col = 0
    for i0, j0, k0 in blocks:
        for u_dict, v_dict, w_dict in term_data:
            # Fill U
            for (li, lj), coeff in u_dict.items():
                gi = i0 + li
                gj = j0 + lj
                U[gi * m + gj][col] = coeff
                
            # Fill V
            for (lj, lk), coeff in v_dict.items():
                gj = j0 + lj
                gk = k0 + lk
                V[gj * p + gk][col] = coeff
                
            # Fill W
            for (lk, li), coeff in w_dict.items():
                gk = k0 + lk
                gi = i0 + li
                W[gk * n + gi][col] = coeff
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
