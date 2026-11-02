import time

def construct():
    n, m, p = 2, 2, 12
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for one 2x2x2 block
    # U indices: i*m + j (0..3)
    # V indices: j*p + k (0..23)
    # W indices: k*n + i (0..23)
    
    # Define the 7 terms for one block
    # Each term is (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict { (i,j): val }
    # V_coeffs: dict { (j,k_local): val }
    # W_coeffs: dict { (k_local,i): val }
    
    terms = []
    
    # M1 = (A00 + A11)(B00 + B11)
    # C00 += M1, C11 += M1
    terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    
    # M2 = (A10 + A11)B00
    # C10 += M2, C11 -= M2
    terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(0,1): 1, (1,1): -1}
    })
    
    # M3 = A00(B01 - B11)
    # C01 += M3, C11 += M3
    terms.append({
        'U': {(0,0): 1},
        'V': {(0,1): 1, (1,1): -1},
        'W': {(1,0): 1, (1,1): 1}
    })
    
    # M4 = A11(B10 - B00)
    # C00 += M4, C10 += M4
    terms.append({
        'U': {(1,1): 1},
        'V': {(1,0): 1, (0,0): -1},
        'W': {(0,0): 1, (0,1): 1}
    })
    
    # M5 = (A00 + A01)B11
    # C00 -= M5, C01 += M5
    terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,0): -1, (1,0): 1}
    })
    
    # M6 = (A10 - A00)(B00 + B01)
    # C11 += M6
    terms.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(1,1): 1}
    })
    
    # M7 = (A01 - A11)(B10 + B11)
    # C00 += M7
    terms.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1}
    })
    
    # Fill the matrices
    # There are p/2 = 6 blocks
    num_blocks = p // 2
    for b in range(num_blocks):
        for t_idx in range(7):
            r = b * 7 + t_idx
            term = terms[t_idx]
            
            # U
            for (i, j), val in term['U'].items():
                U[i * m + j][r] = val
            
            # V
            for (j, k_loc), val in term['V'].items():
                k_glob = b * 2 + k_loc
                V[j * p + k_glob][r] = val
            
            # W
            for (k_loc, i), val in term['W'].items():
                k_glob = b * 2 + k_loc
                W[k_glob * n + i][r] = val
                
    return {"U": U, "V": V, "W": W}
