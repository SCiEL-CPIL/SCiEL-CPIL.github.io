import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 terms
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # Terms:
    # 1: (a+d)(e+f)
    # 2: (b+c)(e+f)
    # 3: (a-b)(h-g)
    # 4: (c+d)(g+h)
    # 5: (a+c)(e-h)
    # 6: (b+d)(f-h)
    # 7: (b-c)(a+d) -- I will use this, it's a common variant.
    
    strassen_terms = [
        ([1, 0, 0, 1], [1, 1, 0, 0], [1, 1, 1, 1]),
        ([0, 1, 1, 0], [1, 1, 0, 0], [0, 1, 1, 0]),
        ([1, -1, 0, 0], [0, 0, -1, 1], [0, 0, 1, 1]),
        ([0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]),
        ([1, 0, 1, 0], [1, 0, 0, -1], [1, 0, 0, 0]),
        ([0, 1, 0, 1], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 1, -1, 0], [1, 0, 0, 1], [1, 0, 0, 0]), # (b-c)(a+d)
    ]
    
    col = 0
    for block in range(3):
        for sub_block in range(2):
            for term in strassen_terms:
                U_coeffs, V_coeffs, W_coeffs = term
                
                # Map A indices
                # i in 0,1. j in 2*sub_block, 2*sub_block+1
                for i in range(2):
                    for j_local in range(2):
                        j = 2 * sub_block + j_local
                        a_idx = i * 4 + j
                        coeff = U_coeffs[i * 2 + j_local]
                        if coeff != 0:
                            U[a_idx][col] = coeff
                            
                # Map B indices
                # j in 2*sub_block, 2*sub_block+1. k in 2*block, 2*block+1
                for j_local in range(2):
                    j = 2 * sub_block + j_local
                    for k_local in range(2):
                        k = 2 * block + k_local
                        b_idx = j * 6 + k
                        coeff = V_coeffs[j_local * 2 + k_local]
                        if coeff != 0:
                            V[b_idx][col] = coeff
                            
                # Map C indices
                # i in 0,1. k in 2*block, 2*block+1
                for i in range(2):
                    for k_local in range(2):
                        k = 2 * block + k_local
                        c_idx = k * 2 + i
                        coeff = W_coeffs[i * 2 + k_local]
                        if coeff != 0:
                            W[c_idx][col] = coeff
                            
                col += 1
                
    return {"U": U, "V": V, "W": W}
