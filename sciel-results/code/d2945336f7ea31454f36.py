import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 11 terms for a 2x3x2 block
    # Each term is (u_coeffs, v_coeffs, w_coeffs)
    # u: dict {(i, j): coeff}
    # v: dict {(j, k_loc): coeff}
    # w: dict {(k_loc, i): coeff}
    
    terms = []
    
    # Term 1: (a11 + a22)(b11 + b22)
    terms.append((
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1, (1, 1): 1}
    ))
    
    # Term 2: (a21 + a22)b11
    terms.append((
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 0): 1, (1, 1): -1}
    ))
    
    # Term 3: a11(b21 - b22)
    terms.append((
        {(0, 0): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1}
    ))
    
    # Term 4: a22(b12 - b11)
    terms.append((
        {(1, 1): 1},
        {(0, 1): 1, (0, 0): -1},
        {(0, 0): 1, (1, 0): 1}
    ))
    
    # Term 5: (a11 + a12)b22
    terms.append((
        {(0, 0): 1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): -1, (0, 1): 1}
    ))
    
    # Term 6: (a21 - a11)(b11 + b12)
    terms.append((
        {(1, 0): 1, (0, 0): -1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 1): 1}
    ))
    
    # Term 7: (a12 - a22)(b21 + b22)
    terms.append((
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1}
    ))
    
    # Term 8: a13 b31
    terms.append((
        {(0, 2): 1},
        {(2, 0): 1},
        {(0, 0): 1}
    ))
    
    # Term 9: a13 b32
    terms.append((
        {(0, 2): 1},
        {(2, 1): 1},
        {(0, 1): 1}
    ))
    
    # Term 10: a23 b31
    terms.append((
        {(1, 2): 1},
        {(2, 0): 1},
        {(1, 0): 1}
    ))
    
    # Term 11: a23 b32
    terms.append((
        {(1, 2): 1},
        {(2, 1): 1},
        {(1, 1): 1}
    ))
    
    col = 0
    for b in range(5):
        for term in terms:
            u_coeffs, v_coeffs, w_coeffs = term
            for (i, j), val in u_coeffs.items():
                U[i * m + j][col] += val
            for (j, k_loc), val in v_coeffs.items():
                k_global = 2 * b + k_loc
                V[j * p + k_global][col] += val
            for (k_loc, i), val in w_coeffs.items():
                k_global = 2 * b + k_loc
                W[k_global * n + i][col] += val
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
