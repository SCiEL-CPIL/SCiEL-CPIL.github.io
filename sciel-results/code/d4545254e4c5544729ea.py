DIMENSIONS = [2, 2, 16]

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 16
    num_blocks = p // 2  # 8
    rank = num_blocks * 7  # 56
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 56)
    V = [[0] * rank for _ in range(m * p)]  # (32, 56)
    W = [[0] * rank for _ in range(p * n)]  # (32, 56)
    
    # Strassen products: (a_coeffs, b_coeffs, c_coeffs)
    S = [
        # M1: (a00+a11)(b00+b11) -> c00+=1, c11+=1
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        # M2: (a10+a11)b00 -> c10+=1
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1}
        ),
        # M3: a00(b01-b11) -> c01+=1, c11-=1
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,1): 1, (1,1): -1}
        ),
        # M4: a11(b10-b00) -> c00-=1, c10+=1
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): -1, (1,0): 1}
        ),
        # M5: (a00+a01)b11 -> c01+=1
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,1): 1}
        ),
        # M6: (a10-a00)(b00+b01) -> c11+=1
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        # M7: (a01-a11)(b10+b11) -> c00+=1
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        ),
    ]
    
    for q in range(num_blocks):
        for s in range(7):
            col = q * 7 + s
            a_coeffs, b_coeffs, c_coeffs = S[s]
            
            for (i, j), coeff in a_coeffs.items():
                U[i * m + j][col] += coeff
            
            for (j, k_local), coeff in b_coeffs.items():
                k_global = 2 * q + k_local
                V[j * p + k_global][col] += coeff
            
            for (i, k_local), coeff in c_coeffs.items():
                k_global = 2 * q + k_local
                W[k_global * n + i][col] += coeff
    
    return {"U": U, "V": V, "W": W}
