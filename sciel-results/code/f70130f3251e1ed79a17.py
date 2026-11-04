"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 84
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    products = [
        (0, 0, 0, 0), 
        (0, 0, 0, 1), 
        (0, 0, 0, 2), 
        (0, 1, 1, 0), 
        (0, 1, 1, 1), 
        (0, 1, 1, 2), 
        (1, 0, 0, 0), 
        (1, 0, 0, 1), 
        (1, 0, 0, 2), 
        (1, 1, 1, 0), 
        (1, 1, 1, 1), 
        (1, 1, 1, 2)  
    ]
    
    strassen_U = [
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0],
        [0, 1, 0, -1] 
    ]
    
    strassen_V = [
        [1, 0, 0, 1], 
        [1, 0, 0, 0], 
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, 1]  
    ]
    
    strassen_W_coeffs = [
        [1, 0, 0, 1], 
        [0, 0, 1, -1],
        [0, 1, 0, 1], 
        [1, 0, 1, 0], 
        [-1, 1, 0, 0],
        [0, 0, 0, 1], 
        [1, 0, 0, 0]  
    ]
    
    for p_idx, (a_br, a_bc, b_br, b_bc) in enumerate(products):
        for s in range(7):
            col = p_idx * 7 + s
            
            for a_r in range(2):
                for a_c in range(2):
                    coeff = strassen_U[s][a_r * 2 + a_c]
                    if coeff != 0:
                        i = a_br * 2 + a_r
                        j = a_bc * 2 + a_c
                        U[i * m + j][col] += coeff
            
            for b_r in range(2):
                for b_c in range(2):
                    coeff = strassen_V[s][b_r * 2 + b_c]
                    if coeff != 0:
                        j = b_br * 2 + b_r
                        k = b_bc * 2 + b_c
                        V[j * p + k][col] += coeff
            
            for c_r in range(2):
                for c_c in range(2):
                    coeff = strassen_W_coeffs[s][c_r * 2 + c_c]
                    if coeff != 0:
                        i = a_br * 2 + c_r
                        k = b_bc * 2 + c_c
                        W[k * n + i][col] += coeff
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
