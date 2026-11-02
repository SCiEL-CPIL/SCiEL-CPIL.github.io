"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen steps definitions
    # A coeffs: dict { (i, j_block): coeff } where j_block 0 is left, 1 is right
    # B coeffs: dict { (j_block, k_block): coeff } where j_block 0 is top, 1 is bottom; k_block 0 is left, 1 is right
    # W coeffs: dict { (i, k_block): coeff } where i is row of C, k_block 0 is left, 1 is right
    
    steps = [
        {
            'A': {(0, 0): 1}, # A11
            'B': {(0, 0): 1, (1, 1): 1}, # B11 + B22
            'W': {(0, 0): 1, (1, 1): 1} # C11 + C22
        },
        {
            'A': {(0, 0): 1, (1, 1): 1}, # A11 + A22
            'B': {(1, 1): 1}, # B22
            'W': {(0, 0): -1, (1, 0): 1} # C11 -1, C21 +1
        },
        {
            'A': {(0, 1): 1, (1, 1): 1}, # A12 + A22
            'B': {(1, 0): 1}, # B21
            'W': {(0, 1): 1, (1, 1): 1} # C12 +1, C22 +1
        },
        {
            'A': {(0, 0): 1}, # A11
            'B': {(1, 0): 1, (1, 1): -1}, # B21 - B22
            'W': {(0, 0): 1, (1, 1): -1} # C11 +1, C22 -1
        },
        {
            'A': {(1, 0): 1, (0, 0): 1}, # A21 + A11
            'B': {(0, 0): 1}, # B11
            'W': {(0, 0): 1} # C11 +1
        },
        {
            'A': {(1, 0): 1, (0, 1): -1}, # A21 - A12
            'B': {(0, 0): 1, (0, 1): 1}, # B11 + B12
            'W': {(1, 0): 1} # C21 +1
        },
        {
            'A': {(1, 1): 1, (0, 1): -1}, # A22 - A12
            'B': {(0, 1): 1, (1, 0): 1}, # B12 + B21
            'W': {(0, 1): 1, (1, 1): 1} # C12 +1, C22 +1
        }
    ]
    
    col = 0
    for step in steps:
        A_coeffs = step['A']
        B_coeffs = step['B']
        W_coeffs = step['W']
        
        for r in range(2): # inner index 0, 1
            for kp in range(3): # output index 0, 1, 2
                # Update U
                for (i, j_block), coeff in A_coeffs.items():
                    # j_block 0 -> j in 0,1; j_block 1 -> j in 2,3
                    j_base = 0 if j_block == 0 else 2
                    j_global = j_base + r
                    U[i * m + j_global][col] += coeff
                
                # Update V
                for (j_block, k_block), coeff in B_coeffs.items():
                    # j_block 0 -> j in 0,1; j_block 1 -> j in 2,3
                    j_base = 0 if j_block == 0 else 2
                    j_global = j_base + r
                    
                    # k_block 0 -> k in 0,1,2; k_block 1 -> k in 3,4,5
                    k_base = 0 if k_block == 0 else 3
                    k_global = k_base + kp
                    
                    V[j_global * p + k_global][col] += coeff
                
                # Update W
                for (i, k_block), coeff in W_coeffs.items():
                    # k_block 0 -> k in 0,1,2; k_block 1 -> k in 3,4,5
                    k_base = 0 if k_block == 0 else 3
                    k_global = k_base + kp
                    
                    W[k_global * n + i][col] += coeff
                    
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
