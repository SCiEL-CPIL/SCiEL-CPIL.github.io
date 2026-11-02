import numpy as np

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 35 (5 blocks of 2x2x2, each rank 7)
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrices
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    
    # Products P1..P7
    # P1 = (a00+a11)(b00+b10)
    # P2 = (a00+a11)b01
    # P3 = a00(b11-b10)
    # P4 = a11(b00-b01)
    # P5 = (a00+a01)b11
    # P6 = (a10-a00)(b00+b01)
    # P7 = (a01-a11)(b10+b11)
    
    # A coefficients (4x7)
    A_coeffs = [
        [1, 1, 1, 0, 1, -1, 0], # a00
        [0, 0, 0, 0, 1, 0, -1], # a01
        [0, 0, 0, 0, 0, 1, 0],  # a10
        [1, 1, 0, 1, 0, 0, -1]  # a11
    ]
    
    # B coefficients for one block (4x7)
    B_coeffs = [
        [1, 0, 0, 1, 0, 1, 0], # b00
        [0, 1, 0, -1, 0, 1, 0],# b01
        [1, 0, -1, 0, 0, 0, 1],# b10
        [0, 0, 1, 0, 1, 0, 1]  # b11
    ]
    
    # Output coefficients (4x7) for C
    # C00 = P1 + P4 - P2 + P6
    # C01 = P2 + P5
    # C10 = P3 + P7
    # C11 = P1 - P3 + P5 - P7
    C_coeffs = [
        [1, -1, 0, 1, 0, 1, 0], # C00
        [0, 1, 0, 0, 1, 0, 0],  # C01
        [0, 0, 1, 0, 0, 0, 1],  # C10
        [1, 0, -1, 0, 1, 0, -1] # C11
    ]
    
    # Construct U, V, W
    # There are 5 blocks of B (since p=10, block size 2)
    for t in range(5):
        for i in range(7):
            col = t * 7 + i
            
            # U: A coefficients
            for r in range(4):
                U[r][col] = A_coeffs[r][i]
                
            # V: B coefficients
            # Block t corresponds to B columns 2*t and 2*t+1
            # B entries are indexed as [row * p + col]
            # b00 is B[0, 2*t] -> index 2*t
            # b01 is B[0, 2*t+1] -> index 2*t+1
            # b10 is B[1, 2*t] -> index p + 2*t
            # b11 is B[1, 2*t+1] -> index p + 2*t+1
            
            b_indices = [2*t, 2*t+1, p + 2*t, p + 2*t+1]
            for r in range(4):
                V[b_indices[r]][col] = B_coeffs[r][i]
                
            # W: C coefficients
            # C entries are indexed as [col * n + row] according to W[k*n+i, r]
            # C00 is C[0, 2*t] -> index 2*t * n + 0
            # C01 is C[0, 2*t+1] -> index 2*t+1 * n + 0
            # C10 is C[1, 2*t] -> index 2*t * n + 1
            # C11 is C[1, 2*t+1] -> index 2*t+1 * n + 1
            
            c_indices = [
                (2*t) * n + 0,
                (2*t+1) * n + 0,
                (2*t) * n + 1,
                (2*t+1) * n + 1
            ]
            for r in range(4):
                W[c_indices[r]][col] = C_coeffs[r][i]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
