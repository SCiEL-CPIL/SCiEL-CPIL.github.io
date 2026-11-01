"""Program for matrix multiplication 2,2,10 using block-diagonal Strassen decomposition."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-35 decomposition for 2x2 * 2x10 matrix multiplication.
    This uses the direct sum of five 2x2 * 2x2 Strassen decompositions.
    """
    n, m, p = DIMENSIONS
    rank = 35
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 Strassen terms for 2x2 multiplication.
    # Each term is a tuple (u_dict, v_dict, w_dict) where keys are (i,j), (j,k), (k,i)
    # and values are the coefficients.
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    strassen_terms = [
        # M1 = (a+d)(e+h)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (c+d)e
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}),
        # M3 = a(f-h)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
        # M4 = d(g-e)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        # M5 = (a+b)h
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1}),
        # M6 = (c-a)(e+f)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (b-d)(g+h)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # Fill the matrices for each of the 5 blocks
    for block_idx in range(p // 2):
        for t_idx in range(7):
            col = block_idx * 7 + t_idx
            u_dict, v_dict, w_dict = strassen_terms[t_idx]
            
            # Fill U
            for (i, j), val in u_dict.items():
                U[i * m + j][col] = val
            
            # Fill V
            for (j, k), val in v_dict.items():
                V[j * p + (2 * block_idx + k)][col] = val
            
            # Fill W
            for (k, i), val in w_dict.items():
                W[(2 * block_idx + k) * n + i][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
