import numpy as np

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 2x2 * 2x10 multiplication using block-diagonal Strassen decomposition.
    
    The multiplication $C = AB$ where $A$ is $2 \times 2$, $B$ is $2 \times 10$, and $C$ is $2 \times 10$
    is decomposed into 5 independent $2 \times 2 \times 2$ blocks.
    Each block uses the rank-7 Strassen decomposition.
    Total rank = 5 * 7 = 35.
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 5 blocks
    
    # Strassen rank-7 decomposition terms for 2x2 matrices
    # Each term is (U_entries, V_entries, W_entries)
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
    
    rank = num_blocks * 7  # 35
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for r in range(num_blocks):
        for t in range(7):
            col = r * 7 + t
            u_entries, v_entries, w_entries = strassen_terms[t]
            
            for (i, j), val in u_entries.items():
                U[i * m + j, col] = val
            
            for (j, k), val in v_entries.items():
                V[j * p + (2 * r + k), col] = val
            
            for (k, i), val in w_entries.items():
                W[(2 * r + k) * n + i, col] = val
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
