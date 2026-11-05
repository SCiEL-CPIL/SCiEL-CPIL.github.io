"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank is 88
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # Each term is (A, B, C)
    # A, B, C are 2x2 matrices represented as list of lists
    strassen_terms = [
        # M1
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
        # M2
        ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, -1]]),
        # M3
        ([[1, 0], [0, 0]], [[0, 1], [0, -1]], [[0, 1], [0, 1]]),
        # M4
        ([[0, 0], [0, 1]], [[-1, 0], [1, 0]], [[1, 0], [1, 0]]),
        # M5
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[-1, 1], [0, 0]]),
        # M6
        ([[-1, 0], [1, 0]], [[1, 1], [0, 0]], [[0, 0], [0, 1]]),
        # M7
        ([[0, 1], [-1, 0]], [[0, 0], [1, 1]], [[1, 0], [0, 0]]),
    ]
    
    # 8 Strassen blocks
    # i_base is always 0 (covering i=0,1)
    # j_base is 0 or 2
    # k_base is 0, 2, 4, 6
    i_base = 0
    j_bases = [0, 2]
    k_bases = [0, 2, 4, 6]
    
    for j_b in j_bases:
        for k_b in k_bases:
            for A, B, C in strassen_terms:
                # Update U
                for u in range(2):
                    for v in range(2):
                        coeff = A[u][v]
                        if coeff != 0:
                            row_u = i_base * m + (j_b + v)
                            U[row_u][col] = coeff
                            
                # Update V
                for v in range(2):
                    for w in range(2):
                        coeff = B[v][w]
                        if coeff != 0:
                            row_v = (j_b + v) * p + (k_b + w)
                            V[row_v][col] = coeff
                            
                # Update W
                for u in range(2):
                    for w in range(2):
                        coeff = C[u][w]
                        if coeff != 0:
                            row_w = (k_b + w) * n + (i_base + u)
                            W[row_w][col] = coeff
                            
                col += 1
                
    # Schoolbook for i=2
    # i=2, j in 0..3, k in 0..7
    for j in range(m):
        for k in range(p):
            row_u = 2 * m + j
            row_v = j * p + k
            row_w = k * n + 2
            
            U[row_u][col] = 1
            V[row_v][col] = 1
            W[row_w][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
