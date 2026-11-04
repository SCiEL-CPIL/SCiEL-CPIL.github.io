import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # Strassen on 2x2 blocks of 3x3 matrices
    # Rank = 7 * 27 = 189
    
    rank = 189
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen products definitions
    # A blocks: (0,0)=A11, (0,1)=A12, (1,0)=A21, (1,1)=A22
    # B blocks: (0,0)=B11, (0,1)=B12, (1,0)=B21, (1,1)=B22
    # C blocks: (0,0)=C11, (0,1)=C12, (1,0)=C21, (1,1)=C22
    
    products = [
        # M1 = (A11 + A22)(B11 + B22)
        {'a': {(0,0): 1, (1,1): 1}, 'b': {(0,0): 1, (1,1): 1}, 'c': {(0,0): 1, (1,1): 1}},
        # M2 = (A21 + A22)B11
        {'a': {(1,0): 1, (1,1): 1}, 'b': {(0,0): 1}, 'c': {(1,0): 1, (1,1): -1}},
        # M3 = A11(B12 - B22)
        {'a': {(0,0): 1}, 'b': {(0,1): 1, (1,1): -1}, 'c': {(0,1): 1, (1,1): 1}},
        # M4 = A22(B21 - B11)
        {'a': {(1,1): 1}, 'b': {(1,0): 1, (0,0): -1}, 'c': {(0,0): 1, (1,0): 1}},
        # M5 = (A11 + A12)B22
        {'a': {(0,0): 1, (0,1): 1}, 'b': {(1,1): 1}, 'c': {(0,0): -1, (0,1): 1}},
        # M6 = (A21 - A11)(B11 + B12)
        {'a': {(1,0): 1, (0,0): -1}, 'b': {(0,0): 1, (0,1): 1}, 'c': {(1,1): 1}},
        # M7 = (A12 - A22)(B21 + B22)
        {'a': {(0,1): 1, (1,1): -1}, 'b': {(1,0): 1, (1,1): 1}, 'c': {(0,0): 1}}
    ]
    
    col = 0
    for prod in products:
        # Iterate over the 3x3 schoolbook terms for this product
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # Term: X[i,j] * Y[j,k] * Z[i,k]
                    
                    # Fill U (A indices)
                    # A is 6x6. Block (br, bc) covers rows 3*br..3*br+2, cols 3*bc..3*bc+2
                    # Global row = 3*br + i, Global col = 3*bc + j
                    # Index in U (row-major) = global_row * 6 + global_col
                    for (br, bc), coeff_a in prod['a'].items():
                        gr = 3 * br + i
                        gc = 3 * bc + j
                        U[gr * 6 + gc, col] = coeff_a
                    
                    # Fill V (B indices)
                    # B is 6x6. Block (br, bc) covers rows 3*br..3*br+2, cols 3*bc..3*bc+2
                    # Global row = 3*br + j, Global col = 3*bc + k
                    # Index in V (row-major) = global_row * 6 + global_col
                    for (br, bc), coeff_b in prod['b'].items():
                        gr = 3 * br + j
                        gc = 3 * bc + k
                        V[gr * 6 + gc, col] = coeff_b
                    
                    # Fill W (C indices)
                    # C is 6x6. Block (br, bc) covers rows 3*br..3*br+2, cols 3*bc..3*bc+2
                    # Global row = 3*br + i, Global col = 3*bc + k
                    # Convention: W[k*n + i, r] where i is row, k is col.
                    # So index = global_col * 6 + global_row
                    for (br, bc), coeff_c in prod['c'].items():
                        gr = 3 * br + i
                        gc = 3 * bc + k
                        W[gc * 6 + gr, col] = coeff_c
                        
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
