import numpy as np

def construct():
    n, m, p = 4, 4, 6

    # Strassen 2x2x2 terms
    strassen_2x2_terms = [
        {'A': [[1, 0], [0, 1]], 'B': [[1, 0], [0, 1]], 'C': [[1, 0], [0, 1]]},
        {'A': [[0, 0], [1, 1]], 'B': [[1, 0], [0, 0]], 'C': [[0, 0], [1, -1]]},
        {'A': [[1, 0], [0, 0]], 'B': [[0, 1], [0, -1]], 'C': [[0, 1], [0, 1]]},
        {'A': [[0, 0], [0, 1]], 'B': [[-1, 0], [1, 0]], 'C': [[1, 0], [1, 0]]},
        {'A': [[1, 1], [0, 0]], 'B': [[0, 0], [0, 1]], 'C': [[-1, 1], [0, 0]]},
        {'A': [[-1, 0], [1, 0]], 'B': [[1, 1], [0, 0]], 'C': [[0, 0], [0, 1]]},
        {'A': [[0, 1], [0, -1]], 'B': [[0, 0], [1, 1]], 'C': [[1, 0], [0, 0]]},
    ]

    terms = []

    def add_product_of_block_sums(a_blocks, b_blocks, c_blocks):
        """
        Add terms for the product of:
        X = sum of a_blocks[i] * A_block[i]
        Y = sum of b_blocks[i] * B_block[i]
        contributing to C = sum of c_blocks[i] * C_block[i]
        
        a_blocks: list of (row_offset, col_offset, coeff) for A blocks
        b_blocks: list of (row_offset, col_offset, coeff) for B blocks
        c_blocks: list of (row_offset, col_offset, coeff) for C blocks
        """
        for t in strassen_2x2_terms:
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            
            # Fill U
            for i in range(2):
                for j in range(2):
                    for (r, c, coeff) in a_blocks:
                        global_i = r + i
                        global_j = c + j
                        u[global_i * m + global_j] += coeff * t['A'][i][j]
            
            # Fill V
            for i in range(2):
                for j in range(2):
                    for (r, c, coeff) in b_blocks:
                        global_i = r + i
                        global_j = c + j
                        v[global_i * p + global_j] += coeff * t['B'][i][j]
            
            # Fill W
            for i in range(2):
                for j in range(2):
                    for (r, c, cc) in c_blocks:
                        global_row = r + i
                        global_col = c + j
                        w[global_col * n + global_row] += cc * t['C'][i][j]
            
            terms.append((u, v, w))

    # 4x4x4 part: C = A * B where A is 4x4, B is 4x4, C is 4x4
    # Using Strassen on 4x4x4:
    # M1 = (A00+A11)(B00+B11) -> C00 (+1), C11 (+1)
    add_product_of_block_sums(
        [(0, 0, 1), (2, 2, 1)],  # A00 + A11
        [(0, 0, 1), (2, 2, 1)],  # B00 + B11
        [(0, 0, 1), (2, 2, 1)]   # C00 + C11
    )
    
    # M2 = (A10+A11)B00 -> C10 (+1), C11 (-1)
    add_product_of_block_sums(
        [(2, 0, 1), (2, 2, 1)],  # A10 + A11
        [(0, 0, 1)],             # B00
        [(2, 0, 1), (2, 2, -1)]  # C10 - C11
    )
    
    # M3 = A00(B01-B11) -> C01 (+1), C11 (+1)
    add_product_of_block_sums(
        [(0, 0, 1)],             # A00
        [(0, 2, 1), (2, 2, -1)], # B01 - B11
        [(0, 2, 1), (2, 2, 1)]   # C01 + C11
    )
    
    # M4 = A11(B10-B00) -> C00 (+1), C10 (+1)
    add_product_of_block_sums(
        [(2, 2, 1)],             # A11
        [(2, 0, 1), (0, 0, -1)], # B10 - B00
        [(0, 0, 1), (2, 0, 1)]   # C00 + C10
    )
    
    # M5 = (A00+A01)B11 -> C00 (-1), C01 (+1)
    add_product_of_block_sums(
        [(0, 0, 1), (0, 2, 1)],  # A00 + A01
        [(2, 2, 1)],             # B11
        [(0, 0, -1), (0, 2, 1)]  # -C00 + C01
    )
    
    # M6 = (A10-A00)(B00+B01) -> C11 (+1)
    add_product_of_block_sums(
        [(2, 0, 1), (0, 0, -1)], # A10 - A00
        [(0, 0, 1), (0, 2, 1)],  # B00 + B01
        [(2, 2, 1)]              # C11
    )
    
    # M7 = (A01-A11)(B10+B11) -> C00 (+1)
    add_product_of_block_sums(
        [(0, 2, 1), (2, 2, -1)], # A01 - A11
        [(2, 0, 1), (2, 2, 1)],  # B10 + B11
        [(0, 0, 1)]              # C00
    )
    
    # 4x4x2 part: C = A * B where A is 4x4, B is 4x2, C is 4x2
    # C_top = A00*B_top + A01*B_bottom
    # C_bottom = A10*B_top + A11*B_bottom
    
    # A00*B_top: A block at (0,0), B block at (0,4), C block at (0,4)
    for t in strassen_2x2_terms:
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        
        for i in range(2):
            for j in range(2):
                u[(0+i) * m + (0+j)] = t['A'][i][j]
        
        for i in range(2):
            for j in range(2):
                v[(0+i) * p + (4+j)] = t['B'][i][j]
        
        for i in range(2):
            for j in range(2):
                w[(4+j) * n + (0+i)] = t['C'][i][j]
        
        terms.append((u, v, w))
    
    # A01*B_bottom: A block at (0,2), B block at (2,4), C block at (0,4)
    for t in strassen_2x2_terms:
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        
        for i in range(2):
            for j in range(2):
                u[(0+i) * m + (2+j)] = t['A'][i][j]
        
        for i in range(2):
            for j in range(2):
                v[(2+i) * p + (4+j)] = t['B'][i][j]
        
        for i in range(2):
            for j in range(2):
                w[(4+j) * n + (0+i)] = t['C'][i][j]
        
        terms.append((u, v, w))
    
    # A10*B_top: A block at (2,0), B block at (0,4), C block at (2,4)
    for t in strassen_2x2_terms:
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        
        for i in range(2):
            for j in range(2):
                u[(2+i) * m + (0+j)] = t['A'][i][j]
        
        for i in range(2):
            for j in range(2):
                v[(0+i) * p + (4+j)] = t['B'][i][j]
        
        for i in range(2):
            for j in range(2):
                w[(4+j) * n + (2+i)] = t['C'][i][j]
        
        terms.append((u, v, w))
    
    # A11*B_bottom: A block at (2,2), B block at (2,4), C block at (2,4)
    for t in strassen_2x2_terms:
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        
        for i in range(2):
            for j in range(2):
                u[(2+i) * m + (2+j)] = t['A'][i][j]
        
        for i in range(2):
            for j in range(2):
                v[(2+i) * p + (4+j)] = t['B'][i][j]
        
        for i in range(2):
            for j in range(2):
                w[(4+j) * n + (2+i)] = t['C'][i][j]
        
        terms.append((u, v, w))
    
    rank = len(terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k, (u, v, w) in enumerate(terms):
        for i in range(n * m):
            U[i][k] = u[i]
        for i in range(m * p):
            V[i][k] = v[i]
        for i in range(p * n):
            W[i][k] = w[i]
    
    return {"U": U, "V": V, "W": W}

BET: {mechanism: "Exact 77-rank Strassen decomposition", q: 0.95, delta: [0.0, 0.0]}
