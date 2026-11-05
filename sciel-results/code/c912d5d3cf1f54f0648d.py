import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=6, m=6, p=6
    # Block size 3x3
    bs = 3
    
    # Strassen coefficients
    # M_s = (sum c_a * A_block_a) * (sum c_b * B_block_b)
    # A_blocks: 00, 01, 10, 11
    # B_blocks: 00, 01, 10, 11
    
    # Define Strassen terms
    # (A_coeffs, B_coeffs)
    # A_coeffs is dict {block: coeff}
    # B_coeffs is dict {block: coeff}
    
    strassen_terms = [
        # M0
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        # M1
        ({2: 1, 3: 1}, {0: 1}),
        # M2
        ({0: 1}, {1: 1, 3: -1}),
        # M3
        ({3: 1}, {2: 1, 0: -1}),
        # M4
        ({0: 1, 1: 1}, {3: 1}),
        # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}),
    ]
    
    # Output coefficients: C_block = sum c_s * M_s
    # C_blocks: 00, 01, 10, 11
    # Indices: 0, 1, 2, 3
    output_coeffs = [
        # C00
        {0: 1, 3: 1, 4: -1, 6: 1},
        # C01
        {2: 1, 4: 1},
        # C10
        {1: 1, 3: 1},
        # C11
        {0: 1, 1: -1, 2: 1, 5: 1},
    ]
    
    # Block mapping to global indices
    # Block 0: 00 (rows 0-2, cols 0-2)
    # Block 1: 01 (rows 0-2, cols 3-5)
    # Block 2: 10 (rows 3-5, cols 0-2)
    # Block 3: 11 (rows 3-5, cols 3-5)
    
    def block_to_global(block, i, j):
        # block 0: r=0, c=0
        # block 1: r=0, c=1
        # block 2: r=1, c=0
        # block 3: r=1, c=1
        r = (block // 2) * bs + i
        c = (block % 2) * bs + j
        return r, c
    
    rank = 0
    U = []
    V = []
    W = []
    
    for s in range(7):
        A_coeffs, B_coeffs = strassen_terms[s]
        
        for i in range(bs):
            for k in range(bs):
                for j in range(bs):
                    for a_block, a_coeff in A_coeffs.items():
                        for b_block, b_coeff in B_coeffs.items():
                            # Create a new column
                            u_row = [0] * 36
                            v_row = [0] * 36
                            w_row = [0] * 36
                            
                            # U index: A_block, i, k
                            # A is n x m (6x6). Index i*m + j?
                            # Interface: U[i*m + j, r]. Here i is row, j is col.
                            # In our term, A element is at row i, col k within block.
                            r_a, c_a = block_to_global(a_block, i, k)
                            u_idx = r_a * m + c_a
                            u_row[u_idx] = a_coeff
                            
                            # V index: B_block, k, j
                            # B is m x p (6x6). Index j*p + k?
                            # Interface: V[j*p + k, r]. Here j is row, k is col.
                            # In our term, B element is at row k, col j within block.
                            r_b, c_b = block_to_global(b_block, k, j)
                            v_idx = r_b * p + c_b
                            v_row[v_idx] = b_coeff
                            
                            # W index: C_block, i, j
                            # C is p x n (6x6). Index k*n + i?
                            # Interface: W[k*n + i, r]. Here k is col, i is row.
                            # M_s[i,j] contributes to C_block[i,j].
                            # Global row r_c, col c_c.
                            # W index is c_c * n + r_c.
                            
                            for c_block, c_coeff in output_coeffs.items():
                                if s in c_coeff:
                                    coeff = c_coeff[s]
                                    r_c, c_c = block_to_global(c_block, i, j)
                                    w_idx = c_c * n + r_c
                                    w_row[w_idx] += coeff
                            
                            U.append(u_row)
                            V.append(v_row)
                            W.append(w_row)
                            rank += 1
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
