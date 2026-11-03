import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen block decomposition for 6x6 = 2x2 blocks of 3x3
    # Rank = 7 * (3*3*3) = 189
    rank = 189
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen definitions for 2x2 block structure
    # Blocks are indexed 0,1 for rows/cols.
    # Local indices u, v, w are 0, 1, 2 for the 3x3 submatrices.
    
    # Define the linear combinations for X_r and Y_r
    # X_r[u, v] is a linear combination of A blocks
    # Y_r[v, w] is a linear combination of B blocks
    
    # We will define a list of 7 tuples: (list of (block_row, block_col, coeff) for X, list of (block_row, block_col, coeff) for Y)
    # X_r is the left factor, Y_r is the right factor.
    
    strassen_terms = [
        # r=1: (A11 + A22) * (B11 + B22)
        {
            'X': [(0, 0, 1), (1, 1, 1)],
            'Y': [(0, 0, 1), (1, 1, 1)],
            'C_coeffs': {
                (0, 0): 1,   # C11
                (1, 1): 1    # C22
            }
        },
        # r=2: (A21 + A22) * B11
        {
            'X': [(1, 0, 1), (1, 1, 1)],
            'Y': [(0, 0, 1)],
            'C_coeffs': {
                (1, 0): 1,   # C21
                (1, 1): -1   # C22
            }
        },
        # r=3: A11 * (B12 - B22)
        {
            'X': [(0, 0, 1)],
            'Y': [(0, 1, 1), (1, 1, -1)],
            'C_coeffs': {
                (0, 1): 1,   # C12
                (1, 1): 1    # C22
            }
        },
        # r=4: A22 * (B21 - B11)
        {
            'X': [(1, 1, 1)],
            'Y': [(1, 0, 1), (0, 0, -1)],
            'C_coeffs': {
                (0, 0): 1,   # C11
                (1, 0): 1    # C21
            }
        },
        # r=5: (A11 + A12) * B22
        {
            'X': [(0, 0, 1), (0, 1, 1)],
            'Y': [(1, 1, 1)],
            'C_coeffs': {
                (0, 0): -1,  # C11
                (0, 1): 1    # C12
            }
        },
        # r=6: (A21 - A11) * (B11 + B12)
        {
            'X': [(1, 0, 1), (0, 0, -1)],
            'Y': [(0, 0, 1), (0, 1, 1)],
            'C_coeffs': {
                (1, 1): 1    # C22
            }
        },
        # r=7: (A12 - A22) * (B21 + B22)
        {
            'X': [(0, 1, 1), (1, 1, -1)],
            'Y': [(1, 0, 1), (1, 1, 1)],
            'C_coeffs': {
                (0, 0): 1    # C11
            }
        }
    ]
    
    k = 0
    for r_data in strassen_terms:
        X_terms = r_data['X']
        Y_terms = r_data['Y']
        C_coeffs = r_data['C_coeffs']
        
        # Iterate over local indices for the 3x3 multiplication
        # X[u, v] * Y[v, w]
        for u in range(3):
            for v in range(3):
                for w in range(3):
                    # This is one scalar multiplication
                    # Fill U for A indices
                    # X[u, v] = sum_{br, bc} coeff * A_{br, bc}[u, v]
                    for br, bc, coeff in X_terms:
                        global_i = br * 3 + u
                        global_j = bc * 3 + v
                        U_idx = global_i * m + global_j
                        U[U_idx, k] += coeff
                    
                    # Fill V for B indices
                    # Y[v, w] = sum_{br, bc} coeff * B_{br, bc}[v, w]
                    for br, bc, coeff in Y_terms:
                        global_j = br * 3 + v
                        global_k = bc * 3 + w
                        V_idx = global_j * p + global_k
                        V[V_idx, k] += coeff
                    
                    # Fill W for C indices
                    # The result M_r[u, w] contributes to C blocks defined in C_coeffs
                    for (c_br, c_bc), c_coeff in C_coeffs.items():
                        global_i = c_br * 3 + u
                        global_k = c_bc * 3 + w
                        W_idx = global_k * n + global_i
                        W[W_idx, k] += c_coeff
                    
                    k += 1
    
    # Convert to lists if necessary, but numpy arrays are usually accepted or can be returned as is
    # The prompt says "factor matrices of shapes ... with integer/half-integer entries"
    # Returning numpy arrays is standard.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
