"""Strassen-based rank-189 decomposition for 6x6 matrix multiplication."""
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    block = 3
    rank = 7 * block * block  # 189
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen steps for 2x2 block matrix multiplication
    # Each step: (A_mask, B_mask, C_mask)
    # A_mask: dict {(block_row, block_col): coeff}
    # B_mask: dict {(block_row, block_col): coeff}
    # C_mask: dict {(block_row, block_col): coeff}
    
    steps = [
        # M1 = (A00 + A11)(B00 + B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A10 + A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = A00(B01 - B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = A11(B10 - B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (A00 + A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (A10 - A00)(B00 + B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (A01 - A11)(B10 + B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col = 0
    for A_mask, B_mask, C_mask in steps:
        for u in range(block):
            for v in range(block):
                for w in range(block):
                    # Fill U: A part
                    for (ar, ac), ca in A_mask.items():
                        i = ar * block + u
                        j = ac * block + w
                        U[i, col] += ca
                    
                    # Fill V: B part
                    for (br, bc), cb in B_mask.items():
                        j = br * block + w
                        k = bc * block + v
                        V[j, col] += cb
                    
                    # Fill W: C part
                    for (cr, cc), cc_coeff in C_mask.items():
                        k = cc * block + v
                        i = cr * block + u
                        W[k, col] += cc_coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
