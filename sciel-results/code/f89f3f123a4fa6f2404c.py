import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    # We are doing 6x6 = (2x2 blocks of 3x3)
    # Block size is 3x3.
    # Total elements 36.
    
    # Strassen steps for 2x2 blocks
    # S_k are linear combos of A blocks
    # T_k are linear combos of B blocks
    
    # A blocks: A00, A01, A10, A11
    # Indices in A (36x1):
    # A00: 0-8
    # A01: 9-17
    # A10: 18-26
    # A11: 27-35
    
    # B blocks: B00, B01, B10, B11
    # Indices in B (36x1):
    # B00: 0-8
    # B01: 9-17
    # B10: 18-26
    # B11: 27-35
    
    # C blocks: C00, C01, C10, C11
    # Indices in C (36x1):
    # C00: 0-8
    # C01: 9-17
    # C10: 18-26
    # C11: 27-35
    
    # Define S_k coefficients for A
    # S_k is a 3x3 matrix. S_k[u, v] is a linear combo of A elements.
    # We represent S_k as a 3x3x36 array? Or just a dict of non-zeros?
    # Since it's sparse, we can just define the block coefficients.
    
    # Block indices: 0:00, 1:01, 2:10, 3:11
    # S_k[u, v] depends on which blocks are summed.
    
    # S1 = A00 + A11
    # S2 = A10 + A11
    # S3 = A00
    # S4 = A11
    # S5 = A00 + A01
    # S6 = A10 - A00
    # S7 = A01 - A11
    
    # T1 = B00 + B11
    # T2 = B00
    # T3 = B01 - B11
    # T4 = B10 - B00
    # T5 = B11
    # T6 = B00 + B01
    # T7 = B10 + B11
    
    # For each k, we define the block coefficients for S_k and T_k.
    # S_k_blocks: dict {block_idx: coeff}
    # T_k_blocks: dict {block_idx: coeff}
    
    S_defs = [
        {0: 1, 3: 1}, # S1
        {2: 1, 3: 1}, # S2
        {0: 1},       # S3
        {3: 1},       # S4
        {0: 1, 1: 1}, # S5
        {2: 1, 0: -1},# S6
        {1: 1, 3: -1} # S7
    ]
    
    T_defs = [
        {0: 1, 3: 1}, # T1
        {0: 1},       # T2
        {1: 1, 3: -1},# T3
        {2: 1, 0: -1},# T4
        {3: 1},       # T5
        {0: 1, 1: 1}, # T6
        {2: 1, 3: 1}  # T7
    ]
    
    # Output combinations
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # C_block_defs: dict {block_idx: {M_idx: coeff}}
    # M_idx is 1-based in formulas, 0-based in list
    C_defs = {
        0: {0: 1, 3: 1, 4: -1, 6: 1}, # C00
        1: {2: 1, 4: 1},               # C01
        2: {1: 1, 3: 1},               # C10
        3: {0: 1, 1: -1, 2: 1, 5: 1}   # C11
    }
    
    rank = 7 * 27
    U = np.zeros((36, rank), dtype=int)
    V = np.zeros((36, rank), dtype=int)
    W = np.zeros((36, rank), dtype=int)
    
    col = 0
    for k in range(7):
        S_blocks = S_defs[k]
        T_blocks = T_defs[k]
        
        # Iterate over local indices u, v, w for 3x3 product
        for u in range(3):
            for v in range(3):
                for w in range(3):
                    # Determine U coefficients
                    # S_k[u, v] is a linear combo of A blocks.
                    # The element A_block[i, j] is at global index block_start + i*3 + j.
                    # S_k[u, v] uses the same local indices u, v?
                    # Yes, S_k is a matrix. S_k[u, v] is formed by summing A_blocks[u, v].
                    # So for a fixed u, v, S_k[u, v] = sum_{b} coeff_b * A_b[u, v].
                    # A_b[u, v] is at global index b*9 + u*3 + v.
                    
                    for b_idx, coeff in S_blocks.items():
                        # A_b[u, v] global index
                        a_idx = b_idx * 9 + u * 3 + v
                        U[a_idx, col] += coeff
                    
                    # Determine V coefficients
                    # T_k[v, w] uses T_blocks[v, w]?
                    # T_k is a matrix. T_k[v, w] = sum_{b} coeff_b * B_b[v, w].
                    # B_b[v, w] is at global index b*9 + v*3 + w.
                    
                    for b_idx, coeff in T_blocks.items():
                        b_idx_global = b_idx * 9 + v * 3 + w
                        V[b_idx_global, col] += coeff
                    
                    # Determine W coefficients
                    # M_k[u, w] contributes to C blocks.
                    # For each C block L, if M_k is in C_L with coeff c_L:
                    # C_L[u, w] gets contribution.
                    # C_L[u, w] global index is L*9 + u*3 + w.
                    
                    for L_idx, m_coeffs in C_defs.items():
                        if k in m_coeffs:
                            c_coeff = m_coeffs[k]
                            c_idx = L_idx * 9 + u * 3 + w
                            W[c_idx, col] += c_coeff
                    
                    col += 1
                    
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # Convert to lists if required, but numpy arrays are usually fine?
    # The task says "factor matrices of shapes...".
    # The seed program returned lists of lists.
    # Let's return lists of lists to be safe, or check if numpy is accepted.
    # "integer/half-integer entries"
    # I used integers.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
