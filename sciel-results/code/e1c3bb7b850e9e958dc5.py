import numpy as np

def construct():
    n, m, p = 2, 2, 13
    rank = 46
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Strassen parameters for 2x2 blocks
    # A indices: 00, 01, 10, 11 -> U indices 0, 1, 2, 3
    # B indices for block s:
    # b00 -> V[2s]
    # b01 -> V[2s+1]
    # b10 -> V[p+2s]
    # b11 -> V[p+2s+1]
    # C indices for block s:
    # c00 -> W[4s]
    # c01 -> W[4s+2]
    # c10 -> W[4s+1]
    # c11 -> W[4s+3]
    
    # Strassen terms definitions:
    # t: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict {index: val}
    # V_coeffs: dict {index: val}
    # W_coeffs: dict {index: val}
    
    strassen_terms = [
        # t=1
        ({0: 1, 3: 1}, {0: 1, 1: 1}, {0: 1, 3: 1}), # M1 = (a00+a11)(b00+b11) -> C00+=1, C11+=1
        # t=2
        ({2: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}),      # M2 = (a10+a11)b00 -> C10+=1, C11-=1
        # t=3
        ({0: 1}, {1: 1, 3: -1}, {2: 1, 3: 1}),      # M3 = a00(b01-b11) -> C01+=1, C11+=1
        # t=4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 1: 1}),      # M4 = a11(b10-b00) -> C00+=1, C10+=1
        # t=5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),      # M5 = (a00+a01)b11 -> C00-=1, C01+=1
        # t=6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),      # M6 = (a10-a00)(b00+b01) -> C11+=1
        # t=7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),      # M7 = (a01-a11)(b10+b11) -> C00+=1
    ]
    
    # Note: The V and W indices in strassen_terms are relative to the block.
    # We need to offset them.
    
    col = 0
    for s in range(6):
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
            
            # Update U
            for u_idx, val in u_coeffs.items():
                U[u_idx, col] = val
                
            # Update V
            # b00 -> 2s, b01 -> 2s+1, b10 -> p+2s, b11 -> p+2s+1
            # Map relative indices 0,1,2,3 to absolute
            v_map = {0: 2*s, 1: 2*s+1, 2: p+2*s, 3: p+2*s+1}
            for v_idx_rel, val in v_coeffs.items():
                v_idx_abs = v_map[v_idx_rel]
                V[v_idx_abs, col] = val
                
            # Update W
            # c00 -> 4s, c01 -> 4s+2, c10 -> 4s+1, c11 -> 4s+3
            # Map relative indices 0,1,2,3 to absolute
            w_map = {0: 4*s, 1: 4*s+1, 2: 4*s+2, 3: 4*s+3}
            for w_idx_rel, val in w_coeffs.items():
                w_idx_abs = w_map[w_idx_rel]
                W[w_idx_abs, col] = val
                
            col += 1
            
    # Handle the last 2x1 block (column 12)
    # Standard multiplication
    # C0,12 = A00*B0,12 + A01*B1,12
    # C1,12 = A10*B0,12 + A11*B1,12
    
    # Term 1: A00 * B0,12
    U[0, col] = 1
    V[12, col] = 1
    W[12*2 + 0, col] = 1 # W[24]
    col += 1
    
    # Term 2: A01 * B1,12
    U[1, col] = 1
    V[p+12, col] = 1 # V[15]
    W[12*2 + 0, col] = 1 # W[24]
    col += 1
    
    # Term 3: A10 * B0,12
    U[2, col] = 1
    V[12, col] = 1
    W[12*2 + 1, col] = 1 # W[25]
    col += 1
    
    # Term 4: A11 * B1,12
    U[3, col] = 1
    V[p+12, col] = 1 # V[15]
    W[12*2 + 1, col] = 1 # W[25]
    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
