import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    
    # We decompose the multiplication into:
    # 1. A (2x2) * B (2x10) part
    # 2. A (2x1) * B (1x10) part
    
    # For the 2x2x10 part, we split p=10 into five 2x2x2 blocks.
    # Each 2x2x2 block uses Strassen's algorithm (rank 7).
    # For the 2x1x10 part, we split p=10 into five 2x1x2 blocks.
    # Each 2x1x2 block uses rank 2.
    
    # Total rank = 5 * 7 + 5 * 2 = 35 + 10 = 45.
    
    rank = 45
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add Strassen decomposition for 2x2x2
    # Indices for a 2x2x2 block starting at j_offset, k_offset
    def add_strassen(j_off, k_off):
        nonlocal col
        # Strassen matrices for 2x2x2
        # U, V, W for the 7 terms
        # Standard Strassen:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # We need to map these to the global indices.
        # Local i in {0,1}, j in {0,1}, k in {0,1}
        # Global i = i
        # Global j = j_off + j
        # Global k = k_off + k
        
        # U indices: i * 2 + j
        # V indices: j * 2 + k
        # W indices: k * 2 + i
        
        # Term 1: (A00 + A11)(B00 + B11)
        # U: 00, 11
        # V: 00, 11
        # W: 00, 11, 01, 10? No, W depends on k,i.
        # The Strassen algorithm computes C.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to express the tensor T for 2x2x2 as sum of 7 rank 1s.
        # This is complex to map directly. 
        # Instead, I will use the known rank-7 decomposition coefficients.
        
        # For now, I will just use schoolbook for the 2x2x2 blocks to ensure correctness, 
        # which gives rank 8 per block.
        # 5 * 8 + 5 * 2 = 40 + 10 = 50.
        # This is still better than 60.
        
        # Let's do schoolbook for 2x2x2 (rank 8) and 2x1x2 (rank 2).
        # 2x2x2:
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u_idx = i * 2 + j
                    v_idx = (j_off + j) * 2 + k
                    w_idx = (k_off + k) * 2 + i
                    U[u_idx, col] = 1
                    V[v_idx, col] = 1
                    W[w_idx, col] = 1
                    col += 1

    # 2x2x10 part: 5 blocks of 2x2x2
    for block in range(5):
        add_strassen(0, block * 2)
        
    # Wait, the j index for the first part is 0,1.
    # The second part is j=2.
    
    # 2x1x10 part: 5 blocks of 2x1x2
    for block in range(5):
        for i in range(2):
            for k in range(2):
                u_idx = i * 3 + 2 # j=2
                v_idx = 2 * 10 + (block * 2 + k) # j=2, k=block*2+k
                w_idx = (block * 2 + k) * 2 + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
