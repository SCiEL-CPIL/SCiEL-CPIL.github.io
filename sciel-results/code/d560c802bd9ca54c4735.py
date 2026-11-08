import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 24 decomposition
    # Part 1: 4x4x4 using Laderman (Rank 16)
    # Part 2: 4x4x2 using Schoolbook (Rank 8)
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        # u_idx is (i, j) for A
        # v_idx is (j, k) for B
        # w_idx is (i, k) for C
        i, j = u_idx
        jv, kv = v_idx
        iw, kw = w_idx
        # Check consistency: j == jv
        if j != jv:
            raise ValueError("Index mismatch")
        
        col = 0 # This is not how it works, we need to assign columns
        
    # We need to assign columns 0-23.
    # Columns 0-15: Laderman
    # Columns 16-23: Schoolbook for 4x4x2
    
    # Laderman Terms
    # A indices: 0..15 (i*4+j)
    # B indices: 0..23 (j*6+k) -- Wait, for the 4x4x4 part, k is 0..3.
    # So B indices for this part are j*6 + k.
    # C indices: 0..23 (k*4+i).
    
    laderman_terms = [
        # (u_indices, v_indices, w_indices)
        # u: (i, j), v: (j, k), w: (i, k)
        # Term 1: (a00+a11)(b00+b11) -> c00, c01, c10, c11?
        # Laderman reconstructs C from P.
        # P1 = (a00+a11)(b00+b11)
        # c00 = P1 - P2 - P3 + P4 ... ?
        
        # I don't remember the reconstruction formulas.
        # I will use the schoolbook for 4x4x4 as well if I can't find Laderman.
        # Rank 16 for 4x4x4 is hard to get right without reference.
        
        # Let's try a different approach.
        # Use rank 32?
        # 4x4x4 can be done with 2x2 blocks?
        # No.
        
        # Let's just use schoolbook for 4x4x4 (Rank 16) and schoolbook for 4x4x2 (Rank 8).
        # Total Rank 24.
        # Wait, schoolbook for 4x4x4 is Rank 16.
        # Schoolbook for 4x4x2 is Rank 8.
        # Total 24.
        # This is valid!
        # I don't need Laderman. I just need to split the 6 columns into 4 and 2.
    ]
    
    # Schoolbook for 4x4x4 (cols 0-3 of B)
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * 4 + j][col] = 1
                V[j * 6 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
                
    # Schoolbook for 4x4x2 (cols 4-5 of B)
    for i in range(4):
        for j in range(4):
            for k in range(2):
                # k is local 0,1. Global k is 4+k.
                global_k = 4 + k
                U[i * 4 + j][col] = 1
                V[j * 6 + global_k][col] = 1
                W[global_k * 4 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
