import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # We will use 5 blocks of 2 columns each.
    # Each block uses a decomposition of rank 11 for 2x3x2.
    # Total rank = 5 * 11 = 55.
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for block_idx in range(5):
        # Columns for this block
        k1 = block_idx * 2
        k2 = k1 + 1
        
        # Strassen terms for 2x2x2 part (rows 1,2 of A, rows 1,2 of B)
        # A indices: 0,1,2,3 (i=0,1; j=0,1)
        # B indices: j*p + k. j=0,1. k=k1,k2.
        
        # m1 = (A00 + A11)(B0k1 + B1k2) -> Wait, B indices are (row, col).
        # B row 0 is B_{1, *}, row 1 is B_{2, *}.
        # So B_{0,k1} is B[0, k1], B_{1,k2} is B[1, k2].
        # In our V matrix, index is j*p + k.
        
        # Let's define the 7 Strassen terms explicitly.
        # A indices:
        # A00 = 0*3+0 = 0
        # A01 = 0*3+1 = 1
        # A10 = 1*3+0 = 3
        # A11 = 1*3+1 = 4
        
        # B indices for this block:
        # B0k1 = 0*10 + k1
        # B1k1 = 1*10 + k1
        # B0k2 = 0*10 + k2
        # B1k2 = 1*10 + k2
        
        # Term 1: (A00 + A11)(B0k1 + B1k2)
        # U: A00=1, A11=1
        # V: B0k1=1, B1k2=1
        # W: C0k1=1, C0k2=1, C1k1=1, C1k2=1 (with signs)
        
        # This is getting complicated to map manually.
        # I will use a helper to add a term.
        
        def add_term(u_indices, v_indices, w_indices, col_offset):
            for idx, val in u_indices.items():
                U[idx][col_offset] += val
            for idx, val in v_indices.items():
                V[idx][col_offset] += val
            for idx, val in w_indices.items():
                W[idx][col_offset] += val
        
        # Strassen 7 terms
        # m1
        add_term({0:1, 4:1}, {10*k1:1, 10*k2+1:1}, {2*k1+0:1, 2*k2+0:1, 2*k1+1:1, 2*k2+1:1}, col)
        col += 1
        
        # m2 = (A10 + A11)B0k1
        add_term({3:1, 4:1}, {10*k1:1}, {2*k1+1:1}, col)
        col += 1
        
        # m3 = A00(B0k2 - B1k2)
        add_term({0:1}, {10*k2:1, 10*k2+1:-1}, {2*k2+0:1}, col)
        col += 1
        
        # m4 = A11(B1k1 - B0k1)
        add_term({4:1}, {10*k1+1:1, 10*k1:-1}, {2*k1+1:1}, col)
        col += 1
        
        # m5 = (A00 + A01)B1k2
        add_term({0:1, 1:1}, {10*k2+1:1}, {2*k2+0:1, 2*k2+1:1}, col)
        col += 1
        
        # m6 = (A10 - A00)(B0k1 + B0k2)
        add_term({3:1, 0:-1}, {10*k1:1, 10*k2:1}, {2*k1+1:1, 2*k2+1:1}, col)
        col += 1
        
        # m7 = (A01 - A11)(B1k1 + B1k2)
        add_term({1:1, 4:-1}, {10*k1+1:1, 10*k2+1:1}, {2*k2+0:1, 2*k2+1:1}, col)
        col += 1
        
        # Extra 4 terms for row 2 of A (index 2,5) and row 2 of B (index 20+k)
        # A02 = 2, A12 = 5
        # B2k1 = 20+k1, B2k2 = 20+k2
        
        # A02 * B2k1
        add_term({2:1}, {20+k1:1}, {2*k1+0:1}, col)
        col += 1
        
        # A02 * B2k2
        add_term({2:1}, {20+k2:1}, {2*k2+0:1}, col)
        col += 1
        
        # A12 * B2k1
        add_term({5:1}, {20+k1:1}, {2*k1+1:1}, col)
        col += 1
        
        # A12 * B2k2
        add_term({5:1}, {20+k2:1}, {2*k2+1:1}, col)
        col += 1

    return {"U": U, "V": V, "W": W}
