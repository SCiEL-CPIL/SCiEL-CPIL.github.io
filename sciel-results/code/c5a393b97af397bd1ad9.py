import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 22
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set values
    def set_val(mat, row, col, val):
        mat[row][col] = val

    # Part 1: j=2 block (Rank 1)
    # T_{i, 2, k} = 1 for all i, k
    # u: i=0,1 -> U[0*3+2, 0]=1, U[1*3+2, 0]=1
    set_val(U, 2, 0, 1)
    set_val(U, 5, 0, 1)
    
    # v: j=2, k=0..5 -> V[2*6+k, 0]=1
    for k in range(6):
        set_val(V, 12 + k, 0, 1)
        
    # w: k=0..5, i=0,1 -> W[k*2+i, 0]=1
    for k in range(6):
        set_val(W, 2*k, 0, 1)
        set_val(W, 2*k+1, 0, 1)

    # Part 2: j in {0,1} blocks (3 * Strassen Rank 7 = 21)
    # Blocks b = 0, 1, 2 corresponding to k in {0,1}, {2,3}, {4,5}
    
    for b in range(3):
        base_col = 1 + b * 7
        
        # Local indices for A (2x2), B (2x2), C (2x2)
        # A indices in U:
        # A00: i=0, j=0 -> U[0, c]
        # A01: i=0, j=1 -> U[1, c]
        # A10: i=1, j=0 -> U[3, c]
        # A11: i=1, j=1 -> U[4, c]
        
        # B indices in V:
        # B00: j=0, k=2b   -> V[2b, c]
        # B01: j=0, k=2b+1 -> V[2b+1, c]
        # B10: j=1, k=2b   -> V[6+2b, c]
        # B11: j=1, k=2b+1 -> V[6+2b+1, c]
        
        # C indices in W:
        # C00: i=0, k=2b   -> W[4b, c]
        # C01: i=0, k=2b+1 -> W[4b+2, c]
        # C10: i=1, k=2b   -> W[4b+1, c]
        # C11: i=1, k=2b+1 -> W[4b+3, c]
        
        # M1: (A00 + A11)(B00 + B11)
        c = base_col + 0
        set_val(U, 0, c, 1)
        set_val(U, 4, c, 1)
        set_val(V, 2*b, c, 1)
        set_val(V, 6+2*b+1, c, 1)
        set_val(W, 4*b, c, 1)     # C00
        set_val(W, 4*b+3, c, 1)   # C11
        
        # M2: (A10 + A11)B00
        c = base_col + 1
        set_val(U, 3, c, 1)
        set_val(U, 4, c, 1)
        set_val(V, 2*b, c, 1)
        set_val(W, 4*b+1, c, 1)   # C10
        set_val(W, 4*b+3, c, -1)  # C11
        
        # M3: A00(B01 - B11)
        c = base_col + 2
        set_val(U, 0, c, 1)
        set_val(V, 2*b+1, c, 1)
        set_val(V, 6+2*b+1, c, -1)
        set_val(W, 4*b+2, c, 1)   # C01
        set_val(W, 4*b+3, c, 1)   # C11
        
        # M4: A11(B10 - B00)
        c = base_col + 3
        set_val(U, 4, c, 1)
        set_val(V, 6+2*b, c, 1)
        set_val(V, 2*b, c, -1)
        set_val(W, 4*b, c, 1)     # C00
        set_val(W, 4*b+1, c, 1)   # C10
        
        # M5: (A00 + A01)B11
        c = base_col + 4
        set_val(U, 0, c, 1)
        set_val(U, 1, c, 1)
        set_val(V, 6+2*b+1, c, 1)
        set_val(W, 4*b, c, -1)    # C00
        set_val(W, 4*b+2, c, 1)   # C01
        
        # M6: (A10 - A00)(B00 + B01)
        c = base_col + 5
        set_val(U, 3, c, 1)
        set_val(U, 0, c, -1)
        set_val(V, 2*b, c, 1)
        set_val(V, 2*b+1, c, 1)
        set_val(W, 4*b+3, c, 1)   # C11
        
        # M7: (A01 - A11)(B10 + B11)
        c = base_col + 6
        set_val(U, 1, c, 1)
        set_val(U, 4, c, -1)
        set_val(V, 6+2*b, c, 1)
        set_val(V, 6+2*b+1, c, 1)
        set_val(W, 4*b, c, 1)     # C00

    return {"U": U, "V": V, "W": W}
