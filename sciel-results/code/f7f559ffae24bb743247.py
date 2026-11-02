import time

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add coefficient to a matrix row at a specific column
    def add_val(mat, row, col, val):
        if val != 0:
            mat[row][col] += val
            
    for b in range(3):
        base = b * 7
        # V indices for this block
        v_11 = 2 * b      # b[0, 2b]
        v_12 = 2 * b + 1  # b[0, 2b+1]
        v_21 = 6 + 2 * b  # b[1, 2b]
        v_22 = 6 + 2 * b + 1 # b[1, 2b+1]
        
        # W indices for this block
        w_11 = 4 * b      # c[0, 2b]
        w_21 = 4 * b + 1  # c[1, 2b]
        w_12 = 4 * b + 2  # c[0, 2b+1]
        w_22 = 4 * b + 3  # c[1, 2b+1]
        
        # M1: (a00 + a11) * (b11 + b22)
        col = base
        add_val(U, 0, col, 1)
        add_val(U, 3, col, 1)
        add_val(V, v_11, col, 1)
        add_val(V, v_22, col, 1)
        add_val(W, w_11, col, 1)
        add_val(W, w_22, col, 1)
        
        # M2: (a10 + a11) * b11
        col = base + 1
        add_val(U, 2, col, 1)
        add_val(U, 3, col, 1)
        add_val(V, v_11, col, 1)
        add_val(W, w_21, col, 1)
        add_val(W, w_22, col, -1)
        
        # M3: a00 * (b12 - b22)
        col = base + 2
        add_val(U, 0, col, 1)
        add_val(V, v_12, col, 1)
        add_val(V, v_22, col, -1)
        add_val(W, w_12, col, 1)
        add_val(W, w_22, col, 1)
        
        # M4: a11 * (b21 - b11)
        col = base + 3
        add_val(U, 3, col, 1)
        add_val(V, v_21, col, 1)
        add_val(V, v_11, col, -1)
        add_val(W, w_11, col, 1)
        add_val(W, w_21, col, 1)
        
        # M5: (a00 + a01) * b22
        col = base + 4
        add_val(U, 0, col, 1)
        add_val(U, 1, col, 1)
        add_val(V, v_22, col, 1)
        add_val(W, w_11, col, -1)
        add_val(W, w_12, col, 1)
        
        # M6: (a10 - a00) * (b11 + b12)
        col = base + 5
        add_val(U, 2, col, 1)
        add_val(U, 0, col, -1)
        add_val(V, v_11, col, 1)
        add_val(V, v_12, col, 1)
        add_val(W, w_22, col, 1)
        
        # M7: (a01 - a11) * (b21 + b22)
        col = base + 6
        add_val(U, 1, col, 1)
        add_val(U, 3, col, -1)
        add_val(V, v_21, col, 1)
        add_val(V, v_22, col, 1)
        add_val(W, w_11, col, 1)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
