"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set values
    def set_vals(matrix, indices, col, val=1):
        for idx in indices:
            matrix[idx][col] = val

    # P1
    set_vals(U, [0, 3, 6], 0)
    set_vals(V, [0, 3, 6], 0)
    set_vals(W, [0, 1, 2, 3, 4, 5, 6, 7, 8], 0)
    
    # P2
    set_vals(U, [1, 4, 7], 1)
    set_vals(V, [1, 4, 7], 1)
    set_vals(W, [0, 1, 2, 3, 4, 5, 6, 7, 8], 1)
    
    # P3
    set_vals(U, [2, 5, 8], 2)
    set_vals(V, [2, 5, 8], 2)
    set_vals(W, [0, 1, 2, 3, 4, 5, 6, 7, 8], 2)
    
    # P4
    set_vals(U, [0, 1], 3)
    set_vals(V, [0, 3], 3)
    set_vals(W, [0, 3], 3, -1)
    
    # P5
    set_vals(U, [0, 2], 4)
    set_vals(V, [0, 6], 4)
    set_vals(W, [2, 6], 4, -1)
    
    # P6
    set_vals(U, [3, 4], 5)
    set_vals(V, [3, 4], 5)
    set_vals(W, [0, 3], 5, -1)
    
    # P7
    set_vals(U, [3, 5], 6)
    set_vals(V, [3, 6], 6)
    set_vals(W, [2, 6], 6, -1)
    
    # P8
    set_vals(U, [6, 7], 7)
    set_vals(V, [6, 7], 7)
    set_vals(W, [0, 6], 7, -1)
    
    # P9
    set_vals(U, [6, 8], 8)
    set_vals(V, [6, 8], 8)
    set_vals(W, [2, 6], 8, -1)
    
    # P10
    set_vals(U, [1, 4], 9)
    set_vals(V, [1, 4], 9)
    set_vals(W, [1, 4], 9, -1)
    
    # P11
    set_vals(U, [1, 7], 10)
    set_vals(V, [1, 7], 10)
    set_vals(W, [1, 7], 10, -1)
    
    # P12
    set_vals(U, [2, 5], 11)
    set_vals(V, [2, 5], 11)
    set_vals(W, [2, 5], 11, -1)
    
    # P13
    set_vals(U, [2, 8], 12)
    set_vals(V, [2, 8], 12)
    set_vals(W, [2, 8], 12, -1)
    
    # P14
    set_vals(U, [4, 7], 13)
    set_vals(V, [4, 7], 13)
    set_vals(W, [4, 7], 13, -1)
    
    # P15
    set_vals(U, [5, 8], 14)
    set_vals(V, [5, 8], 14)
    set_vals(W, [5, 8], 14, -1)
    
    # P16
    set_vals(U, [0], 15)
    set_vals(V, [4], 15)
    set_vals(W, [0, 3, 6], 15, 1)
    
    # P17
    set_vals(U, [0], 16)
    set_vals(V, [8], 16)
    set_vals(W, [1, 4, 7], 16, 1)
    
    # P18
    set_vals(U, [3], 17)
    set_vals(V, [8], 17)
    set_vals(W, [2, 5, 8], 17, 1)
    
    # P19
    set_vals(U, [6], 18)
    set_vals(V, [4], 18)
    set_vals(W, [2, 5, 8], 18, -1)
    
    # P20
    set_vals(U, [1], 19)
    set_vals(V, [5], 19)
    set_vals(W, [1, 4, 7], 19, -1)
    
    # P21
    set_vals(U, [2], 20)
    set_vals(V, [4], 20)
    set_vals(W, [0, 3, 6], 20, -1)
    
    # P22
    set_vals(U, [4], 21)
    set_vals(V, [2], 21)
    set_vals(W, [0, 3, 6], 21, -1)
    
    # P23
    set_vals(U, [7], 22)
    set_vals(V, [2], 22)
    set_vals(W, [1, 4, 7], 22, -1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
