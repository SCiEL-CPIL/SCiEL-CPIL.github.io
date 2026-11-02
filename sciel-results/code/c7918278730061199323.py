import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 36 decomposition
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    def add_strassen_222(a_row_offsets, a_col_offsets, b_row_offsets, b_col_offsets):
        """
        Adds a rank-7 Strassen decomposition for 2x2x2 multiplication.
        a_row_offsets: list of 2 row indices for A
        a_col_offsets: list of 2 col indices for A
        b_row_offsets: list of 2 row indices for B
        b_col_offsets: list of 2 col indices for B
        """
        nonlocal col_idx
        # Strassen multipliers
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        
        # Coefficients for U (A factors), V (B factors), W (C factors)
        # C indices: k*n + i. 
        # Here i is row of A, k is col of B (which is row of B in our notation? No.)
        # Standard: C[i, k] = sum_j A[i, j] B[j, k].
        # W index: k * n + i.
        
        # A indices: a_row_offsets[0], a_row_offsets[1]
        # A cols: a_col_offsets[0], a_col_offsets[1]
        # B rows: b_row_offsets[0], b_row_offsets[1]
        # B cols: b_col_offsets[0], b_col_offsets[1]
        
        # Let's map local indices to global
        # A[0,0] -> U[a_row_offsets[0] * m + a_col_offsets[0]]
        # B[0,0] -> V[b_row_offsets[0] * p + b_col_offsets[0]]
        # C[0,0] -> W[b_col_offsets[0] * n + a_row_offsets[0]]
        
        # M1
        U[a_row_offsets[0] * m + a_col_offsets[0]][col_idx] += 1
        U[a_row_offsets[1] * m + a_col_offsets[1]][col_idx] += 1
        V[b_row_offsets[0] * p + b_col_offsets[0]][col_idx] += 1
        V[b_row_offsets[1] * p + b_col_offsets[1]][col_idx] += 1
        W[b_col_offsets[0] * n + a_row_offsets[0]][col_idx] += 1
        W[b_col_offsets[1] * n + a_row_offsets[1]][col_idx] += 1
        W[b_col_offsets[1] * n + a_row_offsets[0]][col_idx] -= 1
        W[b_col_offsets[0] * n + a_row_offsets[1]][col_idx] -= 1
        col_idx += 1
        
        # M2
        U[a_row_offsets[1] * m + a_col_offsets[0]][col_idx] += 1
        U[a_row_offsets[1] * m + a_col_offsets[1]][col_idx] += 1
        V[b_row_offsets[0] * p + b_col_offsets[0]][col_idx] += 1
        W[b_col_offsets[0] * n + a_row_offsets[1]][col_idx] += 1
        W[b_col_offsets[1] * n + a_row_offsets[1]][col_idx] += 1
        col_idx += 1
        
        # M3
        U[a_row_offsets[0] * m + a_col_offsets[0]][col_idx] += 1
        V[b_row_offsets[0] * p + b_col_offsets[1]][col_idx] += 1
        V[b_row_offsets[1] * p + b_col_offsets[1]][col_idx] -= 1
        W[b_col_offsets[1] * n + a_row_offsets[0]][col_idx] += 1
        W[b_col_offsets[1] * n + a_row_offsets[1]][col_idx] += 1
        col_idx += 1
        
        # M4
        U[a_row_offsets[1] * m + a_col_offsets[1]][col_idx] += 1
        V[b_row_offsets[1] * p + b_col_offsets[0]][col_idx] += 1
        V[b_row_offsets[0] * p + b_col_offsets[0]][col_idx] -= 1
        W[b_col_offsets[0] * n + a_row_offsets[0]][col_idx] += 1
        W[b_col_offsets[1] * n + a_row_offsets[1]][col_idx] += 1
        col_idx += 1
        
        # M5
        U[a_row_offsets[0] * m + a_col_offsets[0]][col_idx] += 1
        U[a_row_offsets[0] * m + a_col_offsets[1]][col_idx] += 1
        V[b_row_offsets[1] * p + b_col_offsets[1]][col_idx] += 1
        W[b_col_offsets[1] * n + a_row_offsets[0]][col_idx] -= 1
        W[b_col_offsets[0] * n + a_row_offsets[1]][col_idx] -= 1
        col_idx += 1
        
        # M6
        U[a_row_offsets[1] * m + a_col_offsets[0]][col_idx] += 1
        U[a_row_offsets[0] * m + a_col_offsets[0]][col_idx] -= 1
        V[b_row_offsets[0] * p + b_col_offsets[0]][col_idx] += 1
        V[b_row_offsets[0] * p + b_col_offsets[1]][col_idx] += 1
        W[b_col_offsets[1] * n + a_row_offsets[1]][col_idx] += 1
        col_idx += 1
        
        # M7
        U[a_row_offsets[0] * m + a_col_offsets[1]][col_idx] += 1
        U[a_row_offsets[1] * m + a_col_offsets[1]][col_idx] -= 1
        V[b_row_offsets[1] * p + b_col_offsets[0]][col_idx] += 1
        V[b_row_offsets[1] * p + b_col_offsets[1]][col_idx] += 1
        W[b_col_offsets[0] * n + a_row_offsets[0]][col_idx] += 1
        col_idx += 1

    def add_schoolbook_221(a_row_offsets, a_col_offsets, b_row_offsets, b_col_offsets):
        """
        Adds a rank-4 schoolbook decomposition for 2x2x1 multiplication.
        """
        nonlocal col_idx
        for i in range(2):
            for j in range(2):
                U[a_row_offsets[i] * m + a_col_offsets[j]][col_idx] += 1
                V[b_row_offsets[j] * p + b_col_offsets[0]][col_idx] += 1
                W[b_col_offsets[0] * n + a_row_offsets[i]][col_idx] += 1
                col_idx += 1

    # Block 1: A rows 0-1, cols 0-1. B rows 0-1, cols 0-4.
    # Split B cols into 0-1 (2 cols) and 2-4 (3 cols).
    
    # Sub-block 1a: 2x2x2 (Strassen)
    add_strassen_222(
        a_row_offsets=[0, 1],
        a_col_offsets=[0, 1],
        b_row_offsets=[0, 1],
        b_col_offsets=[0, 1]
    )
    
    # Sub-block 1b: 2x2x3. Split B cols 2-4 into 2 (1 col) and 3-4 (2 cols).
    # Sub-sub-block 1b1: 2x2x1 (Schoolbook)
    add_schoolbook_221(
        a_row_offsets=[0, 1],
        a_col_offsets=[0, 1],
        b_row_offsets=[0, 1],
        b_col_offsets=[2]
    )
    
    # Sub-sub-block 1b2: 2x2x2 (Strassen)
    add_strassen_222(
        a_row_offsets=[0, 1],
        a_col_offsets=[0, 1],
        b_row_offsets=[0, 1],
        b_col_offsets=[3, 4]
    )

    # Block 2: A rows 0-1, cols 2-3. B rows 2-3, cols 0-4.
    # Split B cols into 0-1 (2 cols) and 2-4 (3 cols).
    
    # Sub-block 2a: 2x2x2 (Strassen)
    add_strassen_222(
        a_row_offsets=[0, 1],
        a_col_offsets=[2, 3],
        b_row_offsets=[2, 3],
        b_col_offsets=[0, 1]
    )
    
    # Sub-block 2b: 2x2x3. Split B cols 2-4 into 2 (1 col) and 3-4 (2 cols).
    # Sub-sub-block 2b1: 2x2x1 (Schoolbook)
    add_schoolbook_221(
        a_row_offsets=[0, 1],
        a_col_offsets=[2, 3],
        b_row_offsets=[2, 3],
        b_col_offsets=[2]
    )
    
    # Sub-sub-block 2b2: 2x2x2 (Strassen)
    add_strassen_222(
        a_row_offsets=[0, 1],
        a_col_offsets=[2, 3],
        b_row_offsets=[2, 3],
        b_col_offsets=[3, 4]
    )

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
