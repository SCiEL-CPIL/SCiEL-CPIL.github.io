import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    
    # Total rank 44
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook 1x2x2 block
    def add_1x2x2(i_off, j_off, k_off):
        nonlocal col
        # Local i=0, j in 0,1, k in 0,1
        for j in range(2):
            for k in range(2):
                U[i_off * m + (j_off + j)][col] = 1
                V[(j_off + j) * p + (k_off + k)][col] = 1
                W[(k_off + k) * n + (i_off + 0)][col] = 1
                col += 1
                
    # Helper to add a Strassen 2x2x2 block
    def add_2x2x2(i_off, j_off, k_off):
        nonlocal col
        # Strassen matrices
        # U rows: i_off+m, i_off+m+1 (local 0,1)
        # V rows: j_off+p, j_off+p+1
        # W rows: k_off+n, k_off+n+1
        
        # M1
        U[i_off * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1
        
        # M2
        U[(i_off+1) * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        W[k_off * n + (i_off+1)][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = -1
        col += 1
        
        # M3
        U[i_off * m + j_off][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = -1
        W[(k_off+1) * n + i_off][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1
        
        # M4
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[(j_off+1) * p + k_off][col] = 1
        V[j_off * p + k_off][col] = -1
        W[k_off * n + i_off][col] = 1
        W[k_off * n + (i_off+1)][col] = 1
        col += 1
        
        # M5
        U[i_off * m + j_off][col] = 1
        U[i_off * m + (j_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = -1
        W[(k_off+1) * n + i_off][col] = 1
        col += 1
        
        # M6
        U[(i_off+1) * m + j_off][col] = 1
        U[i_off * m + j_off][col] = -1
        V[j_off * p + k_off][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1
        
        # M7
        U[i_off * m + (j_off+1)][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = -1
        V[(j_off+1) * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = 1
        col += 1

    # 1. A11 B11: i=0, j=0:2, k=0:4
    # A11 is 1x2, B11 is 2x4.
    # Split B11 into two 2x2 blocks?
    # My previous logic:
    # A11 (1x2) * B11 (2x4).
    # B11 cols 0:2 and 2:4.
    # So two 1x2x2 blocks.
    
    # Block 1.1: A11 (1x2) * B11_part1 (2x2)
    # i_off=0, j_off=0, k_off=0
    add_1x2x2(0, 0, 0)
    
    # Block 1.2: A11 (1x2) * B11_part2 (2x2)
    # i_off=0, j_off=0, k_off=2
    add_1x2x2(0, 0, 2)
    
    # 2. A11 B12: i=0, j=0:2, k=4:8
    # B12 is 2x4. Split into 2x2.
    # Block 2.1: i_off=0, j_off=0, k_off=4
    add_1x2x2(0, 0, 4)
    
    # Block 2.2: i_off=0, j_off=0, k_off=6
    add_1x2x2(0, 0, 6)
    
    # 3. A21 B11: i=1:3, j=0:2, k=0:4
    # A21 is 2x2, B11 is 2x4.
    # Split B11 into 2x2.
    # Block 3.1: i_off=1, j_off=0, k_off=0
    add_2x2x2(1, 0, 0)
    
    # Block 3.2: i_off=1, j_off=0, k_off=2
    add_2x2x2(1, 0, 2)
    
    # 4. A21 B12: i=1:3, j=0:2, k=4:8
    # Block 4.1: i_off=1, j_off=0, k_off=4
    add_2x2x2(1, 0, 4)
    
    # Block 4.2: i_off=1, j_off=0, k_off=6
    add_2x2x2(1, 0, 6)
    
    # 5. A12 B21: i=0, j=2:4, k=0:4
    # A12 is 1x2, B21 is 2x4.
    # Block 5.1: i_off=0, j_off=2, k_off=0
    add_1x2x2(0, 2, 0)
    
    # Block 5.2: i_off=0, j_off=2, k_off=2
    add_1x2x2(0, 2, 2)
    
    # 6. A12 B22: i=0, j=2:4, k=4:8
    # Block 6.1: i_off=0, j_off=2, k_off=4
    add_1x2x2(0, 2, 4)
    
    # Block 6.2: i_off=0, j_off=2, k_off=6
    add_1x2x2(0, 2, 6)
    
    # 7. A22 B21: i=1:3, j=2:4, k=0:4
    # Block 7.1: i_off=1, j_off=2, k_off=0
    add_2x2x2(1, 2, 0)
    
    # Block 7.2: i_off=1, j_off=2, k_off=2
    add_2x2x2(1, 2, 2)
    
    # 8. A22 B22: i=1:3, j=2:4, k=4:8
    # Block 8.1: i_off=1, j_off=2, k_off=4
    add_2x2x2(1, 2, 4)
    
    # Block 8.2: i_off=1, j_off=2, k_off=6
    add_2x2x2(1, 2, 6)
    
    return {"U": U, "V": V, "W": W}
