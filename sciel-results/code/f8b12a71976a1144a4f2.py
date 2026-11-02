import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=7
    # Rank 39 decomposition
    rank = 39
    
    # Initialize U, V, W as lists of lists for easy assignment, then convert if needed
    # U: (n*m, rank) = (6, 39)
    # V: (m*p, rank) = (21, 39)
    # W: (p*n, rank) = (14, 39)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_2 B_2
    # A_2 is column j=2 of A (2x1)
    # B_2 is row j=2 of B (1x7)
    # Product contributes to C[i,k] = A[i,2] * B[2,k]
    # i in 0..1, k in 0..6
    # 14 terms
    for i in range(n):
        for k in range(p):
            # U index: i * m + 2
            # V index: 2 * p + k
            # W index: k * n + i
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A_1 B_1
    # A_1 is columns j=0,1 of A (2x2)
    # B_1 is rows j=0,1 of B (2x7)
    # Split B_1 columns (0..6) into groups:
    # Group 1: k=0 (size 1). Rank 4.
    # Group 2: k=1,2 (size 2). Rank 7.
    # Group 3: k=3,4 (size 2). Rank 7.
    # Group 4: k=5,6 (size 2). Rank 7.
    
    # Helper to add 2x2x1 decomposition
    # A' (2x2) at A rows 0,1 cols 0,1
    # B' (2x1) at B rows 0,1 col k0
    # C' (2x1) at C rows 0,1 col k0
    def add_2x2x1(k0):
        nonlocal col
        # Terms:
        # C'_0 = A'_{00}B'_{00} + A'_{01}B'_{10}
        # C'_1 = A'_{10}B'_{00} + A'_{11}B'_{10}
        
        # r0: A'_{00} B'_{00} -> C'_0
        U[0 * m + 0][col] = 1
        V[0 * p + k0][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
        
        # r1: A'_{01} B'_{10} -> C'_0
        U[0 * m + 1][col] = 1
        V[1 * p + k0][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
        
        # r2: A'_{10} B'_{00} -> C'_1
        U[1 * m + 0][col] = 1
        V[0 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # r3: A'_{11} B'_{10} -> C'_1
        U[1 * m + 1][col] = 1
        V[1 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1

    # Helper to add 2x2x2 decomposition (Strassen)
    # A' (2x2) at A rows 0,1 cols 0,1
    # B' (2x2) at B rows 0,1 cols k0, k0+1
    # C' (2x2) at C rows 0,1 cols k0, k0+1
    def add_2x2x2(k0):
        nonlocal col
        k1 = k0 + 1
        
        # P1 = (A00+A11)(B00+B11)
        U[0 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        # W for P1 contributes to C00, C11
        # C00 += P1, C11 += P1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # P2 = (A10+A11)B00
        U[1 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        # C10 += P2, C11 += -P2
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        # P3 = A00(B01-B11)
        U[0 * m + 0][col] = 1
        V[0 * p + k1][col] = 1
        V[1 * p + k1][col] = -1
        # C01 += P3, C11 += -P3
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        # P4 = A11(B10-B00)
        U[1 * m + 1][col] = 1
        V[1 * p + k0][col] = 1
        V[0 * p + k0][col] = -1
        # C00 += -P4, C10 += P4
        W[k0 * n + 0][col] = -1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # P5 = (A00+A01)B11
        U[0 * m + 0][col] = 1
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        # C01 += P5, C11 += P5
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # P6 = (A10-A00)(B00+B01)
        U[1 * m + 0][col] = 1
        U[0 * m + 0][col] = -1
        V[0 * p + k0][col] = 1
        V[0 * p + k1][col] = 1
        # C00 += P6
        W[k0 * n + 0][col] = 1
        col += 1
        
        # P7 = (A01-A11)(B10+B11)
        U[0 * m + 1][col] = 1
        U[1 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        # C01 += P7, C00 += -P7
        W[k1 * n + 0][col] = 1
        W[k0 * n + 0][col] = -1
        col += 1

    # Apply groups
    # Group 1: k=0
    add_2x2x1(0)
    
    # Group 2: k=1,2
    add_2x2x2(1)
    
    # Group 3: k=3,4
    add_2x2x2(3)
    
    # Group 4: k=5,6
    add_2x2x2(5)
    
    return {"U": U, "V": V, "W": W}
