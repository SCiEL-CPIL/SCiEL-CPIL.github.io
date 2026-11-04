import time
import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = 3, 3, 4
    # Rank 32 decomposition
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a block
    def add_block(local_U, local_V, local_W, i0, j0, k0, ns, ms, ps, start_col):
        # local_U: ns*ms x r
        # local_V: ms*ps x r
        # local_W: ps*ns x r
        r = len(local_U[0])
        for i in range(ns):
            for j in range(ms):
                for c in range(r):
                    if local_U[i*ms+j][c] != 0:
                        U[(i0+i)*m + (j0+j)][start_col+c] += local_U[i*ms+j][c]
        for j in range(ms):
            for k in range(ps):
                for c in range(r):
                    if local_V[j*ps+k][c] != 0:
                        V[(j0+j)*p + (k0+k)][start_col+c] += local_V[j*ps+k][c]
        for k in range(ps):
            for i in range(ns):
                for c in range(r):
                    if local_W[k*ns+i][c] != 0:
                        W[(k0+k)*n + (i0+i)][start_col+c] += local_W[k*ns+i][c]
        return r

    # Strassen 2x2x2
    Strassen_U = [
        [1, 1, 1, 0, 1, -1, 0],
        [0, 0, 1, 0, 1, 0, -1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    Strassen_V = [
        [1, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 1, 0, -1, 1, 0, 1],
        [1, 0, -1, 1, 1, 0, 1]
    ]
    Strassen_W = [
        [1, 0, 0, -1, 0, 1, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, 0, -1, 0, 1, 0, 1]
    ]

    # 2x1x2 (Outer product 2x1 * 1x2) Rank 4
    # U: 2x1 x 4. Rows i,j=0. i=0,1. j=0.
    # V: 1x2 x 4. Rows j=0, k=0,1.
    # W: 2x2 x 4. Rows k,i. k=0,1. i=0,1.
    # X[i,k] = Y[i] Z[k]
    # Term 0: Y0 Z0 -> X00
    # Term 1: Y0 Z1 -> X01
    # Term 2: Y1 Z0 -> X10
    # Term 3: Y1 Z1 -> X11
    Outer_212_U = [
        [1, 1, 0, 0], # i=0
        [0, 0, 1, 1]  # i=1
    ]
    Outer_212_V = [
        [1, 0, 1, 0]  # j=0, k=0,1? No, V rows are j*p+k. p=2.
        # Row 0 (j=0,k=0): 1, 0, 1, 0?
        # We need V[0,0]=1 for term 0, V[0,1]=0 for term 1?
        # Term 0: Y0 Z0. U[0,0]=1, V[0,0]=1, W[0,0]=1.
        # Term 1: Y0 Z1. U[0,1]=1, V[0,1]=1, W[1,1]=1?
        # W rows: k*n+i. n=2.
        # k=0, i=0 -> 0.
        # k=1, i=0 -> 2.
        # k=0, i=1 -> 1.
        # k=1, i=1 -> 3.
        
        # Let's construct carefully.
        # Term 0: X00. U[0,0]=1, V[0,0]=1, W[0,0]=1.
        # Term 1: X01. U[0,1]=1, V[1,1]=1? No, V is for Z.
        # Z is 1x2. V row index is j*p+k. j=0.
        # V[0,0] corresponds to Z[0,0].
        # V[1,0] corresponds to Z[0,1].
        # So for Term 1 (Y0 Z1), we need V[1,1]=1.
        # And W[2,1]=1 (k=1, i=0).
        
        # Term 2: X10. U[1,2]=1, V[0,2]=1, W[1,2]=1 (k=0, i=1).
        # Term 3: X11. U[1,3]=1, V[1,3]=1, W[3,3]=1 (k=1, i=1).
        
    ]
    
    # Correct Outer_212 factors
    Outer_212_U = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    Outer_212_V = [
        [1, 0, 1, 0], # j=0, k=0
        [0, 1, 0, 1]  # j=0, k=1
    ]
    Outer_212_W = [
        [1, 0, 0, 0], # k=0, i=0
        [0, 0, 1, 0], # k=0, i=1
        [0, 1, 0, 0], # k=1, i=0
        [0, 0, 0, 1]  # k=1, i=1
    ]

    # 1x2x1 (Dot product) Rank 2
    # Y 1x2, Z 2x1. X 1x1.
    # X = Y0 Z0 + Y1 Z1.
    Dot_121_U = [
        [1, 1] # i=0, j=0,1
    ]
    Dot_121_V = [
        [1, 0], # j=0, k=0
        [0, 1]  # j=1, k=0
    ]
    Dot_121_W = [
        [1, 1] # k=0, i=0
    ]

    # 1x1x2 Rank 2
    # Y 1x1, Z 1x2. X 1x2.
    # X0 = Y0 Z0. X1 = Y0 Z1.
    Outer_112_U = [
        [1, 1]
    ]
    Outer_112_V = [
        [1, 0], # j=0, k=0
        [0, 1]  # j=0, k=1
    ]
    Outer_112_W = [
        [1, 0], # k=0, i=0
        [0, 1]  # k=1, i=0
    ]

    # 1x2x2 Rank 4
    # Y 1x2, Z 2x2. X 1x2.
    # X0 = Y0 Z00 + Y1 Z10.
    # X1 = Y0 Z01 + Y1 Z11.
    Outer_122_U = [
        [1, 0, 1, 0], # i=0, j=0
        [0, 1, 0, 1]  # i=0, j=1
    ]
    Outer_122_V = [
        [1, 0, 0, 0], # j=0, k=0
        [0, 0, 1, 0], # j=0, k=1
        [0, 1, 0, 0], # j=1, k=0
        [0, 0, 0, 1]  # j=1, k=1
    ]
    Outer_122_W = [
        [1, 0, 1, 0], # k=0, i=0
        [0, 1, 0, 1]  # k=1, i=0
    ]

    # Block 1: A11 B11 (2x2x2) at (0,0,0)
    col += add_block(Strassen_U, Strassen_V, Strassen_W, 0, 0, 0, 2, 2, 2, col)
    # Block 2: A12 B21 (2x1x2) at (0,2,0)
    col += add_block(Outer_212_U, Outer_212_V, Outer_212_W, 0, 2, 0, 2, 1, 2, col)
    # Block 3: A11 B12 (2x2x2) at (0,0,2)
    col += add_block(Strassen_U, Strassen_V, Strassen_W, 0, 0, 2, 2, 2, 2, col)
    # Block 4: A12 B22 (2x1x2) at (0,2,2)
    col += add_block(Outer_212_U, Outer_212_V, Outer_212_W, 0, 2, 2, 2, 1, 2, col)
    # Block 5: A21 B11 (1x2x1) at (2,0,0)
    col += add_block(Dot_121_U, Dot_121_V, Dot_121_W, 2, 0, 0, 1, 2, 1, col)
    # Block 6: A21 B12 (1x2x2) at (2,0,2)
    col += add_block(Outer_122_U, Outer_122_V, Outer_122_W, 2, 0, 2, 1, 2, 2, col)
    # Block 7: A22 B21 (1x1x2) at (2,2,0)
    col += add_block(Outer_112_U, Outer_112_V, Outer_112_W, 2, 2, 0, 1, 1, 2, col)
    # Block 8: A22 B22 (1x1x2) at (2,2,2)
    col += add_block(Outer_112_U, Outer_112_V, Outer_112_W, 2, 2, 2, 1, 1, 2, col)
    
    return {"U": U, "V": V, "W": W}
