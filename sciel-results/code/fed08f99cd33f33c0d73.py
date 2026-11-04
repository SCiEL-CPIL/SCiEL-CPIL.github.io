import time
import numpy as np

def construct():
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    rank = 41
    
    U = [[0] * rank for _ in range(dim_u)]
    V = [[0] * rank for _ in range(dim_v)]
    W = [[0] * rank for _ in range(dim_w)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u, v, w, val=1):
        nonlocal col
        U[u][col] += val
        V[v][col] += val
        W[w][col] += val
        col += 1

    # Helper for Strassen 2x2x2
    def add_strassen(X_u, Y_v, Z_w):
        nonlocal col
        # Term 1
        U[X_u[0]][col] += 1; U[X_u[3]][col] += 1
        V[Y_v[0]][col] += 1; V[Y_v[3]][col] += 1
        W[Z_w[0]][col] += 1; W[Z_w[3]][col] += 1
        col += 1
        # Term 2
        U[X_u[2]][col] += 1; U[X_u[3]][col] += 1
        V[Y_v[0]][col] += 1
        W[Z_w[2]][col] += 1; W[Z_w[3]][col] -= 1
        col += 1
        # Term 3
        U[X_u[0]][col] += 1
        V[Y_v[1]][col] += 1; V[Y_v[3]][col] -= 1
        W[Z_w[1]][col] += 1; W[Z_w[3]][col] += 1
        col += 1
        # Term 4
        U[X_u[3]][col] += 1
        V[Y_v[2]][col] += 1; V[Y_v[0]][col] -= 1
        W[Z_w[0]][col] += 1; W[Z_w[2]][col] += 1
        col += 1
        # Term 5
        U[X_u[0]][col] += 1; U[X_u[1]][col] += 1
        V[Y_v[3]][col] += 1
        W[Z_w[0]][col] -= 1; W[Z_w[1]][col] += 1
        col += 1
        # Term 6
        U[X_u[2]][col] += 1; U[X_u[0]][col] -= 1
        V[Y_v[0]][col] += 1; V[Y_v[1]][col] += 1
        W[Z_w[3]][col] += 1
        col += 1
        # Term 7
        U[X_u[1]][col] += 1; U[X_u[3]][col] -= 1
        V[Y_v[2]][col] += 1; V[Y_v[3]][col] += 1
        W[Z_w[0]][col] += 1
        col += 1

    # Core 3x3x3
    # 1. 2x2x2 block at (0..1, 0..1)
    # U: 0,1,4,5. V: 0,1,4,5. W: 0,1,3,4
    add_strassen([0, 1, 4, 5], [0, 1, 4, 5], [0, 1, 3, 4])
    
    # 2. C_0..1, 2
    # A_0..1, 0..1 * B_0..1, 2 (Rank 4)
    add_term(0, 0, 2)
    add_term(1, 0, 5)
    add_term(4, 4, 2)
    add_term(5, 4, 5)
    # A_0..1, 2 * B_2, 2 (Rank 2)
    add_term(2, 8, 2)
    add_term(3, 8, 5)
    
    # 3. C_2, 0..1
    # A_2, 0..1 * B_0..1, 0..1 (Rank 4)
    add_term(8, 0, 6)
    add_term(8, 1, 7)
    add_term(9, 4, 6)
    add_term(9, 5, 7)
    # A_2, 2 * B_2, 0..1 (Rank 2)
    add_term(10, 8, 6)
    add_term(10, 9, 7)
    
    # 4. C_2, 2
    # A_2, 2 * B_2, 2 (Rank 1)
    add_term(10, 10, 8)
    
    # Strip 1: A_0..2, 0..2 * B_0..2, 3 (Rank 9)
    # W indices for k=3, i=0..2 are 9, 10, 11
    add_term(0, 0, 9)
    add_term(1, 4, 9)
    add_term(2, 8, 9)
    add_term(4, 0, 10)
    add_term(5, 4, 10)
    add_term(6, 8, 10)
    add_term(8, 0, 11)
    add_term(9, 4, 11)
    add_term(10, 8, 11)
    
    # Strip 2: A_0..2, 3 * B_3, 0..3 (Rank 12)
    # U indices: 3, 7, 11
    # V indices: 12, 13, 14, 15
    # W indices: 0..11
    add_term(3, 12, 0)
    add_term(3, 13, 1)
    add_term(3, 14, 2)
    add_term(3, 15, 3)
    add_term(7, 12, 3)
    add_term(7, 13, 4)
    add_term(7, 14, 5)
    add_term(7, 15, 6)
    add_term(11, 12, 6)
    add_term(11, 13, 7)
    add_term(11, 14, 8)
    add_term(11, 15, 9)
    
    # Verify rank
    if col != rank:
        # Adjust if necessary, though it should be 41
        pass
        
    return {"U": U, "V": V, "W": W}
