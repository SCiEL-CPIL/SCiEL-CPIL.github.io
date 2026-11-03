import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Try Rank 24 construction
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    col = 0
    # Strassen for top-left 2x2
    # M1
    add_term(0, 1, 0, 1, 0, 1, col); add_term(4, 1, 4, 1, 4, 1, col); col += 1
    # M2
    add_term(3, 1, 0, 1, 3, 1, col); add_term(4, 1, 0, 0, 4, -1, col); col += 1
    # M3
    add_term(0, 1, 3, 1, 0, 1, col); add_term(0, 0, 4, -1, 0, 0, col); col += 1
    # M4
    add_term(4, 1, 1, 1, 0, -1, col); add_term(4, 0, 0, -1, 3, 1, col); col += 1
    # M5
    add_term(0, 1, 4, 1, 1, 1, col); add_term(1, 1, 4, 0, 4, 1, col); col += 1
    # M6
    add_term(3, 1, 0, 1, 4, 1, col); add_term(0, -1, 1, 1, 4, 0, col); col += 1
    # M7
    add_term(1, 1, 3, 1, 0, -1, col); add_term(4, -1, 4, 1, 1, 1, col); col += 1
    
    # qvT: A02 B20 -> C00, A02 B21 -> C01, A12 B20 -> C10, A12 B21 -> C11
    add_term(2, 1, 6, 1, 0, 1, col); col += 1
    add_term(2, 1, 7, 1, 1, 1, col); col += 1
    add_term(5, 1, 6, 1, 3, 1, col); col += 1
    add_term(5, 1, 7, 1, 4, 1, col); col += 1
    
    # Pu: A00 B02 -> C02, A01 B12 -> C02, A10 B02 -> C12, A11 B12 -> C12
    add_term(0, 1, 2, 1, 2, 1, col); col += 1
    add_term(1, 1, 5, 1, 2, 1, col); col += 1
    add_term(3, 1, 2, 1, 5, 1, col); col += 1
    add_term(4, 1, 5, 1, 5, 1, col); col += 1
    
    # rTQ: A20 B00 -> C20, A21 B10 -> C20, A20 B01 -> C21, A21 B11 -> C21
    add_term(6, 1, 0, 1, 6, 1, col); col += 1
    add_term(7, 1, 3, 1, 6, 1, col); col += 1
    add_term(6, 1, 1, 1, 7, 1, col); col += 1
    add_term(7, 1, 4, 1, 7, 1, col); col += 1
    
    # alpha vT: A22 B20 -> C20, A22 B21 -> C21
    add_term(8, 1, 6, 1, 6, 1, col); col += 1
    add_term(8, 1, 7, 1, 7, 1, col); col += 1
    
    # rT w: A20 B02 -> C22, A21 B12 -> C22
    add_term(6, 1, 2, 1, 8, 1, col); col += 1
    add_term(7, 1, 5, 1, 8, 1, col); col += 1
    
    # alpha beta: A22 B22 -> C22
    add_term(8, 1, 8, 1, 8, 1, col); col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
