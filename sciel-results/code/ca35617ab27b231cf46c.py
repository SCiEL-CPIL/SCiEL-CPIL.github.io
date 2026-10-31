import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 Strassen steps
    # Each step is a tuple of (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: [x00, x01, x10, x11]
    # v_coeffs: [y00, y01, y10, y11]
    # w_coeffs: [z00, z01, z10, z11]
    
    steps = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # P1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # P2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # P3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # P4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # P5
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]), # P6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # P7
    ]
    
    # Block definitions
    # u_idx: [00, 01, 10, 11]
    # v_idx: [00, 01, 10, 11]
    # w_idx: [00, 01, 10, 11] (mapped to Z00, Z01, Z10, Z11)
    
    blocks = [
        # A11 B11 -> C11
        ([0, 1, 4, 5], [0, 1, 4, 5], [0, 4, 1, 5]),
        # A12 B21 -> C11
        ([2, 3, 6, 7], [8, 9, 12, 13], [0, 4, 1, 5]),
        # A11 B12 -> C12
        ([0, 1, 4, 5], [2, 3, 6, 7], [8, 12, 9, 13]),
        # A12 B22 -> C12
        ([2, 3, 6, 7], [10, 11, 14, 15], [8, 12, 9, 13]),
        # A21 B11 -> C21
        ([8, 9, 12, 13], [0, 1, 4, 5], [2, 6, 3, 7]),
        # A22 B21 -> C21
        ([10, 11, 14, 15], [8, 9, 12, 13], [2, 6, 3, 7]),
        # A21 B12 -> C22
        ([8, 9, 12, 13], [2, 3, 6, 7], [10, 14, 11, 15]),
        # A22 B22 -> C22
        ([10, 11, 14, 15], [10, 11, 14, 15], [10, 14, 11, 15]),
    ]
    
    col = 0
    for u_idx, v_idx, w_idx in blocks:
        for u_c, v_c, w_c in steps:
            for i in range(4):
                if u_c[i] != 0:
                    U[u_idx[i]][col] = u_c[i]
                if v_c[i] != 0:
                    V[v_idx[i]][col] = v_c[i]
                if w_c[i] != 0:
                    W[w_idx[i]][col] = w_c[i]
            col += 1
            
    return {"U": U, "V": V, "W": W}
