import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Indices for A (2x4) -> 8 elements
    # A = [[a0, a1, a2, a3], [a4, a5, a6, a7]]
    # Block A1 (cols 0-1): [a0, a1; a4, a5]
    # Block A2 (cols 2-3): [a2, a3; a6, a7]
    
    # Indices for B (4x4) -> 16 elements
    # B = [[b0, b1, b2, b3],
    #      [b4, b5, b6, b7],
    #      [b8, b9, b10, b11],
    #      [b12, b13, b14, b15]]
    # Block B11 (rows 0-1, cols 0-1): [[b0, b1], [b4, b5]]
    # Block B12 (rows 0-1, cols 2-3): [[b2, b3], [b6, b7]]
    # Block B21 (rows 2-3, cols 0-1): [[b8, b9], [b12, b13]]
    # Block B22 (rows 2-3, cols 2-3): [[b10, b11], [b14, b15]]
    
    # Indices for C (2x4) -> 8 elements
    # C = [[c0, c1, c2, c3], [c4, c5, c6, c7]]
    # C1 (cols 0-1): [[c0, c1], [c4, c5]]
    # C2 (cols 2-3): [[c2, c3], [c6, c7]]

    def add_strassen_block(col_offset, A1_idx, B_idx, C_idx):
        # A1_idx: [a11, a12, a21, a22] indices in U
        # B_idx: [b11, b12, b21, b22] indices in V
        # C_idx: [c11, c12, c21, c22] indices in W
        
        a11, a12, a21, a22 = A1_idx
        b11, b12, b21, b22 = B_idx
        c11, c12, c21, c22 = C_idx

        # M1 = (a11 + a22) * (b11 + b22)
        U[a11, col_offset + 0] += 1
        U[a22, col_offset + 0] += 1
        V[b11, col_offset + 0] += 1
        V[b22, col_offset + 0] += 1
        W[c11, col_offset + 0] += 1
        W[c22, col_offset + 0] += 1

        # M2 = (a21 + a22) * b11
        U[a21, col_offset + 1] += 1
        U[a22, col_offset + 1] += 1
        V[b11, col_offset + 1] += 1
        W[c21, col_offset + 1] += 1
        W[c22, col_offset + 1] -= 1

        # M3 = a11 * (b12 - b22)
        U[a11, col_offset + 2] += 1
        V[b12, col_offset + 2] += 1
        V[b22, col_offset + 2] -= 1
        W[c12, col_offset + 2] += 1
        W[c22, col_offset + 2] += 1

        # M4 = a22 * (b21 - b11)
        U[a22, col_offset + 3] += 1
        V[b21, col_offset + 3] += 1
        V[b11, col_offset + 3] -= 1
        W[c11, col_offset + 3] += 1
        W[c21, col_offset + 3] += 1

        # M5 = (a11 + a12) * b22
        U[a11, col_offset + 4] += 1
        U[a12, col_offset + 4] += 1
        V[b22, col_offset + 4] += 1
        W[c11, col_offset + 4] -= 1
        W[c12, col_offset + 4] += 1

        # M6 = (a21 - a11) * (b11 + b12)
        U[a21, col_offset + 5] += 1
        U[a11, col_offset + 5] -= 1
        V[b11, col_offset + 5] += 1
        V[b12, col_offset + 5] += 1
        W[c22, col_offset + 5] += 1

        # M7 = (a12 - a22) * (b21 + b22)
        U[a12, col_offset + 6] += 1
        U[a22, col_offset + 6] -= 1
        V[b21, col_offset + 6] += 1
        V[b22, col_offset + 6] += 1
        W[c11, col_offset + 6] += 1

    # Block 1: A1 * B11 -> contributes to C1
    # A1 indices: a0, a1, a4, a5
    # B11 indices: b0, b1, b4, b5
    # C1 indices: c0, c1, c4, c5
    add_strassen_block(0, [0, 1, 4, 5], [0, 1, 4, 5], [0, 1, 4, 5])

    # Block 2: A2 * B21 -> contributes to C1
    # A2 indices: a2, a3, a6, a7
    # B21 indices: b8, b9, b12, b13
    # C1 indices: c0, c1, c4, c5 (Add to same C1)
    add_strassen_block(7, [2, 3, 6, 7], [8, 9, 12, 13], [0, 1, 4, 5])

    # Block 3: A1 * B12 -> contributes to C2
    # A1 indices: a0, a1, a4, a5
    # B12 indices: b2, b3, b6, b7
    # C2 indices: c2, c3, c6, c7
    add_strassen_block(14, [0, 1, 4, 5], [2, 3, 6, 7], [2, 3, 6, 7])

    # Block 4: A2 * B22 -> contributes to C2
    # A2 indices: a2, a3, a6, a7
    # B22 indices: b10, b11, b14, b15
    # C2 indices: c2, c3, c6, c7 (Add to same C2)
    add_strassen_block(21, [2, 3, 6, 7], [10, 11, 14, 15], [2, 3, 6, 7])

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
