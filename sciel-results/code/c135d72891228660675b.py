import numpy as np

def construct():
    n, m, p = 2, 2, 14
    nb = p // 2
    R = 7 * nb

    # Strassen 2x2x2 coefficients
    # Each term: list of (a_idx, b_idx, c_idx, sign) where
    # a_idx in {0,1,2,3} -> A[row][col] = A[a//2][a%2]
    # b_idx in {0,1,2,3} -> B[col][row] mapped to B[row][2b+b%2] for block b
    # c_idx in {0,1,2,3} -> C[row][col] = C[c//2][c%2]
    # Actually let me use the standard formulation:
    # M1=(a00+a11)(b00+b11), M2=(a10+a11)b00, M3=a00(b01-b11), M4=a11(b10-b00)
    # M5=(a00+a01)b11, M6=(a10-a00)(b00+b01), M7=(a01-a11)(b10+b11)
    # C00=M1-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M6

    # For each term, define: U_vec (4-dim), V_local (4-dim), W_local (4-dim)
    # U index: i*m+j = 2*i+j
    # V local index for block b: row*2 + col_within_block, where col_global = 2*b + col_within
    #   So V[2b+0], V[2b+1], V[2b+2], V[2b+3] correspond to B[0][2b], B[1][2b], B[0][2b+1], B[1][2b+1]
    # W index: k*n+i = 2*k+i (trace convention)
    #   C[i][k] = sum_j A[i][j]B[j][k]
    #   W index for C[i][k]: k*n + i = 2*k + i
    #   Local W: for column k=2b: C[0][2b]->4b, C[1][2b]->4b+1, C[0][2b+1]->4b+2, C[1][2b+1]->4b+3

    # Term definitions: (U_indices with values, V_local_indices with values, W_local_indices with values)
    # U: [a00, a01, a10, a11] -> indices 0,1,2,3
    # V_local: [b00, b10, b01, b11] for a block -> indices 0,1,2,3 (col 0 row 0, col 0 row 1, col 1 row 0, col 1 row 1)
    # Wait, B is 2x14, so B[row][col]. For block b, B[0][2b], B[1][2b], B[0][2b+1], B[1][2b+1]
    # V vector index: row*m + col = 2*row + col. So B[0][2b] -> 2b, B[1][2b] -> 2b+1, B[0][2b+1] -> 2b+2, B[1][2b+1] -> 2b+3
    # W: C is 14x2. C[row][col] where row=2b or 2b+1, col=0 or 1.
    # W index: k*n + i = 2*row + col (since W[k*n+i,r] with k=row, i=col)
    # So C[2b][0] -> 4b, C[2b][1] -> 4b+1, C[2b+1][0] -> 4b+2, C[2b+1][1] -> 4b+3

    # Standard Strassen for C = A*B where A is 2x2, B is 2x2:
    # C[0][0] = M1 - M5 + M7
    # C[0][1] = M3 + M5
    # C[1][0] = M2 + M4
    # C[1][1] = M1 - M2 + M6

    # M1: (a00+a11)*(b00+b11) -> U=[1,0,0,1], V=[1,0,0,1] (b00,b10,b01,b11)
    # M2: (a10+a11)*b00 -> U=[0,0,1,1], V=[1,0,0,0]
    # M3: a00*(b01-b11) -> U=[1,0,0,0], V=[0,0,1,-1]
    # M4: a11*(b10-b00) -> U=[0,0,0,1], V=[0,1,0,-1]
    # M5: (a00+a01)*b11 -> U=[1,1,0,0], V=[0,0,0,1]
    # M6: (a10-a00)*(b00+b01) -> U=[-1,0,1,0], V=[1,0,1,0]
    # M7: (a01-a11)*(b10+b11) -> U=[0,1,0,-1], V=[0,1,0,1]

    terms_U = np.array([
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1],  # M7
    ], dtype=float)

    terms_V = np.array([
        [1, 0, 0, 1],   # M1: b00+b11
        [1, 0, 0, 0],   # M2: b00
        [0, 0, 1, -1],  # M3: b01-b11
        [0, 1, 0, -1],  # M4: b10-b00
        [0, 0, 0, 1],   # M5: b11
        [1, 0, 1, 0],   # M6: b00+b01
        [0, 1, 0, 1],   # M7: b10+b11
    ], dtype=float)

    terms_W = np.array([
        [1, 0, 0, 1],   # M1: C00+C11
        [0, 0, 1, -1],  # M2: C10-C11
        [0, 1, 0, 1],   # M3: C01+C11
        [1, 1, 0, 0],   # M4: C00+C10
        [-1, 0, 1, 0],  # M5: -C00+C01
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, 0],   # M7: C00
    ], dtype=float)

    # Build matrices
    U_mat = np.zeros((4, R))
    V_mat = np.zeros((28, R))
    W_mat = np.zeros((28, R))

    col = 0
    for b in range(nb):
        for t in range(7):
            U_mat[:, col] = terms_U[t]
            # V: local indices 0,1,2,3 -> global 2b, 2b+1, 2b+2, 2b+3
            V_mat[2*b:2*b+4, col] = terms_V[t]
            # W: local indices 0,1,2,3 -> global 4b, 4b+1, 4b+2, 4b+3
            W_mat[4*b:4*b+4, col] = terms_W[t]
            col += 1

    # Convert to lists of lists
    return {
        "U": U_mat.tolist(),
        "V": V_mat.tolist(),
        "W": W_mat.tolist()
    }
