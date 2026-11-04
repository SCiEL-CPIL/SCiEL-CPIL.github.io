<<<<<<< SEARCH
def _block_decomposition():
    """
    Construct a rank-14 decomposition by splitting the 7-wide dimension
    into 3+3+1 blocks, using schoolbook for each sub-block.

    2x3x3 (rank 6) + 2x3x3 (rank 6) + 2x3x1 (rank 2) = rank 14.
    """
    n, m, p = 2, 3, 7
    rank = 6 + 6 + 2  # = 14

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    # Block 1: output columns k=0,1,2 (2x3x3 multiplication)
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Block 2: output columns k=3,4,5 (2x3x3 multiplication)
    for i in range(n):
        for j in range(m):
            for k in range(3, 6):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Block 3: output column k=6 (2x3x1 multiplication)
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 6][col] = 1
            W[6 * n + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
=======
def _block_decomposition():
    """
    Construct a rank-35 decomposition using Strassen for 2x2x2 sub-blocks.
    Split B (3x7) into B1 (3x2), B2 (3x2), B3 (3x3).
    1. A*B1 (2x3x2): Split A cols into 2+1. 2x2x2 (Strassen 7) + 2x1x2 (School 2) = 9.
    2. A*B2 (2x3x2): Same = 9.
    3. A*B3 (2x3x3): Split A cols into 2+1.
       A01*B3 (2x2x3): Split B3 cols into 2+1. 2x2x2 (Strassen 7) + 2x2x1 (School 4) = 11.
       A2*B3 (2x1x3): School 6.
       Total 17.
    Total Rank: 9 + 9 + 17 = 35.
    """
    n, m, p = 2, 3, 7
    rank = 35

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Helper to add a 2x2x2 Strassen block
    # U indices: i*3+j, V indices: j*3+k, W indices: k*2+i
    # Local indices: i' in {0,1}, j' in {0,1}, k' in {0,1}
    # Global offsets: i_off, j_off, k_off
    def add_strassen_2x2x2(i_off, j_off, k_off, start_col):
        U_str, V_str, W_str = _strassen_2x2x2()
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    u_idx = (i + i_off) * 3 + (j + j_off)
                    U[u_idx][start_col + c] += U_str[i * 2 + j][c]
            for j in range(2):
                for k in range(2):
                    v_idx = (j + j_off) * 3 + (k + k_off)
                    V[v_idx][start_col + c] += V_str[j * 2 + k][c]
            for k in range(2):
                for i in range(2):
                    w_idx = (k + k_off) * 2 + (i + i_off)
                    W[w_idx][start_col + c] += W_str[k * 2 + i][c]
        return start_col + 7

    # Helper to add a schoolbook block of size ni x nj x nk
    def add_schoolbook(i_off, j_off, k_off, ni, nj, nk, start_col):
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    u_idx = (i + i_off) * 3 + (j + j_off)
                    v_idx = (j + j_off) * 3 + (k + k_off)
                    w_idx = (k + k_off) * 2 + (i + i_off)
                    U[u_idx][start_col] = 1
                    V[v_idx][start_col] = 1
                    W[w_idx][start_col] = 1
                    start_col += 1
        return start_col

    # Block 1: A * B1 (k=0,1). B1 is 3x2.
    # Split A columns j=0,1 (2x2x2) and j=2 (2x1x2).
    # 1a. j in {0,1}, k in {0,1}: 2x2x2 Strassen
    col = add_strassen_2x2x2(0, 0, 0, col)
    # 1b. j=2, k in {0,1}: 2x1x2 Schoolbook
    col = add_schoolbook(0, 2, 0, 2, 1, 2, col)

    # Block 2: A * B2 (k=2,3). B2 is 3x2.
    # 2a. j in {0,1}, k in {2,3}: 2x2x2 Strassen
    col = add_strassen_2x2x2(0, 0, 2, col)
    # 2b. j=2, k in {2,3}: 2x1x2 Schoolbook
    col = add_schoolbook(0, 2, 2, 2, 1, 2, col)

    # Block 3: A * B3 (k=4,5,6). B3 is 3x3.
    # Split A columns j=0,1 (2x2x3) and j=2 (2x1x3).
    # 3a. j in {0,1}, k in {4,5,6}:
    #     Split B3 cols k=4,5 (2x2x2) and k=6 (2x2x1).
    #     3a1. j in {0,1}, k in {4,5}: 2x2x2 Strassen
    col = add_strassen_2x2x2(0, 0, 4, col)
    #     3a2. j in {0,1}, k=6: 2x2x1 Schoolbook
    col = add_schoolbook(0, 0, 6, 2, 2, 1, col)
    # 3b. j=2, k in {4,5,6}: 2x1x3 Schoolbook
    col = add_schoolbook(0, 2, 4, 2, 1, 3, col)

    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE

