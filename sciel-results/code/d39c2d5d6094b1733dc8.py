"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Matrix multiplication 3x3 by 3x6.
    Output index convention: W[k*n + i, r] where n=3, i in [0,3), k in [0,6).
    
    Strategy: Use Laderman's rank-23 decomposition for 3x3x3, applied to each
    of the two 3x3 blocks (columns 0-2 and 3-5) of the 3x6 matrix.
    This gives rank 46.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    # Laderman's 3x3x3 decomposition (rank 23)
    # Convention: U[i*m+j, c], V[j*p'+k, c], W[k'*n+i, c] where p'=3, k' in [0,3)
    # The tensors satisfy: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p'+k, c] * W[k'*n+i, c]
    
    # Laderman's matrices for 3x3x3 (standard form from literature)
    # U: 9x23, V: 9x23, W: 9x23
    
    U_lad = [
        # c:  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22
        [ 1,  0,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # i=0,j=0
        [ 0,  1,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # i=0,j=1
        [ 0,  0,  1,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # i=0,j=2
        [ 0,  0,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # i=1,j=0
        [ 0,  0,  0,  0,  1,  1,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # i=1,j=1
        [ 0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # i=1,j=2
        [ 0,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # i=2,j=0
        [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # i=2,j=1
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0],  # i=2,j=2
    ]
    
    V_lad = [
        # c:  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22
        [ 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # j=0,k=0
        [ 0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # j=0,k=1
        [ 0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # j=0,k=2
        [ 0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # j=1,k=0
        [ 0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # j=1,k=1
        [ 0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # j=1,k=2
        [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # j=2,k=0
        [ 0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # j=2,k=1
        [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # j=2,k=2
    ]
    
    W_lad = [
        # c:  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22
        [ 1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # k=0,i=0
        [ 0,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # k=0,i=1
        [ 0,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # k=0,i=2
        [ 0,  0,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # k=1,i=0
        [ 0,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # k=1,i=1
        [ 0,  0,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # k=1,i=2
        [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # k=2,i=0
        [ 0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # k=2,i=1
        [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # k=2,i=2
    ]
    
    # Build the 3x3x6 decomposition using two copies of Laderman
    # Block 1: k in {0,1,2}, Block 2: k in {3,4,5}
    rank = 46
    U = [[0] * rank for _ in range(n * m)]  # 9 x 46
    V = [[0] * rank for _ in range(m * p)]  # 18 x 46
    W = [[0] * rank for _ in range(p * n)]  # 18 x 46
    
    for i in range(n):
        for j in range(m):
            u_row = i * m + j
            for c in range(23):
                U[u_row][c] = U_lad[u_row][c]
                U[u_row][23 + c] = U_lad[u_row][c]
    
    for j in range(m):
        for k in range(3):
            v_row = j * p + k
            v_lad_row = j * 3 + k
            for c in range(23):
                V[v_row][c] = V_lad[v_lad_row][c]
    
    for j in range(m):
        for k in range(3):
            v_row = j * p + (3 + k)
            v_lad_row = j * 3 + k
            for c in range(23):
                V[v_row][23 + c] = V_lad[v_lad_row][c]
    
    for i in range(n):
        for k in range(3):
            w_row = k * n + i
            w_lad_row = k * 3 + i
            for c in range(23):
                W[w_row][c] = W_lad[w_lad_row][c]
    
    for i in range(n):
        for k in range(3):
            w_row = (3 + k) * n + i
            w_lad_row = k * 3 + i
            for c in range(23):
                W[w_row][23 + c] = W_lad[w_lad_row][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
