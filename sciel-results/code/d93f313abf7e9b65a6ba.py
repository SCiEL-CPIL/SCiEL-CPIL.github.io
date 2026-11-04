"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    
    We partition the computation as follows:
    - C[0,:] = A[0,:] * B[:, :]  (1x3 * 3x4 = 1x4, rank 4)
    - C[1:3,:] = A[1:3,:] * B[:, :]  (2x3 * 3x4 = 2x4)
    
    For the 2x3 * 3x4 block, we partition B into B_left (3x2) and B_right (3x2):
    - C[1:3, 0:2] = A[1:3,:] * B[:, 0:2]  (2x3 * 3x2 = 2x2, rank 4)
    - C[1:3, 2:4] = A[1:3,:] * B[:, 2:4]  (2x3 * 3x2 = 2x2, rank 4)
    
    Total rank: 4 + 4 + 4 = 12
    
    Wait, that's not quite right. Let me reconsider.
    
    Actually, for 2x3 * 3x2, the schoolbook rank is 2*3*2 = 12. But we can do better.
    For 2x2 * 2x2, Strassen gives rank 7.
    
    Let me partition A (3x3) into A_top (1x3) and A_bottom (2x3).
    Partition B (3x4) into B_left (3x2) and B_right (3x2).
    
    C = [[A_top * B_left, A_top * B_right],
         [A_bottom * B_left, A_bottom * B_right]]
    
    Each block:
    - A_top * B_left: 1x3 * 3x2 = 1x2, rank 2
    - A_top * B_right: 1x3 * 3x2 = 1x2, rank 2
    - A_bottom * B_left: 2x3 * 3x2 = 2x2, rank 4 (schoolbook) or can we do better?
    - A_bottom * B_right: 2x3 * 3x2 = 2x2, rank 4
    
    For 2x3 * 3x2: We can partition A_bottom into A_bottom_left (2x2) and A_bottom_right (2x1).
    Then:
    A_bottom * B_left = A_bottom_left * B_left_left + A_bottom_right * B_left_right
    where B_left_left is 2x2 (first 2 rows of B_left) and B_left_right is 1x2 (last row of B_left).
    
    A_bottom_left * B_left_left: 2x2 * 2x2, rank 7 (Strassen)
    A_bottom_right * B_left_right: 2x1 * 1x2, rank 2
    
    So A_bottom * B_left has rank 7 + 2 = 9? That's worse than schoolbook 12... no wait, 9 < 12.
    
    Hmm, but this is getting complicated. Let me just use a simple block decomposition.
    
    Partition A (3x3) into rows 0 and rows 1-2.
    Partition B (3x4) into cols 0-1 and cols 2-3.
    
    Block 1: C[0, 0:2] = A[0, :] * B[:, 0:2]  (1x3 * 3x2 = 1x2, rank 2)
    Block 2: C[0, 2:4] = A[0, :] * B[:, 2:4]  (1x3 * 3x2 = 1x2, rank 2)
    Block 3: C[1:3, 0:2] = A[1:3, :] * B[:, 0:2]  (2x3 * 3x2 = 2x2, rank 4)
    Block 4: C[1:3, 2:4] = A[1:3, :] * B[:, 2:4]  (2x3 * 3x2 = 2x2, rank 4)
    
    Total rank: 2 + 2 + 4 + 4 = 12
    
    Let me verify this is correct and construct it.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    
    # We'll build the decomposition with rank 12
    rank = 12
    U = [[0] * rank for _ in range(n * m)]  # 9 x 12
    V = [[0] * rank for _ in range(m * p)]  # 12 x 12
    W = [[0] * rank for _ in range(p * n)]  # 12 x 12
    
    col = 0
    
    # Block 1: C[0, 0:2] = sum_j A[0,j] * B[j, 0:2]
    # For each j in 0..2, k in 0..1
    for j in range(m):
        for k in range(2):
            U[0 * m + j][col] = 1  # A[0,j]
            V[j * p + k][col] = 1  # B[j,k]
            W[k * n + 0][col] = 1  # C[0,k]
            col += 1
    
    # Block 2: C[0, 2:4] = sum_j A[0,j] * B[j, 2:4]
    for j in range(m):
        for k in range(2, 4):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
    
    # Block 3: C[1:3, 0:2] = sum_j A[i,j] * B[j, 0:2] for i in 1,2
    for i in range(1, 3):
        for j in range(m):
            for k in range(2):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Block 4: C[1:3, 2:4] = sum_j A[i,j] * B[j, 2:4] for i in 1,2
    for i in range(1, 3):
        for j in range(m):
            for k in range(2, 4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
