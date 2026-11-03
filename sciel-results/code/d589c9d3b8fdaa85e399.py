"""Winograd-style Fourier decomposition for 2x4x5 matrix multiplication."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 5  # A is 2x4, B is 4x5, C is 2x5

    # We decompose using Winograd's approach:
    # Split A into 2 blocks of 2x2 (columns 0-1, 2-3)
    # Split B into 2 blocks of 2x5 (rows 0-1, 2-3)
    # For each (A_block, B_block) pair, we compute a 2x2 * 2x5 product.
    # 
    # For 2x2 * 2x5, we use a Winograd-style decomposition:
    # The 2x2 * 2x5 tensor has rank 10 (schoolbook), but we can do better
    # by using the fact that the 2x2 part can be handled with Strassen-like
    # tricks combined with the 5 columns.
    #
    # Actually, let's think about this more carefully.
    # The full tensor is 2x4 x 4x5 x 2x5, which has 8*20*10 = 1600 entries.
    # The rank of the matrix multiplication tensor for (n,m,p) is at most n*m*p.
    # For (2,4,5), schoolbook gives rank 40.
    # 
    # Let me use a direct approach: decompose into 2x2*2x5 blocks.
    # For 2x2 * 2x5:
    #   A = [[a0, a1], [a2, a3]], B = [[b0, b1, b2, b3, b4], [b5, b6, b7, b8, b9]]
    #   C = [[c0, c1, c2, c3, c4], [c5, c6, c7, c8, c9]]
    #   C[i][j] = A[i][0]*B[0][j] + A[i][1]*B[1][j]
    #
    # This is just 10 independent products, so rank 10 for this sub-tensor.
    # Total rank = 2 (A blocks) * 2 (B blocks) * 10 = 40.
    #
    # But we can do better! Let's use Strassen for the 2x2 part.
    # For 2x2 * 2x2, Strassen gives rank 7.
    # For 2x2 * 2x1, it's rank 4 (schoolbook, since it's just 4 products).
    #
    # So for 2x2 * 2x5, split the 5 columns into 2+2+1:
    #   Rank = 7 + 7 + 4 = 18
    # Total rank = 2 * 2 * 18 = 72. That's worse than 40!
    #
    # So schoolbook at rank 40 is better. Let me just implement schoolbook.
    
    rank = 40
    U = [[0] * rank for _ in range(n * m)]  # 8 x 40
    V = [[0] * rank for _ in range(m * p)]  # 20 x 40
    W = [[0] * rank for _ in range(p * n)]  # 10 x 40
    
    col = 0
    
    def add_rank1_term(u_dict, v_dict, w_dict):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
        col += 1
    
    # Schoolbook decomposition for 2x4 * 4x5 -> 2x5
    # C[i][k] = sum_{j=0}^{3} A[i][j] * B[j][k]
    # 
    # U index: i*m + j = i*4 + j (i in 0..1, j in 0..3)
    # V index: j*p + k = j*5 + k (j in 0..3, k in 0..4)
    # W index: k*n + i = k*2 + i (k in 0..4, i in 0..1)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                add_rank1_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
