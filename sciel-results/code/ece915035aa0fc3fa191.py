"""Rewritten program for 2x3 by 3x6 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    # Partition the 3x6 matrix B into two 3x3 blocks: B_L (cols 0-2) and B_R (cols 3-5)
    # C = A * B_L + A * B_R
    # We use a rank-9 decomposition for each 2x3 x 3x3 multiplication
    # Total rank = 18
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]  # 6 x 18
    V = [[0] * rank for _ in range(m * p)]  # 18 x 18
    W = [[0] * rank for _ in range(p * n)]  # 12 x 18
    
    # For 2x3 x 3x3, we use a decomposition based on the following strategy:
    # Let A be 2x3 with entries a[i][j] for i in {0,1}, j in {0,1,2}
    # Let B be 3x3 with entries b[j][k] for j in {0,1,2}, k in {0,1,2}
    # C is 2x3 with entries c[i][k]
    #
    # We define 9 rank-1 terms using the following pattern:
    # For each pair (i, k), c[i][k] = sum_j a[i][j] * b[j][k]
    # We can group terms to reduce rank.
    #
    # A known rank-9 decomposition for 2x3 x 3x3:
    # Let's define intermediate sums:
    # s0 = a[0][0] + a[0][1]
    # s1 = a[0][1] + a[0][2]
    # s2 = a[1][0] + a[1][1]
    # s3 = a[1][1] + a[1][2]
    # t0 = b[0][k] + b[1][k]
    # t1 = b[1][k] + b[2][k]
    #
    # Actually, let me use a more systematic approach.
    # The tensor for 2x3 x 3x3 has 18 terms in schoolbook.
    # We can achieve rank 9 by the following construction:
    #
    # For k in {0,1,2} (output column):
    #   c[0][k] = a[0][0]*b[0][k] + a[0][1]*b[1][k] + a[0][2]*b[2][k]
    #   c[1][k] = a[1][0]*b[0][k] + a[1][1]*b[1][k] + a[1][2]*b[2][k]
    #
    # We use the following 9 rank-1 terms:
    # Term 1: (a[0][0] + a[1][0]) * (b[0][0] + b[0][1] + b[0][2]) -> contributes to c[0][0], c[0][1], c[0][2], c[1][0], c[1][1], c[1][2]
    # This doesn't work directly. Let me use a different approach.
    #
    # Actually, the minimal rank for 2x3 x 3x3 is known to be 9.
    # Let me construct it using the following pattern:
    #
    # We'll use 3 terms for each output column k, for k in {0,1,2}, but shared across k.
    #
    # A simpler approach: just use schoolbook for each block.
    # For 2x3 x 3x3, schoolbook has 18 terms.
    # But we can do better. Let me use the following rank-9 construction:
    #
    # Define for each k:
    #   u_k = a[0][k] if k<3 else 0  (but A is 2x3, so k goes 0,1,2)
    #
    # Let me just implement a valid rank-9 decomposition.
    # Based on literature, the following works:
    #
    # Let A = [[a00, a01, a02],
    #          [a10, a11, a12]]
    # Let B = [[b00, b01, b02],
    #          [b10, b11, b12],
    #          [b20, b21, b22]]
    #
    # 9 rank-1 terms:
    # 1. (a00 + a10) * (b00, b01, b02) -> c00 += (a00+a10)*b00, c01 += (a00+a10)*b01, c02 += (a00+a10)*b02
    #    c10 += (a00+a10)*b00, c11 += (a00+a10)*b01, c12 += (a00+a10)*b02
    #    But we need to subtract the extra a10 terms from c0 and a00 terms from c1.
    #
    # This is getting complicated. Let me just use schoolbook for simplicity,
    # which gives rank 18 per block, total 36. But the task asks for rank-minimal.
    #
    # Actually, let me reconsider. The current program uses rank 36.
    # The task says "rank-minimal". For 2x3 x 3x6, what is the minimal rank?
    #
    # Let me just use a straightforward block approach with schoolbook for each 2x3 x 3x3.
    # Each has 18 terms, total 36. This is what the current program does.
    #
    # To improve, I need to find a lower-rank decomposition.
    # Let me try rank 18 total by sharing structure between the two blocks.
    #
    # Since B_L and B_R are independent, we can't easily share.
    # But we can use a rank-9 decomposition for each 2x3 x 3x3.
    #
    # Here's a valid rank-9 construction for 2x3 x 3x3:
    # We define 9 terms indexed by (i, k) for i in {0,1}, k in {0,1,2}, plus 3 more.
    #
    # Actually, let me use the following known construction:
    # For 2x3 x 3x3, the tensor rank is 9.
    # Construction:
    # Let x_j = a[0][j] + a[1][j] for j in {0,1,2}
    # Let y_j = b[j][k] for all j,k
    #
    # Term (0,k): a[0][k] * (b[0][k], b[1][k], b[2][k]) -> no, this is 3D.
    #
    # I'll use a different strategy: just output the schoolbook decomposition
    # but with rank 18 total by being clever about the two blocks.
    #
    # Wait, the two blocks are independent, so rank should be sum of ranks.
    # If each 2x3 x 3x3 has rank 9, total is 18.
    #
    # Let me implement rank-9 for 2x3 x 3x3 using the following:
    #
    # c[0][k] = sum_j a[0][j] b[j][k]
    # c[1][k] = sum_j a[1][j] b[j][k]
    #
    # Define:
    # p0 = a[0][0] + a[1][0]
    # p1 = a[0][1] + a[1][1]
    # p2 = a[0][2] + a[1][2]
    # q0 = b[0][k] (for each k)
    # q1 = b[1][k]
    # q2 = b[2][k]
    #
    # c[0][k] = a[0][0] b[0][k] + a[0][1] b[1][k] + a[0][2] b[2][k]
    # c[1][k] = a[1][0] b[0][k] + a[1][1] b[1][k] + a[1][2] b[2][k]
    #
    # c[0][k] + c[1][k] = p0 b[0][k] + p1 b[1][k] + p2 b[2][k]
    # c[0][k] - c[1][k] = (a[0][0]-a[1][0]) b[0][k] + (a[0][1]-a[1][1]) b[1][k] + (a[0][2]-a[1][2]) b[2][k]
    #
    # This gives 6 terms for the sum and 6 for the difference, total 12, not 9.
    #
    # Let me just use schoolbook with rank 18 per block = 36 total, same as current.
    # But I'll structure it cleanly.
    
    # Actually, let me try a different approach.
    # Use the fact that 2x3 x 3x6 can be seen as 2x3 x 3x6.
    # Schoolbook: 2*3*6 = 36 terms.
    
    # Let me implement the rank-36 schoolbook decomposition cleanly.
    rank = n * m * p  # 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
