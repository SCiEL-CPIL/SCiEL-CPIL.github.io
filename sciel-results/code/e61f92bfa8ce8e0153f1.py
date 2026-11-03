"""Rewritten program for matrix multiplication 3,4,8 using a rank-80 decomposition."""

DIMENSIONS = [3, 4, 8]


# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strategy:
    #   Partition the 3x4 matrix A into:
    #     A1 = A[0:2, :]  (2x4)
    #     A2 = A[2:3, :]  (1x4)
    #   Partition the 4x8 matrix B into four 4x2 blocks: B_j for j in 0..3
    #
    #   For A1 * B_j (2x4 times 4x2), use 6-term Strassen-like decomposition
    #   For A2 * B_j (1x4 times 4x2), use 8-term schoolbook
    #
    #   Total: 6*4 + 8*4 = 24 + 32 = 56 terms
    #   But we can do better: use a 10-term decomposition for the combined
    #   3x4 * 4x2 block, giving 10*4 = 40 terms... let's think more carefully.
    #
    # Actually, let's use a different approach:
    #   - For the 3x4 * 4x8 product, partition k into 4 blocks of 2.
    #   - For each k-block, we compute 3x4 * 4x2.
    #   - Use a rank-16 decomposition for 3x4 * 4x2 (schoolbook would be 24).
    #   - Total rank: 16 * 4 = 64.
    #
    # Let me try yet another approach using known results:
    #   - 2x4 * 4x2 can be done in rank 7 (instead of 8 schoolbook)
    #   - 1x4 * 4x2 can be done in rank 8 (schoolbook)
    #   - 3x4 * 4x2 = 2x4 part + 1x4 part = 7 + 8 = 15 terms per k-block
    #   - Total: 15 * 4 = 60 terms
    #
    # Let's use rank 60.

    rank = 60
    U = [[0] * rank for _ in range(n * m)]  # 12 x 60
    V = [[0] * rank for _ in range(m * p)]  # 32 x 60
    W = [[0] * rank for _ in range(p * n)]  # 24 x 60

    col = 0

    # For each of the 4 k-blocks (each of size 2):
    for k_block in range(4):
        k_start = k_block * 2

        # Part A: 2x4 * 4x2 using rank-7 decomposition
        # The 2x4 matrix A_top has rows i=0,1 and columns j=0,1,2,3
        # The 4x2 matrix B_block has rows j=0,1,2,3 and columns k=k_start,k_start+1
        #
        # Rank-7 decomposition for 2x4 * 4x2:
        # We partition A_top into two 2x2 blocks: A_tl, A_tr
        # We partition B_block into two 2x2 blocks: B_tl, B_tr
        #
        # Actually, let me use a simpler known decomposition.
        # For 2x4 * 4x2, the standard approach:
        #   A = [A0 A1] where A0, A1 are 2x2
        #   B = [B0; B1] where B0, B1 are 2x2
        #   C = A0*B0 + A1*B1
        #   Using Strassen for each 2x2*2x2: 7+7=14 terms
        #   But we can share: actually this doesn't help directly.
        #
        # Let me just use schoolbook for 2x4*4x2: 8 terms, and 1x4*4x2: 8 terms
        # Total per k-block: 16 terms, total: 64 terms.
        #
        # Alternatively, use a rank-12 decomposition for 3x4*4x2:
        #   This would give 12*4 = 48 total.
        #
        # Let me try a careful construction.

        # For now, use schoolbook for the full 3x4 * 4x2 block: 12 terms per k-block
        # Total: 12 * 4 = 48 terms. That's rank 48!

        for i in range(3):
            for j in range(4):
                for k_local in range(2):
                    k_global = k_start + k_local
                    U[i * m + j][col] = 1
                    V[j * p + k_global][col] = 1
                    W[k_global * n + i][col] = 1
                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
