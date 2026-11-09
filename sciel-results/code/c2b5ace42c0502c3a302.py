"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    # We aim for rank 32 by exploiting structure
    # Split the middle dimension m=4 into two groups of 2
    # Split the output dimension p=8 into two groups of 4
    # This gives a 2x2 block structure where each block is 1x2 * 2x4
    # Each 1x2 * 2x4 has rank 4 in schoolbook, 4 blocks * 4 = 16? 
    # Let me verify: C[i,k] = sum_j A[i,j] B[j,k]
    # If we group j into {0,1} and {2,3}, and k into {0..3} and {4..7}
    # C[i, k] for k in 0..3: A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    # = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    # Each parenthesis is a 1x2 * 2x4 product (for fixed i and k-group)
    # For fixed i and k-group (4 columns), the 1x2 * 2x4 product needs rank 4
    # So for i=0,1 and 2 k-groups: 2 * 2 * 4 = 16 rank-1 terms
    # But wait, the two j-groups are independent, so total is 2(i) * 2(j-groups) * 4(k-in-group) = 16?
    # Let me check: for each i, each j-group (2 of them), each k in the 4-column group (4 values)
    # That's 2 * 2 * 4 = 16 terms. Each term is A[i,j]*B[j,k] for specific i,j,k.
    # But that's just schoolbook for half the k's. Total would be 16*2 = 32 for all k.
    # Hmm, that's still 32, not 16.
    
    # Let me try a different grouping. What if I use rank 32 directly?
    rank = 32
    U = [[0] * rank for _ in range(n * m)]  # 8 x 32
    V = [[0] * rank for _ in range(m * p)]  # 32 x 32
    W = [[0] * rank for _ in range(p * n)]  # 16 x 32
    
    # Indexing: U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
    # We assign one rank-1 term for each (i, j, k) but merge some
    # For rank 32, we can do: for each i (2), j (4), and k (4 of 8), assign a term
    # That gives 2*4*4 = 32 terms, covering half the k's. We need all 8 k's.
    # So this won't work directly.
    
    # Let me just do schoolbook with rank 64 but see if I can find a better construction.
    # Actually, let me try rank = 32 by using the following:
    # For each pair of adjacent k values, share the A and B factors.
    # k and k+1: A[i,j]*B[j,k] and A[i,j]*B[j,k+1]
    # These can't really be merged in tensor rank terms.
    
    # I'll go with a known decomposition. Let me use rank 32 by splitting p=8 into 4+4
    # and for each half, doing 2x4x4 with rank 32/2=16? No, 2x4x4 schoolbook is 32.
    
    # Let me just implement schoolbook with rank 64 as a safe baseline, 
    # but try to optimize the indexing.
    
    # Actually, let me try a concrete rank-32 construction:
    # Use the fact that B is 4x8. Write B as [B_top; B_bottom] where B_top, B_bottom are 2x8.
    # A = [A_left | A_right] where A_left, A_right are 2x2.
    # C = A_left * B_top + A_right * B_bottom
    # Each 2x2x8 product has schoolbook rank 32. Total 64.
    
    # I'll just do schoolbook rank 64 for now, but structured cleanly.
    rank = n * m * p
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
