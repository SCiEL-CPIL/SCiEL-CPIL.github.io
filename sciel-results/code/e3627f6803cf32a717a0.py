"""Matrix multiplication 3x3 * 3x4: rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a valid exact tensor decomposition for multiplying a 3x3 matrix by a 3x4 matrix.
    
    Uses a block-partitioned approach:
    - Partition A (3x3) into top row (1x3) and bottom two rows (2x3)
    - Partition B (3x4) into left two cols (3x2) and right two cols (3x2)
    
    This gives 4 sub-problems:
    1. C[0,0:2] = A[0,:] * B[:,0:2]  -> 1x3 * 3x2 = 1x2 (rank 6 schoolbook)
    2. C[0,2:4] = A[0,:] * B[:,2:4]  -> 1x3 * 3x2 = 1x2 (rank 6 schoolbook)
    3. C[1:3,0:2] = A[1:3,:] * B[:,0:2]  -> 2x3 * 3x2 = 2x2 (use Strassen on core, rank 7+4=11)
    4. C[1:3,2:4] = A[1:3,:] * B[:,2:4]  -> 2x3 * 3x2 = 2x2 (use Strassen on core, rank 7+4=11)
    
    Total rank: 6 + 6 + 11 + 11 = 34
    
    But let's be more careful. For 2x3 * 3x2:
    Let A_sub = [[a0, a1, a2], [b0, b1, b2]], B_sub = [[p0, p1], [q0, q1], [r0, r1]]
    Output C_sub = [[c00, c01], [c10, c11]] where:
    c00 = a0*p0 + a1*q0 + a2*r0
    c01 = a0*p1 + a1*q1 + a2*r1
    c10 = b0*p0 + b1*q0 + b2*r0
    c11 = b0*p1 + b1*q1 + b2*r1
    
    Schoolbook: 12 terms. Can we do better?
    
    Let's just use a clean construction. The known bound is rank(3,3,4) <= 24.
    Let me construct a rank-24 decomposition.
    
    Actually, let me use a simpler and definitely correct approach first:
    Schoolbook gives rank = 3*3*4 = 36.
    
    Let me try to get a better bound using known results.
    For m x n * n x p, rank <= m*n*p.
    With blocking: if we partition into blocks, we can sometimes save.
    
    Let me just construct a valid decomposition with the best rank I can guarantee.
    I'll use the following approach:
    
    For each output element C[i,k] = sum_j A[i,j] * B[j,k],
    the schoolbook term is one rank-1 term per (i,j,k).
    
    To reduce rank, I need to combine terms. The key insight is that
    if I can express multiple output elements using shared rank-1 terms,
    I save rank.
    
    Let me try a different approach: use the known result that
    rank(2,2,2) = 7 (Strassen), and build up from there.
    
    Partition:
    A = [A_top; A_bot] where A_top is 1x3, A_bot is 2x3
    B = [B_left, B_right] where B_left is 3x2, B_right is 3x2
    
    C = [C_top_left, C_top_right; C_bot_left, C_bot_right]
    where C_top_left is 1x2, C_top_right is 1x2,
    C_bot_left is 2x2, C_bot_right is 2x2.
    
    C_top_left = A_top * B_left (1x3 * 3x2 = 1x2)
    C_top_right = A_top * B_right (1x3 * 3x2 = 1x2)
    C_bot_left = A_bot * B_left (2x3 * 3x2 = 2x2)
    C_bot_right = A_bot * B_right (2x3 * 3x2 = 2x2)
    
    For the 2x2 outputs, I can use a technique similar to Strassen.
    For 2x3 * 3x2, let's see if I can do it in rank 7 or less.
    
    Actually, let me just use schoolbook for all blocks and see what rank I get:
    C_top_left: 1*3*2 = 6 terms
    C_top_right: 1*3*2 = 6 terms
    C_bot_left: 2*3*2 = 12 terms
    C_bot_right: 2*3*2 = 12 terms
    Total: 36 (same as full schoolbook, no savings from blocking)
    
    The savings come from using Strassen-like methods on the 2x2 blocks.
    
    Let me try: for C_bot_left (2x2 output from 2x3 * 3x2):
    A_bot = [[a, b, c], [d, e, f]], B_left = [[g, h], [i, j], [k, l]]
    C_bot_left[0,0] = a*g + b*i + c*k
    C_bot_left[0,1] = a*h + b*j + c*l
    C_bot_left[1,0] = d*g + e*i + f*k
    C_bot_left[1,1] = d*h + e*j + f*l
    
    This is 12 terms in schoolbook. Can I do it in 7?
    
    Strassen works for 2x2 * 2x2 in 7 terms. Here we have 2x3 * 3x2.
    I can write:
    C_bot_left[0,0] = a*g + b*i + c*k
    C_bot_left[0,1] = a*h + b*j + c*l
    C_bot_left[1,0] = d*g + e*i + f*k
    C_bot_left[1,1] = d*h + e*j + f*l
    
    Let me try to find a rank-7 decomposition.
    Define:
    M1 = (a+b)(g+i) -> contributes to [0,0] and [0,1] partially
    This is getting complex. Let me just use schoolbook rank 36 to be safe.
    
    Actually, let me try a rank-24 construction using known results.
    
    From the literature, rank(3,3,4) <= 24. Let me construct it.
    
    One approach: use the fact that rank(3,3,4) <= rank(3,3,3) + rank(3,3,1) = 23 + 9 = 32.
    But that's not 24.
    
    Let me try: rank(3,3,4) <= rank(2,2,2) + rank(1,2,2) + rank(1,2,2) + rank(2,1,2) + rank(2,1,2) + rank(1,1,4)
    Hmm, this is getting complicated.
    
    Let me just use a clean schoolbook decomposition with rank 36. It's correct and valid.
    """
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Schoolbook decomposition: rank = n * m * p = 36
    rank = n * m * p
    
    # U: shape (n*m, rank) = (9, 36)
    # V: shape (m*p, rank) = (12, 36)
    # W: shape (p*n, rank) = (12, 36)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j][col] = A[i,j]
                U[i * m + j][col] = 1
                # V[j*p + k][col] = B[j,k]
                V[j * p + k][col] = 1
                # W[k*n + i][col] = C[i,k]
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
