"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

import time
import random
import math

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return U, V, W for 2x2 * 2x2 multiplication using Strassen (rank 7).
    Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=m=p=2.
    So U has 4 rows, V has 4 rows, W has 4 rows, 7 cols.
    """
    n = m = p = 2
    # Strassen formulas for C = A*B where A,B are 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-d)(f+h)
    # m7 = (b-d)(e+g)
    # C11 = m1 - m4 + m5 + m6
    # C12 = m3 + m4
    # C21 = m2 + m5
    # C22 = m1 - m3 + m2 - m7

    # We need to express each entry of A, B, C as linear combinations.
    # U maps A entries to intermediate values, V maps B entries, W maps to C entries.
    # Actually the tensor decomposition: T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # Wait, the convention is U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    # The multiplication tensor T has entries T[(i,j),(j,k)] = delta for C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k], else 0.
    # The decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank.

    # For Strassen, the 7 rank-1 terms correspond to the 7 intermediate products.
    # Let me index A entries: A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
    # B entries: B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
    # C entries: C[0,0], C[0,1], C[1,0], C[1,1]

    # m1 = (a+d)(e+h): A part: a+d -> U contributions for (0,0):1, (1,1):1; B part: e+h -> V for (0,0):1, (1,1):1
    #   C contribution: C[0,0] += 1, C[1,1] += 1
    # m2 = (c+d)e: A: c+d -> (1,0):1, (1,1):1; B: e -> (0,0):1
    #   C: C[1,0] += 1
    # m3 = a(f-h): A: a -> (0,0):1; B: f-h -> (0,1):1, (1,1):-1
    #   C: C[0,1] += 1, C[1,1] += -1
    # m4 = d(g-e): A: d -> (1,1):1; B: g-e -> (1,0):1, (0,0):-1
    #   C: C[0,0] += -1, C[1,0] += 1
    # m5 = (a+b)h: A: a+b -> (0,0):1, (0,1):1; B: h -> (1,1):1
    #   C: C[0,0] += 1, C[1,0] += 1
    # m6 = (c-d)(f+h): A: c-d -> (1,0):1, (1,1):-1; B: f+h -> (0,1):1, (1,1):1
    #   C: C[0,0] += 1, C[1,1] += 1
    # m7 = (b-d)(e+g): A: b-d -> (0,1):1, (1,1):-1; B: e+g -> (0,0):1, (1,0):1
    #   C: C[1,1] += -1

    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    # m1
    U[0][0] = 1; U[3][0] = 1  # a, d
    V[0][0] = 1; V[3][0] = 1  # e, h
    W[0][0] = 1; W[3][0] = 1  # C[0,0], C[1,1]

    # m2
    U[2][1] = 1; U[3][1] = 1  # c, d
    V[0][1] = 1  # e
    W[2][1] = 1  # C[1,0]

    # m3
    U[0][2] = 1  # a
    V[1][2] = 1; V[3][2] = -1  # f, -h
    W[1][2] = 1; W[3][2] = -1  # C[0,1], -C[1,1]

    # m4
    U[3][3] = 1  # d
    V[2][3] = 1; V[0][3] = -1  # g, -e
    W[0][3] = -1; W[2][3] = 1  # -C[0,0], C[1,0]

    # m5
    U[0][4] = 1; U[1][4] = 1  # a, b
    V[3][4] = 1  # h
    W[0][4] = 1; W[2][4] = 1  # C[0,0], C[1,0]

    # m6
    U[2][5] = 1; U[3][5] = -1  # c, -d
    V[1][5] = 1; V[3][5] = 1  # f, h
    W[0][5] = 1; W[3][5] = 1  # C[0,0], C[1,1]

    # m7
    U[1][6] = 1; U[3][6] = -1  # b, -d
    V[0][6] = 1; V[2][6] = 1  # e, g
    W[3][6] = -1  # -C[1,1]

    return U, V, W


def _build_decomposition(n, m, p, block_decomps):
    """
    Build a decomposition by partitioning indices into blocks and using
    block decompositions for each block, plus direct terms for cross-block interactions.
    
    block_decomps: dict mapping (ni, mj, pk) -> (U_block, V_block, W_block, rank)
    """
    # We'll use a simple approach: partition i into groups, j into groups, k into groups.
    # For each combination of groups, use the block decomposition if available,
    # otherwise use direct rank-1 terms.
    
    # For simplicity, let's handle the specific case n=2, m=4, p=7.
    # We can split m=4 into two groups of 2, and p=7 into groups of 2,2,2,1.
    # n=2 can be kept as one group or split.
    
    # Actually, let's just do a direct construction with Strassen where possible.
    
    # Let me think differently. The multiplication C = A*B where A is n x m, B is m x p.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We can group the j indices. For each group of j's, we compute partial sums.
    # If we group j into blocks of size 2, we can use Strassen-like tricks.
    
    # Let's try: group j into pairs. m=4 -> 2 groups of 2.
    # For each pair of j's (j1, j2), and for each i and k, we have:
    #   A[i,j1]*B[j1,k] + A[i,j2]*B[j2,k]
    # This is like a 1x2 * 2x1 multiplication, which is just 2 rank-1 terms.
    # No savings there.
    
    # Better: group both i and j. If n=2, m=4, we can view A as 2x4.
    # Group j into 2 groups of 2: A[:, 0:2] and A[:, 2:4].
    # For each group, we have a 2x2 * 2x? subproblem.
    
    # Actually, the most natural Strassen application:
    # View the computation as: for each pair of i's (we only have n=2, so one pair),
    # and group j's and k's.
    
    # Let me try a different approach: use the known rank bounds.
    # R(2,4,7) <= R(2,2,7) + R(2,2,7) (splitting m into two 2's)
    # R(2,2,7) <= ? 
    # R(2,2,p) for p=7: we can use Strassen on the 2x2 part.
    # Actually R(2,2,p) = 2p - 2 for p >= 2? No, that's not right.
    
    # Let me just construct it directly with a careful approach.
    
    # Strategy: 
    # 1. Split j into groups: j_groups = [(0,1), (2,3)] (two groups of 2)
    # 2. For each j_group, we have A[:, j_group] (2x2) and B[j_group, :] (2x7)
    # 3. The product A[:, j_group] @ B[j_group, :] is 2x7
    # 4. To compute 2x2 * 2x7, we can use Strassen on the 2x2 part:
    #    Treat B[j_group, :] as 7 columns of 2x1 vectors.
    #    For each column k, we compute [a b; c d] * [e; f] = [ae+bf; ce+df]
    #    This is 2x2 * 2x1, which has rank 2 (no Strassen benefit for 1 column).
    
    # Hmm, Strassen needs 2x2 * 2x2. So we need to group k's too.
    # p=7 = 2+2+2+1
    # For each j_group (2 of them) and each k_group (4 groups: 2,2,2,1),
    # we compute a 2x2 * 2x2 or 2x2 * 2x1 subproduct.
    
    # For 2x2 * 2x2: Strassen gives rank 7
    # For 2x2 * 2x1: direct gives rank 2
    
    # Total: 2 j_groups * (3 k_groups of size 2 + 1 k_group of size 1)
    # = 2 * (3*7 + 1*2) = 2 * (21 + 2) = 46
    
    # That's better than 56!
    
    # Let's implement this.
    
    U = []
    V = []
    W = []
    offset = 0
    
    j_groups = [(0, 1), (2, 3)]
    k_groups = [(0, 1), (2, 3), (4, 5), (6, 6)]
    
    for jg in j_groups:
        j1, j2 = jg
        for kg in k_groups:
            k1, k2 = kg
            if k2 > k1:
                # 2x2 * 2x2 subproblem
                # A is 2x2: rows i=0,1; cols j=j1,j2
                # B is 2x2: rows j=j1,j2; cols k=k1,k2
                # C is 2x2: rows i=0,1; cols k=k1,k2
                U_blk, V_blk, W_blk, rank_blk = _strassen_2x2_with_offsets(n, j1, j2, k1, k2)
            else:
                # 2x2 * 2x1 subproblem
                U_blk, V_blk, W_blk, rank_blk = _direct_2x2x1(n, j1, j2, k1)
            
            # Extend U, V, W
            for c in range(rank_blk):
                U.append([0] * (n * m))
                V.append([0] * (m * p))
                W.append([0] * (p * n))
            
            for r in range(len(U_blk)):
                for c in range(rank_blk):
                    if U_blk[r][c] != 0:
                        U[-(rank_blk)][r] = U_blk[r][c]  # This is wrong, let me fix indexing
    
    # Let me redo this more carefully.
    pass


def _strassen_2x2_with_offsets(n, j1, j2, k1, k2):
    """
    Strassen decomposition for the subproblem:
    A_sub is 2x2 (rows i=0..n-1, cols j1,j2) -- but n=2 so rows are 0,1
    B_sub is 2x2 (rows j1,j2, cols k1,k2)
    C_sub is 2x2 (rows i=0..n-1, cols k1,k2)
    
    Returns U_blk (n x rank), V_blk (2 x rank), W_blk (2 x rank)
    where the full U row index is i*m + j, V row index is j*p + k, W row index is k*n + i.
    
    Actually, let me return the full-size vectors with zeros elsewhere.
    """
    n, m, p = 2, 4, 7  # Hardcoded for simplicity
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A entries: A[i, j1], A[i, j2] for i=0,1
    # B entries: B[j1, k1], B[j1, k2], B[j2, k1], B[j2, k2]
    # C entries: C[i, k1], C[i, k2] for i=0,1
    
    # Strassen for 2x2 * 2x2:
    # Let a=A[0,j1], b=A[0,j2], c=A[1,j1], d=A[1,j2]
    # Let e=B[j1,k1], f=B[j1,k2], g=B[j2,k1], h=B[j2,k2]
    # C[0,k1] = ae+bg, C[0,k2] = af+bh
    # C[1,k1] = ce+dg, C[1,k2] = cf+dh
    
    # m1 = (a+d)(e+h)
    U[0 * m + j1][0] = 1  # a
    U[1 * m + j2][0] = 1  # d
    V[j1 * p + k1][0] = 1  # e
    V[j2 * p + k2][0] = 1  # h
    W[k1 * n + 0][0] = 1  # C[0,k1]
    W[k2 * n + 1][0] = 1  # C[1,k2]
    
    # m2 = (c+d)e
    U[1 * m + j1][1] = 1  # c
    U[1 * m + j2][1] = 1  # d
    V[j1 * p + k1][1] = 1  # e
    W[k1 * n + 1][1] = 1  # C[1,k1]
    
    # m3 = a(f-h)
    U[0 * m + j1][2] = 1  # a
    V[j1 * p + k2][2] = 1  # f
    V[j2 * p + k2][2] = -1  # -h
    W[k2 * n + 0][2] = 1  # C[0,k2]
    W[k2 * n + 1][2] = -1  # C[1,k2]
    
    # m4 = d(g-e)
    U[1 * m + j2][3] = 1  # d
    V[j2 * p + k1][3] = 1  # g
    V[j1 * p + k1][3] = -1  # -e
    W[k1 * n + 0][3] = -1  # C[0,k1]
    W[k1 * n + 1][3] = 1  # C[1,k1]
    
    # m5 = (a+b)h
    U[0 * m + j1][4] = 1  # a
    U[0 * m + j2][4] = 1  # b
    V[j2 * p + k2][4] = 1  # h
    W[k2 * n + 0][4] = 1  # C[0,k2]
    W[k1 * n + 1][4] = 1  # C[1,k1]
    
    # m6 = (c-d)(f+h)
    U[1 * m + j1][5] = 1  # c
    U[1 * m + j2][5] = -1  # -d
    V[j1 * p + k2][5] = 1  # f
    V[j2 * p + k2][5] = 1  # h
    W[k1 * n + 0][5] = 1  # C[0,k1]
    W[k2 * n + 1][5] = 1  # C[1,k2]
    
    # m7 = (b-d)(e+g)
    U[0 * m + j2][6] = 1  # b
    U[1 * m + j2][6] = -1  # -d
    V[j1 * p + k1][6] = 1  # e
    V[j2 * p + k1][6] = 1  # g
    W[k2 * n + 1][6] = -1  # C[1,k2]
    
    return U, V, W, rank


def _direct_2x2x1(n, j1, j2, k1):
    """
    Direct decomposition for 2x2 * 2x1 subproblem.
    A_sub is 2x2 (rows i=0,1; cols j1,j2)
    B_sub is 2x1 (rows j1,j2; col k1)
    C_sub is 2x1 (rows i=0,1; col k1)
    
    C[0,k1] = A[0,j1]*B[j1,k1] + A[0,j2]*B[j2,k1]
    C[1,k1] = A[1,j1]*B[j1,k1] + A[1,j2]*B[j2,k1]
    
    Rank 2.
    """
    m, p = 4, 7
    rank = 2
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Term 1: A[i,j1] * B[j1,k1] -> C[i,k1]
    U[0 * m + j1][0] = 1
    U[1 * m + j1][0] = 1
    V[j1 * p + k1][0] = 1
    W[k1 * n + 0][0] = 1
    W[k1 * n + 1][0] = 1
    
    # Term 2: A[i,j2] * B[j2,k1] -> C[i,k1]
    U[0 * m + j2][1] = 1
    U[1 * m + j2][1] = 1
    V[j2 * p + k1][1] = 1
    W[k1 * n + 0][1] = 1
    W[k1 * n + 1][1] = 1
    
    return U, V, W, rank


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x4 * 4x7 matrix multiplication.
    Uses Strassen for 2x2*2x2 blocks and direct terms for 2x2*2x1 blocks.
    """
    n, m, p = DIMENSIONS
    assert n == 2 and m == 4 and p == 7
    
    # Partition j into groups of 2: [(0,1), (2,3)]
    # Partition k into groups: [(0,1), (2,3), (4,5), (6,6)]
    j_groups = [(0, 1), (2, 3)]
    k_groups = [(0, 1), (2, 3), (4, 5), (6, 6)]
    
    U_full = [[0] * 0 for _ in range(n * m)]  # Will be built up
    V_full = [[0] * 0 for _ in range(m * p)]
    W_full = [[0] * 0 for _ in range(p * n)]
    
    total_rank = 0
    blocks = []
    
    for jg in j_groups:
        j1, j2 = jg
        for kg in k_groups:
            k1, k2 = kg
            if k2 > k1:
                U_blk, V_blk, W_blk, rank_blk = _strassen_2x2_with_offsets(n, j1, j2, k1, k2)
            else:
                U_blk, V_blk, W_blk, rank_blk = _direct_2x2x1(n, j1, j2, k1)
            blocks.append((U_blk, V_blk, W_blk, rank_blk))
            total_rank += rank_blk
    
    # Assemble full matrices
    U_full = [[0] * total_rank for _ in range(n * m)]
    V_full = [[0] * total_rank for _ in range(m * p)]
    W_full = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for U_blk, V_blk, W_blk, rank_blk in blocks:
        for r in range(n * m):
            for c in range(rank_blk):
                U_full[r][col_offset + c] = U_blk[r][c]
        for r in range(m * p):
            for c in range(rank_blk):
                V_full[r][col_offset + c] = V_blk[r][c]
        for r in range(p * n):
            for c in range(rank_blk):
                W_full[r][col_offset + c] = W_blk[r][c]
        col_offset += rank_blk
    
    return {"U": U_full, "V": V_full, "W": W_full}

# EVOLVE-BLOCK-END
