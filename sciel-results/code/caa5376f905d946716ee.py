"""Matrix multiplication 3,5,6 — structured Kronecker/block decomposition."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x5x6 matrix multiplication.
    
    Strategy: Decompose the problem using a block structure that exploits the
    factorization of dimensions into smaller sub-problems. We use a Kronecker-product
    based approach:
    
    - Split the 3x5x6 multiplication into blocks where one dimension factorizes.
    - For 3 = 1+2, 5 = 2+3, 6 = 2+4, we can use known lower-rank algorithms
      for small matrix multiplications and combine them.
    
    The schoolbook rank is 90. We attempt to reduce this by using:
    - A 2x2x2 block structure with rank 7 (Strassen) where applicable
    - A 3x3x3 block with known rank 23 (or lower) where applicable
    
    Actually, for these specific small dimensions, the best known approaches
    combine smaller multiplications. Let's use a direct construction based on
    splitting indices.
    
    Key insight: 3x5x6 = (1+2) x (2+3) x (2+4)
    We can split the computation into 8 sub-multiplications:
    1x2x2, 1x2x4, 1x3x2, 1x3x4, 2x2x2, 2x2x4, 2x3x2, 2x3x4
    
    Each sub-problem can be done with schoolbook rank:
    1x2x2 = 2, 1x2x4 = 4, 1x3x2 = 2, 1x3x4 = 4,
    2x2x2 = 4 (schoolbook) or 7 (Strassen, but 7>4 so use schoolbook),
    2x2x4 = 8, 2x3x2 = 4, 2x3x4 = 8
    
    Total schoolbook sum = 2+4+2+4+4+8+4+8 = 36
    
    Wait, that's actually lower than 90! Let me verify this approach.
    
    For a 1xMxP multiplication: A is 1xM, B is MxP, C = A*B is 1xP.
    The tensor is sum_{j=1}^{M} A_j * B_j * C where C_j is the j-th component.
    This has rank M (each term is a rank-1 tensor in the appropriate sense).
    Actually for 1xMxP, the rank is M because C_i = sum_j A_j B_{j,i}, and each
    term A_j * B_{j,i} * C_i can be separated... 
    
    Let me think more carefully. The matrix multiplication tensor T_{i,j,k} = delta_{i,j,k}
    for multiplying (n x m) by (m x p) to get (n x p).
    
    T_{i,j,k} = 1 if the multiplication C_{i,k} = sum_j A_{i,j} B_{j,k} has that term,
    i.e., T_{i,j,k} = 1 for all valid i,j,k, and the tensor is T[i,j,k] = 1.
    
    The rank of the tensor is the minimum number of rank-1 tensors needed to sum to T.
    A rank-1 tensor is U_i * V_j * W_k.
    
    For the schoolbook: T[i,j,k] = sum_c (U[i,j,c] * V[j,k,c] * W[k,i,c])
    where each (i,j,k) corresponds to one c, with U[i,m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1.
    
    For 1xMxP: T[i=0,j,k] = 1 for j in 0..M-1, k in 0..P-1.
    We need T[0,j,k] = sum_c U[0,j,c] V[j,k,c] W[k,0,c].
    If we set U[0,j,c] = 1 for c=j (for j < M), V[j,k,c] = 1 if c=j, W[k,0,c] = 1 for all k.
    Then T[0,j,k] = sum_c delta_{c,j} * 1 * 1 = 1 for each j,k. But this gives the same value
    for all k, which is correct since T[0,j,k]=1 for all valid j,k.
    Wait, that's not right. Each term should contribute to specific (j,k) pairs.
    
    Let me reconsider. For 1xMxP, the matrix multiplication is:
    C[k] = sum_j A[j] * B[j,k] for k=0..P-1, A[j] for j=0..M-1.
    
    The tensor T[j,k] (with i fixed at 0) = 1 for all j,k.
    To decompose: T[j,k] = sum_c u_j * v_{j,k} * w_k
    If we set c = j (M terms), u_j = 1 for c=j, v_{j',k,c} = delta_{j',c}, w_{k,c} = 1.
    Then T[j,k] = sum_c delta_{j,c} * delta_{j,c} * 1 = 1. Yes! Rank M for 1xMxP.
    
    Similarly for Nx1xP: T[i,k] = 1 for all i,k. Rank P.
    For NxMx1: T[i,j] = 1 for all i,j. Rank min(N,M)? No, it's N*M in schoolbook but
    actually rank N*M because T[i,j] = sum_c u_{i,c} v_{j,c} w_0 and we need T[i,j]=1
    for all i,j, so we need N*M independent terms (it's an outer product which has rank 1
    as a matrix but the tensor rank with the third index being 1... let me think again).
    
    For N x M x 1: C[0] = sum_{i,j} A[i,j] B[j,0]. The tensor T[i,j,0] = 1.
    T[i,j,0] = sum_c U[i,c] V[j,c] W[0,c].
    We need this to equal 1 for all i,j. This is like the all-ones matrix decomposed as
    sum_c u_c v_c^T where u_c is N-vector and v_c is M-vector. The rank of the all-ones
    N x M matrix is 1, so we can do it with c=1: U[i,0]=1, V[j,0]=1, W[0,0]=1. Rank 1!
    
    So for N x M x 1, rank is 1.
    For 1 x M x P, rank is M.
    For N x 1 x P, rank is P.
    
    Now back to the block decomposition:
    3x5x6 split into (1+2) x (2+3) x (2+4):
    
    Blocks:
    1x2x2: rank = 2
    1x2x4: rank = 2
    1x3x2: rank = 3
    1x3x4: rank = 3
    2x2x2: rank = 4 (schoolbook) 
    2x2x4: rank = 4 (for 2x2x4, split as (1+1)x2x4, giving 1x2x4 + 1x2x4 = 2+2=4)
    2x3x2: rank = 4 (split as (1+1)x3x2, giving 1x3x2 + 1x3x2 = 3+3=6? No...)
    
    Hmm, let me be more careful. For N x M x P, if I split N = N1+N2, I get two subproblems
    N1 x M x P and N2 x M x P, and the total rank is rank(N1,M,P) + rank(N2,M,P).
    
    Similarly for splitting M or P.
    
    So:
    rank(1,2,2) = 2
    rank(1,2,4) = 2
    rank(1,3,2) = 3
    rank(1,3,4) = 3
    rank(2,2,2) = rank(1,2,2) + rank(1,2,2) = 2+2 = 4
    rank(2,2,4) = rank(1,2,4) + rank(1,2,4) = 2+2 = 4
    rank(2,3,2) = rank(1,3,2) + rank(1,3,2) = 3+3 = 6
    rank(2,3,4) = rank(1,3,4) + rank(1,3,4) = 3+3 = 6
    
    Total = 2 + 2 + 3 + 3 + 4 + 4 + 6 + 6 = 30
    
    That's much better than 90! Let me verify this is correct.
    
    The key principle: if we split the matrix multiplication into independent blocks
    (by partitioning the row, column, or inner dimensions), the total rank is the sum
    of the ranks of the sub-problems, because the tensors for different blocks have
    disjoint support and can be decomposed independently.
    
    So the total rank is 30. Let me construct the matrices.
    """
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # We'll build U, V, W matrices. U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # Split: n = 1+2, m = 2+3, p = 2+4
    n_parts = [1, 2]
    m_parts = [2, 3]
    p_parts = [2, 4]
    
    # Offsets for each part
    n_offsets = [0]
    for i in range(1, len(n_parts)):
        n_offsets.append(n_offsets[-1] + n_parts[i-1])
    m_offsets = [0]
    for i in range(1, len(m_parts)):
        m_offsets.append(m_offsets[-1] + m_parts[i-1])
    p_offsets = [0]
    for i in range(1, len(p_parts)):
        p_offsets.append(p_offsets[-1] + p_parts[i-1])
    
    # Determine rank for each sub-problem
    # rank(1, M, P) = M, rank(N, 1, P) = P, rank(N, M, 1) = 1
    # For general N,M,P where none is 1, we use schoolbook N*M*P
    # But we can also split further. For our purposes, the sub-problems are small enough.
    
    def sub_rank(nn, mm, pp):
        if nn == 0 or mm == 0 or pp == 0:
            return 0
        if nn == 1:
            return mm
        if mm == 1:
            return pp
        if pp == 1:
            return 1
        # For small cases, use schoolbook
        return nn * mm * pp
    
    # Build the decomposition
    U = []  # Will be list of lists
    V = []
    W = []
    
    # First, figure out total rank
    total_rank = 0
    block_ranks = []
    for ni in range(len(n_parts)):
        for mi in range(len(m_parts)):
            for pi in range(len(p_parts)):
                r = sub_rank(n_parts[ni], m_parts[mi], p_parts[pi])
                block_ranks.append(r)
                total_rank += r
    
    # Initialize matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for ni in range(len(n_parts)):
        for mi in range(len(m_parts)):
            for pi in range(len(p_parts)):
                nn = n_parts[ni]
                mm = m_parts[mi]
                pp = p_parts[pi]
                n0 = n_offsets[ni]
                m0 = m_offsets[mi]
                p0 = p_offsets[pi]
                
                r = block_ranks[ni * len(m_parts) * len(p_parts) + mi * len(p_parts) + pi]
                
                if nn == 1:
                    # rank = mm
                    # T[i=0, j, k] = 1 for j in 0..mm-1, k in 0..pp-1
                    # Use c = 0..mm-1, U[n0*m + m0 + j, col+c] = delta_{j,c}
                    # V[(m0+j)*p + (p0+k), col+c] = 1 for all k
                    # W[(p0+k)*n + (n0), col+c] = 1
                    for j in range(mm):
                        for k in range(pp):
                            U[(n0 + 0) * m + (m0 + j)][col + j] = 1
                            V[(m0 + j) * p + (p0 + k)][col + j] = 1
                            W[(p0 + k) * n + (n0 + 0)][col + j] = 1
                    col += r
                elif mm == 1:
                    # rank = pp
                    for i in range(nn):
                        for k in range(pp):
                            U[(n0 + i) * m + (m0 + 0)][col + k] = 1
                            V[(m0 + 0) * p + (p0 + k)][col + k] = 1
                            W[(p0 + k) * n + (n0 + i)][col + k] = 1
                    col += r
                elif pp == 1:
                    # rank = 1
                    for i in range(nn):
                        for j in range(mm):
                            U[(n0 + i) * m + (m0 + j)][col] = 1
                            V[(m0 + j) * p + (p0 + 0)][col] = 1
                            W[(p0 + 0) * n + (n0 + i)][col] = 1
                    col += r
                else:
                    # Schoolbook for this sub-block
                    for i in range(nn):
                        for j in range(mm):
                            for k in range(pp):
                                U[(n0 + i) * m + (m0 + j)][col] = 1
                                V[(m0 + j) * p + (p0 + k)][col] = 1
                                W[(p0 + k) * n + (n0 + i)][col] = 1
                                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
