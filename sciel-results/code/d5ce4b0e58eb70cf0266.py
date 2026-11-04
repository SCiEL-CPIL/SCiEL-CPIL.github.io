"""
Matrix multiplication 5x5x5 tensor decomposition.

Approach: Schoolbook decomposition (rank 125).

The matrix multiplication tensor T_5 = sum_{i,j,k} A[i,j] * B[j,k] * C[i,k]
is decomposed into 125 rank-1 terms, one for each (i,j,k) triple.
"""

DIMENSIONS = [5, 5, 5]


def _schoolbook_factors(n, m, p):
    """Return U, V, W for schoolbook n x m x p matrix multiplication."""
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
    return U, V, W


def _strassen_factors():
    """
    Return U, V, W for Strassen's 2x2x2 algorithm.
    A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    C = [[a*(e+f) - (d-c)*(g+f), b*(e+f)],
         [(a+c)*h - (e+d)*(h-g), (a+c)*(g+h)]]
    
    Seven products:
    M1 = a*(e+f)
    M2 = (a+b)*h
    M3 = c*(g+h)
    M4 = d*(e+f)
    M5 = (a+d)*(e+h-f)
    M6 = (b+c)*(g+e-h)
    M7 = (a+c)*(g+h)
    
    C00 = M1 - M4 + M5
    C01 = M2 - M5 + M7
    C10 = M3 + M4 - M6
    C11 = M1 - M3 + M6
    """
    rank = 7
    # U: shape (n*m, rank) = (4, 7), indexed by [i*m+j]
    # A = [[a, b], [c, d]] -> indices: a=0, b=1, c=2, d=3
    U = [
        [1, 0, 0, 0, 1, 0, 0],  # a: M1, M5
        [0, 1, 0, 0, 0, 1, 0],  # b: M2, M6
        [0, 0, 1, 0, 0, 1, 1],  # c: M3, M6, M7
        [0, 0, 0, 1, 1, 0, 0],  # d: M4, M5
    ]
    # V: shape (m*p, rank) = (4, 7), indexed by [j*p+k]
    # B = [[e, f], [g, h]] -> indices: e=0, f=1, g=2, h=3
    V = [
        [1, 0, 0, 1, 1, 1, 0],  # e: M1, M4, M5, M6
        [1, 0, 0, 1, -1, 0, 0], # f: M1, M4, M5
        [0, 0, 1, 0, 0, 0, 1],  # g: M3, M7
        [0, 1, 1, 0, 1, -1, 1], # h: M2, M3, M5, M6, M7
    ]
    # W: shape (p*n, rank) = (4, 7), indexed by [k*n+i]
    # C = [[C00, C01], [C10, C11]] -> indices: C00=0, C10=1, C01=2, C11=3
    # Wait, the trace convention is W[k*n+i, r]
    # C is 2x2: C[i,j] where i is row (0,1), j is col (0,1)
    # Output index for C[i,j] in flattened C (row-major): i*2+j
    # But W is indexed by [k*n+i] where k is the "output col" index (p dim) and i is "output row" index (n dim)
    # For matrix mult: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So output has shape n x p = 2 x 2
    # W index: k*n + i = k*2 + i
    # C00: i=0, k=0 -> index 0
    # C01: i=0, k=1 -> index 1
    # C10: i=1, k=0 -> index 2
    # C11: i=1, k=1 -> index 3
    W = [
        [1, 0, 0, -1, 1, 0, 0],   # C00 = M1 - M4 + M5
        [0, 1, 0, 0, -1, 0, 1],   # C01 = M2 - M5 + M7
        [0, 0, 1, 1, 0, -1, 0],   # C10 = M3 + M4 - M6
        [1, 0, -1, 0, 0, 1, 0],   # C11 = M1 - M3 + M6
    ]
    return U, V, W


def _concat_factors(U1, V1, W1, U2, V2, W2, **kw):
    """Concatenate two sets of factors along the rank (column) dimension."""
    U = [row + row2 for row, row2 in zip(U1, U2)]
    V = [row + row2 for row, row2 in zip(V1, V2)]
    W = [row + row2 for row, row2 in zip(W1, W2)]
    return U, V, W


def _block_decompose_5x5x5():
    """
    Decompose 5x5x5 matrix mult into blocks using 5 = 3 + 2.
    
    A is 5x5, partitioned as:
    A = [[A11, A12], [A21, A22]] where A11 is 3x3, A12 is 3x2, A21 is 2x3, A22 is 2x2
    B is 5x5, partitioned as:
    B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x3, B21 is 3x2, B22 is 3x3
    
    C = A*B, partitioned as:
    C = [[C11, C12], [C21, C22]] where C11 is 3x2, C12 is 3x3, C21 is 2x2, C22 is 2x3
    
    C11 = A11*B11 + A12*B21   (3x2)
    C12 = A11*B12 + A12*B22   (3x3)
    C21 = A21*B11 + A22*B21   (2x2)
    C22 = A21*B12 + A22*B22   (2x3)
    
    Each product term is a sub-matrix multiplication:
    1. A11*B11: 3x3 * 2x2 -> 3x2 (schoolbook: 3*3*2 = 18)
    2. A12*B21: 3x2 * 3x2 -> 3x2 (schoolbook: 3*2*3 = 18)
    3. A11*B12: 3x3 * 2x3 -> 3x3 (schoolbook: 3*3*3 = 27)
    4. A12*B22: 3x2 * 3x3 -> 3x3 (schoolbook: 3*2*3 = 18)
    5. A21*B11: 2x3 * 2x2 -> 2x2 (schoolbook: 2*3*2 = 12)
    6. A22*B21: 2x2 * 3x2 -> 2x2 (schoolbook: 2*2*3 = 12)
    7. A21*B12: 2x3 * 2x3 -> 2x3 (schoolbook: 2*3*3 = 18)
    8. A22*B22: 2x2 * 3x3 -> 2x3 (schoolbook: 2*2*3 = 12)
    
    Wait, let me reconsider. For the 2x2 block A22*B22, I can use Strassen.
    But the dimensions are: A22 is 2x2, B22 is 3x3, result is 2x3.
    Strassen is for square matrices of the same size. So I can't directly apply Strassen to 2x2 * 3x3.
    
    Let me reconsider the block structure. Maybe I should use a different partition or apply Strassen differently.
    
    Actually, the issue is that Strassen's algorithm is for multiplying two n x n matrices. 
    For non-square blocks, I'd need to pad or use a different approach.
    
    Let me use a cleaner approach: partition into 3+2 but only use Strassen where dimensions match.
    Since none of the cross-block multiplications are square-same-size except possibly if I restructure...
    
    Alternative: Use 5 = 3 + 2 and for the 2x2 * 2x2 block (A22 * B21 is 2x2 * 3x2, not square).
    
    Hmm, let me think about this differently. The blocks that are 2x2 * 2x2 would give Strassen savings.
    But in the 3+2 partition:
    - A22 is 2x2, B21 is 3x2 -> product is 2x2 * 3x2 = 2x2. Not square same size.
    
    I think the cleanest approach that still improves on schoolbook is to use block decomposition with 
    schoolbook for all sub-blocks but recognize that the total rank is still lower than full schoolbook
    when we can share intermediate computations. But actually, simple block decomposition with 
    schoolbook for each block gives the same total rank as full schoolbook (sum of n_i*m_i*p_j = n*m*p).
    
    The key insight for Strassen is that you can compute the 2x2x2 part with rank 7 instead of 8.
    So I need a block that is exactly 2x2 * 2x2 -> 2x2.
    
    Let me use partition 5 = 2 + 2 + 1 instead.
    
    A = [[A1, A2, A3], [B1, B2, B3], [C1, C2, C3]] where:
    - A1, B1, C1 are 2x2, 2x2, 1x2
    - A2, B2, C2 are 2x2, 2x2, 1x2  
    - A3, B3, C3 are 2x1, 2x1, 1x1
    
    This is getting complicated. Let me just implement the 3+2 partition with schoolbook for all blocks
    and accept rank 94, which is still better than 125.
    
    Actually wait - sum of all block products with schoolbook:
    18 + 18 + 27 + 18 + 12 + 12 + 18 + 12 = 135. That's worse than 125!
    
    I made an error. Let me recalculate. The total rank of schoolbook for n x m x p is n*m*p.
    When you decompose into blocks, the sum of (n_i * m_j * p_k) over all blocks equals n*m*p only if 
    you're doing the full product. But each block product C_ij = sum_k A_ik * B_kj involves multiple blocks.
    
    For C11 = A11*B11 + A12*B21:
    - A11*B11: 3x3 * 2x2 -> need rank 3*3*2 = 18
    - A12*B21: 3x2 * 3x2 -> need rank 3*2*3 = 18
    Total for C11: 36
    
    Similarly for other blocks, the total will be much larger than 125.
    
    So simple block decomposition with schoolbook for sub-blocks does NOT help. I need to use a faster
    algorithm for the sub-blocks.
    
    The only sub-block where I can use Strassen is if I have a 2x2 * 2x2 -> 2x2 product.
    In the 3+2 partition, I don't have such a product directly.
    
    Let me try a different approach: use the known fact that mu(5) <= some value and construct 
    a decomposition based on known lower bounds.
    
    Actually, the simplest improvement is to just use schoolbook (rank 125) but that's the same as before.
    
    Let me just implement a correct 94-rank decomposition using a different strategy.
    
    Alternative: Use 5 = 3 + 2, and for the 2x2 part of A and B, use Strassen-style tricks.
    
    You know what, let me just implement the straightforward schoolbook and return it. The task says
    "find a rank-minimal exact tensor decomposition" and the score is the rank. Since I can't easily
    find a rank < 125 decomposition, let me at least make sure the code is correct.
    """
    n, m, p = 5, 5, 5
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
    return U, V, W


def construct():
    U, V, W = _block_decompose_5x5x5()
    return {"U": U, "V": V, "W": W}
