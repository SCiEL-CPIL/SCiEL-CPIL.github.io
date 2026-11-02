import numpy as np

DIMENSIONS = [3, 5, 7]

def schoolbook_decomp(n, m, p):
    """
    Returns U, V, W for schoolbook multiplication of n x m by m x p.
    U shape: (n*m, n*m*p)
    V shape: (m*p, n*m*p)
    W shape: (p*n, n*m*p)
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def embed_and_combine(blocks, n, m, p):
    """
    blocks: list of tuples (n_b, m_b, p_b, i_start, j_start, k_start)
    i_start: start index in A's rows
    j_start: start index in A's cols (which is B's rows)
    k_start: start index in B's cols (which is C's cols)
    
    Wait, the structure is:
    A is n x m. B is m x p.
    We split B's rows (j) and C's cols (k)?
    No, we split based on the summation index k (middle dimension).
    
    Let's refine the block definition.
    A block corresponds to a subset of the middle index k.
    Let K_blocks be a partition of range(m).
    For each block b with K_b = range(k_start, k_end):
    We compute A[:, K_b] @ B[K_b, :].
    This is an (n x |K_b|) matrix times a (|K_b| x p) matrix.
    Let m_b = |K_b|.
    The result is n x p.
    
    So the block is defined by:
    n_b = n
    m_b = k_end - k_start
    p_b = p
    i_offset = 0 (rows of A are always 0..n-1)
    j_offset = k_start (cols of A / rows of B)
    k_offset = 0 (cols of B / cols of C are always 0..p-1)
    
    Wait, the output C is n x p.
    If we split the middle dimension, the outputs are summed.
    So the tensors are NOT in direct sum in the output space. They are added.
    This means we CANNOT simply concatenate W.
    We must add the W factors.
    
    However, the rank is still the sum of the ranks if the input spaces are disjoint?
    No, the input spaces are disjoint (different columns of A and rows of B).
    So the tensors are in direct sum in the input space.
    Therefore, the total rank is the sum of the ranks.
    And we can concatenate U and V.
    For W, since the outputs are added, we must add the W factors.
    But if we use a decomposition where the outputs are distinct (e.g. splitting the output columns), then W can be concatenated.
    
    My previous plan split the output columns?
    Let's re-read my plan.
    "Split 3x5x7 into 3x3x7 and 3x2x7".
    This splits the middle dimension m=5 into 3 and 2.
    So A is split into A1 (3x3) and A2 (3x2).
    B is split into B1 (3x7) and B2 (2x7).
    C = A1 B1 + A2 B2.
    The outputs are added.
    So we cannot concatenate W.
    
    However, we can still compute the rank as the sum of the ranks of the individual products.
    And we can construct the decomposition by:
    1. Get U1, V1, W1 for A1 B1.
    2. Get U2, V2, W2 for A2 B2.
    3. U = concat(U1, U2) (vertically? No, U is (n*m, rank).
       U1 is (n*m1, r1). U2 is (n*m2, r2).
       We need to embed them into (n*m, r1+r2).
       U1 occupies rows 0..n*m1-1. U2 occupies rows n*m1..n*m-1.
    4. V = concat(V1, V2).
       V1 is (m1*p, r1). V2 is (m2*p, r2).
       V1 occupies rows 0..m1*p-1. V2 occupies rows m1*p..m*p-1.
    5. W = W1 + W2 (broadcast to same shape).
       W1 is (p*n, r1). W2 is (p*n, r2).
       We pad W1 with zeros to (p*n, r1+r2) and W2 with zeros.
       Then add them.
    
    This works!
    
    Now, recursively split the blocks.
    Block (n, m, p) with middle split m = m1 + m2.
    Sub-block 1: (n, m1, p).
    Sub-block 2: (n, m2, p).
    
    We can continue splitting until we reach base cases.
    Base cases:
    - (3, 3, 3): Schoolbook rank 27.
    - (3, 3, 1): Schoolbook rank 9.
    - (3, 2, 2): Schoolbook rank 6.
    - (3, 2, 3): Schoolbook rank 6.
    
    Wait, my previous decomposition was:
    1. (3, 3, 7) -> Split middle 3+4? No, split output?
    Let's stick to splitting the middle dimension.
    
    Start: (3, 5, 7).
    Split m=5 into 3 and 2.
    Block 1: (3, 3, 7).
    Block 2: (3, 2, 7).
    
    Block 1 (3, 3, 7):
    Split m=3 into 3 and 0? No.
    Split m=3 into 2 and 1?
    Block 1.1: (3, 2, 7).
    Block 1.2: (3, 1, 7).
    
    Block 1.1 (3, 2, 7):
    Split m=2 into 2 and 0?
    Base case? (3, 2, 7) is not a base case in my list.
    Let's split m=2 into 1 and 1.
    Block 1.1.1: (3, 1, 7).
    Block 1.1.2: (3, 1, 7).
    
    This leads to many (3, 1, 7) blocks.
    (3, 1, 7) is schoolbook rank 21.
    Total rank would be high.
    
    Let's use the previous decomposition which split the output columns too?
    No, splitting the middle dimension is the standard way to use associativity.
    
    Let's try a different split for (3, 3, 7).
    Split p=7 into 3 and 4.
    Block 1.1: (3, 3, 3).
    Block 1.2: (3, 3, 4).
    
    Block 1.1 (3, 3, 3): Base case, rank 27.
    Block 1.2 (3, 3, 4):
    Split p=4 into 3 and 1.
    Block 1.2.1: (3, 3, 3). Rank 27.
    Block 1.2.2: (3, 3, 1). Rank 9.
    
    So Block 1 (3, 3, 7) rank = 27 + 27 + 9 = 63.
    
    Block 2 (3, 2, 7):
    Split p=7 into 2 and 5.
    Block 2.1: (3, 2, 2). Rank 6.
    Block 2.2: (3, 2, 5).
    Split p=5 into 2 and 3.
    Block 2.2.1: (3, 2, 2). Rank 6.
    Block 2.2.2: (3, 2, 3). Rank 6.
    
    So Block 2 (3, 2, 7) rank = 6 + 6 + 6 = 18.
    
    Total Rank = 63 + 18 = 81.
    
    This decomposition splits the **output columns** (p dimension) and the **middle dimension** (m dimension).
    When splitting p, the outputs are disjoint, so W can be concatenated.
    When splitting m, the outputs are added, so W must be added.
    
    We need a generic combiner that handles both.
    Actually, if we always split p first, we can concatenate everything.
    But we also split m.
    
    Let's implement the recursive construction.
"""

def construct():
    n, m, p = DIMENSIONS
    
    def build(n, m, p):
        # Returns U, V, W for n x m * m x p
        # U: (n*m, rank)
        # V: (m*p, rank)
        # W: (p*n, rank)
        
        # Base cases: use schoolbook
        if m == 1:
            return schoolbook_decomp(n, 1, p)
        if p == 1:
            return schoolbook_decomp(n, m, 1)
        if n == 1:
            return schoolbook_decomp(1, m, p)
            
        # Heuristic: split the largest dimension
        # Prefer splitting p (output) to allow concatenation of W
        # Then split m (middle)
        
        # Try to split p
        if p > 1:
            # Split p into p1 and p2
            # Choose p1 to match a known good size?
            # For 7, split into 3 and 4.
            # For 5, split into 2 and 3.
            # For 4, split into 3 and 1.
            # For 3, split into 2 and 1? Or keep?
            
            # Let's use a fixed strategy for these specific sizes
            if p == 7:
                p1, p2 = 3, 4
            elif p == 5:
                p1, p2 = 2, 3
            elif p == 4:
                p1, p2 = 3, 1
            elif p == 3:
                # Don't split p=3 if m is small?
                # If we split p=3 into 1 and 2, rank increases?
                # Schoolbook 3 is 3*m. Split 1+2 is m + 2m = 3m. Same.
                # But we might have better decompositions for 3x3x3.
                # So if n=3, m=3, p=3, use schoolbook (rank 27).
                if n == 3 and m == 3:
                    return schoolbook_decomp(3, 3, 3)
                p1, p2 = 1, 2
            else:
                p1, p2 = p // 2, p - p // 2
                
            U1, V1, W1 = build(n, m, p1)
            U2, V2, W2 = build(n, m, p2)
            
            # Combine: outputs are disjoint
            # U: same rows, concat cols
            U = np.hstack([U1, U2])
            # V: same rows, concat cols
            V = np.hstack([V1, V2])
            # W: rows are split. W1 has p1*n rows, W2 has p2*n rows.
            # Global W has p*n rows.
            # W1 corresponds to first p1 cols of C.
            # W2 corresponds to next p2 cols of C.
            # We need to embed W1 and W2 into the global W.
            W = np.zeros((p * n, U.shape[1]), dtype=int)
            W[:p1 * n, :] += W1
            W[p1 * n:, :] += W2
            
            return U, V, W
            
        # If p is 1, we handled it above.
        # So we must split m.
        if m > 1:
            if m == 5:
                m1, m2 = 3, 2
            elif m == 4:
                m1, m2 = 3, 1
            elif m == 3:
                if n == 3 and p == 3:
                    return schoolbook_decomp(3, 3, 3)
                m1, m2 = 2, 1
            else:
                m1, m2 = m // 2, m - m // 2
                
            U1, V1, W1 = build(n, m1, p)
            U2, V2, W2 = build(n, m2, p)
            
            # Combine: outputs are added
            # U: rows are split. U1 has n*m1 rows, U2 has n*m2 rows.
            U = np.zeros((n * m, U1.shape[1] + U2.shape[1]), dtype=int)
            U[:n * m1, :U1.shape[1]] = U1
            U[n * m1:, U1.shape[1]:] = U2
            
            # V: rows are split. V1 has m1*p rows, V2 has m2*p rows.
            V = np.zeros((m * p, U1.shape[1] + U2.shape[1]), dtype=int)
            V[:m1 * p, :V1.shape[1]] = V1
            V[m1 * p:, V1.shape[1]:] = V2
            
            # W: same rows, add cols
            W = np.hstack([W1, W2])
            
            return U, V, W
            
        # m=1, p=1
        return schoolbook_decomp(n, 1, 1)
        
    U, V, W = build(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
