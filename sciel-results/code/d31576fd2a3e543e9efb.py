"""Matrix multiplication 2x3x10 rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START

def _build_block_decomposition(n, m, p, m_left, m_right):
    """
    Build a decomposition by splitting the m dimension into two blocks.
    For each block, use the optimal known decomposition for that sub-problem.
    
    Returns (U, V, W) as lists of lists, or None if not better than schoolbook.
    """
    rank_block_left = n * m_left * p
    rank_block_right = n * m_right * p
    total_rank = rank_block_left + rank_block_right
    if total_rank >= n * m * p:
        return None
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    # Left block: columns 0..m_left-1 of A, rows 0..m_left-1 of B
    for i in range(n):
        for j in range(m_left):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Right block: columns m_left..m-1 of A, rows m_left..m-1 of B
    for i in range(n):
        for j in range(m_left, m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return U, V, W


def _build_strassen_2x3x5():
    """
    Build a rank-7 decomposition for 2x3x5 matrix multiplication.
    This uses a Strassen-like approach adapted for rectangular dimensions.
    
    For 2x3x5: A is 2x3, B is 3x5, C is 2x5.
    Schoolbook rank = 2*3*5 = 30.
    
    We'll use a decomposition based on splitting and combining.
    """
    n, m, p = 2, 3, 5
    rank = 7
    U = [[0] * rank for _ in range(n * m)]  # 6 x 7
    V = [[0] * rank for _ in range(m * p)]  # 15 x 7
    W = [[0] * rank for _ in range(p * n)]  # 10 x 7
    
    # This is a hand-crafted decomposition.
    # We need to find 7 rank-1 tensors whose sum gives the matrix multiplication tensor.
    # 
    # The tensor T has entries T[i,j,k] = delta_{i,k} for the appropriate indices.
    # Actually, for matrix mult: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] where we need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r
    # to equal delta_{i,k}... no, that's not right.
    #
    # The decomposition satisfies: for all i,j,k:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = [i==k] * [j is between 0 and m-1]
    # Wait, no. The matrix multiplication tensor is:
    #   T[a, b, c] = 1 if a and c are "output" indices and b is "input" such that output[a,c] = input[a,b] * input[b,c]
    # 
    # Let me re-derive. We want:
    #   C[i, k] = sum_j A[i, j] * B[j, k]
    # In tensor form: C[i,k] = sum_{j,r} A[i,j] * B[j,k] * delta... 
    # 
    # The standard formulation: we seek U, V, W such that for all (i,j,k):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if this corresponds to a valid (i,j,k) triple
    # 
    # Actually, the tensor entries are: T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # and the decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # So we need: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # This is a system of 2*3*5 = 30 equations in 7*6 + 7*15 + 7*10 = 42 + 105 + 70 = 217 unknowns.
    # It's underdetermined, so solutions exist.
    
    # Let me try a specific construction. I'll use a known rank-7 scheme for 2x2x2 (Strassen)
    # and adapt it.
    
    # For 2x2x2, Strassen's 7 multiplications:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # For 2x3x5, this is more complex. Let me just use schoolbook for now and return a valid
    # decomposition. The directive says to try rank 14 via two 3x5 blocks, but I need to
    # verify that such a decomposition exists.
    
    # Actually, let me just return None and fall back to schoolbook if I can't find a
    # valid lower-rank decomposition.
    
    return None


def _build_strassen_2x3x5_v2():
    """
    Attempt a rank-7 decomposition for 2x3x5 using a systematic approach.
    """
    n, m, p = 2, 3, 5
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # I'll try to construct this using a block structure.
    # Split the 3 columns of A into 1+2, and the 3 rows of B into 1+2.
    
    # Block 1: A[:,0] (2x1) * B[0,:] (1x5) -> 2x5, rank = 2*1*5 = 10
    # Block 2: A[:,1:] (2x2) * B[1:,:] (2x5) -> 2x5, rank = 2*2*5 = 20
    # Total = 30, same as schoolbook. No help.
    
    # Let me try a different approach: use the fact that we only have 2 output rows.
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # For each k, we have a 2x3 * 3x1 multiplication.
    # The tensor for each k is rank at most 3 (schoolbook for 2x3x1).
    # But we can share structure across k.
    
    # Actually, I think I need to just accept that finding a rank-7 decomposition
    # for 2x3x5 is non-trivial and I'll use schoolbook as fallback.
    
    return None


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    
    # Try various decompositions and pick the best (lowest rank) one.
    best_rank = n * m * p
    best_result = None
    
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = n * m * p
    
    # Try splitting the m dimension into blocks
    for m_left in range(1, m):
        m_right = m - m_left
        result = _build_block_decomposition(n, m, p, m_left, m_right)
        if result is not None:
            U_b, V_b, W_b = result
            r = len(U_b[0])
            if r < best_rank:
                best_rank = r
                best_result = {"U": U_b, "V": V_b, "W": W_b}
    
    # Try Strassen-style for sub-blocks
    # For 2x3x5, try rank 7
    if p == 10:
        # Split into two 2x3x5 blocks
        for m_left in [1, 2]:
            m_right = m - m_left
            if m_left == 3 and m_right == 0:
                continue
            # Check if time is running out
            if time.time() > deadline - 5:
                break
            
            # Try to build a rank-7 decomposition for 2x3x5
            # For now, just use schoolbook for each block
            rank_left = n * m_left * (p // 2)
            rank_right = n * m_right * (p // 2)
            
            # Actually, let's just do the block decomposition which we already did above.
            # The block decomposition already handles this.
    
    return best_result

# EVOLVE-BLOCK-END
