"""
Find a rank-minimal exact tensor decomposition for 3x5 * 5x5 matrix multiplication.
Uses known fast matrix multiplication algorithms (Strassen and variants) to reduce
the rank below the schoolbook 75.
"""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2(A, B):
    """
    Strassen's 7-multiplication algorithm for 2x2 matrices.
    Returns C = A @ B for 2x2 matrices.
    """
    a, b = A[0, 0], A[0, 1]
    c, d = A[1, 0], A[1, 1]
    e, f = B[0, 0], B[0, 1]
    g, h = B[1, 0], B[1, 1]
    
    # The 7 Strassen products
    m1 = (a + d) * (e + h)
    m2 = (c + d) * e
    m3 = a * (f - h)
    m4 = d * (g - e)
    m5 = (a + b) * h
    m6 = (c - a) * (e + f)
    m7 = (b - d) * (g + h)
    
    C = np.array([[
        m1 + m4 - m5 + m7,
        m3 + m5
    ], [
        m2 + m4,
        m1 - m2 + m3 + m6
    ]])
    return C

def _strassen_decomposition_2x2(n=2, m=2, p=2):
    """
    Build the rank-7 Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    Returns U, V, W with shapes (4, 7), (4, 7), (4, 7).
    """
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen's 7 products as linear combinations
    # m1 = (A00 + A11) * (B00 + B11)
    # m2 = (A10 + A11) * B00
    # m3 = A00 * (B01 - B11)
    # m4 = A11 * (B10 - B00)
    # m5 = (A00 + A01) * B11
    # m6 = (A10 - A00) * (B00 + B01)
    # m7 = (A01 - A11) * (B10 + B11)
    
    # For each Strassen product r, define U_r, V_r, W_r
    # U_r is a 2x2 matrix, V_r is a 2x2 matrix, W_r is a 2x2 matrix
    # such that C = sum_r (sum_{i,j} U_r[i,j] A[i,j]) * (sum_{j,k} V_r[j,k] B[j,k]) * W_r
    
    # m1: U = [[1,0],[0,1]], V = [[1,0],[0,1]]
    # C1 = (A00+A11)(B00+B11) * W1 where W1 contributes to all 4 entries
    # Actually, let me think about this differently.
    
    # The standard Strassen decomposition:
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # Where:
    # m1 = (A00+A11)(B00+B11)
    # m2 = (A10+A11)(B00)
    # m3 = (A00)(B01-B11)
    # m4 = (A11)(B10-B00)
    # m5 = (A00+A01)(B11)
    # m6 = (A10-A00)(B00+B01)
    # m7 = (A01-A11)(B10+B11)
    
    # For each m_r, we need U_r, V_r such that m_r = (sum U_r[i,j] A[i,j]) * (sum V_r[j,k] B[j,k])
    # And W_r is the 2x2 matrix that m_r multiplies.
    
    # m1: U1 = [[1,0],[0,1]], V1 = [[1,0],[0,1]], W1 = [[1,0],[0,1]] (contributes to all via combination)
    # Actually W_r should be such that m_r * W_r gives the right contribution.
    
    # Let me use the standard formulation where each rank-1 term is:
    # tr(U_r^T A) * tr(V_r^T B) * W_r
    
    # m1 = tr([[1,0],[0,1]]^T A) * tr([[1,0],[0,1]]^T B) = (A00+A11)(B00+B11)
    # Contribution to C: m1 * [[1,0],[0,1]]? No, that's not right.
    
    # The correct decomposition:
    # C = sum_r alpha_r * A_r * B_r where... no.
    
    # Let me just directly construct U, V, W from the Strassen formulas.
    
    # For m1: factor A00+A11 and B00+B11
    # U1 = [[1,0],[0,1]] (so tr(U1^T A) = A00+A11)
    # V1 = [[1,0],[0,1]] (so tr(V1^T B) = B00+B11)
    # W1 should satisfy: m1 * W1 contributes to C.
    # From C00 = ... + m1 ... and C11 = ... + m1 ...
    # C00 gets +m1, C11 gets +m1, C01 gets 0, C10 gets 0
    # So W1 = [[1,0],[0,1]]
    
    U[:, 0] = np.array([1, 0, 0, 1])  # A00+A11
    V[:, 0] = np.array([1, 0, 0, 1])  # B00+B11
    W[:, 0] = np.array([1, 0, 0, 1])  # W[k*n+i]: k=0,i=0 -> idx 0; k=0,i=1 -> idx 1; k=1,i=0 -> idx 2; k=1,i=1 -> idx 3
    # W[0*n+0] = W[0] = 1 (i=0,k=0)
    # W[0*n+1] = W[1] = 0 (i=1,k=0)
    # W[1*n+0] = W[2] = 0 (i=0,k=1)
    # W[1*n+1] = W[3] = 1 (i=1,k=1)
    
    # m2: (A10+A11) * B00
    # U2 = [[0,0],[1,1]] (so tr(U2^T A) = A10+A11)
    # V2 = [[1,0],[0,0]] (so tr(V2^T B) = B00)
    # C10 gets +m2, so W2[1*2+0] = W2[2] = 1
    U[:, 1] = np.array([0, 0, 1, 1])
    V[:, 1] = np.array([1, 0, 0, 0])
    W[:, 1] = np.array([0, 0, 1, 0])  # C10 contribution
    
    # m3: A00 * (B01-B11)
    # U3 = [[1,0],[0,0]]
    # V3 = [[0,1],[0,-1]]
    # C01 gets +m3, so W3[0*2+1] = W3[1] = 1
    U[:, 2] = np.array([1, 0, 0, 0])
    V[:, 2] = np.array([0, 1, 0, -1])
    W[:, 2] = np.array([0, 1, 0, 0])
    
    # m4: A11 * (B10-B00)
    # U4 = [[0,0],[0,1]]
    # V4 = [[-1,0],[1,0]]
    # C00 gets +m4, C10 gets +m4
    # W4[0] = 1 (C00), W4[2] = 1 (C10)
    U[:, 3] = np.array([0, 0, 0, 1])
    V[:, 3] = np.array([-1, 0, 1, 0])
    W[:, 3] = np.array([1, 0, 1, 0])
    
    # m5: (A00+A01) * B11
    # U5 = [[1,1],[0,0]]
    # V5 = [[0,0],[0,1]]
    # C00 gets -m5, C01 gets +m5
    # W5[0] = -1 (C00), W5[1] = 1 (C01)
    U[:, 4] = np.array([1, 1, 0, 0])
    V[:, 4] = np.array([0, 0, 0, 1])
    W[:, 4] = np.array([-1, 1, 0, 0])
    
    # m6: (A10-A00) * (B00+B01)
    # U6 = [[-1,0],[1,0]]
    # V6 = [[1,1],[0,0]]
    # C11 gets +m6
    # W6[3] = 1 (C11)
    U[:, 5] = np.array([-1, 0, 1, 0])
    V[:, 5] = np.array([1, 1, 0, 0])
    W[:, 5] = np.array([0, 0, 0, 1])
    
    # m7: (A01-A11) * (B10+B11)
    # U7 = [[0,1],[0,-1]]
    # V7 = [[0,0],[1,1]]
    # C00 gets +m7
    # W7[0] = 1 (C00)
    U[:, 6] = np.array([0, 1, 0, -1])
    V[:, 6] = np.array([0, 0, 1, 1])
    W[:, 6] = np.array([1, 0, 0, 0])
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify the tensor decomposition exactly reconstructs matrix multiplication."""
    rank = U.shape[1]
    
    # Reshape factor matrices
    U_reshaped = U.T.reshape(rank, n, m)  # (rank, n, m)
    V_reshaped = V.T.reshape(rank, m, p)  # (rank, m, p)
    W_reshaped_temp = W.T.reshape(rank, p, n)  # (rank, p, n)
    W_reshaped = W_reshaped_temp.transpose(0, 2, 1)  # (rank, n, p)
    
    # Test with multiple random matrices
    rng = np.random.RandomState(12345)
    for trial in range(10):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        AB_direct = A @ B
        
        U_A = np.einsum('rij,ij->r', U_reshaped, A)
        V_B = np.einsum('rjk,jk->r', V_reshaped, B)
        AB_decomp = np.einsum('r,r,rik->ik', U_A, V_B, W_reshaped)
        
        if not np.allclose(AB_direct, AB_decomp, atol=1e-10):
            return False
    
    return True

def _construct_3x5_5x5_decomposition():
    """
    Construct a decomposition for 3x5 * 5x5 matrix multiplication.
    
    Strategy: Partition the 5x5 matrix B into blocks and use fast matrix
    multiplication where possible.
    
    One approach: Write the 3x5 matrix A as having 3 rows. Each row is a 1x5 vector.
    C_i = A_i * B where A_i is 1x5 and B is 5x5.
    
    For each row i, we need to compute a 1x5 times 5x5 multiplication.
    The tensor rank for 1x5 * 5x5 is 25 (schoolbook) or potentially less.
    
    Alternative: Use the fact that 5 = 2 + 2 + 1 and apply Strassen to the 2x2
    blocks.
    
    Let me try a different approach: partition into 2x2 and 1x1 blocks.
    
    Actually, let me try to use a known result: mu(1, n, n) = n (for any n).
    So each row multiplication has rank 5.
    Total rank = 3 * 5 = 15? No, that's not right because the rows share B.
    
    Wait, the tensor T(A, B) = AB where A is 3x5 and B is 5x5.
    We can write T = T_1 + T_2 + T_3 where T_i(A, B) gives row i of AB.
    T_i(A, B) = A_i * B where A_i is the i-th row of A.
    
    The tensor T_i has rank at most 5 (since it's a 1x5 times 5x5 multiplication,
    and mu(1, 5, 5) = 5).
    
    But can we do better than 5 for 1x5 * 5x5? No, because:
    T_i(A_i, B)_{k} = sum_j A_i[j] B[j,k]
    This requires 5 independent multiplications.
    
    So the total rank is at most 3 * 5 = 15? Let me verify this.
    
    Actually, I think the issue is that the tensor decomposition is over the full
    space, and the rows do interact through B. Let me just construct the schoolbook
    and try to optimize.
    
    For now, let me try to construct a decomposition of rank 15 by treating each
    row independently.
    """
    n, m, p = 3, 5, 5
    
    # Try rank 15: treat each row independently
    # For row i, we have C_i[k] = sum_j A_i[j] * B[j,k]
    # The tensor for this is: sum_{j,k} A_i[j] B[j,k] C_i[k]
    # Rank-1 decomposition: for each (j,k) pair, term = A_i[j] * B[j,k] * C_i[k]
    # But that's 25 terms per row, total 75.
    
    # To get rank 5 per row, we need:
    # C_i[k] = sum_r alpha_{i,r}(A_i) * beta_{k,r}(B)
    # where alpha depends only on A_i and beta depends on B and k.
    
    # One way: alpha_{i,r}(A_i) = A_i[r] for r=1..5
    # beta_{k,r}(B) = B[r,k]
    # Then C_i[k] = sum_r A_i[r] * B[r,k]. Correct!
    
    # So for each row i and each r in 1..5:
    # U_{i,j,r} = delta_{j,r} (for row i, column j, rank r)
    # V_{j',k',r} = delta_{j',r} (for row j', column k', rank r) -- wait, this doesn't work
    # because V should be the same for all i.
    
    # Actually, the rank index should be (i, r) for i=1..3, r=1..5, total rank 15.
    
    rank = 15
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for r in range(m):  # r = 0..4, corresponds to column r of A_i and row r of B
            rank_idx = i * m + r
            # U: row i*m + j, column rank_idx
            # We want sum_j U[i*m+j, rank_idx] * A[i,j] = A[i, r]
            # So U[i*m+r, rank_idx] = 1
            U[i * m + r, rank_idx] = 1
            
            # V: row j'*p + k', column rank_idx
            # We want sum_{j',k'} V[j'*p+k', rank_idx] * B[j',k'] = B[r, k] for each k
            # But V should give a scalar, not depend on k.
            
            # Hmm, this is the issue. The standard tensor decomposition has:
            # (AB)_{i,k} = sum_r [sum_j U[i*m+j, r] A[i,j]] * [sum_{j'} V[j'*p+k, r] B[j',k]] * W[k*n+i, r]
            
            # So V depends on k. Let me redefine.
            
            # For rank term (i, r):
            # U[i*m+j, rank_idx] should be such that sum_j U[i*m+j, rank_idx] A[i,j] = A[i,r]
            # So U[i*m+r, rank_idx] = 1, others 0.
            
            # V[j'*p+k, rank_idx] should be such that sum_{j'} V[j'*p+k, rank_idx] B[j',k] = B[r,k]
            # So V[r*p+k, rank_idx] = 1 for all k, others 0.
            
            # W[k*n+i, rank_idx] should be 1 for all k (since C[i,k] gets B[r,k] * A[i,r] * 1)
            # Actually C[i,k] = sum_r A[i,r] * B[r,k], so each term contributes A[i,r]*B[r,k] to C[i,k].
            # So W[k*n+i, rank_idx] = 1.
            
            V[r * p + range(p), rank_idx] = 1  # This won't work with numpy indexing like this
            # Let me fix this
            for k in range(p):
                V[r * p + k, rank_idx] = 1
            
            for k in range(p):
                W[k * n + i, rank_idx] = 1
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly reconstructs the matrix multiplication tensor."""
    return _verify_decomposition(U, V, W, n, m, p)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # First, try the rank-15 construction
    U15, V15, W15 = _construct_3x5_5x5_decomposition()
    
    if _verify(U15, V15, W15, n, m, p):
        return {"U": U15.tolist(), "V": V15.tolist(), "W": W15.tolist()}
    
    # If rank 15 doesn't work, try schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
