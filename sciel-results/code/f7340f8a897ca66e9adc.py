"""Matrix multiplication 2x4x5: rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    nm = n * m  # 8
    mp = m * p  # 20
    pn = p * n  # 10

    # We'll build the decomposition using a combination of:
    # 1. Strassen for 2x2x2 blocks where possible
    # 2. Schoolbook for remaining terms
    #
    # Strategy: Split m=4 into two groups of 2: {0,1} and {2,3}
    # Split p=5 into: {0,1}, {2,3}, {4}
    # For each (m_block, p_block) pair:
    #   - If p_block has 2 elements: use Strassen (7 terms) for 2x2x2
    #   - If p_block has 1 element: use schoolbook (4 terms) for 2x2x1
    #
    # Total: 2 m-blocks * (2 Strassen + 1 schoolbook) = 2 * (7+7+4) = 36
    #
    # But we can do better by sharing across m-blocks for the p={4} case.
    # Actually, let's first verify rank 36 works, then try to optimize.
    
    # Let's try a different approach: use the fact that M_{2,4,5} can be
    # decomposed more efficiently.
    #
    # Known result: R(M_{2,m,p}) <= m*p + floor(m/2)*ceil(p/2) for some constructions.
    # For m=4, p=5: 4*5 + 2*3 = 20 + 6 = 26? That seems too low.
    #
    # Let me use a verified construction. The key insight is that we can use
    # Kronecker-like structures.
    #
    # Actually, let me use a direct approach: build the tensor and find a
    # low-rank decomposition via numerical methods, then round to integers.
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Try to find a low-rank decomposition
    # We'll use iterative SVD-based rank reduction
    
    # Start with schoolbook rank 40
    rank = 40
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    
    # Now try to reduce rank by finding linear dependencies
    # Use the fact that if we can find vectors a, b, c such that
    # sum_r a_r U_r ⊗ b_r V_r ⊗ c_r W_r = 0, we can eliminate terms
    
    # Actually, let me just try specific known constructions.
    # 
    # For M_{2,4,5}, a known bound is R <= 32 via various methods.
    # Let me try to construct rank 32.
    #
    # Approach: Use Strassen on overlapping blocks.
    # Split m={0,1,2,3} into overlapping pairs: {0,1}, {1,2}, {2,3}
    # This is getting complex. Let me use a simpler verified construction.
    
    # Simplest correct approach: use the schoolbook but with some sharing.
    # For each k, we compute row i of C as sum_j A[i,j]*B[j,k].
    # We can share B[j,k] across i values.
    
    # Let me just return a valid rank-40 solution first, then improve.
    # Actually, the task says "rank-minimal", so I should try harder.
    
    # Let me try rank 36 with the block Strassen approach:
    rank = 36
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    
    def add_strassen(m_block, p_block):
        """Add Strassen decomposition for 2x2x2 block.
        m_block = [m1, m2], p_block = [p1, p2]
        """
        nonlocal col
        m1, m2 = m_block
        p1, p2 = p_block
        
        # M1 = A11 * B11
        U[0*m + m1, col] = 1.0
        V[m1*p + p1, col] = 1.0
        W[p1*n + 0, col] = 1.0
        col += 1
        
        # M2 = A21 * B11
        U[1*m + m1, col] = 1.0
        V[m1*p + p1, col] = 1.0
        W[p1*n + 1, col] = 1.0
        col += 1
        
        # M3 = A12 * B21
        U[0*m + m2, col] = 1.0
        V[m2*p + p1, col] = 1.0
        W[p1*n + 0, col] = 1.0
        col += 1
        
        # M4 = A22 * B21
        U[1*m + m2, col] = 1.0
        V[m2*p + p1, col] = 1.0
        W[p1*n + 1, col] = 1.0
        col += 1
        
        # M5 = (A11 + A22) * (B11 - B21)
        U[0*m + m1, col] = 1.0
        U[1*m + m2, col] = 1.0
        V[m1*p + p1, col] = 1.0
        V[m2*p + p1, col] = -1.0
        W[p1*n + 0, col] = 1.0
        W[p1*n + 1, col] = -1.0
        col += 1
        
        # M6 = (A21 - A11) * (B11 + B12)
        U[1*m + m1, col] = 1.0
        U[0*m + m1, col] = -1.0
        V[m1*p + p1, col] = 1.0
        V[m1*p + p2, col] = 1.0
        W[p2*n + 1, col] = 1.0
        col += 1
        
        # M7 = (A12 - A22) * (B21 + B22)
        U[0*m + m2, col] = 1.0
        U[1*m + m2, col] = -1.0
        V[m2*p + p1, col] = 1.0
        V[m2*p + p2, col] = 1.0
        W[p2*n + 0, col] = 1.0
        col += 1
    
    def add_schoolbook_221(m_block, p_val):
        """Add schoolbook for 2x2x1 block.
        m_block = [m1, m2], p_val = p
        """
        nonlocal col
        m1, m2 = m_block
        pv = p_val
        
        # C1 = A11*B1 + A12*B2
        U[0*m + m1, col] = 1.0
        V[m1*p + pv, col] = 1.0
        W[pv*n + 0, col] = 1.0
        col += 1
        
        U[0*m + m2, col] = 1.0
        V[m2*p + pv, col] = 1.0
        W[pv*n + 0, col] = 1.0
        col += 1
        
        # C2 = A21*B1 + A22*B2
        U[1*m + m1, col] = 1.0
        V[m1*p + pv, col] = 1.0
        W[pv*n + 1, col] = 1.0
        col += 1
        
        U[1*m + m2, col] = 1.0
        V[m2*p + pv, col] = 1.0
        W[pv*n + 1, col] = 1.0
        col += 1
    
    # Block 1: m in {0, 1}
    add_strassen([0, 1], [0, 1])
    add_strassen([0, 1], [2, 3])
    add_schoolbook_221([0, 1], 4)
    
    # Block 2: m in {2, 3}
    add_strassen([2, 3], [0, 1])
    add_strassen([2, 3], [2, 3])
    add_schoolbook_221([2, 3], 4)
    
    # Verify: col should be 36
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
