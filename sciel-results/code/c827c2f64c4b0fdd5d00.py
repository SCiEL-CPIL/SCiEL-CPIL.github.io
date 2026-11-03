"""Matrix multiplication 2x3x6 - block decomposition approach."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np

def _build_schoolbook(n, m, p):
    """Build the trivial schoolbook decomposition for n x m times m x p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    # T[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r
    # We check a set of random integer matrices for exactness
    rng = np.random.RandomState(42)
    for trial in range(20):
        A = rng.randint(-3, 4, size=(n, m))
        B = rng.randint(-3, 4, size=(m, p))
        C_expected = A @ B
        # Compute via tensor: C[i,k] = sum_{j,r} U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
        # More efficiently: build the bilinear map
        C = np.zeros((n, p))
        for r in range(U.shape[1]):
            u_r = U[:, r]  # length n*m
            v_r = V[:, r]  # length m*p
            w_r = W[:, r]  # length p*n
            # u_r reshaped to (n, m), v_r to (m, p), w_r to (p, n)
            u_r = u_r.reshape(n, m)
            v_r = v_r.reshape(m, p)
            w_r = w_r.reshape(p, n)
            # contribution: sum_j A[i,j]*u_r[i,j] * v_r[j,k] * B[j,k] * w_r[k,i]
            # = sum_{j,k} w_r[k,i] * (sum_j u_r[i,j]*A[i,j]*v_r[j,k]) * B[j,k]
            # Let's just do it directly:
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += u_r[i, j] * A[i, j] * v_r[j, k] * B[j, k]
                    C[i, k] += w_r[k, i] * s
        if not np.allclose(C, C_expected, atol=1e-9):
            return False
    return True

def _try_block_decomp(n, m, p, n_blocks, m_blocks, p_blocks):
    """
    Try to decompose n x m * m x p by partitioning into blocks and using
    rank-7 Strassen-like for 2x2 blocks where possible, schoolbook elsewhere.
    We partition the m dimension (inner) and the i,k dimensions.
    """
    # We'll partition A into blocks of size (n_b, m_b) and B into (m_b, p_b)
    # Simplest: partition along m (the shared dimension) into m_blocks chunks,
    # and along n into n_blocks, p into p_blocks.
    # For each block combination (block_i, block_j, block_k) we need a decomposition.
    # This gets complex; instead use a simpler scheme:
    # Partition m into 3 chunks of size 1 (since m=3).
    # Then C = sum_{j=0}^{2} A[:, j:j+1] @ B[j:j+1, :]
    # Each term is (n x 1) * (1 x p) = rank n*p each, total 3*n*p.
    # That's just schoolbook. Not helpful.
    
    # Alternative: use the fact that R(n,m,p) <= sum over partitions.
    # For 2x3x6, one known bound: R(2,3,6) <= 2*R(1,3,6) + ... 
    # Actually R(2,3,6) <= R(2,3,3) + R(2,3,3) and R(2,3,3) <= ?
    # R(2,3,3): partition m=3 into 1+2: R(2,1,3)+R(2,2,3) = 2*3 + R(2,2,3)
    # R(2,2,3) <= R(2,2,2)+R(2,2,1) = 8 + 4 = 12, or use Strassen 7 + 4 = 11?
    # R(2,2,2) with Strassen = 7, R(2,2,1) = 4 (schoolbook, since 2*2*1=4)
    # So R(2,2,3) <= 7 + 4 = 11.
    # R(2,1,3) = 2*1*3 = 6 (rank 6, it's just outer products).
    # R(2,3,3) <= 6 + 11 = 17.
    # R(2,3,6) <= 2 * 17 = 34.
    # That's an improvement over 36!
    
    # Let's implement: partition p=6 into two chunks of 3.
    # C[:, 0:3] = A @ B[:, 0:3]  (2x3 * 3x3)
    # C[:, 3:6] = A @ B[:, 3:6]  (2x3 * 3x3)
    # Each 2x3*3x3 needs rank <= 17.
    # For 2x3*3x3: partition m=3 into m1=1, m2=2.
    # A @ B = A[:, 0:1] @ B[0:1, :] + A[:, 1:3] @ B[1:3, :]
    # First term: 2x1 * 1x3, rank = 2*1*3 = 6.
    # Second term: 2x2 * 2x3, rank <= R(2,2,2) + R(2,2,1) = 7 + 4 = 11.
    # Total for 2x3*3x3: 6 + 11 = 17.
    # Total for 2x3*3x6: 17 + 17 = 34.
    
    # Now build the actual decomposition.
    # We'll construct U, V, W as concatenations.
    
    def _schoolbook_small(n_s, m_s, p_s):
        """Schoolbook for small dimensions."""
        rank_s = n_s * m_s * p_s
        U_s = np.zeros((n_s * m_s, rank_s))
        V_s = np.zeros((m_s * p_s, rank_s))
        W_s = np.zeros((p_s * n_s, rank_s))
        col = 0
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    U_s[i * m_s + j, col] = 1.0
                    V_s[j * p_s + k, col] = 1.0
                    W_s[k * n_s + i, col] = 1.0
                    col += 1
        return U_s, V_s, W_s
    
    def _strassen_222():
        """Rank-7 Strassen decomposition for 2x2*2x2.
        Returns U (4x7), V (4x7), W (4x7).
        Convention: A is 2x2, B is 2x2, C = A@B is 2x2.
        U index: i*2+j for A[i,j]
        V index: j*2+k for B[j,k]
        W index: k*2+i for C[i,k] (trace convention)
        
        Strassen:
        M1 = (a11+a22)(b11+b22)
        M2 = (a21+a22)(b11)
        M3 = (a11)(b12-b22)
        M4 = (a22)(b21-b11)
        M5 = (a11+a12)(b22)
        M6 = (a21-a11)(b11+b12)
        M7 = (a12-a22)(b21+b22)
        
        c11 = M1 - M4 + M5 - M7
        c12 = M3 + M7
        c21 = M2 + M6
        c22 = M1 - M3 + M4 + M6
        """
        # Let me verify this Strassen formula.
        # Standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)(B11)
        # M3 = (A11)(B12-B22)
        # M4 = (A22)(B21-B11)
        # M5 = (A11+A12)(B22)
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M7
        # C21 = M2 + M6
        # C22 = M1 - M3 + M4 + M6
        # Let me verify with A=I, B=I:
        # M1 = (1+0)(1+0) = 1
        # M2 = (0+0)(1) = 0
        # M3 = (1)(0-0) = 0
        # M4 = (0)(0-1) = 0
        # M5 = (1+0)(0) = 0
        # M6 = (0-1)(1+0) = -1
        # M7 = (0-0)(0+0) = 0
        # C11 = 1 - 0 + 0 - 0 = 1 ✓
        # C12 = 0 + 0 = 0 ✓
        # C21 = 0 + (-1) = -1 ✗ (should be 0)
        # This is wrong. Let me use a different standard form.
        
        # Let me use the standard Strassen from Wikipedia:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)(B11)
        # M3 = (A11)(B12-B22)
        # M4 = (A22)(B21-B11)
        # M5 = (A11+A12)(B22)
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M7
        # C21 = M2 + M6
        # C22 = M1 - M3 + M4 + M6
        
        # The issue is with M6. Let me recheck.
        # Actually the standard Strassen is:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M7
        # C21 = M2 + M6
        # C22 = M1 - M3 + M4 + M6
        
        # For A=I, B=I:
        # M1 = (1+0)(1+0) = 1
        # M2 = (0+0)(1) = 0
        # M3 = 1*(0-0) = 0
        # M4 = 0*(0-1) = 0
        # M5 = (1+0)*0 = 0
        # M6 = (0-1)*(1+0) = -1
        # M7 = (0-0)*(0+0) = 0
        # C11 = 1 - 0 + 0 - 0 = 1 ✓
        # C12 = 0 + 0 = 0 ✓
        # C21 = 0 + (-1) = -1 ✗
        
        # Something is wrong. Let me look up the correct formula.
        # The correct Strassen formulas are:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)(B11)
        # M3 = (A11)(B12 - B22)
        # M4 = (A22)(B21 - B11)
        # M5 = (A11 + A12)(B22)
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M7
        # C21 = M2 + M6
        # C22 = M1 - M3 + M4 + M6
        
        # I think the issue is that this particular set of 7 products doesn't work.
        # Let me use a verified set. The standard Strassen uses:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)(B11)
        # M3 = (A11)(B12-B22)
        # M4 = (A22)(B21-B11)
        # M5 = (A11+A12)(B22)
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # And the reconstruction is:
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M7
        # C21 = M2 + M6
        # C22 = M1 - M3 + M4 + M6
        
        # Let me test with A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        # C = [[19,22],[43,50]]
        # M1 = (1+4)(5+8) = 5*13 = 65
        # M2 = (3+4)(5) = 7*5 = 35
        # M3 = (1)(6-8) = 1*(-2) = -2
        # M4 = (4)(7-5) = 4*2 = 8
        # M5 = (1+2)(8) = 3*8 = 24
        # M6 = (3-1)(5+6) = 2*11 = 22
        # M7 = (2-4)(7+8) = (-2)*15 = -30
        # C11 = 65 - 8 + 24 - (-30) = 65 - 8 + 24 + 30 = 111 ✗ (should be 19)
        
        # This is clearly wrong. Let me use a completely different approach.
        # I'll just use schoolbook for 2x2x2 (rank 8) to be safe.
        return _schoolbook_small(2, 2, 2)
    
    # Given the difficulty of getting Strassen right, let's just use schoolbook
    # for all sub-problems. This gives rank 36 which is the same as before.
    # But let's at least try the block partitioning to see if we can do better
    # with a verified small decomposition.
    
    # For now, fall back to schoolbook.
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try block decomposition: partition p=6 into two 3s.
    # C[:, 0:3] = A @ B[:, 0:3]  (2x3 * 3x3)
    # C[:, 3:6] = A @ B[:, 3:6]  (2x3 * 3x3)
    # For each 2x3*3x3, partition m=3 into 1+2:
    #   Term1: A[:, 0:1] @ B[0:1, :]  (2x1 * 1x3) rank 6
    #   Term2: A[:, 1:3] @ B[1:3, :]  (2x2 * 2x3) rank <= 11 (Strassen 7 + schoolbook 4)
    # Total per 2x3*3x3: 6 + 11 = 17
    # Total: 34
    
    # Let's build this carefully.
    # We need a rank-7 decomposition for 2x2*2x2.
    # Let me hardcode a verified one.
    
    # After much deliberation, let me just use schoolbook for 2x2x2 (rank 8)
    # to ensure correctness. Total rank: 2*(6 + 8 + 4) = 2*18 = 36. Same as before.
    
    # Actually, let me try to get Strassen right by constructing it from scratch
    # and verifying.
    
    def _strassen_222_verified():
        """Construct and verify a rank-7 decomposition for 2x2*2x2."""
        # Strassen's 7 products:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)(B11)
        # M3 = (A11)(B12-B22)
        # M4 = (A22)(B21-B11)
        # M5 = (A11+A12)(B22)
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # U matrix: rows indexed by (i,j) for A, cols by product index
        # U[i*2+j, r] = coefficient of A[i,j] in product r
        U = np.zeros((4, 7))
        # M1: (A11+A22)(B11+B22) -> A part: A11 + A22
        U[0, 0] = 1  # A11 in M1
        U[3, 0] = 1  # A22 in M1
        # M2: (A21+A22)(B11) -> A part: A21 + A22
        U[2, 1] = 1  # A21 in M2
        U[3, 1] = 1  # A22 in M2
        # M3: (A11)(B12-B22) -> A part: A11
        U[0, 2] = 1  # A11 in M3
        # M4: (A22)(B21-B11) -> A part: A22
        U[3, 3] = 1  # A22 in M4
        # M5: (A11+A12)(B22) -> A part: A11 + A12
        U[0, 4] = 1  # A11 in M5
        U[1, 4] = 1  # A12 in M5
        # M6: (A21-A11)(B11+B12) -> A part: A21 - A11
        U[2, 5] = 1  # A21 in M6
        U[0, 5] = -1  # A11 in M6
        # M7: (A12-A22)(B21+B22) -> A part: A12 - A22
        U[1, 6] = 1  # A12 in M7
        U[3, 6] = -1  # A22 in M7
        
        # V matrix: rows indexed by (j,k) for B, cols by product index
        V = np.zeros((4, 7))
        # M1: B part: B11 + B22
        V[0, 0] = 1  # B11
        V[3, 0] = 1  # B22
        # M2: B part: B11
        V[0, 1] = 1  # B11
        # M3: B part: B12 - B22
        V[1, 2] = 1  # B12
        V[3, 2] = -1  # B22
        # M4: B part: B21 - B11
        V[2, 3] = 1  # B21
        V[0, 3] = -1  # B11
        # M5: B part: B22
        V[3, 4] = 1  # B22
        # M6: B part: B11 + B12
        V[0, 5] = 1  # B11
        V[1, 5] = 1  # B12
        # M7: B part: B21 + B22
        V[2, 6] = 1  # B21
        V[3, 6] = 1  # B22
        
        # W matrix: rows indexed by (k,i) for C (trace convention C[i,k]), cols by product index
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M7
        # C21 = M2 + M6
        # C22 = M1 - M3 + M4 + M6
        W = np.zeros((4, 7))
        # W[k*2+i, r]
        # C11: k=0,i=0 -> W[0, r]
        W[0, 0] = 1   # M1
        W[0, 3] = -1  # -M4
        W[0, 4] = 1   # M5
        W[0, 6] = -1  # -M7
        # C12: k=1,i=0 -> W[1, r]
        W[1, 2] = 1   # M3
        W[1, 6] = 1   # M7
        # C21: k=0,i=1 -> W[2, r]
        W[2, 1] = 1   # M2
        W[2, 5] = 1   # M6
        # C22: k=1,i=1 -> W[3, r]
        W[3, 0] = 1   # M1
        W[3, 2] = -1  # -M3
        W[3, 3] = 1   # M4
        W[3, 5] = 1   # M6
        
        # Verify
        rng = np.random.RandomState(123)
        for trial in range(50):
            A = rng.randint(-3, 4, size=(2, 2))
            B = rng.randint(-3, 4, size=(2, 2))
            C_expected = A @ B
            C = np.zeros((2, 2))
            for r in range(7):
                u_r = U[:, r].reshape(2, 2)
                v_r = V[:, r].reshape(2, 2)
                w_r = W[:, r].reshape(2, 2)
                # Contribution: sum_{j,k} w_r[k,i] * (sum_j u_r[i,j]*A[i,j]*v_r[j,k]) * B[j,k]
                for i in range(2):
                    for k in range(2):
                        s = 0.0
                        for j in range(2):
                            s += u_r[i, j] * A[i, j] * v_r[j, k] * B[j, k]
                        C[i, k] += w_r[k, i] * s
            if not np.allclose(C, C_expected, atol=1e-9):
                return None  # Strassen failed verification
        
        return U, V, W
    
    strassen = _strassen_222_verified()
    
    if strassen is not None:
        # Use Strassen for 2x2*2x2 blocks
        U_s, V_s, W_s = strassen
        # Build the full decomposition
        # Structure:
        # C[:, 0:3] = A @ B[:, 0:3]
        #   = A[:, 0:1] @ B[0:1, 0:3] + A[:, 1:3] @ B[1:3, 0:3]
        # C[:, 3:6] = A @ B[:, 3:6]
        #   = A[:, 0:1] @ B[0:1, 3:6] + A[:, 1:3] @ B[1:3, 3:6]
        
        # Block 1: A[:, 0:1] @ B[0:1, 0:3]  (2x1 * 1x3) rank 6
        # Block 2: A[:, 1:3] @ B[1:3, 0:3]  (2x2 * 2x3) rank 11
        # Block 3: A[:, 0:1] @ B[0:1, 3:6]  (2x1 * 1x3) rank 6
        # Block 4: A[:, 1:3] @ B[1:3, 3:6]  (2x2 * 2x3) rank 11
        # Total rank: 6 + 11 + 6 + 11 = 34
        
        # For 2x2 * 2x3: partition p=3 into 2+1
        #   2x2*2x2 (Strassen, rank 7) + 2x2*2x1 (schoolbook, rank 4) = 11
        
        total_rank = 34
        U_full = np.zeros((n * m, total_rank))
        V_full = np.zeros((m * p, total_rank))
        W_full = np.zeros((p * n, total_rank))
        
        col = 0
        # Block 1: A[:, 0:1] @ B[0:1, 0:3]  (2x1 * 1x3)
        # A indices: i in 0..1, j=0. U index: i*3+0
        # B indices: j=0, k in 0..2. V index: 0*6+k = k
        # C indices: k in 0..2, i in 0..1. W index: k*2+i
        for i in range(2):
            for k in range(3):
                U_full[i * m + 0, col] = 1.0
                V_full[0 * p + k, col] = 1.0
                W_full[k * n + i, col] = 1.0
                col += 1
        # col = 6
        
        # Block 2: A[:, 1:3] @ B[1:3, 0:3]  (2x2 * 2x3)
        # A indices: i in 0..1, j in 1..2. U index: i*3+j
        # B indices: j in 1..2, k in 0..2. V index: j*6+k
        # C indices: k in 0..2, i in 0..1. W index: k*2+i
        # Partition: 2x2*2x2 (j=1..2, k=0..1) + 2x2*2x1 (j=1..2, k=2)
        
        # Sub-block 2a: 2x2*2x2 using Strassen
        # Local A: rows i=0..1, cols j=1..2 -> local (i, j') where j'=j-1
        # Local B: rows j'=0..1 (j=1..2), cols k=0..1
        # Local C: rows k=0..1, cols i=0..1
        for r in range(7):
            for i in range(2):
                for j_local in range(2):
                    j = j_local + 1  # j=1,2
                    U_full[i * m + j, col] = U_s[i * 2 + j_local, r]
            for j_local in range(2):
                for k in range(2):
                    j = j_local + 1
                    V_full[j * p + k, col] = V_s[j_local * 2 + k, r]
            for k in range(2):
                for i in range(2):
                    W_full[k * n + i, col] = W_s[k * 2 + i, r]
            col += 1
        # col = 6 + 7 = 13
        
        # Sub-block 2b: 2x2*2x1 (j=1..2, k=2) schoolbook rank 4
        for i in range(2):
            for j_local in range(2):
                j = j_local + 1
                U_full[i * m + j, col] = 1.0
                V_full[j * p + 2, col] = 1.0
                W_full[2 * n + i, col] = 1.0
                col += 1
        # col = 13 + 4 = 17
        
        # Block 3: A[:, 0:1] @ B[0:1, 3:6]  (2x1 * 1x3)
        for i in range(2):
            for k_local in range(3):
                k = k_local + 3
                U_full[i * m + 0, col] = 1.0
                V_full[0 * p + k, col] = 1.0
                W_full[k * n + i, col] = 1.0
                col += 1
        # col = 17 + 6 = 23
        
        # Block 4: A[:, 1:3] @ B[1:3, 3:6]  (2x2 * 2x3)
        # Sub-block 4a: 2x2*2x2 (j=1..2, k=3..4)
        for r in range(7):
            for i in range(2):
                for j_local in range(2):
                    j = j_local + 1
                    U_full[i * m + j, col] = U_s[i * 2 + j_local, r]
            for j_local in range(2):
                for k_local in range(2):
                    j = j_local + 1
                    k = k_local + 3
                    V_full[j * p + k, col] = V_s[j_local * 2 + k_local, r]
            for k_local in range(2):
                for i in range(2):
                    k = k_local + 3
                    W_full[k * n + i, col] = W_s[k_local * 2 + i, r]
            col += 1
        # col = 23 + 7 = 30
        
        # Sub-block 4b: 2x2*2x1 (j=1..2, k=5)
        for i in range(2):
            for j_local in range(2):
                j = j_local + 1
                U_full[i * m + j, col] = 1.0
                V_full[j * p + 5, col] = 1.0
                W_full[5 * n + i, col] = 1.0
                col += 1
        # col = 30 + 4 = 34
        
        # Verify the full decomposition
        if _verify(U_full, V_full, W_full, n, m, p):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
