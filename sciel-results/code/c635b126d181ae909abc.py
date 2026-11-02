"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    start_time = time.time()
    deadline = start_time + 140.0

    def schoolbook(n, m, p):
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
        return U, V, W

    def strassen_2x2x2():
        """Strassen's 2x2x2 rank-7 decomposition."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1
        W[3, 0] = 1
        
        # M2 = (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        W[3, 1] = -1
        
        # M3 = A11(B12-B22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        W[3, 2] = 1
        
        # M4 = A22(B21-B11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1
        W[2, 3] = 1
        
        # M5 = (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1
        W[1, 4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def construct_3x2x2():
        """3x2x2 decomposition, rank 11.
        Split n=2+1.
        Part 1: 2x2x2 (Strassen, rank 7)
        Part 2: 1x2x2 (Schoolbook, rank 4)
        """
        n_sub, m_sub, p_sub = 3, 2, 2
        rank = 11
        U = np.zeros((n_sub * m_sub, rank))
        V = np.zeros((m_sub * p_sub, rank))
        W = np.zeros((p_sub * n_sub, rank))
        
        # Part 1: n=0:2, m=0:2, p=0:2 (Strassen)
        Us, Vs, Ws = strassen_2x2x2()
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * m_sub + j, r] = Us[i * 2 + j, r]
            for j in range(2):
                for k in range(2):
                    V[j * p_sub + k, r] = Vs[j * 2 + k, r]
            for k in range(2):
                for i in range(2):
                    W[k * n_sub + i, r] = Ws[k * 2 + i, r]
        
        # Part 2: n=2:3, m=0:2, p=0:2 (Schoolbook)
        col_off = 7
        col = 0
        for i in range(1):
            for j in range(2):
                for k in range(2):
                    U[(2 + i) * m_sub + j, col_off + col] = 1
                    V[j * p_sub + k, col_off + col] = 1
                    W[k * n_sub + (2 + i), col_off + col] = 1
                    col += 1
        
        return U, V, W

    def construct_3x2x4():
        """3x2x4 decomposition, rank 22.
        Split p=2+2.
        Part 1: 3x2x2 (rank 11)
        Part 2: 3x2x2 (rank 11)
        """
        n_sub, m_sub, p_sub = 3, 2, 4
        rank = 22
        U = np.zeros((n_sub * m_sub, rank))
        V = np.zeros((m_sub * p_sub, rank))
        W = np.zeros((p_sub * n_sub, rank))
        
        # Part 1: n=0:3, m=0:2, p=0:2
        U1, V1, W1 = construct_3x2x2()
        for r in range(11):
            for i in range(3):
                for j in range(2):
                    U[i * m_sub + j, r] = U1[i * 2 + j, r]
            for j in range(2):
                for k in range(2):
                    V[j * p_sub + k, r] = V1[j * 2 + k, r]
            for k in range(2):
                for i in range(3):
                    W[k * n_sub + i, r] = W1[k * 3 + i, r]
        
        # Part 2: n=0:3, m=0:2, p=2:4
        col_off = 11
        for r in range(11):
            for i in range(3):
                for j in range(2):
                    U[i * m_sub + j, col_off + r] = U1[i * 2 + j, r]
            for j in range(2):
                for k in range(2):
                    V[j * p_sub + (2 + k), col_off + r] = V1[j * 2 + k, r]
            for k in range(2):
                for i in range(3):
                    W[(2 + k) * n_sub + i, col_off + r] = W1[k * 3 + i, r]
        
        return U, V, W

    def construct_3x3x2():
        """3x3x2 decomposition, rank 17.
        Split n=2+1.
        Part 1: 2x3x2 (split m=2+1: 2x2x2(7) + 2x1x2(4) = 11)
        Part 2: 1x3x2 (Schoolbook, rank 6)
        """
        n_sub, m_sub, p_sub = 3, 3, 2
        rank = 17
        U = np.zeros((n_sub * m_sub, rank))
        V = np.zeros((m_sub * p_sub, rank))
        W = np.zeros((p_sub * n_sub, rank))
        
        # Part 1: n=0:2, m=0:3, p=0:2
        # 1.1: 2x2x2 (Strassen)
        Us, Vs, Ws = strassen_2x2x2()
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * m_sub + j, r] = Us[i * 2 + j, r]
            for j in range(2):
                for k in range(2):
                    V[j * p_sub + k, r] = Vs[j * 2 + k, r]
            for k in range(2):
                for i in range(2):
                    W[k * n_sub + i, r] = Ws[k * 2 + i, r]
        
        # 1.2: 2x1x2 (Schoolbook) - n=0:2, m=2:3, p=0:2
        col_off = 7
        col = 0
        for i in range(2):
            for j in range(1):
                for k in range(2):
                    U[i * m_sub + (2 + j), col_off + col] = 1
                    V[(2 + j) * p_sub + k, col_off + col] = 1
                    W[k * n_sub + i, col_off + col] = 1
                    col += 1
        
        # Part 2: n=2:3, m=0:3, p=0:2 (Schoolbook)
        col_off = 11
        col = 0
        for i in range(1):
            for j in range(3):
                for k in range(2):
                    U[(2 + i) * m_sub + j, col_off + col] = 1
                    V[j * p_sub + k, col_off + col] = 1
                    W[k * n_sub + (2 + i), col_off + col] = 1
                    col += 1
        
        return U, V, W

    def construct_3x3x4():
        """3x3x4 decomposition, rank 34.
        Split p=2+2.
        Part 1: 3x3x2 (rank 17)
        Part 2: 3x3x2 (rank 17)
        """
        n_sub, m_sub, p_sub = 3, 3, 4
        rank = 34
        U = np.zeros((n_sub * m_sub, rank))
        V = np.zeros((m_sub * p_sub, rank))
        W = np.zeros((p_sub * n_sub, rank))
        
        # Part 1: n=0:3, m=0:3, p=0:2
        U1, V1, W1 = construct_3x3x2()
        for r in range(17):
            for i in range(3):
                for j in range(3):
                    U[i * m_sub + j, r] = U1[i * 3 + j, r]
            for j in range(3):
                for k in range(2):
                    V[j * p_sub + k, r] = V1[j * 2 + k, r]
            for k in range(2):
                for i in range(3):
                    W[k * n_sub + i, r] = W1[k * 3 + i, r]
        
        # Part 2: n=0:3, m=0:3, p=2:4
        col_off = 17
        for r in range(17):
            for i in range(3):
                for j in range(3):
                    U[i * m_sub + j, col_off + r] = U1[i * 3 + j, r]
            for j in range(3):
                for k in range(2):
                    V[j * p_sub + (2 + k), col_off + r] = V1[j * 2 + k, r]
            for k in range(2):
                for i in range(3):
                    W[(2 + k) * n_sub + i, col_off + r] = W1[k * 3 + i, r]
        
        return U, V, W

    def verify(U, V, W, n, m, p):
        """Verify the decomposition."""
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_decomp = np.zeros((n, p))
        for r in range(U.shape[1]):
            M_r = np.zeros(n)
            N_r = np.zeros(p)
            for i in range(n):
                for j in range(m):
                    M_r[i] += U[i*m+j, r] * A[i, j]
            for k in range(p):
                for j in range(m):
                    N_r[k] += V[j*p+k, r] * B[j, k]
            for i in range(n):
                for k in range(p):
                    C_decomp[i, k] += W[k*n+i, r] * M_r[i] * N_r[k]
        
        return np.allclose(C_decomp, C_expected, atol=1e-10)

    # Build 3x5x6
    # Split m=2+3, p=2+4
    # Block 1: 3x2x2 (m=0:2, p=0:2) - rank 11
    # Block 2: 3x2x4 (m=0:2, p=2:6) - rank 22
    # Block 3: 3x3x2 (m=2:5, p=0:2) - rank 17
    # Block 4: 3x3x4 (m=2:5, p=2:6) - rank 34
    
    U1, V1, W1 = construct_3x2x2()
    U2, V2, W2 = construct_3x2x4()
    U3, V3, W3 = construct_3x3x2()
    U4, V4, W4 = construct_3x3x4()
    
    rank_total = U1.shape[1] + U2.shape[1] + U3.shape[1] + U4.shape[1]
    
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # Embed Block 1: m=0:2, p=0:2
    col_off = 0
    for r in range(U1.shape[1]):
        for i in range(3):
            for j in range(2):
                U[i * m + j, col_off + r] = U1[i * 2 + j, r]
        for j in range(2):
            for k in range(2):
                V[j * p + k, col_off + r] = V1[j * 2 + k, r]
        for k in range(2):
            for i in range(3):
                W[k * n + i, col_off + r] = W1[k * 3 + i, r]
    
    # Embed Block 2: m=0:2, p=2:6
    col_off += U1.shape[1]
    for r in range(U2.shape[1]):
        for i in range(3):
            for j in range(2):
                U[i * m + j, col_off + r] = U2[i * 2 + j, r]
        for j in range(2):
            for k in range(4):
                V[j * p + (2 + k), col_off + r] = V2[j * 4 + k, r]
        for k in range(4):
            for i in range(3):
                W[(2 + k) * n + i, col_off + r] = W2[(2 + k) * 3 + i, r]
    
    # Embed Block 3: m=2:5, p=0:2
    col_off += U2.shape[1]
    for r in range(U3.shape[1]):
        for i in range(3):
            for j in range(3):
                U[i * m + (2 + j), col_off + r] = U3[i * 3 + j, r]
        for j in range(3):
            for k in range(2):
                V[(2 + j) * p + k, col_off + r] = V3[j * 2 + k, r]
        for k in range(2):
            for i in range(3):
                W[k * n + i, col_off + r] = W3[k * 3 + i, r]
    
    # Embed Block 4: m=2:5, p=2:6
    col_off += U3.shape[1]
    for r in range(U4.shape[1]):
        for i in range(3):
            for j in range(3):
                U[i * m + (2 + j), col_off + r] = U4[i * 3 + j, r]
        for j in range(3):
            for k in range(4):
                V[(2 + j) * p + (2 + k), col_off + r] = V4[j * 4 + k, r]
        for k in range(4):
            for i in range(3):
                W[(2 + k) * n + i, col_off + r] = W4[(2 + k) * 3 + i, r]
    
    if verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback
    U, V, W = schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
