import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Split m=5 into 2+3, p=6 into 3+3
    # Blocks:
    # 1. A(4x2) B(2x3)
    # 2. A(4x2) B(2x3)
    # 3. A(4x3) B(3x3)
    # 4. A(4x3) B(3x3)
    
    def get_block_decomp(n_b, m_b, p_b):
        # Returns U, V, W for block of size n_b x m_b x p_b
        # U: (n_b*m_b, rank)
        # V: (m_b*p_b, rank)
        # W: (p_b*n_b, rank)
        
        if n_b == 1 or m_b == 1 or p_b == 1:
            rank = n_b * m_b * p_b
            U = np.zeros((n_b * m_b, rank))
            V = np.zeros((m_b * p_b, rank))
            W = np.zeros((p_b * n_b, rank))
            col = 0
            for i in range(n_b):
                for j in range(m_b):
                    for k in range(p_b):
                        U[i * m_b + j, col] = 1
                        V[j * p_b + k, col] = 1
                        W[k * n_b + i, col] = 1
                        col += 1
            return U, V, W, rank

        # Specific decompositions
        if n_b == 4 and m_b == 2 and p_b == 3:
            # Split p=3 into 2+1
            # Block 1: 4x2x2, Block 2: 4x2x1
            U1, V1, W1, r1 = get_block_decomp(4, 2, 2)
            U2, V2, W2, r2 = get_block_decomp(4, 2, 1)
            
            U = np.hstack([U1, U2])
            # V for 4x2x3: V1 is 2x2, V2 is 2x1.
            # V1 corresponds to k=0,1. V2 corresponds to k=2.
            V = np.zeros((m_b * p_b, r1 + r2))
            V[:2*2, :r1] = V1
            V[2*2:, r1:] = V2
            # W for 4x2x3: W1 is 2x4, W2 is 1x4.
            # W1 corresponds to k=0,1. W2 corresponds to k=2.
            W = np.zeros((p_b * n_b, r1 + r2))
            W[:2*4, :r1] = W1
            W[2*4:, r1:] = W2
            return U, V, W, r1 + r2

        if n_b == 4 and m_b == 3 and p_b == 3:
            # Split n=4 into 2+2
            # Block 1: 2x3x3, Block 2: 2x3x3
            U1, V1, W1, r1 = get_block_decomp(2, 3, 3)
            U2, V2, W2, r2 = get_block_decomp(2, 3, 3)
            
            U = np.vstack([U1, U2])
            # V for 4x3x3: V1 is 3x3, V2 is 3x3.
            V = np.hstack([V1, V2])
            # W for 4x3x3: W1 is 3x2, W2 is 3x2.
            # W1 corresponds to i=0,1. W2 corresponds to i=2,3.
            W = np.zeros((p_b * n_b, r1 + r2))
            W[:3*2, :r1] = W1
            W[3*2:, r1:] = W2
            return U, V, W, r1 + r2

        if n_b == 4 and m_b == 2 and p_b == 2:
            # Split n=4 into 2+2
            # Block 1: 2x2x2, Block 2: 2x2x2
            U1, V1, W1, r1 = get_block_decomp(2, 2, 2)
            U2, V2, W2, r2 = get_block_decomp(2, 2, 2)
            
            U = np.vstack([U1, U2])
            V = np.hstack([V1, V2])
            W = np.zeros((p_b * n_b, r1 + r2))
            W[:2*2, :r1] = W1
            W[2*2:, r1:] = W2
            return U, V, W, r1 + r2

        if n_b == 2 and m_b == 3 and p_b == 3:
            # Split m=3 into 2+1
            # Block 1: 2x2x3, Block 2: 2x1x3
            U1, V1, W1, r1 = get_block_decomp(2, 2, 3)
            U2, V2, W2, r2 = get_block_decomp(2, 1, 3)
            
            U = np.hstack([U1, U2])
            # V for 2x3x3: V1 is 2x3, V2 is 1x3.
            V = np.zeros((m_b * p_b, r1 + r2))
            V[:2*3, :r1] = V1
            V[2*3:, r1:] = V2
            W = np.hstack([W1, W2])
            return U, V, W, r1 + r2

        if n_b == 2 and m_b == 2 and p_b == 3:
            # Split p=3 into 2+1
            # Block 1: 2x2x2, Block 2: 2x2x1
            U1, V1, W1, r1 = get_block_decomp(2, 2, 2)
            U2, V2, W2, r2 = get_block_decomp(2, 2, 1)
            
            U = np.hstack([U1, U2])
            V = np.zeros((m_b * p_b, r1 + r2))
            V[:2*2, :r1] = V1
            V[2*2:, r1:] = V2
            W = np.zeros((p_b * n_b, r1 + r2))
            W[:2*2, :r1] = W1
            W[2*2:, r1:] = W2
            return U, V, W, r1 + r2

        if n_b == 2 and m_b == 2 and p_b == 2:
            # Strassen
            rank = 7
            U = np.zeros((4, 7))
            V = np.zeros((4, 7))
            W = np.zeros((4, 7))
            
            # M1 = (A11+A22)(B11+B22)
            # M2 = (A21+A22)B11
            # M3 = A11(B21-B22)
            # M4 = A22(B11-B21)
            # M5 = (A11+A12)B22
            # M6 = (A21-A11)(B11+B12)
            # M7 = (A12-A22)(B21+B22)
            
            # U rows: 00, 01, 10, 11
            # V rows: 00, 01, 10, 11
            # W rows: 00, 01, 10, 11
            
            # M1
            U[0, 0] = 1; U[3, 0] = 1
            V[0, 0] = 1; V[3, 0] = 1
            W[0, 0] = 1; W[3, 0] = 1
            
            # M2
            U[2, 1] = 1; U[3, 1] = 1
            V[0, 1] = 1
            W[1, 1] = 1; W[3, 1] = 1
            
            # M3
            U[0, 2] = 1
            V[2, 2] = 1; V[3, 2] = -1
            W[0, 2] = 1
            
            # M4
            U[3, 3] = 1
            V[0, 3] = 1; V[2, 3] = -1
            W[3, 3] = 1
            
            # M5
            U[0, 4] = 1; U[1, 4] = 1
            V[3, 4] = 1
            W[1, 4] = 1; W[2, 4] = 1
            
            # M6
            U[2, 5] = 1; U[0, 5] = -1
            V[0, 5] = 1; V[1, 5] = 1
            W[0, 5] = 1; W[2, 5] = 1
            
            # M7
            U[1, 6] = 1; U[3, 6] = -1
            V[2, 6] = 1; V[3, 6] = 1
            W[1, 6] = 1; W[3, 6] = 1
            
            return U, V, W, rank

        # Fallback: Schoolbook
        rank = n_b * m_b * p_b
        U = np.zeros((n_b * m_b, rank))
        V = np.zeros((m_b * p_b, rank))
        W = np.zeros((p_b * n_b, rank))
        col = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U[i * m_b + j, col] = 1
                    V[j * p_b + k, col] = 1
                    W[k * n_b + i, col] = 1
                    col += 1
        return U, V, W, rank

    U1, V1, W1, r1 = get_block_decomp(4, 2, 3)
    U2, V2, W2, r2 = get_block_decomp(4, 2, 3)
    U3, V3, W3, r3 = get_block_decomp(4, 3, 3)
    U4, V4, W4, r4 = get_block_decomp(4, 3, 3)
    
    total_rank = r1 + r2 + r3 + r4
    
    U = np.hstack([U1, U2, U3, U4])
    
    # V: m=5, p=6
    # Block 1: m=2, p=3. Rows 0-5 (2x3)
    # Block 2: m=2, p=3. Rows 0-5 (2x3)
    # Block 3: m=3, p=3. Rows 6-14 (3x3)
    # Block 4: m=3, p=3. Rows 6-14 (3x3)
    
    V = np.zeros((5 * 6, total_rank))
    # V1, V2 correspond to A(4x2) B(2x3). B is 2x3.
    # V1 shape (6, r1).
    V[:6, :r1] = V1
    V[:6, r1:r1+r2] = V2
    
    # V3, V4 correspond to A(4x3) B(3x3). B is 3x3.
    # V3 shape (9, r3).
    V[6:15, r1+r2:r1+r2+r3] = V3
    V[6:15, r1+r2+r3:] = V4
    
    # W: p=6, n=4
    # Block 1: p=3, n=4. Rows 0-11 (3x4)
    # Block 2: p=3, n=4. Rows 0-11 (3x4)
    # Block 3: p=3, n=4. Rows 12-23 (3x4)
    # Block 4: p=3, n=4. Rows 12-23 (3x4)
    
    W = np.zeros((6 * 4, total_rank))
    W[:12, :r1] = W1
    W[:12, r1:r1+r2] = W2
    W[12:24, r1+r2:r1+r2+r3] = W3
    W[12:24, r1+r2+r3:] = W4
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

