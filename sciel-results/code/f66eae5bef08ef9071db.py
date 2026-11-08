import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140

def construct():
    """
    Exact tensor decomposition for 3x4x5 matrix multiplication.
    Returns {'U', 'V', 'W'} where U is (12, rank), V is (20, rank), W is (15, rank).
    W uses convention W[k*n+i, r] where n=3.
    """
    n, m, p = 3, 4, 5
    
    if time.time() > _DEADLINE:
        return _schoolbook(n, m, p)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk):
        """Add a block decomposition with index mappings."""
        for r in range(U_blk.shape[1]):
            u = np.zeros(n * m)
            for i, idx in enumerate(u_idx):
                u[idx] = U_blk[i, r]
            v = np.zeros(m * p)
            for i, idx in enumerate(v_idx):
                v[idx] = V_blk[i, r]
            w = np.zeros(p * n)
            for i, idx in enumerate(w_idx):
                w[idx] = W_blk[i, r]
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)
    
    # Strassen 2x2x2: U(4x7), V(4x7), W(4x7)
    # U: [A00, A01, A10, A11], V: [B00, B01, B10, B11], W: [C00, C01, C10, C11]
    def strassen():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # M1 = (A00+A11)(B00+B11)
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[3,0]=1
        # M2 = (A10+A11)B00
        U[2,1]=1; U[3,1]=1; V[0,1]=1
        W[2,1]=1; W[3,1]=-1
        # M3 = A00(B01-B11)
        U[0,2]=1; V[1,2]=1; V[3,2]=-1
        W[1,2]=1; W[3,2]=1
        # M4 = A11(B10-B00)
        U[3,3]=1; V[2,3]=1; V[0,3]=-1
        W[0,3]=1; W[2,3]=1
        # M5 = (A00+A01)B11
        U[0,4]=1; U[1,4]=1; V[3,4]=1
        W[0,4]=-1; W[1,4]=1
        # M6 = (A10-A00)(B00+B01)
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1
        W[3,5]=1
        # M7 = (A01-A11)(B10+B11)
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1
        W[0,6]=1
        return U, V, W
    
    # Schoolbook 2x2x1: U(4x4), V(2x4), W(2x4)
    # A(2x2)*B(2x1)=C(2x1)
    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        U[0,0]=1; V[0,0]=1; W[0,0]=1  # A00*B0 -> C0
        U[1,1]=1; V[1,1]=1; W[0,1]=1  # A01*B1 -> C0
        U[2,2]=1; V[0,2]=1; W[1,2]=1  # A10*B0 -> C1
        U[3,3]=1; V[1,3]=1; W[1,3]=1  # A11*B1 -> C1
        return U, V, W
    
    # Schoolbook 1x2x2: U(2x4), V(4x4), W(2x4)
    # A(1x2)*B(2x2)=C(1x2)
    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        U[0,0]=1; V[0,0]=1; W[0,0]=1  # A0*B00 -> C0
        U[1,1]=1; V[2,1]=1; W[0,1]=1  # A1*B10 -> C0
        U[0,2]=1; V[1,2]=1; W[1,2]=1  # A0*B01 -> C1
        U[1,3]=1; V[3,3]=1; W[1,3]=1  # A1*B11 -> C1
        return U, V, W
    
    # Schoolbook 1x2x1: U(2x2), V(2x2), W(1x2)
    # A(1x2)*B(2x1)=C(1x1)
    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        U[0,0]=1; V[0,0]=1; W[0,0]=1  # A0*B0
        U[1,1]=1; V[1,1]=1; W[0,1]=1  # A1*B1
        return U, V, W
    
    # Index mappings:
    # U[i*m+j] for A[i,j], i in 0..2, j in 0..3
    # V[j*p+k] for B[j,k], j in 0..3, k in 0..4
    # W[k*n+i] for C[i,k], i in 0..2, k in 0..4
    
    # Split A into rows: A_top (rows 0,1), A_bot (row 2)
    # Split B columns into: B_left (cols 0,1), B_mid (cols 2,3), B_right (col 4)
    # Split B rows into: B_top (rows 0,1), B_bot (rows 2,3)
    
    # A_top_left: rows 0,1; cols 0,1 -> U indices [0,1,4,5]
    # A_top_right: rows 0,1; cols 2,3 -> U indices [2,3,6,7]
    # A_bot_left: row 2; cols 0,1 -> U indices [8,9]
    # A_bot_right: row 2; cols 2,3 -> U indices [10,11]
    
    # B_top_left: rows 0,1; cols 0,1 -> V indices [0,1,5,6]
    # B_top_mid: rows 0,1; cols 2,3 -> V indices [2,3,7,8]
    # B_top_right: rows 0,1; col 4 -> V indices [4,9]
    # B_bot_left: rows 2,3; cols 0,1 -> V indices [10,11,15,16]
    # B_bot_mid: rows 2,3; cols 2,3 -> V indices [12,13,17,18]
    # B_bot_right: rows 2,3; col 4 -> V indices [14,19]
    
    # W indices for C[i,k] = k*n + i = 3k + i
    # C row 0, cols 0,1: W[0], W[3]
    # C row 0, cols 2,3: W[6], W[9]
    # C row 0, col 4: W[12]
    # C row 1, cols 0,1: W[1], W[4]
    # C row 1, cols 2,3: W[7], W[10]
    # C row 1, col 4: W[13]
    # C row 2, cols 0,1: W[2], W[5]
    # C row 2, cols 2,3: W[8], W[11]
    # C row 2, col 4: W[14]
    
    # Block 1: A_top_left * B_top_left (2x2x2) -> C rows 0,1; cols 0,1
    U_s, V_s, W_s = strassen()
    add_block([0,1,4,5], [0,1,5,6], [0,3,1,4], U_s, V_s, W_s)
    
    # Block 2: A_top_left * B_top_mid (2x2x2) -> C rows 0,1; cols 2,3
    U_s, V_s, W_s = strassen()
    add_block([0,1,4,5], [2,3,7,8], [6,9,7,10], U_s, V_s, W_s)
    
    # Block 3: A_top_left * B_top_right (2x2x1) -> C rows 0,1; col 4
    U_s, V_s, W_s = schoolbook_2x2x1()
    add_block([0,1,4,5], [4,9], [12,13], U_s, V_s, W_s)
    
    # Block 4: A_top_right * B_bot_left (2x2x2) -> C rows 0,1; cols 0,1
    U_s, V_s, W_s = strassen()
    add_block([2,3,6,7], [10,11,15,16], [0,3,1,4], U_s, V_s, W_s)
    
    # Block 5: A_top_right * B_bot_mid (2x2x2) -> C rows 0,1; cols 2,3
    U_s, V_s, W_s = strassen()
    add_block([2,3,6,7], [12,13,17,18], [6,9,7,10], U_s, V_s, W_s)
    
    # Block 6: A_top_right * B_bot_right (2x2x1) -> C rows 0,1; col 4
    U_s, V_s, W_s = schoolbook_2x2x1()
    add_block([2,3,6,7], [14,19], [12,13], U_s, V_s, W_s)
    
    # Block 7: A_bot_left * B_top_left (1x2x2) -> C row 2; cols 0,1
    U_s, V_s, W_s = schoolbook_1x2x2()
    add_block([8,9], [0,1,5,6], [2,5], U_s, V_s, W_s)
    
    # Block 8: A_bot_left * B_top_mid (1x2x2) -> C row 2; cols 2,3
    U_s, V_s, W_s = schoolbook_1x2x2()
    add_block([8,9], [2,3,7,8], [8,11], U_s, V_s, W_s)
    
    # Block 9: A_bot_left * B_top_right (1x2x1) -> C row 2; col 4
    U_s, V_s, W_s = schoolbook_1x2x1()
    add_block([8,9], [4,9], [14], U_s, V_s, W_s)
    
    # Block 10: A_bot_right * B_bot_left (1x2x2) -> C row 2; cols 0,1
    U_s, V_s, W_s = schoolbook_1x2x2()
    add_block([10,11], [10,11,15,16], [2,5], U_s, V_s, W_s)
    
    # Block 11: A_bot_right * B_bot_mid (1x2x2) -> C row 2; cols 2,3
    U_s, V_s, W_s = schoolbook_1x2x2()
    add_block([10,11], [12,13,17,18], [8,11], U_s, V_s, W_s)
    
    # Block 12: A_bot_right * B_bot_right (1x2x1) -> C row 2; col 4
    U_s, V_s, W_s = schoolbook_1x2x1()
    add_block([10,11], [14,19], [14], U_s, V_s, W_s)
    
    rank = len(U_cols)
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    # Verify correctness
    _verify(U, V, W, n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    # Build the tensor T[a_idx, b_idx, c_idx] = sum_r U[a_idx,r] * V[b_idx,r] * W[c_idx,r]
    # Check against the matrix multiplication tensor
    for trial in range(3):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for r in range(rank):
            # Extract the A and B factors for this rank-1 term
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            # Reconstruct the contribution
            A_r = u.reshape(n, m)
            B_r = v.reshape(m, p)
            W_r = w.reshape(p, n)
            # C += sum over the bilinear form
            # Actually, the tensor decomposition means:
            # C[i,k] = sum_j A[i,j]*B[j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # So we need to compute this sum
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    C_computed[i,k] += s
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            raise ValueError(f"Verification failed: max diff = {np.max(np.abs(C_computed - C_expected))}")

def _schoolbook(n, m, p):
    """Schoolbook decomposition with rank n*m*p."""
    U_cols = []
    V_cols = []
    W_cols = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n * m)
                u[i * m + j] = 1
                v = np.zeros(m * p)
                v[j * p + k] = 1
                w = np.zeros(p * n)
                w[k * n + i] = 1
                U_cols.append(u)
                V_cols.append(v)
                W_cols.append(w)
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
