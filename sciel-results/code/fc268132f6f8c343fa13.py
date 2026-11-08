import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Build the matrix multiplication tensor T with indices (i, j, k)
    # where i in [0,n), j in [0,m), k in [0,p)
    # T[i*m+j, k*p+l, ...] -- actually let's use the standard form
    # The tensor for matrix mult (n x m) * (m x p) = (n x p)
    # T has entries T[i, j, k, l] = 1 if j == k else 0
    # We want: sum_r U[i,j,r] * V[j,k,r] * W[k,l,r] = T[i,j,k,l]
    
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)  
    # W shape: (p*n, rank) with W[k*n+i, r]
    
    # Let's build a systematic decomposition using block decomposition
    # with Strassen for 2x2x2 blocks where possible.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper: add a block of size (n_b, m_b, p_b) at offsets
    def add_block(n_off, m_off, p_off, n_b, m_b, p_b, U_b, V_b, W_b):
        """
        U_b: (n_b * m_b, rank_b)
        V_b: (m_b * p_b, rank_b)
        W_b: (p_b * n_b, rank_b)
        """
        for r in range(U_b.shape[1]):
            u_col = np.zeros(n * m)
            v_col = np.zeros(m * p)
            w_col = np.zeros(p * n)
            
            # Fill U: indices (n_off + i, m_off + j) -> (n_off+i)*m + (m_off+j)
            for i in range(n_b):
                for j in range(m_b):
                    u_idx = (n_off + i) * m + (m_off + j)
                    u_col[u_idx] = U_b[i * m_b + j, r]
            
            # Fill V: indices (m_off + j, p_off + k) -> (m_off+j)*p + (p_off+k)
            for j in range(m_b):
                for k in range(p_b):
                    v_idx = (m_off + j) * p + (p_off + k)
                    v_col[v_idx] = V_b[j * p_b + k, r]
            
            # Fill W: indices (p_off + k, n_off + i) -> (p_off+k)*n + (n_off+i)
            for k in range(p_b):
                for i in range(n_b):
                    w_idx = (p_off + k) * n + (n_off + i)
                    w_col[w_idx] = W_b[k * n_b + i, r]
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
    
    # Strassen 2x2x2 decomposition
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    # Schoolbook 2x2x1
    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        return U, V, W
    
    # Schoolbook 1x2x2
    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        return U, V, W
    
    # Schoolbook 1x2x1
    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        return U, V, W
    
    # Schoolbook 2x1x2
    def schoolbook_2x1x2():
        U = np.zeros((2, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[0, 1] = 1; W[1, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[0, 2] = 1
        U[1, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        return U, V, W
    
    # Schoolbook 1x1x2
    def schoolbook_1x1x2():
        U = np.zeros((1, 2))
        V = np.zeros((2, 2))
        W = np.zeros((2, 2))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[0, 1] = 1; V[1, 1] = 1; W[1, 1] = 1
        return U, V, W
    
    # Schoolbook 2x1x1
    def schoolbook_2x1x1():
        U = np.zeros((2, 2))
        V = np.zeros((1, 2))
        W = np.zeros((1, 2))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[0, 1] = 1; W[0, 1] = 1
        return U, V, W
    
    # Schoolbook 1x1x1
    def schoolbook_1x1x1():
        U = np.zeros((1, 1))
        V = np.zeros((1, 1))
        W = np.zeros((1, 1))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        return U, V, W
    
    # Now decompose 3x4x5
    # Split n=3 into 2+1
    # Split m=4 into 2+2
    # Split p=5 into 2+2+1
    
    # Block (2,2,2) at (0,0,0): Strassen, rank 7
    U_b, V_b, W_b = strassen_2x2x2()
    add_block(0, 0, 0, 2, 2, 2, U_b, V_b, W_b)
    
    # Block (2,2,2) at (0,2,0): Strassen, rank 7
    U_b, V_b, W_b = strassen_2x2x2()
    add_block(0, 2, 0, 2, 2, 2, U_b, V_b, W_b)
    
    # Block (2,2,1) at (0,0,4): schoolbook, rank 4
    U_b, V_b, W_b = schoolbook_2x2x1()
    add_block(0, 0, 4, 2, 2, 1, U_b, V_b, W_b)
    
    # Block (2,2,1) at (0,2,4): schoolbook, rank 4
    U_b, V_b, W_b = schoolbook_2x2x1()
    add_block(0, 2, 4, 2, 2, 1, U_b, V_b, W_b)
    
    # Block (1,2,2) at (2,0,0): schoolbook, rank 4
    U_b, V_b, W_b = schoolbook_1x2x2()
    add_block(2, 0, 0, 1, 2, 2, U_b, V_b, W_b)
    
    # Block (1,2,2) at (2,2,0): schoolbook, rank 4
    U_b, V_b, W_b = schoolbook_1x2x2()
    add_block(2, 2, 0, 1, 2, 2, U_b, V_b, W_b)
    
    # Block (1,2,1) at (2,0,4): schoolbook, rank 2
    U_b, V_b, W_b = schoolbook_1x2x1()
    add_block(2, 0, 4, 1, 2, 1, U_b, V_b, W_b)
    
    # Block (1,2,1) at (2,2,4): schoolbook, rank 2
    U_b, V_b, W_b = schoolbook_1x2x1()
    add_block(2, 2, 4, 1, 2, 1, U_b, V_b, W_b)
    
    # Total rank: 7+7+4+4+4+4+2+2 = 34
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    # Verify correctness
    if time.time() < deadline:
        # Check: sum_r U[:,r] outer V[:,r] outer W[:,r] should match the matmul tensor
        # For random test
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_test = A_test @ B_test
        
        # Compute from decomposition
        C_decomp = np.zeros((n, p))
        for r in range(U.shape[1]):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            C_decomp += np.trace(u * v * w.T)  # This isn't right, let me think...
        
        # Actually, the decomposition is:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        C_decomp = np.zeros((n, p))
        for r in range(U.shape[1]):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        C_decomp[i, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        if not np.allclose(C_decomp, C_test, atol=1e-10):
            # Fallback to schoolbook
            U_cols = []
            V_cols = []
            W_cols = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_col = np.zeros(n * m)
                        v_col = np.zeros(m * p)
                        w_col = np.zeros(p * n)
                        u_col[i*m+j] = 1
                        v_col[j*p+k] = 1
                        w_col[k*n+i] = 1
                        U_cols.append(u_col)
                        V_cols.append(v_col)
                        W_cols.append(w_col)
            U = np.column_stack(U_cols)
            V = np.column_stack(V_cols)
            W = np.column_stack(W_cols)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
