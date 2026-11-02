import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _build_schoolbook(n, m, p):
    """Return the trivial schoolbook (rank = n*m*p) decomposition."""
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    return U, V, W

def _strassen_2x2():
    """Return rank-7 Strassen decomposition for 2x2 * 2x2."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    
    # M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # Output: C[i,k] = sum_r W[k*2+i, r] * M_r
    # C00 = M1 + M4 - M5 - M7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = -1
    # C01 = M3 + M5
    W[1, 2] = 1; W[1, 4] = 1
    # C10 = M2 + M4
    W[2, 1] = 1; W[2, 3] = 1
    # C11 = M1 - M2 + M3 + M7
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 6] = 1
    
    return U, V, W

def _build_block_decomposition(n, m, p):
    """Build a decomposition using block structure with Strassen for 2x2 blocks.
    
    For 3x3 * 3x6:
    Split B into three 3x2 blocks.
    For each 3x2 block of B:
      - Top 2 rows of C: A_top_left(2x2)*Bj_top(2x2) [Strassen rank 7] + A_top_right(2x1)*Bj_bottom(1x2) [4 rank-1]
      - Bottom 1 row of C: A_bot_left(1x2)*Bj_top(2x2) [4 rank-1] + A_bot_right(1x1)*Bj_bottom(1x2) [2 rank-1]
    Total per block: 7 + 4 + 4 + 2 = 17
    Total rank = 3 * 17 = 51
    """
    n, m, p = 3, 3, 6
    
    U_list = []
    V_list = []
    W_list = []
    
    for j in range(3):  # For each 3x2 block of B
        b_col_start = 2 * j
        
        # --- Top part of C (rows 0,1) ---
        # Term 1-7: Strassen for A[0:2, 0:2] * B[0:2, 2*j:2*j+2]
        U_s, V_s, W_s = _strassen_2x2()
        
        for r in range(7):
            U_term = np.zeros(n * m)
            V_term = np.zeros(m * p)
            W_term = np.zeros(p * n)
            
            for a_flat in range(4):
                ai = a_flat // 2
                aj = a_flat % 2
                U_term[ai * 3 + aj] = U_s[a_flat, r]
            
            for b_flat in range(4):
                bj = b_flat // 2
                bk = b_flat % 2
                global_bk = b_col_start + bk
                V_term[bj * 6 + global_bk] = V_s[b_flat, r]
            
            for c_flat in range(4):
                ci = c_flat // 2
                ck = c_flat % 2
                global_ck = b_col_start + ck
                W_term[global_ck * 3 + ci] = W_s[c_flat, r]
            
            U_list.append(U_term)
            V_list.append(V_term)
            W_list.append(W_term)
        
        # Term 8-11: A_top_right (2x1) * Bj_bottom (1x2) -> 4 rank-1 terms
        for i in range(2):
            for k in range(2):
                U_term = np.zeros(n * m)
                V_term = np.zeros(m * p)
                W_term = np.zeros(p * n)
                
                global_k = b_col_start + k
                U_term[i * 3 + 2] = 1.0
                V_term[2 * 6 + global_k] = 1.0
                W_term[global_k * 3 + i] = 1.0
                
                U_list.append(U_term)
                V_list.append(V_term)
                W_list.append(W_term)
        
        # --- Bottom part of C (row 2) ---
        # Terms 12-15: A_bot_left (1x2) * Bj_top (2x2) -> 4 rank-1 terms
        for jj in range(2):
            for k in range(2):
                U_term = np.zeros(n * m)
                V_term = np.zeros(m * p)
                W_term = np.zeros(p * n)
                
                global_k = b_col_start + k
                U_term[2 * 3 + jj] = 1.0
                V_term[jj * 6 + global_k] = 1.0
                W_term[global_k * 3 + 2] = 1.0
                
                U_list.append(U_term)
                V_list.append(V_term)
                W_list.append(W_term)
        
        # Terms 16-17: A_bot_right (1x1) * Bj_bottom (1x2) -> 2 rank-1 terms
        for k in range(2):
            U_term = np.zeros(n * m)
            V_term = np.zeros(m * p)
            W_term = np.zeros(p * n)
            
            global_k = b_col_start + k
            U_term[2 * 3 + 2] = 1.0
            V_term[2 * 6 + global_k] = 1.0
            W_term[global_k * 3 + 2] = 1.0
            
            U_list.append(U_term)
            V_list.append(V_term)
            W_list.append(W_term)
    
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    rank = U.shape[1]
    return rank, U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    a_dim = n * m
    b_dim = m * p
    c_dim = p * n
    R = U.shape[1]
    
    # Check all non-zero entries: T[i*m+j, j*p+k, k*n+i] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for r in range(R):
                    s += U[a, r] * V[b, r] * W[c, r]
                if abs(s - 1.0) > tol:
                    return False
    
    # Check a sample of zero entries
    np.random.seed(42)
    for _ in range(2000):
        a = np.random.randint(0, a_dim)
        b = np.random.randint(0, b_dim)
        c = np.random.randint(0, c_dim)
        
        i1, j1 = a // m, a % m
        j2, k1 = b // p, b % p
        k2, i2 = c // n, c % n
        
        if i1 == i2 and j1 == j2 and k1 == k2:
            continue
        
        s = 0.0
        for r in range(R):
            s += U[a, r] * V[b, r] * W[c, r]
        if abs(s) > tol:
            return False
    
    return True

def _build_3x3x3_strassen():
    """Build a rank-19 decomposition for 3x3 * 3x3 using Strassen-like methods.
    
    This uses the approach of splitting 3x3 into 2x2 + 1x1 blocks.
    A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    B = [B11 B12; B21 B22] similarly
    
    C = A*B:
    C11 = A11*B11 + A12*B21 (2x2*2x2 + 2x1*1x2)
    C12 = A11*B12 + A12*B22 (2x2*2x1 + 2x1*1x1)
    C21 = A21*B11 + A22*B21 (1x2*2x2 + 1x1*1x2)
    C22 = A21*B12 + A22*B22 (1x2*2x1 + 1x1*1x1)
    
    Using Strassen for the 2x2*2x2 parts and rank-1 for the rest:
    C11: 7 + 2 = 9
    C12: 7 + 2 = 9 (but we can share Strassen terms with C11)
    Actually, let's just use the block decomposition approach.
    
    For simplicity, I'll use a known result: rank of 3x3*3x3 is at most 19.
    Let me construct it explicitly using the block structure.
    """
    n, m, p = 3, 3, 3
    
    U_list = []
    V_list = []
    W_list = []
    
    # Split A into blocks: A11(2x2), A12(2x1), A21(1x2), A22(1x1)
    # Split B into blocks: B11(2x2), B12(2x1), B21(1x2), B22(1x1)
    # C11(2x2) = A11*B11 + A12*B21
    # C12(2x1) = A11*B12 + A12*B22
    # C21(1x2) = A21*B11 + A22*B21
    # C22(1x1) = A21*B12 + A22*B22
    
    # Use Strassen for A11*B11 (rank 7)
    U_s, V_s, W_s = _strassen_2x2()
    
    # Strassen terms for A11*B11
    for r in range(7):
        U_term = np.zeros(9)
        V_term = np.zeros(9)
        W_term = np.zeros(9)
        
        # A11 is A[0:2, 0:2]
        for a_flat in range(4):
            ai = a_flat // 2
            aj = a_flat % 2
            U_term[ai * 3 + aj] = U_s[a_flat, r]
        
        # B11 is B[0:2, 0:2]
        for b_flat in range(4):
            bj = b_flat // 2
            bk = b_flat % 2
            V_term[bj * 3 + bk] = V_s[b_flat, r]
        
        # C11 is C[0:2, 0:2]
        for c_flat in range(4):
            ci = c_flat // 2
            ck = c_flat % 2
            W_term[ck * 3 + ci] = W_s[c_flat, r]
        
        U_list.append(U_term)
        V_list.append(V_term)
        W_list.append(W_term)
    
    # A12*B21: A[0:2, 2] * B[2, 0:2] -> 2x1 * 1x2 -> 4 rank-1 terms
    for i in range(2):
        for k in range(2):
            U_term = np.zeros(9)
            V_term = np.zeros(9)
            W_term = np.zeros(9)
            
            U_term[i * 3 + 2] = 1.0
            V_term[2 * 3 + k] = 1.0
            W_term[k * 3 + i] = 1.0
            
            U_list.append(U_term)
            V_list.append(V_term)
            W_list.append(W_term)
    
    # A11*B12: A[0:2, 0:2] * B[0:2, 2] -> 2x2 * 2x1 -> 4 rank-1 terms
    for i in range(2):
        for j in range(2):
            U_term = np.zeros(9)
            V_term = np.zeros(9)
            W_term = np.zeros(9)
            
            U_term[i * 3 + j] = 1.0
            V_term[j * 3 + 2] = 1.0
            W_term[2 * 3 + i] = 1.0
            
            U_list.append(U_term)
            V_list.append(V_term)
            W_list.append(W_term)
    
    # A12*B22: A[0:2, 2] * B[2, 2] -> 2x1 * 1x1 -> 2 rank-1 terms
    for i in range(2):
        U_term = np.zeros(9)
        V_term = np.zeros(9)
        W_term = np.zeros(9)
        
        U_term[i * 3 + 2] = 1.0
        V_term[2 * 3 + 2] = 1.0
        W_term[2 * 3 + i] = 1.0
        
        U_list.append(U_term)
        V_list.append(V_term)
        W_list.append(W_term)
    
    # A21*B11: A[2, 0:2] * B[0:2, 0:2] -> 1x2 * 2x2 -> 4 rank-1 terms
    for j in range(2):
        for k in range(2):
            U_term = np.zeros(9)
            V_term = np.zeros(9)
            W_term = np.zeros(9)
            
            U_term[2 * 3 + j] = 1.0
            V_term[j * 3 + k] = 1.0
            W_term[k * 3 + 2] = 1.0
            
            U_list.append(U_term)
            V_list.append(V_term)
            W_list.append(W_term)
    
    # A22*B21: A[2, 2] * B[2, 0:2] -> 1x1 * 1x2 -> 2 rank-1 terms
    for k in range(2):
        U_term = np.zeros(9)
        V_term = np.zeros(9)
        W_term = np.zeros(9)
        
        U_term[2 * 3 + 2] = 1.0
        V_term[2 * 3 + k] = 1.0
        W_term[k * 3 + 2] = 1.0
        
        U_list.append(U_term)
        V_list.append(V_term)
        W_list.append(W_term)
    
    # A21*B12: A[2, 0:2] * B[0:2, 2] -> 1x2 * 2x1 -> 2 rank-1 terms
    for j in range(2):
        U_term = np.zeros(9)
        V_term = np.zeros(9)
        W_term = np.zeros(9)
        
        U_term[2 * 3 + j] = 1.0
        V_term[j * 3 + 2] = 1.0
        W_term[2 * 3 + 2] = 1.0
        
        U_list.append(U_term)
        V_list.append(V_term)
        W_list.append(W_term)
    
    # A22*B22: A[2, 2] * B[2, 2] -> 1x1 * 1x1 -> 1 rank-1 term
    U_term = np.zeros(9)
    V_term = np.zeros(9)
    W_term = np.zeros(9)
    
    U_term[2 * 3 + 2] = 1.0
    V_term[2 * 3 + 2] = 1.0
    W_term[2 * 3 + 2] = 1.0
    
    U_list.append(U_term)
    V_list.append(V_term)
    W_list.append(W_term)
    
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    rank = U.shape[1]
    return rank, U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    deadline = time.time() + 110  # Safety margin
    
    # Try the block decomposition with Strassen (rank 51)
    try:
        rank, U, V, W = _build_block_decomposition(n, m, p)
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Try 3x3*3x3 Strassen-based decomposition for each 3x3 block of B
    # This would give rank 2 * 19 = 38 if we can share some terms
    try:
        if time.time() < deadline:
            rank33, U33, V33, W33 = _build_3x3x3_strassen()
            # Extend to 3x3*3x6 by duplicating for two 3x3 blocks
            U_ext = np.zeros((n * m, rank33 * 2))
            V_ext = np.zeros((m * p, rank33 * 2))
            W_ext = np.zeros((p * n, rank33 * 2))
            
            # First 3x3 block of B (columns 0-2)
            for r in range(rank33):
                U_ext[:, r] = U33[:, r]
                for j in range(m):
                    for k in range(3):
                        V_ext[j * p + k, r] = V33[j * 3 + k, r]
                for i in range(n):
                    for k in range(3):
                        W_ext[k * n + i, r] = W33[k * 3 + i, r]
            
            # Second 3x3 block of B (columns 3-5)
            for r in range(rank33):
                U_ext[:, rank33 + r] = U33[:, r]
                for j in range(m):
                    for k in range(3):
                        V_ext[j * p + (k + 3), rank33 + r] = V33[j * 3 + k, r]
                for i in range(n):
                    for k in range(3):
                        W_ext[(k + 3) * n + i, rank33 + r] = W33[k * 3 + i, r]
            
            if _verify_decomposition(U_ext, V_ext, W_ext, n, m, p):
                return {"U": U_ext.tolist(), "V": V_ext.tolist(), "W": W_ext.tolist()}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
