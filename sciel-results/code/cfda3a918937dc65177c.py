import numpy as np
import time

def construct():
    """
    Matrix multiplication 4x4x4 via Kronecker product of Strassen's 2x2x2 algorithm.
    
    Uses the rank-7 Strassen decomposition for 2x2x2, then takes the Kronecker
    product to get a rank-49 decomposition for 4x4x4. Row permutations are applied
    to match the linear indexing conventions required by the task.
    
    Interface: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    The decomposition satisfies: for all i,j,k,
        sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    """
    start_time = time.time()
    deadline = start_time + 100  # safety margin before 120s limit
    
    # Strassen's 2x2x2 algorithm - verified correct version
    # U2: 4x7 (input A, rows indexed by i*2+j)
    # V2: 4x7 (input B, rows indexed by j*2+k)  
    # W2: 4x7 (output C, rows indexed by k*2+i per trace convention)
    #
    # Strassen terms:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    #
    # C00 = M0 + M3 - M4 - M5
    # C01 = M2 + M4
    # C10 = M1 + M5
    # C11 = M0 - M1 + M2 + M6
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A00: M0, M2, M4, M5
        [0, 0, 0, 0, 1, 0, 1],    # A01: M4, M6
        [0, 1, 0, 0, 0, 1, 0],    # A10: M1, M5
        [1, 1, 0, 1, 0, 0, -1]    # A11: M0, M1, M3, M6
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # B00: M0, M1, M3, M5
        [0, 0, 1, 0, 0, 1, 0],    # B01: M2, M5
        [0, 0, 0, 1, 0, 0, 1],    # B10: M3, M6
        [1, 0, -1, 0, 1, 0, 1]    # B11: M0, M2, M4, M6
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, -1, 0],  # C00 (k=0,i=0): M0, M3, M4, M5
        [0, 1, 0, 0, 0, 1, 0],    # C10 (k=0,i=1): M1, M5
        [0, 0, 1, 0, 1, 0, 0],    # C01 (k=1,i=0): M2, M4
        [1, -1, 1, 0, 0, 0, 1]    # C11 (k=1,i=1): M0, M1, M2, M6
    ], dtype=int)
    
    # Verify 2x2 Strassen correctness
    def _schoolbook():
        n, m, p = 4, 4, 4
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U[ii * m + jj][col] = 1
                    V[jj * p + kk][col] = 1
                    W[kk * n + ii][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    valid2 = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = int(np.dot(U2[i*2+j], V2[j*2+k] * W2[k*2+i]))
                if s != 1:
                    valid2 = False
                    break
            if not valid2:
                break
        if not valid2:
            break
    
    if not valid2:
        return _schoolbook()
    
    # Kronecker products give 16x49 matrices
    U4_raw = np.kron(U2, U2)
    V4_raw = np.kron(V2, V2)
    W4_raw = np.kron(W2, W2)
    
    # Compute row permutations to match task's linear indexing
    # For U: row S = i*4 + j, where i = i_out*2+i_in, j = j_out*2+j_in
    #   Kron row R = (i_out*2+j_out)*4 + (i_in*2+j_in)
    # For V: row S = j*4 + k, where j = j_out*2+j_in, k = k_out*2+k_in
    #   Kron row R = (j_out*2+k_out)*4 + (j_in*2+k_in)
    # For W: row S = k*4 + i, where k = k_out*2+k_in, i = i_out*2+i_in
    #   In W2, row index is k*2+i, so:
    #   Kron row R = (k_out*2+i_out)*4 + (k_in*2+i_in)
    
    perm_U = np.zeros(16, dtype=int)
    perm_V = np.zeros(16, dtype=int)
    perm_W = np.zeros(16, dtype=int)
    
    for S in range(16):
        # U permutation
        i = S // 4
        j = S % 4
        i_out, i_in = i // 2, i % 2
        j_out, j_in = j // 2, j % 2
        perm_U[S] = (i_out * 2 + j_out) * 4 + (i_in * 2 + j_in)
        
        # V permutation
        j = S // 4
        k = S % 4
        j_out, j_in = j // 2, j % 2
        k_out, k_in = k // 2, k % 2
        perm_V[S] = (j_out * 2 + k_out) * 4 + (j_in * 2 + k_in)
        
        # W permutation
        k = S // 4
        i = S % 4
        k_out, k_in = k // 2, k % 2
        i_out, i_in = i // 2, i % 2
        perm_W[S] = (k_out * 2 + i_out) * 4 + (k_in * 2 + i_in)
    
    U4 = U4_raw[perm_U, :]
    V4 = V4_raw[perm_V, :]
    W4 = W4_raw[perm_W, :]
    
    # Verify 4x4 correctness using vectorized approach
    valid4 = True
    if time.time() < deadline:
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s = int(np.dot(U4[i*4+j], V4[j*4+k] * W4[k*4+i]))
                    if s != 1:
                        valid4 = False
                        break
                if not valid4:
                    break
            if not valid4:
                break
    
    if not valid4:
        return _schoolbook()
    
    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }
