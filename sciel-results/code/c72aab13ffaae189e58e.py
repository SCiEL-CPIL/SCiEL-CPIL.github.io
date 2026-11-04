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
    
    # Strassen's 2x2x2 algorithm
    # U2: 4x7 (input A, rows indexed by i*2+j)
    # V2: 4x7 (input B, rows indexed by j*2+k)
    # W2: 4x7 (output C, rows indexed by k*2+i per trace convention)
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    # Quick verification of 2x2 Strassen (should always pass)
    # T[i,j,k] = sum_c U2[i*2+j, c] * V2[j*2+k, c] * W2[k*2+i, c] = 1
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
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = int(np.dot(U2[i*2+j], V2[j*2+k] * W2[k*2+i]))
                if s != 1:
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
        # Vectorized verification: check all 64 (i,j,k) triples
        # U4[i*4+j, :], V4[j*4+k, :], W4[k*4+i, :]
        # We need: sum_t U4[i*4+j,t] * V4[j*4+k,t] * W4[k*4+i,t] == 1 for all i,j,k
        valid4 = True
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
