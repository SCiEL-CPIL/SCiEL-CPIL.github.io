import numpy as np
import time

def construct():
    """
    Matrix multiplication 4x4x4 via Kronecker product of Strassen's 2x2x2 algorithm.
    
    Uses rank-7 Strassen decomposition for 2x2x2 and Kronecker product to get
    rank-49 decomposition for 4x4x4. Careful row permutations ensure the linear
    indexing matches the task's convention: U[i*m+j, :], V[j*p+k, :], W[k*n+i, :].
    
    Score = 64/49 ≈ 1.306.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # Strassen's 2x2x2 algorithm matrices
    # U2: 4x7, rows indexed by (i,j) of A, linear index i*2+j
    # V2: 4x7, rows indexed by (j,k) of B, linear index j*2+k  
    # W2: 4x7, rows indexed by (k,i) of C, linear index k*2+i
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
    
    # Verify 2x2 Strassen decomposition first
    # T2[i,j,k] should be 1 for all i,j,k in 0..1
    # sum_r U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r] == 1
    valid_2x2 = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                u = U2[i*2+j]
                v = V2[j*2+k]
                w = W2[k*2+i]
                s = int(np.dot(u, v * w))
                if s != 1:
                    valid_2x2 = False
                    break
            if not valid_2x2:
                break
        if not valid_2x2:
            break
    
    if not valid_2x2:
        # Fallback to schoolbook rank 64
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
    
    # Kronecker products: rank = 7*7 = 49
    U4_raw = np.kron(U2, U2)  # 16x49
    V4_raw = np.kron(V2, V2)  # 16x49
    W4_raw = np.kron(W2, W2)  # 16x49
    
    # Permutation to map Kronecker row indices to task's linear indices
    # np.kron(A, B) row index R = i_A * B.shape[0] + i_B = i_A * 4 + i_B
    # where i_A, i_B are 0..3 (linear indices of the 2x2 sub-matrix entries)
    # i_A = r_out*2 + c_out (for U: row block, col block of A)
    # i_B = r_in*2 + c_in (for U: row inner, col inner of A)
    # Task index S = r*4 + c where r = r_out*2 + r_in, c = c_out*2 + c_in
    # So: r_out = r//2, r_in = r%2, c_out = c//2, c_in = c%2
    #     i_A = r_out*2 + c_out, i_B = r_in*2 + c_in
    #     R = i_A*4 + i_B
    perm = np.zeros(16, dtype=int)
    for S in range(16):
        r = S // 4
        c = S % 4
        r_out = r // 2
        r_in = r % 2
        c_out = c // 2
        c_in = c % 2
        i_A = r_out * 2 + c_out
        i_B = r_in * 2 + c_in
        R = i_A * 4 + i_B
        perm[S] = R
    
    U4 = U4_raw[perm, :]
    V4 = V4_raw[perm, :]
    W4 = W4_raw[perm, :]
    
    # Verify 4x4 decomposition: sum_r U4[i*4+j, r] * V4[j*4+k, r] * W4[k*4+i, r] == 1
    valid = True
    if time.time() < deadline:
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    u = U4[i*4+j]
                    v = V4[j*4+k]
                    w = W4[k*4+i]
                    s = int(np.dot(u, v * w))
                    if s != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
    
    if not valid:
        # Fallback to schoolbook (Rank 64)
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
    
    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }
