import numpy as np
import time

def construct():
    n, m, p = 4, 4, 4
    
    # Deadline guard
    start_time = time.time()
    
    # Strassen's 2x2x2 algorithm
    # U2: 4x7 (input A), V2: 4x7 (input B), W2: 4x7 (output C)
    # Indices for 2x2:
    # U2 row index: i*2 + j (i,j in 0..1)
    # V2 row index: j*2 + k (j,k in 0..1)
    # W2 row index: k*2 + i (k,i in 0..1) -- Note: Trace convention W[k, i]
    
    # Standard Strassen matrices for C = A * B
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M0 = (a00+a11)(b00+b11)
    # M1 = (a10+a11)b00
    # M2 = a00(b01-b11)
    # M3 = a11(b10-b00)
    # M4 = (a00+a01)b11
    # M5 = (a10-a00)(b00+b01)
    # M6 = (a01-a11)(b10+b11)
    
    # U2[i,j] contributes to M_c
    # V2[j,k] contributes to M_c
    # W2[k,i] contributes to M_c (coefficient in C[i,k])
    
    # Let's build the 2x2 factors explicitly based on the definition of tensor decomposition
    # sum_c U2[i,j,c] * V2[j,k,c] * W2[k,i,c] = delta(i,j,k) ? No, it equals 1 if (i,j,k) is a valid product term?
    # Actually, the tensor T_{i,j,k} = 1 if C[i,k] depends on A[i,j] and B[j,k].
    # For matrix multiplication, C[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor T has T[i,j,k] = 1 for all i,j,k.
    # We need U, V, W such that sum_t U[i,j,t] V[j,k,t] W[k,i,t] = 1 for all i,j,k.
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # i=0, j=0
        [0, 0, 0, 0, 1, 0, 1],   # i=0, j=1
        [0, 1, 0, 0, 0, 1, 0],   # i=1, j=0
        [1, 1, 0, 1, 0, 0, -1]   # i=1, j=1
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # j=0, k=0
        [0, 0, 1, 0, 0, 1, 0],   # j=0, k=1
        [0, 0, 0, 1, 0, 0, 1],   # j=1, k=0
        [1, 0, -1, 0, 1, 0, 1]   # j=1, k=1
    ], dtype=int)
    
    # W2 row index is k*2 + i
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # k=0, i=0
        [0, 1, 0, 1, 0, 0, 0],   # k=0, i=1
        [0, 0, 1, 0, 1, 0, 0],   # k=1, i=0
        [1, -1, 1, 0, 0, 1, 0]   # k=1, i=1
    ], dtype=int)
    
    # Verify 2x2 decomposition
    # sum_t U2[i*2+j, t] * V2[j*2+k, t] * W2[k*2+i, t] == 1
    valid_2x2 = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = np.dot(U2[i*2+j, :], V2[j*2+k, :] * W2[k*2+i, :])
                if val != 1:
                    valid_2x2 = False
                    break
            if not valid_2x2: break
        if not valid_2x2: break
    
    if not valid_2x2:
        # Fallback to schoolbook if Strassen matrices are wrong (should not happen)
        return _schoolbook(n, m, p)
    
    # Construct 4x4 via Kronecker Product
    # The 4x4 tensor is the Kronecker product of the 2x2 tensor with itself.
    # T4[i,j,k] = T2[i1,j1,k1] * T2[i2,j2,k2] where i=2*i1+i2, etc.
    # Since T2 is all 1s, T4 is all 1s.
    # The factors are Kronecker products of the 2x2 factors, but we need to permute the rows
    # to match the specific row ordering:
    # U row: i*m + j = i*4 + j
    # V row: j*p + k = j*4 + k
    # W row: k*n + i = k*4 + i
    
    # Kronecker product of matrices A (m1 x n1) and B (m2 x n2) results in (m1*m2) x (n1*n2).
    # Row index of A⊗B is (rA, rB) -> rA*m2 + rB.
    # Here, U2 is 4x7. U4_raw = U2 ⊗ U2 is 16x49.
    # Row index of U4_raw is (rU2_outer, rU2_inner) -> rU2_outer * 4 + rU2_inner.
    # rU2_outer corresponds to the "first" 2x2 block indices.
    # rU2_inner corresponds to the "second" 2x2 block indices.
    
    # We need to map the target row index S = i*4 + j to the Kronecker row index K = r_out * 4 + r_in.
    # Let i = 2*i1 + i2, j = 2*j1 + j2.
    # The "outer" 2x2 multiplication involves indices i1, j1.
    # The "inner" 2x2 multiplication involves indices i2, j2.
    
    # For U (input A):
    # The factor U2 depends on (i, j).
    # In the Kronecker product U2 ⊗ U2, the first U2 factor corresponds to the outer indices (i1, j1) and the second to the inner indices (i2, j2).
    # So, U4_raw row K corresponds to U2[2*i1+j1] (outer) and U2[2*i2+j2] (inner).
    # K = (2*i1 + j1) * 4 + (2*i2 + j2).
    
    # We need to fill U4 such that U4[S, :] = U4_raw[K, :] where S = i*4 + j.
    # So we need a permutation perm_U such that U4 = U4_raw[perm_U, :].
    # perm_U[S] = K.
    
    perm_U = np.zeros(16, dtype=int)
    for i in range(4):
        for j in range(4):
            S = i * 4 + j
            i1, i2 = i // 2, i % 2
            j1, j2 = j // 2, j % 2
            r_out = 2 * i1 + j1
            r_in = 2 * i2 + j2
            K = r_out * 4 + r_in
            perm_U[S] = K
            
    perm_V = np.zeros(16, dtype=int)
    for j in range(4):
        for k in range(4):
            S = j * 4 + k
            j1, j2 = j // 2, j % 2
            k1, k2 = k // 2, k % 2
            r_out = 2 * j1 + k1
            r_in = 2 * j2 + k2
            K = r_out * 4 + r_in
            perm_V[S] = K
            
    perm_W = np.zeros(16, dtype=int)
    for k in range(4):
        for i in range(4):
            S = k * 4 + i
            k1, k2 = k // 2, k % 2
            i1, i2 = i // 2, i % 2
            r_out = 2 * k1 + i1
            r_in = 2 * k2 + i2
            K = r_out * 4 + r_in
            perm_W[S] = K
            
    U4_raw = np.kron(U2, U2)
    V4_raw = np.kron(V2, V2)
    W4_raw = np.kron(W2, W2)
    
    U4 = U4_raw[perm_U, :]
    V4 = V4_raw[perm_V, :]
    W4 = W4_raw[perm_W, :]
    
    # Verify 4x4 decomposition
    # sum_t U4[i*4+j, t] * V4[j*4+k, t] * W4[k*4+i, t] == 1
    valid_4x4 = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = np.dot(U4[i*4+j, :], V4[j*4+k, :] * W4[k*4+i, :])
                if val != 1:
                    valid_4x4 = False
                    break
            if not valid_4x4: break
        if not valid_4x4: break
        
    if not valid_4x4:
        # If the construction failed (due to permutation errors), fall back to schoolbook
        return _schoolbook(n, m, p)
        
    # Convert to lists of integers
    U_list = U4.tolist()
    V_list = V4.tolist()
    W_list = W4.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
