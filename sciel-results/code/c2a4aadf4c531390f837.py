import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def get_strassen_2x2_factors():
    # Returns U, V, W for 2x2x2 multiplication with rank 7
    # U: (4, 7) for A (2x2)
    # V: (4, 7) for B (2x2)
    # W: (4, 7) for C (2x2)
    
    # Indices for 2x2 matrix A:
    # 0: A00, 1: A01, 2: A10, 3: A11
    # Same for B, C.
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[1, 2] = 1
    
    # M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[3, 3] = 1
    
    # M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1; W[1, 5] = 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[3, 6] = 1; W[0, 6] = 1
    
    return U, V, W

def get_strassen_4x4_factors():
    # Use Kronecker product of 2x2 factors
    U2, V2, W2 = get_strassen_2x2_factors()
    
    # The 4x4 matrix is composed of 2x2 blocks.
    # A_4x4 = [A11 A12; A21 A22] where each is 2x2.
    # The tensor for 4x4x4 is the Kronecker product of the 2x2x2 tensor with itself.
    # U_4x4 = U2 (kron) U2
    # V_4x4 = V2 (kron) V2
    # W_4x4 = W2 (kron) W2
    
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    return U4, V4, W4

def get_schoolbook_4x4x1_factors():
    # Rank 16
    # A: 4x4, B: 4x1, C: 4x1
    U = np.zeros((16, 16))
    V = np.zeros((4, 16))
    W = np.zeros((4, 16))
    
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            U[i * 4 + j, idx] = 1
            V[j, idx] = 1
            W[i, idx] = 1
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    t0 = time.time()
    deadline = t0 + 140  # Safety margin before 150s limit

    # Get 4x4x4 factors (Strassen, rank 49)
    U4, V4, W4 = get_strassen_4x4_factors()
    # U4: (16, 49), V4: (16, 49), W4: (16, 49)

    # Get 4x4x1 factors (schoolbook, rank 16)
    U1, V1, W1 = get_schoolbook_4x4x1_factors()
    # U1: (16, 16), V1: (4, 16), W1: (4, 16)

    rank = 49 + 49 + 16
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Part 1: C[:, 0:4] = A * B[:, 0:4]
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4

    # Permutation for W4: row-major to column-major for 4x4
    perm = np.zeros(16, dtype=int)
    for i in range(4):
        for k in range(4):
            row_major_idx = i * 4 + k
            col_major_idx = k * 4 + i
            perm[row_major_idx] = col_major_idx

    W4_perm = W4[perm, :]
    W[0:16, 0:49] = W4_perm

    # Part 2: C[:, 4:8] = A * B[:, 4:8]
    v_perm = np.zeros(16, dtype=int)
    for j in range(4):
        for k in range(4):
            local_idx = j * 4 + k
            global_idx = j * 9 + (k + 4)
            v_perm[local_idx] = global_idx

    V[v_perm, 49:98] = V4
    W[16:32, 49:98] = W4_perm

    # Part 3: C[:, 8:9] = A * B[:, 8:9]
    v1_indices = [8, 17, 26, 35]
    for i in range(4):
        V[v1_indices[i], 98:114] = V1[i, :]

    U[:, 98:114] = U1

    w1_indices = [32, 33, 34, 35]
    for i in range(4):
        W[w1_indices[i], 98:114] = W1[i, :]

    # Verify
    if time.time() < deadline:
        if verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook if verification fails or time is up
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

def verify(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    # Compute S_ijk = sum_t U_ijt V_jkt W_kit
    # Using einsum
    S = np.einsum('ijt,jkt,kit->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    if not np.allclose(S, 1.0, atol=1e-9):
        return False
    return True
