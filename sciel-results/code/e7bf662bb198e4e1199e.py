"""Matrix multiplication 4x4x4 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-28 decomposition for 4x4 matrix multiplication
    using the Kronecker product of Strassen (rank 7) and Naive (rank 4) decompositions.
    Falls back to rank-49 or rank-64 if verification fails.
    """
    start_time = time.time()
    DEADLINE = 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # 1. Strassen 2x2 decomposition (rank 7)
    U2_s = np.zeros((4, 7), dtype=int)
    V2_s = np.zeros((4, 7), dtype=int)
    W2_s = np.zeros((4, 7), dtype=int)

    # M0 = (A00+A11)(B00+B11)
    U2_s[0, 0] = 1; U2_s[3, 0] = 1
    V2_s[0, 0] = 1; V2_s[3, 0] = 1
    W2_s[0, 0] = 1; W2_s[3, 0] = 1

    # M1 = (A10+A11)B00
    U2_s[2, 1] = 1; U2_s[3, 1] = 1
    V2_s[0, 1] = 1
    W2_s[1, 1] = 1

    # M2 = A00(B10-B11)
    U2_s[0, 2] = 1
    V2_s[2, 2] = 1; V2_s[3, 2] = -1
    W2_s[2, 2] = 1

    # M3 = A11(B01-B00)
    U2_s[3, 3] = 1
    V2_s[1, 3] = 1; V2_s[0, 3] = -1
    W2_s[2, 3] = 1; W2_s[3, 3] = 1

    # M4 = (A00+A01)B11
    U2_s[0, 4] = 1; U2_s[1, 4] = 1
    V2_s[3, 4] = 1
    W2_s[0, 4] = -1; W2_s[2, 4] = 1

    # M5 = (A10-A00)(B00+B01)
    U2_s[2, 5] = 1; U2_s[0, 5] = -1
    V2_s[0, 5] = 1; V2_s[1, 5] = 1
    W2_s[1, 5] = -1

    # M6 = (A01-A11)(B10+B11)
    U2_s[1, 6] = 1; U2_s[3, 6] = -1
    V2_s[2, 6] = 1; V2_s[3, 6] = 1
    W2_s[1, 6] = 1

    # 2. Naive 2x2 decomposition (rank 4)
    U2_n = np.zeros((4, 4), dtype=int)
    V2_n = np.zeros((4, 4), dtype=int)
    W2_n = np.zeros((4, 4), dtype=int)
    for i in range(2):
        for j in range(2):
            idx = i * 2 + j
            U2_n[idx, idx] = 1
            V2_n[idx, idx] = 1
            W2_n[idx, idx] = 1

    # 3. Kronecker product to get 4x4 decomposition (Rank 28)
    U4_kron = np.kron(U2_s, U2_n)  # shape (16, 28)
    V4_kron = np.kron(V2_s, V2_n)  # shape (16, 28)
    W4_kron = np.kron(W2_s, W2_n)  # shape (16, 28)

    # 4. Apply bit-interleaving permutation to reorder rows
    # U row index in kron: (i1*2+j1)*4 + (i2*2+j2) where i=i1*2+i2, j=j1*2+j2
    # Target U row index: i*4+j = (i1*2+i2)*4 + (j1*2+j2)
    
    def compute_perm_u(S):
        i = S // 4
        j = S % 4
        i1 = i // 2
        i2 = i % 2
        j1 = j // 2
        j2 = j % 2
        K = (i1 * 2 + j1) * 4 + (i2 * 2 + j2)
        return K

    def compute_perm_v(S):
        j = S // 4
        k = S % 4
        j1 = j // 2
        j2 = j % 2
        k1 = k // 2
        k2 = k % 2
        K = (j1 * 2 + k1) * 4 + (j2 * 2 + k2)
        return K

    def compute_perm_w(S):
        k = S // 4
        i = S % 4
        k1 = k // 2
        k2 = k % 2
        i1 = i // 2
        i2 = i % 2
        K = (k1 * 2 + i1) * 4 + (k2 * 2 + i2)
        return K

    perm_U = [compute_perm_u(S) for S in range(16)]
    perm_V = [compute_perm_v(S) for S in range(16)]
    perm_W = [compute_perm_w(S) for S in range(16)]

    U4 = U4_kron[perm_U, :]
    V4 = V4_kron[perm_V, :]
    W4 = W4_kron[perm_W, :]

    # 5. Verify the 4x4 decomposition
    for i in range(4):
        for j in range(4):
            for k in range(4):
                expected = 1 if i == k else 0
                actual = 0
                for c in range(28):
                    actual += U4[i * 4 + j, c] * V4[j * 4 + k, c] * W4[k * 4 + i, c]
                if actual != expected:
                    # Fallback to rank-49 (Kron of two Strassen)
                    return _strassen_kron_49(n, m, p)

    if time.time() - start_time > DEADLINE:
        return _strassen_kron_49(n, m, p)

    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }

def _strassen_kron_49(n, m, p):
    """Fallback to rank-49 decomposition."""
    # Re-use the logic from the previous program
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    U2[0, 0] = 1; U2[3, 0] = 1; V2[0, 0] = 1; V2[3, 0] = 1; W2[0, 0] = 1; W2[3, 0] = 1
    U2[2, 1] = 1; U2[3, 1] = 1; V2[0, 1] = 1; W2[1, 1] = 1
    U2[0, 2] = 1; V2[2, 2] = 1; V2[3, 2] = -1; W2[2, 2] = 1
    U2[3, 3] = 1; V2[1, 3] = 1; V2[0, 3] = -1; W2[2, 3] = 1; W2[3, 3] = 1
    U2[0, 4] = 1; U2[1, 4] = 1; V2[3, 4] = 1; W2[0, 4] = -1; W2[2, 4] = 1
    U2[2, 5] = 1; U2[0, 5] = -1; V2[0, 5] = 1; V2[1, 5] = 1; W2[1, 5] = -1
    U2[1, 6] = 1; U2[3, 6] = -1; V2[2, 6] = 1; V2[3, 6] = 1; W2[1, 6] = 1

    U4_kron = np.kron(U2, U2)
    V4_kron = np.kron(V2, V2)
    W4_kron = np.kron(W2, W2)

    def compute_perm_u(S):
        i = S // 4; j = S % 4
        i1 = i // 2; i2 = i % 2; j1 = j // 2; j2 = j % 2
        return (i1 * 2 + j1) * 4 + (i2 * 2 + j2)

    def compute_perm_v(S):
        j = S // 4; k = S % 4
        j1 = j // 2; j2 = j % 2; k1 = k // 2; k2 = k % 2
        return (j1 * 2 + k1) * 4 + (j2 * 2 + k2)

    def compute_perm_w(S):
        k = S // 4; i = S % 4
        k1 = k // 2; k2 = k % 2; i1 = i // 2; i2 = i % 2
        return (k1 * 2 + i1) * 4 + (k2 * 2 + i2)

    perm_U = [compute_perm_u(S) for S in range(16)]
    perm_V = [compute_perm_v(S) for S in range(16)]
    perm_W = [compute_perm_w(S) for S in range(16)]

    U4 = U4_kron[perm_U, :]
    V4 = V4_kron[perm_V, :]
    W4 = W4_kron[perm_W, :]

    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }

def _schoolbook(n, m, p):
    """Return the naive rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
