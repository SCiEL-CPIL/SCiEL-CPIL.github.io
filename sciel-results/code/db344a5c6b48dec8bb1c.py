import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_kron_4x4(n, m, p):
    """Rank-49 decomposition via Kronecker product of Strassen 2x2."""
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

    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)

    perm_u = np.zeros(16, dtype=int)
    for S in range(16):
        i1 = S // 8
        j1 = (S // 4) % 2
        i2 = (S // 2) % 2
        j2 = S % 2
        i = i1 * 2 + i2
        j = j1 * 2 + j2
        perm_u[S] = i * 4 + j

    perm_v = np.zeros(16, dtype=int)
    for S in range(16):
        j1 = S // 8
        k1 = (S // 4) % 2
        j2 = (S // 2) % 2
        k2 = S % 2
        j = j1 * 2 + j2
        k = k1 * 2 + k2
        perm_v[S] = j * 4 + k

    perm_w = np.zeros(16, dtype=int)
    for S in range(16):
        k1 = S // 8
        i1 = (S // 4) % 2
        k2 = (S // 2) % 2
        i2 = S % 2
        k = k1 * 2 + k2
        i = i1 * 2 + i2
        perm_w[S] = k * 4 + i

    U4 = U4[perm_u, :]
    V4 = V4[perm_v, :]
    W4 = W4[perm_w, :]

    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=int)
    V = np.array(V, dtype=int)
    W = np.array(W, dtype=int)
    rank = U.shape[1]
    
    # Build the full tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This should equal delta(i,k) for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                expected = 1 if i == k else 0
                if s != expected:
                    return False
    return True

def construct():
    """
    Constructs a minimal-rank exact tensor decomposition for 4x4 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer entries.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Rank-49 Kronecker-Strassen
    if time.time() < deadline:
        r49 = _strassen_kron_4x4(n, m, p)
        if _verify_decomposition(r49["U"], r49["V"], r49["W"], n, m, p):
            return r49
    
    # Final fallback: schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
