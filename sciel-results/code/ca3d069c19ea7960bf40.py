import numpy as np
import time

DIMENSIONS = [4, 4, 4]

_DEADLINE = time.time() + 120.0
_MARGIN = 5.0

def _time_ok():
    return time.time() < _DEADLINE - _MARGIN

def _build_target_tensor():
    n, m, p = 4, 4, 4
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_c U[:,c] outer V[:,c] outer W[:,c] == T"""
    n, m, p = 4, 4, 4
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    rank = U.shape[1]
    
    # Build the tensor from the decomposition
    T_recon = np.zeros((dim_u, dim_v, dim_w))
    for c in range(rank):
        uc = np.array(U[:, c], dtype=float)
        vc = np.array(V[:, c], dtype=float)
        wc = np.array(W[:, c], dtype=float)
        T_recon += np.einsum('i,j,k->ijk', uc, vc, wc)
    
    diff = np.max(np.abs(T_recon - T))
    return diff < tol

def _build_rank49_solution():
    """Build the rank-49 Kronecker product of two 2x2 Strassen decompositions."""
    n, m, p = 4, 4, 4
    
    # 2x2 Strassen decomposition (rank 7)
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]

    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1

    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1

    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1

    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1

    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1

    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1

    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1

    rank = 49
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    U2_sparse = [[(c, v) for c, v in enumerate(row) if v != 0] for row in U2]
    V2_sparse = [[(c, v) for c, v in enumerate(row) if v != 0] for row in V2]
    W2_sparse = [[(c, v) for c, v in enumerate(row) if v != 0] for row in W2]

    for i in range(n):
        for j in range(m):
            i1 = i // 2
            i2 = i % 2
            j1 = j // 2
            j2 = j % 2
            row_u2_1 = i1 * 2 + j1
            row_u2_2 = i2 * 2 + j2
            row_u = U[i*m + j]
            for c1, v1 in U2_sparse[row_u2_1]:
                for c2, v2 in U2_sparse[row_u2_2]:
                    c = c1 * 7 + c2
                    row_u[c] = v1 * v2

    for j in range(m):
        for k in range(p):
            j1 = j // 2
            j2 = j % 2
            k1 = k // 2
            k2 = k % 2
            row_v2_1 = j1 * 2 + k1
            row_v2_2 = j2 * 2 + k2
            row_v = V[j*p + k]
            for c1, v1 in V2_sparse[row_v2_1]:
                for c2, v2 in V2_sparse[row_v2_2]:
                    c = c1 * 7 + c2
                    row_v[c] = v1 * v2

    for k in range(p):
        for i in range(n):
            k1 = k // 2
            k2 = k % 2
            i1 = i // 2
            i2 = i % 2
            row_w2_1 = k1 * 2 + i1
            row_w2_2 = k2 * 2 + i2
            row_w = W[k*n + i]
            for c1, v1 in W2_sparse[row_w2_1]:
                for c2, v2 in W2_sparse[row_w2_2]:
                    c = c1 * 7 + c2
                    row_w[c] = v1 * v2

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Build the known valid rank-49 solution
    U49, V49, W49 = _build_rank49_solution()
    
    # Verify the solution
    if _time_ok():
        T = _build_target_tensor()
        if _verify_decomposition(U49, V49, W49, T, tol=1e-9):
            # Solution is valid, return it
            return {"U": U49, "V": V49, "W": W49}
        else:
            # Solution is invalid, but still return it as fallback
            # (this should not happen)
            pass
    
    # Return the rank-49 solution
    return {"U": U49, "V": V49, "W": W49}
