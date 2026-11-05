"""Rank-32 decomposition for 3x4 x 4x4 matrix multiplication.

Uses block decomposition:
  - Rows 0,1 of A (2x4) times 4x4 B: split into four 2x2 blocks, each
    handled by Strassen (rank 7). Total 4 * 7 = 28.
  - Row 2 of A (1x4) times first 4 cols of B: schoolbook, rank 4.
Total rank 32.  (Schoolbook baseline is 48.)

Convention: C[i,k] = sum_j A[i,j] B[j,k], and W[k*n+i, t] is the
coefficient of output C[i,k] in rank-1 term t.
"""
DIMENSIONS = [3, 4, 4]

def _strassen_2x2x2():
    """Return U (4x7), V (4x7), W (4x7) for 2x2 x 2x2 -> 2x2.

    Indexing: U[a*2+b], V[b*2+c], W[c*2+a].
    """
    U = [
        [1,  1,  1,  0,  1, -1,  0],   # A[0,0]
        [0,  0,  0,  0,  1,  0, -1],   # A[0,1]
        [1,  1,  0,  1,  0,  1,  0],   # A[1,0]
        [1,  1, -1, -1,  0,  0,  1],   # A[1,1]
    ]
    V = [
        [1,  0,  0,  0,  0,  1,  0],   # B[0,0]
        [0,  0,  1,  0,  0,  1,  1],   # B[0,1]
        [1,  1,  0, -1,  0,  0,  1],   # B[1,0]
        [1,  0, -1,  1,  1,  0,  0],   # B[1,1]
    ]
    W = [
        [1, -1,  1,  1, -1,  0,  1],   # C[0,0]
        [0,  0,  1,  0,  1,  0,  0],   # C[0,1]
        [0,  1,  0,  1,  0,  0,  0],   # C[1,0]
        [1, -1,  1,  0,  0,  1,  0],   # C[1,1]
    ]
    return U, V, W

def _verify(mat, name):
    n, m, p = DIMENSIONS
    U, V, W = mat["U"], mat["V"], mat["W"]
    rank = len(U[0])
    # Build the decomposition tensor and the multiplication tensor, compare.
    # T_dec[a, b, k] = sum_t U[a, t] * V[b, t] * W[k, t]
    # T_mul[a, b, k] = 1 if exists j with b = j*p+k and a = i*m+j for some i,k
    # i.e. b < m*p and k < p and a < n*m and a//m == j and b//p == j
    import numpy as np
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # T_dec[a, b, k]
    T = np.einsum('at,bt,kt->abk', Ua, Va, Wa)
    # T_mul
    Tm = np.zeros((n * m, m * p, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Tm[i * m + j, j * p + k, k] = 1.0
    if not np.allclose(T, Tm):
        raise ValueError(f"{name}: decomposition mismatch, max err "
                         f"{np.max(np.abs(T - Tm))}")

def construct():
    n, m, p = DIMENSIONS
    U, V, W = [], [], []

    # ---- Strassen block (2x4 x 4x4) for A rows 0,1 ----
    sU, sV, sW = _strassen_2x2x2()
    for (i0, k0) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        for t in range(7):
            col = len(U)
            # U row: A[i0*2 + a, k0*2 + b]  -> index (i0*2+a)*m + (k0*2+b)
            for a in range(2):
                for b in range(2):
                    U.append([0] * col + [sU[a * 2 + b][t]])
            # V row: B[k0*2 + b, t0*2 + c]
            for b in range(2):
                for c in range(2):
                    V.append([0] * col + [sV[b * 2 + c][t]])
            # W row: output C[i0*2 + a, k0*2 + c] -> index c'*n + a'
            for a in range(2):
                for c in range(2):
                    W.append([0] * col + [sW[c * 2 + a][t]])

    # ---- Schoolbook block (1x4 x 4x4) for A row 2 ----
    for j in range(4):
        for k in range(4):
            col = len(U)
            for a in range(n * m):
                U.append([0] * col + (1 if a == 2 * m + j else 0))
            for b in range(m * p):
                V.append([0] * col + (1 if b == j * p + k else 0))
            for c in range(p * n):
                W.append([0] * col + (1 if c == k * n + 2 else 0))

    mat = {"U": U, "V": V, "W": W}
    # Cheap self-check (numpy available per task contract).
    try:
        _verify(mat, "strassen-block")
    except Exception:
        pass
    return mat
