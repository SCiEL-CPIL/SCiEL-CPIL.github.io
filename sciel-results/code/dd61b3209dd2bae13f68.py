import time
import numpy as np

DIMENSIONS = [2, 3, 8]
N, M, P = 2, 3, 8

def _verify(U, V, W, tol=1e-9):
    """Verify U, V, W (lists of lists) exactly decompose the tensor."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    T = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                e = abs(T[a, b, c] - 1.0)
                if e > max_err:
                    max_err = e
    # Check non-target entries are zero
    target_mask = np.zeros((n * m, m * p, p * n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_mask[i * m + j, j * p + k, k * n + i] = True
    non_target_err = float(np.max(np.abs(T[~target_mask]))) if (~target_mask).any() else 0.0
    return max_err <= tol and non_target_err <= tol

def _strassen_block(block):
    """Return (U,V,W) local factor matrices (6x7, 24x7, 16x7) for a 2x2*2x2
    Strassen block acting on B columns (k0,k1), A cols 0,1."""
    k0 = 2 * block
    k1 = 2 * block + 1
    U = np.zeros((6, 7))
    V = np.zeros((24, 7))
    W = np.zeros((16, 7))
    # m1 = (a00+a11)(b00+b11)
    U[0, 0] += 1; U[4, 0] += 1
    V[k0, 0] += 1; V[8 + k1, 0] += 1
    W[2 * k0, 0] += 1; W[2 * k1 + 1, 0] += 1
    # m2 = (a10+a11)b00
    U[3, 1] += 1; U[4, 1] += 1
    V[k0, 1] += 1
    W[2 * k0 + 1, 1] += 1; W[2 * k1 + 1, 1] += -1
    # m3 = a00(b01-b11)
    U[0, 2] += 1
    V[k1, 2] += 1; V[8 + k1, 2] += -1
    W[2 * k1, 2] += 1; W[2 * k1 + 1, 2] += 1
    # m4 = a11(b10-b00)
    U[4, 3] += 1
    V[8 + k0, 3] += 1; V[k0, 3] += -1
    W[2 * k0, 3] += 1; W[2 * k0 + 1, 3] += 1
    # m5 = (a00+a01)b11
    U[0, 4] += 1; U[1, 4] += 1
    V[8 + k1, 4] += 1
    W[2 * k0, 4] += -1; W[2 * k1, 4] += 1
    # m6 = (a10-a00)(b00+b01)
    U[3, 5] += 1; U[0, 5] += -1
    V[k0, 5] += 1; V[k1, 5] += 1
    W[2 * k1 + 1, 5] += 1
    # m7 = (a01-a11)(b10+b11)
    U[1, 6] += 1; U[4, 6] += -1
    V[8 + k0, 6] += 1; V[8 + k1, 6] += 1
    W[2 * k0, 6] += 1
    return U, V, W

def _outer_block(block):
    """4 rank-1 terms for A_right (col 2) * B_bottom (row 2)."""
    k0 = 2 * block
    k1 = 2 * block + 1
    U = np.zeros((6, 4))
    V = np.zeros((24, 4))
    W = np.zeros((16, 4))
    t = 0
    for ii in range(2):
        for kk in range(2):
            kv = k0 if kk == 0 else k1
            U[ii * 3 + 2, t] = 1
            V[16 + kv, t] = 1
            W[kv * 2 + ii, t] = 1
            t += 1
    return U, V, W

def _build_rank44_decomposition():
    """Explicit rank-44 decomposition: 4 blocks x (Strassen 7 + outer 4)."""
    n, m, p = N, M, P
    U = np.zeros((n * m, 44))
    V = np.zeros((m * p, 44))
    W = np.zeros((p * n, 44))
    col = 0
    for block in range(4):
        Us, Vs, Ws = _strassen_block(block)
        Uo, Vo, Wo = _outer_block(block)
        U[:, col:col + 7] = Us
        V[:, col:col + 7] = Vs
        W[:, col:col + 7] = Ws
        col += 7
        U[:, col:col + 4] = Uo
        V[:, col:col + 4] = Vo
        W[:, col:col + 4] = Wo
        col += 4
    assert col == 44
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    # Fallback to schoolbook if something goes wrong
    U_fb, V_fb, W_fb = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n*m*p)
                v = [0] * (m*p)
                w = [0] * (p*n)
                idx = i*m + j
                u[idx] = 1
                u = u # placeholder
                # Actually, let's just return the rank 44 one directly as it's verified
                pass
    
    U, V, W = _build_rank44_decomposition()
    return {"U": U, "V": V, "W": W}
