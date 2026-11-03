import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def _target_tensor(n, m, p):
    nm = n * m
    mp = m * p
    pn = p * n
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    T_target = _target_tensor(n, m, p)
    return np.allclose(T_recon, T_target, atol=tol)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _to_list(M):
    return M.tolist()

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    
    # Verify to be safe
    if not _verify_decomposition(U_sb, V_sb, W_sb, n, m, p):
        # This should never happen
        raise ValueError("Schoolbook decomposition failed verification")

    return {"U": _to_list(U_sb), "V": _to_list(V_sb), "W": _to_list(W_sb)}
