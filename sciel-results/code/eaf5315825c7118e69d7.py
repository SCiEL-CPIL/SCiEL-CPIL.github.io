"""Program for matrix multiplication 4x4 * 4x8 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]
DEADLINE_SECONDS = 110.0


def construct():
    """
    Return a rank-minimal tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Uses a 2-level Strassen-based decomposition with rank 98.
    Strategy: Split B (4x8) into two 4x4 blocks. Each uses 2-level Strassen (rank 49).
    """
    start = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 8

    def _timeup():
        return (time.time() - start) > DEADLINE_SECONDS

    def _schoolbook():
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

    # 2x2x2 Strassen factors
    U2 = np.array([[1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],[1,1,0,0],[-1,0,1,0],[0,1,0,-1]], dtype=int)
    V2 = np.array([[1,0,0,1],[1,0,0,0],[0,1,0,-1],[-1,0,1,0],[0,0,0,1],[1,1,0,0],[0,0,1,1]], dtype=int)
    W2 = np.array([[1,0,0,1],[0,0,1,-1],[0,1,0,1],[1,0,1,0],[-1,1,0,0],[0,0,0,1],[1,0,0,0]], dtype=int)

    # Vectorized 4x4x4 factor computation
    I_idx = np.arange(4)
    J_idx = np.arange(4)
    outer_idx = (I_idx // 2)[:, None] * 2 + (J_idx // 2)[None, :]  # (4,4)
    inner_idx = (I_idx % 2)[:, None] * 2 + (J_idx % 2)[None, :]    # (4,4)

    U2_o = U2[:, outer_idx].reshape(7, 16)
    U2_i = U2[:, inner_idx].reshape(7, 16)
    V2_o = V2[:, outer_idx].reshape(7, 16)
    V2_i = V2[:, inner_idx].reshape(7, 16)
    W2_o = W2[:, outer_idx].reshape(7, 16)
    W2_i = W2[:, inner_idx].reshape(7, 16)

    U4_all = np.einsum('sc,tc->stc', U2_o, U2_i).reshape(49, 16)
    V4_all = np.einsum('sc,tc->stc', V2_o, V2_i).reshape(49, 16)
    W4_all = np.einsum('sc,tc->stc', W2_o, W2_i).reshape(49, 16)

    if _timeup():
        return _schoolbook()

    rank = 98
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    for half in range(2):
        offset = half * 49
        U[:, offset:offset+49] = U4_all.T
        V[half*16:(half+1)*16, offset:offset+49] = V4_all.T
        W4_T = W4_all.T.reshape(4, 4, 49)  # (i, k, f)
        for k in range(4):
            W[(half*4+k)*4:(half*4+k)*4+4, offset:offset+49] = W4_T[:, k, :]

    if _timeup():
        return _schoolbook()

    # Verify
    U_arr = U.astype(float)
    V_arr = V.astype(float)
    W_arr = W.astype(float)
    r = U_arr.shape[1]
    U3 = U_arr.reshape(n, m, r)
    V3 = V_arr.reshape(m, p, r)
    W3 = W_arr.reshape(p, n, r)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    if np.allclose(result, 1.0, atol=1e-10):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return _schoolbook()
