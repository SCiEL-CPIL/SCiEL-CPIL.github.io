import time
import numpy as np

_N, _M, _P = 2, 3, 10
_HARD_LIMIT = 140.0
_BUILD_BUDGET = 2.0
_T0 = time.monotonic()

# Strassen rank-7 kernel for 2x2 * 2x2
# Each term is (u_coeffs, v_coeffs, w_coeffs)
# u_coeffs: dict {(i,j): val}
# v_coeffs: dict {(j,k): val}
# w_coeffs: dict {(i,k): val}

_STRASSEN_TERMS = [
    # M1 = (A00+A11)(B00+B11)
    ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
    # M2 = (A10+A11)B00
    ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
    # M3 = A00(B01-B11)
    ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
    # M4 = A11(B10-B00)
    ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
    # M5 = (A00+A01)B11
    ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
    # M6 = (A10-A00)(B00+B01)
    ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
    # M7 = (A01-A11)(B10+B11)
    ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
]

def _build_rank55():
    rank = 55
    U = np.zeros((_N * _M, rank))
    V = np.zeros((_M * _P, rank))
    W = np.zeros((_P * _N, rank))
    
    col = 0
    # 5 blocks of 2 columns
    for g in range(_P // 2):
        k0 = 2 * g
        k1 = k0 + 1
        for u_dict, v_dict, w_dict in _STRASSEN_TERMS:
            # Fill U
            for (i, j), val in u_dict.items():
                U[i * _M + j, col] = val
            # Fill V
            for (j, k_local), val in v_dict.items():
                k_global = k0 + k_local
                V[j * _P + k_global, col] = val
            # Fill W
            for (i, k_local), val in w_dict.items():
                k_global = k0 + k_local
                W[k_global * _N + i, col] = val
            col += 1
            
    # 20 outer product terms for j=2
    for i in range(_N):
        for k in range(_P):
            U[i * _M + 2, col] = 1.0
            V[2 * _P + k, col] = 1.0
            W[k * _N + i, col] = 1.0
            col += 1
            
    assert col == rank
    return U, V, W

def _verify(U, V, W):
    n, m, p = _N, _M, _P
    r = U.shape[1]
    C = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            total = 0.0
            for j in range(m):
                total += np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
            C[i, k] = total
    return np.allclose(C, np.ones((n, p)), atol=1e-9)

def construct():
    if time.monotonic() - _T0 > _HARD_LIMIT - _BUILD_BUDGET:
        return {"U": np.zeros((_N * _M, 1)).tolist(), "V": np.zeros((_M * _P, 1)).tolist(), "W": np.zeros((_P * _N, 1)).tolist()}
        
    U, V, W = _build_rank55()
    
    if not _verify(U, V, W):
        # Fallback: Standard rank 60
        rank = _N * _M * _P
        U = np.zeros((_N * _M, rank))
        V = np.zeros((_M * _P, rank))
        W = np.zeros((_P * _N, rank))
        idx = 0
        for i in range(_N):
            for j in range(_M):
                for k in range(_P):
                    U[i * _M + j, idx] = 1.0
                    V[j * _P + k, idx] = 1.0
                    W[k * _N + i, idx] = 1.0
                    idx += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
