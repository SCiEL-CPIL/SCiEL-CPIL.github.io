import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 120.0

def _time_remaining():
    return _DEADLINE - time.time()

_TRUE_TENSOR = None

def _get_true_tensor():
    global _TRUE_TENSOR
    if _TRUE_TENSOR is None:
        n, m, p = 2, 2, 7
        T = np.zeros((n*m, m*p, p*n))
        for a in range(n*m):
            i = a // m
            j = a % m
            for b in range(m*p):
                jb = b // p
                k = b % p
                if j == jb:
                    c = k * n + i
                    T[a, b, c] = 1.0
        _TRUE_TENSOR = T
    return _TRUE_TENSOR

def _verify_decomposition(U, V, W):
    """Verify tensor identity for 2x2 * 2x7 matrix multiplication."""
    T_true = _get_true_tensor()
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_true, T_decomp, atol=1e-10)

def _build_rank25():
    """Build the known-good rank-25 decomposition."""
    n, m, p = 2, 2, 7
    rank = 25
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_strassen(col_start, k_start):
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1
        V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1
        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1
        V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        W[2*(k_start+1) + 1, col_start+2] += 1
        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1
        V[7*0 + k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1
        W[2*k_start + 1, col_start+3] += 1
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1
        V[7*0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1
        V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1
    
    add_strassen(0, 0)
    add_strassen(7, 2)
    add_strassen(14, 4)
    
    col_start = 21
    U[0, col_start+0] += 1
    V[7*0 + 6, col_start+0] += 1
    W[2*6 + 0, col_start+0] += 1
    U[1, col_start+1] += 1
    V[7*1 + 6, col_start+1] += 1
    W[2*6 + 0, col_start+1] += 1
    U[2, col_start+2] += 1
    V[7*0 + 6, col_start+2] += 1
    W[2*6 + 1, col_start+2] += 1
    U[3, col_start+3] += 1
    V[7*1 + 6, col_start+3] += 1
    W[2*6 + 1, col_start+3] += 1
    
    return U, V, W

def _try_rank24():
    """
    Attempt rank-24: share one term between the cols 4-5 Strassen block and col 6.
    Modify M2 and M5 of the cols 4-5 Strassen to also cover col 6,
    then only need 2 extra terms for col 6 instead of 4.
    """
    n, m, p = 2, 2, 7
    rank = 24
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # First 14 terms: standard Strassen for cols 0-1 and cols 2-3
    def add_strassen(col_start, k_start):
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1
        V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1
        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1
        V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        W[2*(k_start+1) + 1, col_start+2] += 1
        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1
        V[7*0 + k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1
        W[2*k_start + 1, col_start+3] += 1
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1
        V[7*0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1
        V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1
    
    add_strassen(0, 0)
    add_strassen(7, 2)
    
    # Terms 14-23: modified Strassen for cols 4,5,6 (10 terms)
    cs = 14
    k4, k5, k6 = 4, 5, 6
    
    # M1 = (A00+A11)*(B0,4+B1,5)  -- standard
    U[0, cs+0] += 1
    U[3, cs+0] += 1
    V[7*0+k4, cs+0] += 1
    V[7*1+k5, cs+0] += 1
    W[2*k4+0, cs+0] += 1
    W[2*k5+1, cs+0] += 1
    
    # M2 = (A10+A11)*(B0,4+B0,6)  -- extended to col 6
    U[2, cs+1] += 1
    U[3, cs+1] += 1
    V[7*0+k4, cs+1] += 1
    V[7*0+k6, cs+1] += 1
    W[2*k4+1, cs+1] += 1
    W[2*k5+1, cs+1] -= 1
    W[2*k6+1, cs+1] += 1
    
    # M3 = A00*(B0,5-B1,5)  -- standard
    U[0, cs+2] += 1
    V[7*0+k5, cs+2] += 1
    V[7*1+k5, cs+2] -= 1
    W[2*k5+0, cs+2] += 1
    W[2*k5+1, cs+2] += 1
    
    # M4 = A11*(B1,4-B0,4)  -- standard
    U[3, cs+3] += 1
    V[7*1+k4, cs+3] += 1
    V[7*0+k4, cs+3] -= 1
    W[2*k4+0, cs+3] += 1
    W[2*k4+1, cs+3] += 1
    
    # M5 = (A00+A01)*(B1,5+B1,6)  -- extended to col 6
    U[0, cs+4] += 1
    U[1, cs+4] += 1
    V[7*1+k5, cs+4] += 1
    V[7*1+k6, cs+4] += 1
    W[2*k4+0, cs+4] -= 1
    W[2*k5+0, cs+4] += 1
    W[2*k6+0, cs+4] += 1
    
    # M6 = (A10-A00)*(B0,4+B0,5)  -- standard
    U[2, cs+5] += 1
    U[0, cs+5] -= 1
    V[7*0+k4, cs+5] += 1
    V[7*0+k5, cs+5] += 1
    W[2*k5+1, cs+5] += 1
    
    # M7 = (A01-A11)*(B1,4+B1,5)  -- standard
    U[1, cs+6] += 1
    U[3, cs+6] -= 1
    V[7*1+k4, cs+6] += 1
    V[7*1+k5, cs+6] += 1
    W[2*k4+0, cs+6] += 1
    
    # M8 = A01*B1,6  -- covers remaining C[0,6] piece
    U[1, cs+7] += 1
    V[7*1+k6, cs+7] += 1
    W[2*k6+0, cs+7] += 1
    
    # M9 = A10*B0,6  -- covers remaining C[1,6] piece
    U[2, cs+8] += 1
    V[7*0+k6, cs+8] += 1
    W[2*k6+1, cs+8] += 1
    
    # M10: check if needed - verify will tell us
    # Leave cs+9 unused for now, verify if rank 23 works
    
    return U, V, W

def construct():
    """
    Construct a minimal-rank exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    Tries rank-24 first, falls back to rank-25.
    """
    if _time_remaining() < 5.0:
        return _schoolbook_decomposition()
    
    # Try rank-24
    try:
        U, V, W = _try_rank24()
        if _verify_decomposition(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fall back to rank-25
    try:
        U, V, W = _build_rank25()
        if _verify_decomposition(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    return _schoolbook_decomposition()

def _schoolbook_decomposition():
    """Fallback rank-28 schoolbook decomposition."""
    n, m, p = 2, 2, 7
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1
                V[j*p + k, idx] = 1
                W[k*n + i, idx] = 1
                idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    A = np.random.rand(2, 2)
    B = np.random.rand(2, 7)
    C_expected = A @ B
    
    C_W = np.zeros(14)
    for r in range(U.shape[1]):
        SA = np.dot(U[:, r], A.flatten())
        SB = np.dot(V[:, r], B.flatten())
        for idx in range(14):
            C_W[idx] += W[idx, r] * SA * SB
            
    C_mat = np.zeros((2,7))
    for k in range(7):
        for i in range(2):
            C_mat[i, k] = C_W[2*k+i]
            
    if np.allclose(C_mat, C_expected):
        print("Verification PASSED, rank =", U.shape[1])
    else:
        print("Verification FAILED")
        print(C_mat)
        print(C_expected)
