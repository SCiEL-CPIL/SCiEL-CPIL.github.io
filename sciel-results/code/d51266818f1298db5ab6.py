import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0

def _check_deadline():
    return (time.time() - _start_time) > (_DEADLINE - _SAFETY_MARGIN)

def _verify(U, V, W, n=2, m=3, p=5, tol=1e-9):
    """Verify exact decomposition."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    tensor = np.zeros((n, m, p))
    for r in range(rank):
        u = Ua[:, r].reshape(n, m)
        v = Va[:, r].reshape(m, p)
        w = Wa[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    return np.allclose(tensor, np.ones((n, m, p)), atol=tol)

def _naive(n=2, m=3, p=5):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2(r_offset, u_base, v_base, w_base):
    """
    Add 7 Strassen terms for a 2x2 sub-problem.
    u_base: indices into U for A[0,0],A[0,1],A[1,0],A[1,1]
    v_base: indices into V for B[0,0],B[0,1],B[1,0],B[1,1]
    w_base: indices into W for C[0,0],C[1,0],C[0,1],C[1,1]
    """
    U, V, W = _strassen_2x2.U, _strassen_2x2.V, _strassen_2x2.W
    r = r_offset
    a00, a01, a10, a11 = u_base
    b00, b01, b10, b11 = v_base
    c00, c10, c01, c11 = w_base
    
    # M1 = (A00+A11)(B00+B11): +C00, +C11
    U[a00][r] += 1; U[a11][r] += 1; V[b00][r] += 1; V[b11][r] += 1; W[c00][r] += 1; W[c11][r] += 1; r += 1
    # M2 = (A10+A11)B00: +C10, -C11
    U[a10][r] += 1; U[a11][r] += 1; V[b00][r] += 1; W[c10][r] += 1; W[c11][r] += -1; r += 1
    # M3 = A00(B10-B00): +C01, +C11
    U[a00][r] += 1; V[b10][r] += 1; V[b00][r] += -1; W[c01][r] += 1; W[c11][r] += 1; r += 1
    # M4 = A11(B01-B10): +C00, +C10
    U[a11][r] += 1; V[b01][r] += 1; V[b10][r] += -1; W[c00][r] += 1; W[c10][r] += 1; r += 1
    # M5 = (A00+A01)B11: -C00, +C01
    U[a00][r] += 1; U[a01][r] += 1; V[b11][r] += 1; W[c00][r] += -1; W[c01][r] += 1; r += 1
    # M6 = (A10-A00)(B00+B10): +C11
    U[a10][r] += 1; U[a00][r] += -1; V[b00][r] += 1; V[b10][r] += 1; W[c11][r] += 1; r += 1
    # M7 = (A01-A11)(B01+B11): -C00, -C01
    U[a01][r] += 1; U[a11][r] += -1; V[b01][r] += 1; V[b11][r] += 1; W[c00][r] += -1; W[c01][r] += -1; r += 1
    return r

def construct():
    if _check_deadline():
        n, m, p = DIMENSIONS
        return _naive(n, m, p)
    
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    _strassen_2x2.U = U
    _strassen_2x2.V = V
    _strassen_2x2.W = W
    
    r = 0
    # Part 1: j=2, all k (10 terms)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    # Part 2: j=0,1, k=4 (4 terms)
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
    # Part 3: Strassen for j=0,1 x k=0,1 (7 terms)
    # U indices: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # V indices: B[0,0]=0, B[0,1]=1, B[1,0]=5, B[1,1]=6
    # W indices: C[0,0]=0, C[1,0]=1, C[0,1]=2, C[1,1]=3
    r = _strassen_2x2(r, [0,1,3,4], [0,1,5,6], [0,1,2,3])
    # Part 4: Strassen for j=0,1 x k=2,3 (7 terms)
    # V indices: B[0,2]=2, B[0,3]=3, B[1,2]=7, B[1,3]=8
    # W indices: C[0,2]=4, C[1,2]=5, C[0,3]=6, C[1,3]=7
    r = _strassen_2x2(r, [0,1,3,4], [2,3,7,8], [4,5,6,7])
    
    if not _verify(U, V, W):
        return _naive(n, m, p)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
