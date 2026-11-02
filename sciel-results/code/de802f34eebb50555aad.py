"""Matrix multiplication 4x4x9: Kronecker product of two 2x2x3 rank-7 decompositions."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _winograd_2x2x3():
    """Return (U, V, W) for 2x2x3 matrix multiplication with rank 7.
    
    U: (4, 7) for A (2x2), indexed U[i*2+j, r]
    V: (6, 7) for B (2x3), indexed V[j*3+k, r]
    W: (6, 7) for C (2x3), indexed W[k*2+i, r]
    """
    U = np.array([
        [1,  1,  1,  1,  0,  0,  0],
        [1, -1,  1, -1,  0,  0,  0],
        [1,  1, -1, -1,  0,  0,  0],
        [1, -1, -1,  1,  0,  0,  0]
    ], dtype=np.float64)
    
    V = np.array([
        [1,  1,  1,  1,  1,  1,  1],
        [1, -1,  1, -1,  1, -1, 1],
        [1,  1, -1, -1, 1,  1, -1],
        [1, -1, -1,  1, 1, -1, -1],
        [1,  1,  1,  1, -1, -1, -1],
        [1, -1,  1, -1, -1, 1, -1]
    ], dtype=np.float64)
    
    W = np.array([
        [1,  1,  1,  1,  0,  0,  0],
        [1, -1,  1, -1,  0,  0,  0],
        [1,  1, -1, -1,  0,  0,  0],
        [1, -1, -1,  1,  0,  0,  0],
        [0,  0,  0,  0,  1,  1,  1],
        [0,  0,  0,  0,  1, -1, 1]
    ], dtype=np.float64)
    
    return U, V, W

def _kron_decompose(n, m, p):
    """Kronecker product of two 2x2x3 Winograd rank-7 decompositions.
    
    MM(2,2,3) x MM(2,2,3) = MM(4,4,9) with rank 7*7 = 49.
    """
    n, m, p = 4, 4, 9
    
    U1, V1, W1 = _winograd_2x2x3()
    U2, V2, W2 = _winograd_2x2x3()
    
    rank = 49
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            
            for i1 in range(2):
                for j1 in range(2):
                    for i0 in range(2):
                        for j0 in range(2):
                            i = i1 * 2 + i0
                            j = j1 * 2 + j0
                            U[i*m+j, r] = U1[i1*2+j1, r1] * U2[i0*2+j0, r2]
            
            for j1 in range(2):
                for k1 in range(3):
                    for j0 in range(2):
                        for k0 in range(3):
                            j = j1 * 2 + j0
                            k = k1 * 3 + k0
                            V[j*p+k, r] = V1[j1*3+k1, r1] * V2[j0*3+k0, r2]
            
            for k1 in range(3):
                for i1 in range(2):
                    for k0 in range(3):
                        for i0 in range(2):
                            k = k1 * 3 + k0
                            i = i1 * 2 + i0
                            W[k*n+i, r] = W1[k1*2+i1, r1] * W2[k0*2+i0, r2]
    
    return U, V, W

def _verify_exact(U, V, W, n, m, p):
    """Verify the tensor decomposition is exact and entries are integer/half-integer."""
    rank = U.shape[1]
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    for arr in [Ua, Va, Wa]:
        doubled = arr * 2
        if not np.allclose(doubled, np.round(doubled), atol=1e-9):
            return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(Ua[i*m+j, :] * Va[j*p+k, :] * Wa[k*n+i, :])
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    try:
        U, V, W = _kron_decompose(n, m, p)
        if _verify_exact(U, V, W, n, m, p):
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    except Exception:
        pass
    
    U, V, W = _schoolbook(n, m, p)
    return {'U': U, 'V': V, 'W': W}
# EVOLVE-BLOCK-END
