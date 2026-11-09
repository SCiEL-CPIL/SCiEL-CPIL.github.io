import time
import numpy as np

DIMENSIONS = [4, 4, 4]
TIME_LIMIT = 145

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """
    2x2 Strassen decomposition, rank 7.
    
    U2: shape (4,7), index row*2+col. U2[idx, a] = coeff of A element in product a.
    V2: shape (4,7), index row*2+col. V2[idx, a] = coeff of B element in product a.
    W2: shape (4,7), index col*2+row. W2[idx, a] = coeff for C element from product a.
    
    Products:
      P0 = (A00+A11)(B00+B11)
      P1 = (A10+A11)(B00)
      P2 = (A00)(B01-B11)
      P3 = (A11)(B10-B00)
      P4 = (A00+A01)(B11)
      P5 = (A10-A00)(B00+B10)
      P6 = (A01-A11)(B01+B11)
    
    C00 = P0 - P3 + P5 + P6
    C10 = P1 + P3
    C01 = P2 + P4
    C11 = P0 - P2 + P1 + P4
    """
    U2 = np.array([
        [1, 0, 1, 0, 1, 0, 0],   # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [1, 1, 0, 0, 0, -1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1],  # A11
    ], dtype=np.int64)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 0, 1],   # B01
        [0, 0, 0, 1, 0, 1, 0],   # B10
        [1, 0, -1, 0, 1, 0, 1],  # B11
    ], dtype=np.int64)
    
    W2 = np.array([
        [1, 0, 0, -1, 0, 1, 1],  # C00 (col=0,row=0)
        [0, 1, 0, 1, 0, 0, 0],   # C10 (col=0,row=1)
        [0, 0, 1, 0, 1, 0, 0],   # C01 (col=1,row=0)
        [1, 1, -1, 0, 1, 0, 0],  # C11 (col=1,row=1)
    ], dtype=np.int64)
    
    return U2, V2, W2

def _build_4x4():
    """
    Rank-49 decomposition for 4x4 matmul via tensor product of Strassen.
    
    U: (16,49) idx=i*4+j
    V: (16,49) idx=j*4+k  
    W: (16,49) idx=k*4+i
    rank_idx = ao*7 + ai
    """
    U2, V2, W2 = _strassen_2x2()
    
    U = np.zeros((16, 49), dtype=np.int64)
    V = np.zeros((16, 49), dtype=np.int64)
    W = np.zeros((16, 49), dtype=np.int64)
    
    for I in range(2):
        for J in range(2):
            for K in range(2):
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            ai_ = (2*I + i)*4 + (2*J + j)
                            bi_ = (2*J + j)*4 + (2*K + k)
                            ci_ = (2*K + k)*4 + (2*I + i)
                            
                            for ao in range(7):
                                ub = U2[I*2+J, ao]
                                vb = V2[J*2+K, ao]
                                wb = W2[K*2+I, ao]
                                if ub == 0 and vb == 0 and wb == 0:
                                    continue
                                for ai in range(7):
                                    ui = U2[i*2+j, ai]
                                    vi = V2[j*2+k, ai]
                                    wi = W2[k*2+i, ai]
                                    if ui == 0 and vi == 0 and wi == 0:
                                        continue
                                    r = ao*7 + ai
                                    if ub != 0 and ui != 0:
                                        U[ai_, r] = ub * ui
                                    if vb != 0 and vi != 0:
                                        V[bi_, r] = vb * vi
                                    if wb != 0 and wi != 0:
                                        W[ci_, r] = wb * wi
    
    # Prune
    keep = [r for r in range(49)
            if np.any(U[:,r]) and np.any(V[:,r]) and np.any(W[:,r])]
    if len(keep) < 49:
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
    
    return U.tolist(), V.tolist(), W.tolist()

def _verify(U, V, W):
    n = m = p = 4
    rank = len(U[0])
    T = np.zeros((n, m, p), dtype=np.int64)
    for r in range(rank):
        ur = np.array([U[i][r] for i in range(16)], dtype=np.int64)
        vr = np.array([V[i][r] for i in range(16)], dtype=np.int64)
        wr = np.array([W[i][r] for i in range(16)], dtype=np.int64)
        for i in range(n):
            for j in range(m):
                uval = ur[i*m + j]
                if uval == 0:
                    continue
                for k in range(p):
                    T[i,j,k] += uval * vr[j*p+k] * wr[k*n+i]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[i,j,k] != (1 if i==k else 0):
                    return False
    return True

def construct():
    t0 = time.time()
    U, V, W = _build_4x4()
    if not _verify(U, V, W):
        raise ValueError("Verification failed")
    if time.time() - t0 > TIME_LIMIT - 5:
        pass
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
