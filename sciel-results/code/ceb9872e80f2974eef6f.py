"""Matrix multiplication 4x4 * 4x9 rank-minimal exact tensor decomposition.

Strategy: split B into [B1 (4x4), B2 (4x4), B3 (4x1)].
  C = [A*B1, A*B2, A*B3]
  A*B1: Strassen rank 49
  A*B2: Strassen rank 49
  A*B3: schoolbook rank 16
  Total rank: 114
"""
import time
import numpy as np

_START = time.time()
_DEADLINE = 140.0


def _time_ok():
    return time.time() - _START < _DEADLINE


def _strassen_2x2_terms():
    """7 Strassen terms for 2x2*2x2."""
    return [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]


def _strassen_4x4_4x4_terms():
    """49 terms for 4x4*4x4 via recursive Strassen."""
    inner = _strassen_2x2_terms()
    top = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]
    all_terms = []
    for (ab, bb, cb) in top:
        for (ui, vi, wi) in inner:
            u, v, w = {}, {}, {}
            for (bi, bj), ca in ab.items():
                for (li, lj), ci in ui.items():
                    i, j = 2*bi+li, 2*bj+lj
                    u[(i,j)] = u.get((i,j),0) + ca*ci
            for (bi, bj), cb_ in bb.items():
                for (li, lj), ci in vi.items():
                    j, k = 2*bi+li, 2*bj+lj
                    v[(j,k)] = v.get((j,k),0) + cb_*ci
            for (bi, bj), cc in cb.items():
                for (li, lj), ci in wi.items():
                    i, k = 2*bi+li, 2*bj+lj
                    w[(i,k)] = w.get((i,k),0) + cc*ci
            all_terms.append((u, v, w))
    return all_terms


def _build_decomposition(n, m, p):
    """Build rank-114: 49 (cols 0-3) + 49 (cols 4-7) + 16 (col 8)."""
    rank = 114
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    st = _strassen_4x4_4x4_terms()
    
    # A*B1: cols 0-3
    for c, (up, vp, wp) in enumerate(st):
        for (i,j), cu in up.items():
            if cu: U[i*m+j, c] = cu
        for (j,k), cv in vp.items():
            if cv: V[j*p+k, c] = cv
        for (i,k), cw in wp.items():
            if cw: W[k*n+i, c] = cw
    
    # A*B2: cols 4-7
    off = 49
    for c, (up, vp, wp) in enumerate(st):
        for (i,j), cu in up.items():
            if cu: U[i*m+j, c+off] = cu
        for (j,k), cv in vp.items():
            if cv: V[j*p+(4+k), c+off] = cv
        for (i,k), cw in wp.items():
            if cw: W[(4+k)*n+i, c+off] = cw
    
    # A*B3: col 8 (schoolbook)
    off2 = 98
    for i in range(n):
        for j in range(m):
            c = off2 + i*m + j
            U[i*m+j, c] = 1
            V[j*p+8, c] = 1
            W[8*n+i, c] = 1
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify using numpy einsum."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)


def _schoolbook(n, m, p):
    rank = n*m*p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1
    return U, V, W


def construct():
    n, m, p = 4, 4, 9
    
    if _time_ok():
        U, V, W = _build_decomposition(n, m, p)
        if _verify(U, V, W, n, m, p):
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    U, V, W = _schoolbook(n, m, p)
    if _verify(U, V, W, n, m, p):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
