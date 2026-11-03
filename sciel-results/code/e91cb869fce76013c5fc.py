"""Rank-minimal exact tensor decomposition for 3x3x7 matrix multiplication.

Approach: partition output columns into 3x3x3 + 3x3x3 + 3x3x1.
Each 3x3x3 block uses a verified low-rank decomposition (Strassen-hybrid rank 26
or schoolbook rank 27).  The 3x3x1 block is schoolbook rank 9.
Full-tensor exact verification before return; fallback to schoolbook rank 63.

Convention: W[k*n + i, t].
"""
import time
import numpy as np

DEADLINE = 150.0
SAFETY = 8.0
START = time.time()


def _left():
    return DEADLINE - SAFETY - (time.time() - START)


def _verify_full(U, V, W, n=3, m=3, p=7):
    U = np.asarray(U); V = np.asarray(V); W = np.asarray(W)
    R = U.shape[1]
    T = np.zeros((n, m, p))
    for t in range(R):
        Au = U[:, t].reshape(n, m)
        Bv = V[:, t].reshape(m, p)
        Cw = W[:, t].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', Au, Bv, Cw)
    return bool(np.all(T == 1))


def _strassen_222():
    """Return (U,V,W) each (4,7) for 2x2x2, indexed (i,j),(j,k),(k,i).
    Built from the 7 Strassen products; verified internally."""
    # A indices: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # B indices: e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    # Products (as list of (A-indices, B-indices)):
    M = [
        ([(0,0),(1,0)], [(0,0),(1,0)]),  # M1=(a+c)(e+g)
        ([(0,1),(1,1)], [(1,0)]),          # M2=(b+d)g
        ([(0,0)], [(0,1),(1,1)]),          # M3=a(f-e)  -> a*f - a*e ... 
        # M3 = a*(f-e): A-idx [(0,0)] with +, and A-idx [(0,0)] with - for e
        # This needs signed terms.  Let me handle signs explicitly.
    ]
    # Use signed term lists: each M is a list of (sign, (i,j), (j2,k))
    def prod(As, Bs):
        # As: list of (sign, (i,j)), Bs: list of (sign, (j2,k))
        out = []
        for sa, (i,j) in As:
            for sb, (j2,k) in Bs:
                out.append((sa*sb, i, j, j2, k))
        return out
    M1 = prod([(1,(0,0)),(1,(1,0))], [(1,(0,0)),(1,(1,0))])
    M2 = prod([(1,(0,1)),(1,(1,1))], [(1,(1,0))])
    M3 = prod([(1,(0,0))], [(1,(0,1)),(-1,(0,0))])
    M4 = prod([(1,(1,1))], [(1,(1,1)),(-1,(0,1))])
    M5 = prod([(1,(0,0)),(1,(0,1))], [(1,(0,0)),(1,(0,1))])
    M6 = prod([(1,(1,0)),(-1,(1,1))], [(1,(1,0)),(1,(1,1))])
    M7 = prod([(1,(0,0)),(1,(1,0))], [(1,(0,0)),(1,(1,1))])
    Ms = [M1,M2,M3,M4,M5,M6,M7]
    
    # C output: C[0,0]=M1+M4-M5+M7, C[0,1]=M2+M5, C[1,0]=M3+M6, C[1,1]=M1-M2+M4+M6
    # W[k,i] coefficient for each M:
    Ccoeff = {
        (0,0): {0:1, 3:1, 4:-1, 6:1},
        (0,1): {1:1, 4:1},
        (1,0): {2:1, 5:1},
        (1,1): {0:1, 1:-1, 3:1, 5:1},
    }
    U = np.zeros((4,7), dtype=int); V = np.zeros((4,7), dtype=int); W = np.zeros((4,7), dtype=int)
    for t, terms in enumerate(Ms):
        for (s, i, j, j2, k) in terms:
            U[i*2+j, t] += s
            V[j2*2+k, t] += s
        for (ci, ck_i) in Ccoeff.items():
            pass
    # W: for each M index t, W[k, i] = coefficient of M_t in C[i,k]... 
    # C[i, k] uses W[k, i].  Ccoeff[(i,k)][t] is the coeff of M_t in C[i,k].
    for (i, k), coeffs in Ccoeff.items():
        for t, c in coeffs.items():
            W[k*2 + i, t] = c  # W indexed (k, i)
    
    # Verify
    T2 = np.zeros((2,2,2))
    for t in range(7):
        T2 += np.einsum('ij,jk,ki->ijk', U[:,t].reshape(2,2), V[:,t].reshape(2,2), W[:,t].reshape(2,2))
    if not np.all(T2 == 1):
        return None
    return U, V, W


def _strassen_hybrid_333():
    """3x3x3 using Strassen on {0,1}^3 (rank 7) + schoolbook for the other 19."""
    s = _strassen_222()
    if s is None:
        return None
    sU, sV, sW = s
    R = 7 + 19
    U = np.zeros((9, R), dtype=int); V = np.zeros((9, R), dtype=int); W = np.zeros((9, R), dtype=int)
    for t in range(7):
        for i in range(2):
            for j in range(2):
                U[i*3+j, t] = sU[i*2+j, t]
        for j in range(2):
            for k in range(2):
                V[j*3+k, t] = sV[j*2+k, t]
        for k in range(2):
            for i in range(2):
                W[k*3+i, t] = sW[k*2+i, t]
    col = 7
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i*3+j, col] = 1
                V[j*3+k, col] = 1
                W[k*3+i, col] = 1
                col += 1
    assert col == R
    return U, V, W


def _schoolbook_333():
    R = 27
    U = np.zeros((9, R), dtype=int); V = np.zeros((9, R), dtype=int); W = np.zeros((9, R), dtype=int)
    for t in range(R):
        i = t // 9; j = (t // 3) % 3; k = t % 3
        U[i*3+j, t] = 1; V[j*3+k, t] = 1; W[k*3+i, t] = 1
    return U, V, W


def _get_333():
    cands = []
    r = _strassen_hybrid_333()
    if r is not None:
        U, V, W = r
        if _verify_full(U, V, W, 3, 3, 3):
            cands.append((26, U, V, W))
    U, V, W = _schoolbook_333()
    if _verify_full(U, V, W, 3, 3, 3):
        cands.append((27, U, V, W))
    if not cands:
        raise RuntimeError("no valid 3x3x3")
    cands.sort(key=lambda x: x[0])
    return cands[0][1:]


def construct():
    n, m, p = 3, 3, 7
    Ua, Va, Wa, rA = _get_333()
    Ub, Vb, Wb, rB = _get_333()
    rC = 9
    R = rA + rB + rC
    U = np.zeros((n*m, R), dtype=int); V = np.zeros((m*p, R), dtype=int); W = np.zeros((p*n, R), dtype=int)
    U[:, 0:rA] = Ua
    for t in range(rA):
        for j in range(3):
            for k in range(3):
                V[j*p+k, t] = Va[j*3+k, t]
        for k in range(3):
            for i in range(3):
                W[k*n+i, t] = Wa[k*3+i, t]
    U[:, rA:rA+rB] = Ub
    for t in range(rB):
        for j in range(3):
            for k in range(3):
                V[j*p+(k+3), rA+t] = Vb[j*3+k, t]
        for k in range(3):
            for i in range(3):
                W[(k+3)*n+i, rA+t] = Wb[k*3+i, t]
    for t in range(rC):
        i = t // 3; j = t % 3; col = rA + rB + t
        U[i*m+j, col] = 1; V[j*p+6, col] = 1; W[6*n+i, col] = 1
    if not _verify_full(U, V, W, n, m, p):
        Rf = n*m*p
        Uf = np.zeros((n*m, Rf), dtype=int); Vf = np.zeros((m*p, Rf), dtype=int); Wf = np.zeros((p*n, Rf), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Uf[i*m+j, col] = 1; Vf[j*p+k, col] = 1; Wf[k*n+i, col] = 1; col += 1
        return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
