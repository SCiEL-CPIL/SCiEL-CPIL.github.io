import time
import numpy as np

DEADLINE = 145.0

def construct():
    t0 = time.time()
    n, m, p = 2, 4, 7
    Na, Nb, Nc = n * m, m * p, p * n

    def _build_target():
        tgt = np.zeros((Na, Nb, Nc))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tgt[i * m + j, j * p + k, k * n + i] = 1.0
        return tgt

    target = _build_target()
    tf = target.ravel()

    def _strassen_222():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
        U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
        U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
        return U, V, W

    def _school_221():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        U[1,1]=1; V[1,1]=1; W[0,1]=1
        U[2,2]=1; V[0,2]=1; W[1,2]=1
        U[3,3]=1; V[1,3]=1; W[1,3]=1
        return U, V, W

    def _assemble_base():
        U = np.zeros((Na, 50))
        V = np.zeros((Nb, 50))
        W = np.zeros((Nc, 50))
        sU, sV, sW = _strassen_222()
        hU, hV, hW = _school_221()

        def _place_S(ar, ac, br, bc, cr, cc, c0):
            for rr in range(7):
                for u in range(4):
                    if sU[u,rr]:
                        U[ar[u]*m+ac[u], c0+rr] += sU[u,rr]
                for v in range(4):
                    if sV[v,rr]:
                        V[br[v]*p+bc[v], c0+rr] += sV[v,rr]
                for w in range(4):
                    if sW[w,rr]:
                        W[cc[w]*n+cr[w], c0+rr] += sW[w,rr]
            return c0+7

        def _place_H(ar, ac, br, bc, cr, cc, c0):
            for rr in range(4):
                for u in range(4):
                    if hU[u,rr]:
                        U[ar[u]*m+ac[u], c0+rr] += hU[u,rr]
                for v in range(2):
                    if hV[v,rr]:
                        V[br[v]*p+bc[v], c0+rr] += hV[v,rr]
                for w in range(2):
                    if hW[w,rr]:
                        W[cc[w]*n+cr[w], c0+rr] += hW[w,rr]
            return c0+4

        c = 0
        c = _place_S([0,1],[0,1],[0,1],[0,1],[0,1],[0,1], c)
        c = _place_H([0,1],[0,1],[0,1],[2],[0,1],[2], c)
        c = _place_S([0,1],[2,3],[2,3],[0,1],[0,1],[0,1], c)
        c = _place_H([0,1],[2,3],[2,3],[2],[0,1],[2], c)
        c = _place_S([0,1],[0,1],[0,1],[3,4],[0,1],[3,4], c)
        c = _place_S([0,1],[0,1],[0,1],[5,6],[0,1],[5,6], c)
        c = _place_S([0,1],[2,3],[2,3],[3,4],[0,1],[3,4], c)
        c = _place_S([0,1],[2,3],[2,3],[5,6],[0,1],[5,6], c)
        return U, V, W

    def _verify(U, V, W, r):
        for rr in range(r):
            ucol = U[:,rr]
            vcol = V[:,rr]
            wcol = W[:,rr]
            nz_u = np.nonzero(ucol)[0]
            nz_v = np.nonzero(vcol)[0]
            nz_w = np.nonzero(wcol)[0]
            if len(nz_u)==0 or len(nz_v)==0 or len(nz_w)==0:
                continue
            for u in nz_u:
                for v in nz_v:
                    base = u*Nb*Nc + v*Nc
                    for w in nz_w:
                        tf[base+w] += ucol[u]*vcol[v]*wcol[w]
        return np.allclose(tf, target.ravel(), atol=1e-9)

    def _term_supports(U, V, W, R):
        sup = []
        for rr in range(R):
            su = set(np.nonzero(U[:,rr])[0].tolist())
            sv = set(np.nonzero(V[:,rr])[0].tolist())
            sw = set(np.nonzero(W[:,rr])[0].tolist())
            sup.append((su, sv, sw))
        return sup

    def _try_reduce(U0, V0, W0, R):
        if time.time() - t0 > DEADLINE - 12:
            return None
        sups = _term_supports(U0, V0, W0, R)
        for e in range(R):
            if time.time() - t0 > DEADLINE - 10:
                return None
            for f in range(R):
                if e == f:
                    continue
                se, sf = sups[e], sups[f]
                if not (se[0] & sf[0] and se[1] & sf[1] and se[2] & sf[2]):
                    continue
                U = U0.copy()
                V = V0.copy()
                W = W0.copy()
                U[:,e] = U[:,e] + U[:,f]
                V[:,e] = V[:,e] + V[:,f]
                W[:,e] = W[:,e] + W[:,f]
                U[:,f] = 0.0
                V[:,f] = 0.0
                W[:,f] = 0.0
                if _verify(U, V, W, R):
                    nz = np.all(U != 0, axis=0) | np.all(V != 0, axis=0) | np.all(W != 0, axis=0)
                    idx = np.where(nz)[0]
                    return U[:,idx].tolist(), V[:,idx].tolist(), W[:,idx].tolist()
        return None

    U, V, W = _assemble_base()
    R = 50
    if not _verify(U, V, W, R):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best = (U.tolist(), V.tolist(), W.tolist())
    res = _try_reduce(U, V, W, R)
    if res is not None:
        U, V, W = res
        best = (U, V, W)

    if time.time() - t0 < DEADLINE - 10:
        res2 = _try_reduce(np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float), len(U[0]))
        if res2 is not None:
            U, V, W = res2
            best = (U, V, W)

    return {"U": U, "V": V, "W": W}
