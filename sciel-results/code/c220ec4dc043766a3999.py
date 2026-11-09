import time

DIMENSIONS = [5, 5, 5]

def _build_block_terms(i0, i1, j0, j1, k0, k1):
    # Returns list of (U_dict, V_dict, W_dict) for one 2x2 Strassen block.
    # U keyed by (i,j), V keyed by (j,k), W keyed by (k,i).
    def make_U(a=None, b=None, c=None, d=None):
        dd = {}
        if a is not None: dd[(i0,j0)] = dd.get((i0,j0),0)+a
        if b is not None: dd[(i0,j1)] = dd.get((i0,j1),0)+b
        if c is not None: dd[(i1,j0)] = dd.get((i1,j0),0)+c
        if d is not None: dd[(i1,j1)] = dd.get((i1,j1),0)+d
        return dd
    def make_V(e=None, f=None, g=None, h=None):
        dd = {}
        if e is not None: dd[(j0,k0)] = dd.get((j0,k0),0)+e
        if f is not None: dd[(j0,k1)] = dd.get((j0,k1),0)+f
        if g is not None: dd[(j1,k0)] = dd.get((j1,k0),0)+g
        if h is not None: dd[(j1,k1)] = dd.get((j1,k1),0)+h
        return dd
    def make_W(p=None, q=None, r=None, s=None):
        dd = {}
        if p is not None: dd[(k0,i0)] = dd.get((k0,i0),0)+p
        if q is not None: dd[(k1,i0)] = dd.get((k1,i0),0)+q
        if r is not None: dd[(k0,i1)] = dd.get((k0,i1),0)+r
        if s is not None: dd[(k1,i1)] = dd.get((k1,i1),0)+s
        return dd
    terms = []
    # m1
    terms.append((make_U(a=1,d=1), make_V(e=1,h=1), make_W(p=1,s=1)))
    # m2
    terms.append((make_U(c=1,d=1), make_V(e=1), make_W(r=1)))
    # m3
    terms.append((make_U(a=1), make_V(f=1,h=-1), make_W(q=1)))
    # m4
    terms.append((make_U(d=1), make_V(g=1,e=-1), make_W(p=1,r=1)))
    # m5
    terms.append((make_U(a=1,b=1), make_V(h=1), make_W(p=-1)))
    # m6
    terms.append((make_U(c=1,a=-1), make_V(e=1,f=1), make_W(s=1)))
    # m7
    terms.append((make_U(b=1,d=-1), make_V(g=1,h=1), make_W(p=1)))
    return terms

def construct():
    start = time.time()
    DEADLINE = 145.0  # safety margin before 150s
    n, m, p = DIMENSIONS
    rank = n * m * p

    # Collect rank-1 terms as dicts: U: {(i,j):val}, V: {(j,k):val}, W: {(k,i):val}
    terms = []

    # Disjoint 2x2 Strassen blocks
    blocks = [((0,1),(0,1),(0,1)), ((2,3),(2,3),(2,3))]
    block_indices = set()
    for (i0,i1),(j0,j1),(k0,k1) in blocks:
        for i in (i0,i1):
            for j in (j0,j1):
                for k in (k0,k1):
                    block_indices.add((i,j,k))
        terms.extend(_build_block_terms(i0,i1,j0,j1,k0,k1))

    # Schoolbook for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i,j,k) in block_indices:
                    continue
                terms.append(
                    ({(i,j):1}, {(j,k):1}, {(k,i):1})
                )

    # Build matrices
    rank = len(terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for c, (ud, vd, wd) in enumerate(terms):
        for (i,j), val in ud.items():
            U[i*m+j][c] += val
        for (j,k), val in vd.items():
            V[j*p+k][c] += val
        for (k,i), val in wd.items():
            W[k*n+i][c] += val

    # Verify exactly (integer arithmetic)
    import numpy as np
    Ua = np.array(U, dtype=object)
    Va = np.array(V, dtype=object)
    Wa = np.array(W, dtype=object)
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                if s != 1:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break

    if not ok:
        # Fallback to schoolbook
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
