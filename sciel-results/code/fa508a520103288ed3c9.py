import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def _add_term(U, V, W, a_coeffs, b_coeffs, c_coeffs, col, n, m, p):
    # a_coeffs: dict (i,j)->val ; b_coeffs: (j,k)->val ; c_coeffs: (k,i)->val
    for (i, j), val in a_coeffs.items():
        U[i*m + j][col] += val
    for (j, k), val in b_coeffs.items():
        V[j*p + k][col] += val
    for (k, i), val in c_coeffs.items():
        W[k*n + i][col] += val

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    # Build candidate rank-35 decomposition
    try:
        rank = 35
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        # Strassen 2x2x2 for C[0:2,0:2]
        terms = [
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(0,1,-1),(1,0,-1),(1,1,1)]),
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
            ([(0,0,1)], [(1,0,1),(1,1,-1)], [(0,1,1)]),
            ([(1,1,1)], [(0,1,1),(0,0,-1)], [(0,1,1),(1,0,1)]),
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
            ([(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)], [(1,1,1)]),
            ([(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ]
        for a, b, c in terms:
            _add_term(U, V, W, dict((x[0],x[1]) for x in a) and { (x[0],x[1]):x[2] for x in a }, { (x[0],x[1]):x[2] for x in b }, { (x[0],x[1]):x[2] for x in c }, col, n, m, p)
            col += 1
        # row 2 of C: C[2,0],C[2,1],C[2,2] = A[2,0:2] . B[0:2,0:3]
        for j in range(2):
            for k in range(3):
                _add_term(U, V, W, {(2,j):1}, {(j,k):1}, {(k,2):1}, col, n, m, p)
                col += 1
        # col 2 of C, rows 0,1: C[0,2],C[1,2] = A[0:2,0:2] . B[0:2,2]
        for i in range(2):
            for j in range(2):
                _add_term(U, V, W, {(i,j):1}, {(j,2):1}, {(2,i):1}, col, n, m, p)
                col += 1
        # columns 3,4 of C: C[i,3],C[i,4] = A[i,0:3] . B[0:3, k]
        for i in range(3):
            for k in (3,4):
                for j in range(3):
                    _add_term(U, V, W, {(i,j):1}, {(j,k):1}, {(k,i):1}, col, n, m, p)
                    col += 1
        # verify
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    # fallback schoolbook
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

def _verify(U, V, W, n, m, p, trials=20):
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for _ in range(trials):
        A = np.random.randint(-5, 6, size=(n, m)).astype(float)
        B = np.random.randint(-5, 6, size=(m, p)).astype(float)
        C = A @ B
        # term products
        prod = (Ua.reshape(n*m, rank) @ A.reshape(n*m, 1)).reshape(n*m)
        # actually Ua @ A_flat
        Aflat = A.reshape(n*m)
        Bflat = B.reshape(m*p)
        u = Ua @ Aflat  # (rank,)
        v = Va @ Bflat  # (rank,)
        Crec = (Wa * (u * v)) @ np.ones(1)  # sum over r of W[c,r]*u[r]*v[r]
        Crec = (Wa * (u*v)).sum(axis=1)  # (p*n,)
        Crec = Crec.reshape(p, n).T  # Crec[k*n+i] -> C[i,k]
        if not np.allclose(Crec, C, atol=1e-6):
            return False
    return True
