def construct():
    import time
    import numpy as np

    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = 3, 4, 8

    # Target tensor T[a,b,c], a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def build_schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    def check(U, V, W, tol=1e-7):
        recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(U.shape[1]):
            recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.max(np.abs(recon - T)) < tol

    def als(r, iters=200, seed=0):
        if time.time() > deadline - 8:
            return None
        rng = np.random.RandomState(seed)
        U = rng.randn(n*m, r)
        V = rng.randn(m*p, r)
        W = rng.randn(p*n, r)
        for r_ in range(r):
            nrm = np.sqrt(np.sum(U[:,r_]**2)*np.sum(V[:,r_]**2)*np.sum(W[:,r_]**2))
            if nrm > 1e-12:
                s = nrm ** (1.0/3.0)
                U[:,r_] /= s; V[:,r_] /= s; W[:,r_] /= s
        for it in range(iters):
            if time.time() > deadline - 8:
                return None
            VW = np.kron(V, W)
            UW = np.kron(U, W)
            UV = np.kron(U, V)
            Tr_a = T.reshape(n*m, m*p*p*n)
            Tr_b = T.reshape(m*p, n*m*p*n)
            Tr_c = T.reshape(p*n, n*m*m*p)
            U = Tr_a @ np.linalg.pinv(VW)
            V = Tr_b @ np.linalg.pinv(UW)
            W = Tr_c @ np.linalg.pinv(UV)
        if check(U, V, W):
            return U, V, W
        return None

    best = None
    for r in [48, 60, 64, 72, 80, 88, 96]:
        if time.time() > deadline - 12:
            break
        found = False
        for seed in range(4):
            res = als(r, seed=seed)
            if res is not None:
                U, V, W = res
                if best is None or r < best[0]:
                    best = (r, U, V, W)
                found = True
                break
        if found:
            # found a valid rank r; try smaller already done by loop order
            pass

    if best is not None:
        r, U, V, W = best
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U, V, W = build_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
