import numpy as np
import time
import scipy.optimize

DIMENSIONS = [3, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 110.0 # Safety margin

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    schoolbook_rank = n * m * p
    
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def check_decomposition(U, V, W):
        Res = np.einsum('ar,br,cr->abc', U, V, W)
        if not np.allclose(Res, T, atol=1e-9):
            return False
        return True

    def round_to_half_int(M):
        M2 = M * 2
        M_rounded = np.round(M2)
        if not np.allclose(M2, M_rounded, atol=1e-6):
            return None
        return M_rounded / 2.0

    def solve_rank(R):
        def residuals(x):
            U = x[:nm*R].reshape(nm, R)
            V = x[nm*R:nm*R+mp*R].reshape(mp, R)
            W = x[nm*R+mp*R:].reshape(pn, R)
            Res = np.einsum('ar,br,cr->abc', U, V, W)
            return (Res - T).flatten()
        
        # Try a few random initializations
        for _ in range(5):
            if time.time() > START_TIME + TIME_LIMIT:
                return None
            x0 = np.random.randn(nm*R + mp*R + pn*R)
            try:
                res = scipy.optimize.least_squares(residuals, x0, method='lm', max_nfev=50)
                if res.cost < 1e-10:
                    U = res.x[:nm*R].reshape(nm, R)
                    V = res.x[nm*R:nm*R+mp*R].reshape(mp, R)
                    W = res.x[nm*R+mp*R:].reshape(pn, R)
                    Ur = round_to_half_int(U)
                    Vr = round_to_half_int(V)
                    Wr = round_to_half_int(W)
                    if Ur is not None and Vr is not None and Wr is not None:
                        if check_decomposition(Ur, Vr, Wr):
                            return Ur, Vr, Wr
            except Exception:
                pass
        return None

    # Try ranks from 47 down to 40
    for R in range(47, 39, -1):
        if time.time() > START_TIME + TIME_LIMIT:
            break
        start_time = time.time()
        # Limit time per rank to ensure we don't get stuck
        while time.time() - start_time < 10 and time.time() < START_TIME + TIME_LIMIT:
            sol = solve_rank(R)
            if sol is not None:
                return {"U": sol[0].tolist(), "V": sol[1].tolist(), "W": sol[2].tolist()}
        
    # Fallback to schoolbook
    U = [[0] * schoolbook_rank for _ in range(nm)]
    V = [[0] * schoolbook_rank for _ in range(mp)]
    W = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
