import time
import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    DEADLINE = 110.0

    def to_actual(doubled_2d):
        # doubled_2d: list of lists of ints (2x value)
        out = []
        for row in doubled_2d:
            nr = []
            for x in row:
                if x % 2 == 0:
                    nr.append(x // 2)
                else:
                    nr.append(x / 2.0)
            out.append(nr)
        return out

    def schoolbook():
        rank = n*m*p
        Ud = [[0]*rank for _ in range(n*m)]
        Vd = [[0]*rank for _ in range(m*p)]
        Wd = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Ud[i*m+j][col] = 2
                    Vd[j*p+k][col] = 2
                    Wd[k*n+i][col] = 2
                    col += 1
        return {"U": to_actual(Ud), "V": to_actual(Vd), "W": to_actual(Wd)}

    def verify_d(Ud, Vd, Wd):
        rank = len(Ud[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += Ud[i*m+j][r] * Vd[j*p+k][r] * Wd[k*n+i][r]
                    if s != 8:
                        return False
        return True

    def search():
        # candidate ranks to try (descending), limited
        rng = np.random.default_rng(12345)
        ranks = list(range(35, 11, -1))  # 35 down to 12
        # We'll iterate with a global budget.
        best = None
        # Precompute row/col mapping
        # rows: (i,j,k) -> idx
        def row_idx(i, j, k):
            return (i * p + j) * p + k  # wait p for k count
        # Actually k in [0,p), j in [0,m), i in [0,n)
        # row index = i*(m*p) + j*p + k
        def ridx(i, j, k):
            return i * (m * p) + j * p + k
        num_rows = n * m * p
        for r in ranks:
            if time.time() - start > DEADLINE:
                break
            num_cols = p * n * r
            trials = 40
            for t in range(trials):
                if time.time() - start > DEADLINE:
                    break
                # random Ud, Vd in {-2,-1,0,1,2}
                Ud = rng.integers(-2, 3, size=(n*m, r)).tolist()
                Vd = rng.integers(-2, 3, size=(m*p, r)).tolist()
                # Build A (num_rows x num_cols)
                A = np.zeros((num_rows, num_cols), dtype=np.int64)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row = ridx(i, j, k)
                            wbase = k * n + i  # w_idx
                            for rr in range(r):
                                val = Ud[i*m+j][rr] * Vd[j*p+k][rr]
                                if val != 0:
                                    A[row, wbase * r + rr] = val
                b = np.full(num_rows, 8, dtype=np.float64)
                # Solve least squares
                try:
                    Wd_real, *_ = np.linalg.lstsq(A, b, rcond=None)
                except Exception:
                    continue
                Wd_int = np.rint(Wd_real).astype(np.int64)
                # check residual with integer
                resid = A @ Wd_int - b
                if np.max(np.abs(resid)) < 1e-6:
                    # convert to 2D
                    Wd2 = Wd_int.reshape(p*n, r).tolist()
                    if verify_d(Ud, Vd, Wd2):
                        best = (r, Ud, Vd, Wd2)
                        break
            if best is not None:
                break
        if best is not None:
            r, Ud, Vd, Wd = best
            return {"U": to_actual(Ud), "V": to_actual(Vd), "W": to_actual(Wd)}
        return None

    res = search()
    if res is None:
        res = schoolbook()
    return res
