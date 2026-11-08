import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 110  # safety margin before 120s
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)  # (8, 28)
    V = np.zeros((m * p, rank), dtype=int)  # (16, 28)
    W = np.zeros((p * n, rank), dtype=int)  # (8, 28)
    
    strassen = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def apply_block(a_col_off, b_row_off, b_col_off, c_col_off, start_col):
        for t, (u_loc, v_loc, w_loc) in enumerate(strassen):
            col = start_col + t
            for loc, coeff in u_loc.items():
                lr, lc = divmod(loc, 2)
                gr, gc = lr, a_col_off * 2 + lc
                U[gr * m + gc, col] += coeff
            for loc, coeff in v_loc.items():
                lr, lc = divmod(loc, 2)
                gr, gc = b_row_off * 2 + lr, b_col_off * 2 + lc
                V[gr * p + gc, col] += coeff
            for loc, coeff in w_loc.items():
                lr, lc = divmod(loc, 2)
                gr, gc = lr, c_col_off * 2 + lc
                W[gc * n + gr, col] += coeff
    
    apply_block(0, 0, 0, 0, 0)
    apply_block(1, 1, 0, 0, 7)
    apply_block(0, 0, 1, 1, 14)
    apply_block(1, 1, 1, 1, 21)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    np.random.seed(42)
    for trial in range(100):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        Cf = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                acc = 0.0
                for r in range(U.shape[1]):
                    for j in range(m):
                        acc += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                Cf[i, k] = acc
        assert np.allclose(C, Cf, atol=1e-10), (trial, C, Cf)
    
    print("OK rank", U.shape[1])
