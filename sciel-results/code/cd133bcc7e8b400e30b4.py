import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = start_time + 135

    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_result = (U_sb, V_sb, W_sb)

    for r in [50, 52, 54, 55]:
        if time.time() > deadline:
            break
        for trial in range(3):
            if time.time() > deadline:
                break
            np.random.seed(42 + r * 10 + trial)
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            success = False
            for it in range(50):
                if time.time() > deadline:
                    break
                for a in range(n*m):
                    A_a = np.array([np.outer(V[:,rr], W[:,rr]).flatten() for rr in range(r)]).T
                    U[a] = np.linalg.lstsq(A_a, T[a].flatten(), rcond=None)[0]
                for b in range(m*p):
                    A_b = np.array([np.outer(U[:,rr], W[:,rr]).flatten() for rr in range(r)]).T
                    V[b] = np.linalg.lstsq(A_b, T[:,b,:].flatten(), rcond=None)[0]
                for c in range(p*n):
                    A_c = np.array([np.outer(U[:,rr], V[:,rr]).flatten() for rr in range(r)]).T
                    W[c] = np.linalg.lstsq(A_c, T[:,:,c].flatten(), rcond=None)[0]
                recon = sum(np.outer(U[:,rr], np.outer(V[:,rr], W[:,rr])) for rr in range(r))
                if np.max(np.abs(recon - T)) < 1e-10:
                    best_result = (U, V, W)
                    success = True
                    break
            if success:
                break

    U, V, W = best_result
    return {"U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]}
# EVOLVE-BLOCK-END
