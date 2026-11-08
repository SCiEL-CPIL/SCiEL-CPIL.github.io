"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n

    # Precompute target
    target = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0

    def err(U, V, W):
        # T = sum_c U[:,c] outer V[:,c] outer W[:,c]
        # Use einsum for speed
        T = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.sum((T - target)**2)

    def als(rank, iters=300, trials=5):
        for t in range(trials):
            if time.time() > deadline - 20:
                return None
            np.random.seed(t * 31 + rank)
            U = np.random.randn(N, rank)
            V = np.random.randn(M, rank)
            W = np.random.randn(P, rank)
            best = (float('inf'), None, None, None)
            for it in range(iters):
                if time.time() > deadline - 20:
                    break
                # Update U
                for a in range(N):
                    A = (V * W[None, :, :]).reshape(M*P, rank)  # A[b,c,col] = V[b,col]*W[c,col]
                    y = target[a].flatten()
                    try:
                        U[a] = np.linalg.lstsq(A, y, rcond=None)[0]
                    except:
                        pass
                # Update V
                for b in range(M):
                    A = (U * W[None, :, :]).reshape(N*P, rank)
                    y = target[:, b, :].reshape(N*P)
                    try:
                        V[b] = np.linalg.lstsq(A, y, rcond=None)[0]
                    except:
                        pass
                # Update W
                for c in range(P):
                    A = (U * V[None, :, :]).reshape(N*M, rank)
                    y = target[:, :, c].reshape(N*M)
                    try:
                        W[c] = np.linalg.lstsq(A, y, rcond=None)[0]
                    except:
                        pass
                if it % 20 == 0:
                    e = err(U, V, W)
                    if e < best[0]:
                        best = (e, U.copy(), V.copy(), W.copy())
                    if e < 1e-10:
                        break
            if best[0] < 1e-8:
                U_b, V_b, W_b = best[1], best[2], best[3]
                # Round to half-integers
                U_r = np.round(U_b * 2) / 2
                V_r = np.round(V_b * 2) / 2
                W_r = np.round(W_b * 2) / 2
                if err(U_r, V_r, W_r) < 1e-9:
                    return (U_r.tolist(), V_r.tolist(), W_r.tolist())
        return None

    for r in range(23, 37):
        if time.time() > deadline - 30:
            break
        res = als(r)
        if res:
            return {"U": res[0], "V": res[1], "W": res[2]}

    # Schoolbook fallback
    r = N * M * P // (m * m)  # No, just n*m*p
    r = n * m * p
    U = np.zeros((N, r)); V = np.zeros((M, r)); W = np.zeros((P, r))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1; V[j*p+k, c] = 1; W[k*n+i, c] = 1; c += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
