import numpy as np

def construct():
    n, m, p = 2, 3, 4
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i, j, k] = 1 if (i % m == j // p) and (j % p == k // n)
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n * m):
        i1, i2 = divmod(i, m)
        for j in range(m * p):
            j1, j2 = divmod(j, p)
            if i2 == j1:
                for k in range(p * n):
                    k1, k2 = divmod(k, n)
                    if j2 == k2:
                        T[i, j, k] = 1.0

    T_flat_u = T.reshape(n * m, -1)
    T_flat_v = T.transpose(1, 0, 2).reshape(m * p, -1)
    T_flat_w = T.transpose(2, 0, 1).reshape(p * n, -1)

    def try_rank(r, seed):
        np.random.seed(seed)
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        for _ in range(200): # Increase iterations for better convergence
            for c in range(r):
                vw = (V[:, c] * W[:, c]).flatten()
                den = np.dot(vw, vw)
                if den < 1e-10:
                    continue
                U[:, c] = T_flat_u @ vw / den
            
            for c in range(r):
                uw = (U[:, c] * W[:, c]).flatten()
                den = np.dot(uw, uw)
                if den < 1e-10:
                    continue
                V[:, c] = T_flat_v @ uw / den
            
            for c in range(r):
                uv = (U[:, c] * V[:, c]).flatten()
                den = np.dot(uv, uv)
                if den < 1e-10:
                    continue
                W[:, c] = T_flat_w @ uv / den

        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        T_recon = np.zeros_like(T)
        for c in range(r):
            T_recon += np.einsum('i,j,k->ijk', U_rounded[:, c], V_rounded[:, c], W_rounded[:, c])
        
        if np.allclose(T_recon, T, atol=1e-9):
            return {
                "U": U_rounded.tolist(),
                "V": V_rounded.tolist(),
                "W": W_rounded.tolist()
            }
        return None

    # Try ranks 20, 21, 22, 23
    for r in [20, 21, 22, 23]:
        for seed in range(5):
            res = try_rank(r, seed)
            if res is not None:
                return res

    # Fallback to schoolbook rank 24
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
